"""Linear-regression mu estimator — the simplest ML arm.

Sits between the no-learning baseline (sample mean) and the non-linear
arms (GBT, LSTM) on the model-complexity ladder. Trained on the *same*
feature panel that the GBT consumes, with the *same* 21-day-forward target.

The only thing that differs across arms is the function class:
    sample_mean : flat constant per asset
    linreg      : linear function of features (this module)
    gbt         : ensemble of decision trees
    lstm        : recurrent neural net over return sequences

What this module does carefully — and *why*
-------------------------------------------
A naive Ridge is a weak baseline because the feature panel mixes scales
that span four orders of magnitude (returns ≈ 0.01, VIX ≈ 20, dispersion
≈ 0.005). Without standardization the L2 penalty falls almost entirely
on the small-scale features, leaving the linear model unable to use
return-derived signals at all.

We address this by:

1. **Standardizing all numeric features** (z-score, fit on train only).
2. **Selecting alpha via RidgeCV** with leave-one-fold-out on a grid.
3. **Adding hand-picked interaction features** — regime x momentum,
   VIX x momentum, asset x momentum — to give the linear model some of
   the non-linear structure GBT learns from splits.

Categorical features (regime, asset_id) are one-hot encoded. The single
global model pools all (date, asset) rows so coefficients are shared
across assets except where ``asset_id`` dummies modulate the intercept
and the asset x momentum interactions.
"""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.config import GBT_FORECAST_HORIZON

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def _add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Add hand-picked interaction features to the panel.

    Linear models cannot discover interactions on their own. We feed the
    most theoretically motivated pairs as explicit cross-features:

    - **regime x momentum:** the marginal effect of past returns may flip
      sign across volatility regimes (mean-reversion in calm, momentum
      in high-vol).
    - **VIX level x own momentum:** scaling momentum by current implied
      vol captures "how much should I trust this trend signal."
    - **VIX delta x dispersion:** rising VIX in a high-dispersion market
      tends to precede sector rotation.

    Args:
        X: Feature DataFrame indexed by MultiIndex(date, asset).

    Returns:
        Same DataFrame with added interaction columns. Categorical
        columns are left as-is for downstream one-hot encoding.
    """
    X = X.copy()
    if "vix_level" in X.columns and "ret_lag_21" in X.columns:
        X["vix_x_mom21"] = X["vix_level"] * X["ret_lag_21"]
    if "vix_delta_5" in X.columns and "dispersion_21" in X.columns:
        X["vixchg_x_disp"] = X["vix_delta_5"] * X["dispersion_21"]
    if "ret_lag_21" in X.columns and "ret_lag_63" in X.columns:
        X["mom21_x_mom63"] = X["ret_lag_21"] * X["ret_lag_63"]
    if "vol_21" in X.columns and "ret_lag_21" in X.columns:
        X["mom21_x_vol21"] = X["ret_lag_21"] * X["vol_21"]
    return X


def _one_hot_encode(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns; pass numeric columns through."""
    return pd.get_dummies(X, columns=["regime", "asset_id"], drop_first=False)


def train_linreg(
    X: pd.DataFrame,
    y: pd.Series,
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
) -> tuple[RidgeCV, StandardScaler, list[str], list[str]]:
    """Fit a standardized Ridge with cross-validated alpha.

    Pipeline:
      1. Add hand-picked interaction features.
      2. One-hot encode categoricals.
      3. Standardize ALL numeric columns (one-hot stays binary).
      4. Fit RidgeCV with the given alpha grid (leave-one-out CV
         internally — fast and well-suited to small grids).

    Args:
        X: Feature DataFrame indexed by MultiIndex(date, asset).
        y: Target Series with the same index.
        alpha_grid: Grid of L2 regularization strengths to search.

    Returns:
        Tuple of (fitted RidgeCV, fitted StandardScaler, list of
        numeric column names that were standardized, full feature
        column ordering).
    """
    X_int = _add_interactions(X)
    X_enc = _one_hot_encode(X_int).astype(float)
    feature_cols = X_enc.columns.tolist()

    # Numeric columns = everything that wasn't a one-hot dummy. The dummies
    # are bool-like 0/1; we leave them alone so coefficients on regime/asset
    # are interpretable as level shifts.
    onehot_cols = [c for c in feature_cols if c.startswith(("regime_", "asset_id_"))]
    numeric_cols = [c for c in feature_cols if c not in onehot_cols]

    scaler = StandardScaler()
    X_scaled = X_enc.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X_enc[numeric_cols].values)

    model = RidgeCV(alphas=alpha_grid)
    model.fit(X_scaled.values, y.values)
    logger.info(
        "Trained RidgeCV: rows=%d, features=%d, best_alpha=%.4f",
        len(X_scaled), len(feature_cols), model.alpha_,
    )
    return model, scaler, numeric_cols, feature_cols


def linreg_mu_estimator(
    model: RidgeCV,
    scaler: StandardScaler,
    numeric_cols: list[str],
    feature_cols: list[str],
    full_panel: pd.DataFrame,
    asset_order: list[str],
    horizon: int = GBT_FORECAST_HORIZON,
) -> Callable[[pd.DataFrame, pd.Timestamp], np.ndarray]:
    """Build a mu-estimator callable for ``rolling_backtest``.

    Replicates the train-time preprocessing exactly: add interactions,
    one-hot encode, standardize numeric columns using the train scaler,
    align columns to the train ordering, predict, annualize.

    Args:
        model: Fitted RidgeCV model.
        scaler: Fitted StandardScaler from training.
        numeric_cols: Column names that were standardized (used to
            apply the scaler at prediction time without touching one-hot
            dummies).
        feature_cols: Full column ordering used at training.
        full_panel: Pre-built feature panel.
        asset_order: Canonical asset ordering returned to MVO.
        horizon: Forecast horizon used for annualization.

    Returns:
        Callable matching the rolling_backtest mu_estimator interface.
    """
    annualization = _TRADING_DAYS_PER_YEAR / float(horizon)

    def estimator(
        window_returns: pd.DataFrame,  # noqa: ARG001 — interface symmetry
        current_date: pd.Timestamp,
    ) -> np.ndarray:
        try:
            row = full_panel.xs(current_date, level="date")
        except KeyError:
            logger.warning(
                "Linreg estimator: no panel row for %s; falling back to zero mu",
                current_date,
            )
            return np.zeros(len(asset_order))

        X_int = _add_interactions(row)
        X_enc = _one_hot_encode(X_int).astype(float)
        X_enc = X_enc.reindex(columns=feature_cols, fill_value=0.0)
        # Standardize numeric columns using the training scaler.
        X_enc[numeric_cols] = scaler.transform(X_enc[numeric_cols].values)
        preds = pd.Series(
            np.asarray(model.predict(X_enc.values)),
            index=row.index,
        )
        preds = preds.reindex(asset_order)
        if preds.isna().any():
            missing = preds.index[preds.isna()].tolist()
            raise ValueError(
                f"Linreg estimator missing predictions for assets {missing} "
                f"on {current_date}"
            )
        return np.asarray(preds.values) * annualization

    return estimator
