"""Linear-regression mu estimator — the simplest ML arm.

Sits between the no-learning baseline (sample mean) and the non-linear
arms (GBT, LSTM) on the model-complexity ladder. Trained on the *same*
feature panel that the GBT consumes, with the *same* 21-day-forward target.

The only thing that differs across arms is the function class:
    sample_mean : flat constant per asset
    linreg      : linear function of features
    gbt         : ensemble of decision trees
    lstm        : recurrent neural net over return sequences

Categorical features (regime, asset_id) are one-hot encoded. The single
global model pools all (date, asset) rows so the linear coefficients are
shared across assets except where ``asset_id`` dummies modulate them.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import GBT_FORECAST_HORIZON

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def _one_hot_encode(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns; pass numeric columns through."""
    return pd.get_dummies(X, columns=["regime", "asset_id"], drop_first=False)


def train_linreg(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 1.0,
) -> tuple[Ridge, list[str]]:
    """Fit a Ridge regression on the (date x asset) feature panel.

    Ridge (L2-regularized linear regression) is used instead of OLS to keep
    coefficients well-conditioned when one-hot encoding produces collinear
    dummies. ``alpha=1.0`` is a mild default — small enough to behave like
    OLS on uncorrelated features, large enough to absorb the dummy
    collinearity.

    Args:
        X: Feature DataFrame indexed by MultiIndex(date, asset).
        y: Target Series with the same index.
        alpha: Ridge regularization strength.

    Returns:
        Tuple of (fitted Ridge model, ordered list of feature column names
        — needed at prediction time so the encoded matrix has identical
        columns as at training).
    """
    X_enc = _one_hot_encode(X).astype(float)
    feature_cols = X_enc.columns.tolist()
    model = Ridge(alpha=alpha)
    model.fit(X_enc.values, y.values)
    logger.info(
        "Trained Ridge: rows=%d, features=%d, alpha=%.3f",
        len(X_enc), len(feature_cols), alpha,
    )
    return model, feature_cols


def linreg_mu_estimator(
    model: Ridge,
    feature_cols: list[str],
    full_panel: pd.DataFrame,
    asset_order: list[str],
    horizon: int = GBT_FORECAST_HORIZON,
) -> Callable[[pd.DataFrame, pd.Timestamp], np.ndarray]:
    """Build a mu-estimator callable for ``rolling_backtest``.

    Args:
        model: Fitted Ridge model.
        feature_cols: Column ordering used at training. Predictions
            reindex to this same ordering to handle one-hot mismatch
            between train and test slices.
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

        X_enc = _one_hot_encode(row).astype(float)
        # Align prediction-time columns with training-time ordering.
        X_enc = X_enc.reindex(columns=feature_cols, fill_value=0.0)
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
