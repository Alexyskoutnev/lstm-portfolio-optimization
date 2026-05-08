"""Gradient-boosting tree (LightGBM) model for forecasting forward returns.

Why this exists
---------------
Classical Markowitz uses the *sample mean* of historical returns as the
expected-return vector mu. The sample mean is unbiased but extremely noisy:
with 252 daily observations, the standard error on mu is roughly
sigma / sqrt(252), which for a 20% annual-vol ETF is ~1.3% — often larger
than the signal itself. Markowitz then *amplifies* this noise (Michaud
1989, "error maximization"), producing portfolios that look great on
training data and collapse out-of-sample.

This module replaces the sample mean with a learned, conditional mu. A
single global LightGBM regressor is trained on a stacked (date x asset)
feature panel with a forward-return target. Features include lagged
returns, rolling vol, VIX level/changes, regime labels, cross-asset
signals, and a categorical asset id. The trained model exposes a
``gbt_mu_estimator(...)`` callable that plugs into ``rolling_backtest``
exactly where the sample-mean estimator would.

Sigma stays sample-based across all three project arms (classical MVO,
GBT-MVO, LSTM-MVO) so that the comparison isolates the contribution of mu.
"""

import logging
from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import GBT_DEFAULT_PARAMS, GBT_FORECAST_HORIZON

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def build_target(
    log_returns: pd.DataFrame,
    horizon: int = GBT_FORECAST_HORIZON,
) -> pd.Series:
    """Build a stacked forward-return target series.

    For each (date t, asset i), the target is the cumulative log return
    over [t+1, t+horizon]. Rows where the full target window is not yet
    observed are dropped.

    Args:
        log_returns: DataFrame of daily log returns (T x N).
        horizon: Forecast horizon in trading days.

    Returns:
        Series indexed by MultiIndex(date, asset) with the cumulative
        forward log return as the value.
    """
    forward = log_returns.rolling(window=horizon).sum().shift(-horizon)
    forward = forward.dropna(how="any")
    target = forward.stack()
    target.index.names = ["date", "asset"]
    target.name = "target"
    return target


def train_gbt(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    val_fraction: float = 0.2,
) -> lgb.Booster:
    """Train a LightGBM regressor on the (date x asset) panel.

    Uses a simple time-based hold-out for early stopping: the last
    ``val_fraction`` of unique dates becomes the validation set.

    Args:
        X: Feature DataFrame indexed by MultiIndex(date, asset).
        y: Target Series indexed identically to X.
        params: Optional override of LightGBM hyperparameters.
        val_fraction: Fraction of the most recent dates held out for
            early stopping.

    Returns:
        Fitted LightGBM Booster.
    """
    params = dict(GBT_DEFAULT_PARAMS) if params is None else dict(params)
    num_iterations = params.pop("num_iterations", 200)
    early_stopping_round = params.pop("early_stopping_round", None)

    categorical_features = ["regime", "asset_id"]

    if early_stopping_round:
        # Optional path: caller explicitly opted into early stopping. The
        # default training path skips it because our natural val window
        # overlaps COVID and would cut training short.
        unique_dates = X.index.get_level_values("date").unique().sort_values()
        cutoff_idx = max(1, int(len(unique_dates) * (1 - val_fraction)))
        cutoff = unique_dates[cutoff_idx]
        date_level = X.index.get_level_values("date")
        train_mask = date_level < cutoff
        val_mask = ~train_mask
        train_set = lgb.Dataset(
            X[train_mask], label=y[train_mask],
            categorical_feature=categorical_features,
        )
        val_set = lgb.Dataset(
            X[val_mask], label=y[val_mask], reference=train_set,
            categorical_feature=categorical_features,
        )
        logger.info(
            "Training LightGBM: train=%d, val=%d, num_iter=%d, early_stop=%d",
            int(train_mask.sum()), int(val_mask.sum()),
            num_iterations, early_stopping_round,
        )
        model = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=num_iterations,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(early_stopping_round, verbose=False)],
        )
    else:
        # Default path: no early stopping. Train for fixed num_iterations on
        # all data, rely on L2 regularization + bagging to avoid overfit.
        train_set = lgb.Dataset(
            X, label=y, categorical_feature=categorical_features,
        )
        logger.info(
            "Training LightGBM: rows=%d, num_iter=%d (no early stopping)",
            len(X), num_iterations,
        )
        model = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=num_iterations,
        )

    logger.info(
        "LightGBM trained: num_trees=%d, best_iteration=%s",
        model.num_trees(), model.best_iteration,
    )
    return model


def predict_returns(
    model: lgb.Booster,
    X: pd.DataFrame,
) -> pd.Series:
    """Predict raw forward returns for each row in X.

    Args:
        model: Fitted LightGBM Booster.
        X: Feature DataFrame indexed by MultiIndex(date, asset).

    Returns:
        Series of predicted ``horizon``-day cumulative log returns,
        indexed identically to X.
    """
    raw = model.predict(X, num_iteration=model.best_iteration)
    return pd.Series(np.asarray(raw), index=X.index, name="prediction")


def gbt_mu_estimator(
    model: lgb.Booster,
    full_panel: pd.DataFrame,
    asset_order: list[str],
    horizon: int = GBT_FORECAST_HORIZON,
) -> Callable[[pd.DataFrame, pd.Timestamp], np.ndarray]:
    """Build a mu-estimator callable compatible with ``rolling_backtest``.

    The returned callable signature is::

        mu_estimator(window_returns: pd.DataFrame,
                     current_date: pd.Timestamp) -> np.ndarray

    matching the interface ``rolling_backtest`` expects. ``window_returns``
    is ignored — the GBT consumes pre-built features from ``full_panel``
    keyed by ``current_date``.

    Args:
        model: Fitted LightGBM Booster.
        full_panel: Pre-built feature panel covering all dates the
            backtest may query.
        asset_order: Canonical asset ordering returned to MVO. Must match
            the column order of the returns DataFrame fed to
            ``rolling_backtest``.
        horizon: Forecast horizon in trading days. Used to annualize the
            raw prediction.

    Returns:
        Callable that takes (window_returns, current_date) and returns
        an annualized mu vector of shape (len(asset_order),).
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
                "GBT estimator: no panel row for %s; falling back to zero mu",
                current_date,
            )
            return np.zeros(len(asset_order))

        preds = predict_returns(model, row)
        preds = preds.reindex(asset_order)
        if preds.isna().any():
            missing = preds.index[preds.isna()].tolist()
            raise ValueError(
                f"GBT estimator missing predictions for assets {missing} "
                f"on {current_date}"
            )
        return np.asarray(preds.values) * annualization

    return estimator
