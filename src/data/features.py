"""Feature engineering for tree-based and recurrent return-forecasting models.

Builds a stacked (date x asset) panel of strictly-lagged features. Every
feature at time t for asset i depends only on data observed at-or-before
t-1 (returns, VIX, regime), or on calendar metadata at t (which is known
at the start of trading on day t).

Feature groups (window/lag constants live in src/config.py):

- Own-asset lagged returns: short-term reversal, weekly/monthly momentum.
- Own-asset rolling stats: local volatility, mean drift, skew.
- VIX features: implied-vol level and multi-horizon changes.
- Regime label: discrete low/medium/high VIX bucket.
- Cross-asset features: market-wide return, dispersion, this asset's spread.
- Calendar: month-of-year, day-of-week (seasonality).
- Asset id: categorical, lets a global model learn per-asset adjustments.

The panel is the input to src/models/gbt.py.
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    CROSS_ASSET_WINDOW,
    RETURN_LAGS,
    ROLLING_MEAN_WINDOWS,
    ROLLING_SKEW_WINDOWS,
    ROLLING_VOL_WINDOWS,
    VIX_DELTA_HORIZONS,
)

logger = logging.getLogger(__name__)


def _broadcast_series_to_panel(
    series: pd.Series,
    columns: pd.Index,
) -> pd.DataFrame:
    """Broadcast a (date) Series to a (date x asset) DataFrame with given columns."""
    arr = np.asarray(series.values).reshape(-1, 1)
    broadcast = np.broadcast_to(arr, (len(series), len(columns)))
    return pd.DataFrame(broadcast, index=series.index, columns=columns)


def build_feature_panel(
    log_returns: pd.DataFrame,
    vix: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Build a stacked (date x asset) feature panel.

    All features are strictly lagged. The row at (date=t, asset=i) uses
    only returns / VIX / regime values observed at-or-before t-1. Calendar
    features come from t itself.

    Args:
        log_returns: DataFrame of daily log returns (T x N).
        vix: Series of daily VIX values aligned to ``log_returns.index``.
        regimes: Series of regime labels ("low", "medium", "high") aligned
            to ``log_returns.index``.

    Returns:
        DataFrame indexed by MultiIndex(date, asset). Warmup rows where the
        longest lookback is undefined are dropped, leaving a NaN-free panel.
    """
    logger.info(
        "Building feature panel: %d dates, %d assets",
        len(log_returns),
        log_returns.shape[1],
    )

    feature_frames: dict[str, pd.DataFrame] = {}
    asset_cols = log_returns.columns

    for k in RETURN_LAGS:
        feature_frames[f"ret_lag_{k}"] = log_returns.shift(k)

    for w in ROLLING_VOL_WINDOWS:
        feature_frames[f"vol_{w}"] = log_returns.rolling(window=w).std().shift(1)
    for w in ROLLING_MEAN_WINDOWS:
        feature_frames[f"mean_{w}"] = log_returns.rolling(window=w).mean().shift(1)
    for w in ROLLING_SKEW_WINDOWS:
        feature_frames[f"skew_{w}"] = log_returns.rolling(window=w).skew().shift(1)

    market_window_returns = (
        log_returns.rolling(window=CROSS_ASSET_WINDOW).mean().shift(1)
    )
    market_mean = market_window_returns.mean(axis=1)
    dispersion = market_window_returns.std(axis=1)
    feature_frames[f"market_mean_{CROSS_ASSET_WINDOW}"] = _broadcast_series_to_panel(
        market_mean, asset_cols
    )
    feature_frames[f"dispersion_{CROSS_ASSET_WINDOW}"] = _broadcast_series_to_panel(
        dispersion, asset_cols
    )
    feature_frames[f"spread_vs_market_{CROSS_ASSET_WINDOW}"] = (
        market_window_returns.sub(market_mean, axis=0)
    )

    vix_lagged = vix.shift(1)
    feature_frames["vix_level"] = _broadcast_series_to_panel(vix_lagged, asset_cols)
    for h in VIX_DELTA_HORIZONS:
        feature_frames[f"vix_delta_{h}"] = _broadcast_series_to_panel(
            vix.diff(h).shift(1), asset_cols
        )

    long_frames: dict[str, pd.Series] = {}
    for name, wide in feature_frames.items():
        stacked = wide.stack()
        stacked.index.names = ["date", "asset"]
        long_frames[name] = stacked

    panel = pd.DataFrame(long_frames)
    panel = panel.reset_index()

    regime_lagged = regimes.shift(1)
    panel["regime"] = panel["date"].map(regime_lagged)
    panel["month"] = panel["date"].dt.month.astype("int8")
    panel["dow"] = panel["date"].dt.dayofweek.astype("int8")
    panel["asset_id"] = pd.Categorical(
        panel["asset"], categories=list(asset_cols)
    )
    panel["regime"] = pd.Categorical(
        panel["regime"], categories=["low", "medium", "high"]
    )

    panel = panel.set_index(["date", "asset"]).sort_index()

    pre_drop = len(panel)
    # Drop rows where any non-categorical feature is NaN (categorical NaNs
    # would be retained by dropna because of how pandas handles categories).
    numeric_cols = [c for c in panel.columns if panel[c].dtype.name != "category"]
    panel = panel.dropna(subset=numeric_cols)
    # Drop any remaining rows where regime is NaN (warmup of the regime series).
    panel = panel[panel["regime"].notna()]
    logger.info(
        "Feature panel: %d rows after dropping %d warmup rows",
        len(panel),
        pre_drop - len(panel),
    )

    return panel
