"""Volatility regime identification for financial time series.

Provides two approaches for labeling market regimes:
1. VIX-based thresholds (direct fear gauge)
2. Rolling standard deviation of returns (model-free)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Annualization factor: daily volatility -> annual volatility.
# sqrt(252) because there are ~252 trading days per year and
# volatility scales with the square root of time.
_ANNUALIZATION_FACTOR = np.sqrt(252)


def label_regimes_vix(
    vix: pd.Series,
    low_threshold: float = 15.0,
    high_threshold: float = 25.0,
) -> pd.Series:
    """Label volatility regimes based on VIX levels.

    Uses fixed VIX thresholds to bucket each observation into a regime.
    Default thresholds (15 / 25) correspond to historically calm vs.
    stressed market conditions.

    Args:
        vix: Series of VIX close values with DatetimeIndex.
        low_threshold: VIX level below which regime is "low".
        high_threshold: VIX level above which regime is "high".

    Returns:
        Series of regime labels ("low", "medium", "high") aligned
        with the input index.
    """
    logger.info(
        "Labeling VIX regimes (low < %.1f, high >= %.1f) on %d observations",
        low_threshold,
        high_threshold,
        len(vix),
    )

    # np.select evaluates conditions in order; the first True wins.
    # Anything that is neither low nor high falls through to "medium".
    conditions = [
        vix < low_threshold,
        vix >= high_threshold,
    ]
    choices = ["low", "high"]
    labels = np.select(conditions, choices, default="medium")

    result = pd.Series(labels, index=vix.index, name="regime")
    logger.debug(
        "VIX regime distribution: %s",
        result.value_counts().to_dict(),
    )
    return result


def _compute_rolling_annualized_vol(
    returns: pd.Series, window: int
) -> pd.Series:
    """Compute rolling annualized volatility from a return series.

    Args:
        returns: Series of asset returns with DatetimeIndex.
        window: Rolling window size in trading days.

    Returns:
        Series of annualized rolling volatility values.  The first
        ``window - 1`` entries will be NaN (insufficient data).
    """
    # Rolling std gives daily vol; multiply by sqrt(252) to annualize.
    return returns.rolling(window=window).std() * _ANNUALIZATION_FACTOR


def label_regimes_rolling_std(
    returns: pd.Series,
    window: int = 252,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> pd.Series:
    """Label volatility regimes using rolling standard deviation.

    Computes rolling annualized volatility and assigns regime labels
    based on quantile thresholds of the historical distribution.

    Args:
        returns: Series of asset returns with DatetimeIndex.
        window: Rolling window size in trading days.
        low_quantile: Quantile below which regime is "low".
        high_quantile: Quantile above which regime is "high".

    Returns:
        Series of regime labels ("low", "medium", "high"). Values
        during the warmup period (first ``window - 1`` rows) are NaN.
    """
    logger.info(
        "Labeling rolling-std regimes (window=%d, quantiles=%.2f/%.2f) "
        "on %d observations",
        window,
        low_quantile,
        high_quantile,
        len(returns),
    )

    rolling_vol = _compute_rolling_annualized_vol(returns, window)

    # Quantile thresholds split the *observed* vol distribution into
    # terciles (by default).  This is data-adaptive: "high" means the
    # top third of realised volatility for this particular series.
    low_thresh = rolling_vol.quantile(low_quantile)
    high_thresh = rolling_vol.quantile(high_quantile)
    logger.debug(
        "Rolling-vol quantile thresholds: low=%.4f, high=%.4f",
        low_thresh,
        high_thresh,
    )

    def _classify(vol: float) -> str | float:
        """Map a single volatility value to its regime label."""
        if pd.isna(vol):
            return np.nan
        if vol < low_thresh:
            return "low"
        if vol >= high_thresh:
            return "high"
        return "medium"

    labels = rolling_vol.map(_classify)
    result = pd.Series(labels, index=returns.index, name="regime")
    logger.debug(
        "Rolling-std regime distribution: %s",
        result.value_counts().to_dict(),
    )
    return result
