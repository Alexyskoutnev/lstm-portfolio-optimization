"""Volatility regime identification for financial time series.

Provides two approaches for labeling market regimes:
1. VIX-based thresholds (direct fear gauge)
2. Rolling standard deviation of returns (model-free)

Background: what is the VIX and why do we care?
-------------------------------------------------
The CBOE Volatility Index (VIX) is computed from the prices of S&P 500 index
options.  It represents the market's *implied* (expected) annualized volatility
over the next 30 days.  It is NOT a measure of what has already happened --
it is a *forward-looking* consensus of how turbulent traders *expect* the
market to be.  When traders are nervous, they bid up the price of protective
put options, which mechanically pushes the VIX higher.  This is why the VIX
is often called the "fear gauge."

Why does volatility regime matter for portfolio optimization?
--------------------------------------------------------------
During **low-volatility** regimes (VIX < 15), sector returns are relatively
uncorrelated -- Tech might go up while Utilities go down -- and mean-variance
optimization works well because diversification genuinely reduces risk.

During **high-volatility** regimes (VIX >= 25), correlations across all
sectors spike toward 1.0 ("everything sells off together").  The covariance
matrix estimated from calm-period data becomes dangerously wrong, and a
portfolio that was "optimally diversified" in normal times may offer no
protection at all.  By identifying the current regime, we can:
  - Switch to more conservative (e.g., equal-weight or minimum-variance)
    portfolios during stressed periods.
  - Use a different covariance estimator (e.g., shrinkage, exponential
    weighting) tuned to the regime.
  - Avoid over-fitting to calm-period statistics.

VIX thresholds (15 / 25)
--------------------------
These are widely used in industry and academia (see Whaley 2000, "The
Investor Fear Gauge"):
  - VIX < 15:   Calm markets.  ~35% of trading days since 2010 fall here.
  - VIX 15-25:  Normal uncertainty.  The long-run VIX median is ~17-18.
  - VIX >= 25:  Elevated fear.  Historically associated with corrections,
                 bear markets, and liquidity crises (COVID, Euro debt, etc.).

Rolling standard deviation vs. VIX
------------------------------------
The VIX is *implied* (forward-looking, derived from option prices).  Rolling
standard deviation is *realized* (backward-looking, computed from actual
returns over a trailing window).  These two signals complement each other:
  - VIX can spike *before* a crash materializes (because options traders
    anticipate it), giving an early warning.
  - Rolling realized vol captures what *has* happened and is useful when
    VIX data is unavailable (e.g., for non-US markets).
We provide both so downstream models can choose the most appropriate signal.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Annualization factor: daily volatility -> annual volatility.
# WHY sqrt(252)?  Volatility (standard deviation) scales with the *square root*
# of time, not linearly.  This comes from the property of independent random
# variables: Var(X1 + X2 + ... + Xn) = n * Var(X), so Std = sqrt(n) * daily_std.
# Since there are ~252 trading days per year, annual_vol = daily_vol * sqrt(252).
# This lets us express volatility on the same annualized scale as the VIX,
# making the two directly comparable.
_ANNUALIZATION_FACTOR = np.sqrt(252)


def label_regimes_vix(
    vix: pd.Series,
    low_threshold: float = 15.0,
    high_threshold: float = 25.0,
) -> pd.Series:
    """Label volatility regimes based on VIX levels.

    Uses fixed VIX thresholds to bucket each observation into a regime.
    Default thresholds (15 / 25) correspond to historically calm vs.
    stressed market conditions (see module docstring for full rationale).

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

    This is "realized" (historical) volatility -- a backward-looking measure
    of how much the asset's return has actually fluctuated over the trailing
    ``window`` days.  Contrast with the VIX, which is forward-looking.

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

    Unlike VIX-based regimes (which use fixed thresholds), this method uses
    **quantile-based** thresholds computed from the asset's own historical
    volatility distribution.  This is data-adaptive: "high vol" means the
    top third of what *this particular asset* has experienced, regardless of
    absolute level.  This makes the method applicable to any asset class
    (equities, bonds, commodities) without needing to manually choose
    thresholds.

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
