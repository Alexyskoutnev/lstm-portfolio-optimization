"""Data module for portfolio optimization project.

Provides a single entry point (``load_dataset``) for loading and preprocessing
financial time series data.  The full pipeline is:

1. **Fetch prices** (``fetcher.fetch_prices``):
   Download adjusted close prices for each sector ETF from Yahoo Finance,
   with local CSV caching.  "Adjusted close" accounts for dividends and stock
   splits so that returns computed from consecutive prices reflect true total
   returns.

2. **Fetch VIX** (``fetcher.fetch_vix``):
   Download the CBOE Volatility Index separately.  The VIX is not an
   investable asset -- it is an implied-volatility index derived from S&P 500
   option prices.  We use it purely as a *regime label* (see step 5).

3. **Compute returns** (``preprocessing.compute_log_returns`` / ``compute_simple_returns``):
   Convert prices to returns.  Models should never be trained on raw prices
   because prices are non-stationary (they trend upward over time).  Returns
   are approximately stationary and mean-zero.

4. **Reindex VIX to the returns calendar**:
   The VIX and sector ETFs can trade on slightly different days (the CBOE
   options market and NYSE equity market have marginally different holiday
   schedules).  ``vix.reindex(log_returns.index)`` aligns VIX to the exact
   dates present in our return data.  Forward-fill carries the last known VIX
   value into any gap (e.g., VIX was published on a day an ETF did not trade),
   and back-fill covers leading NaNs.  Without this step, we would have NaN
   regime labels on some trading days, breaking downstream model training.

5. **Label volatility regimes** (``regime.label_regimes_vix``):
   Classify each day as low / medium / high volatility using VIX thresholds.
   This lets us train regime-aware models or evaluate how portfolio strategies
   perform under different market conditions.

6. **Temporal train/val/test split** (``preprocessing.split_data``):
   Split chronologically -- never randomly -- to prevent look-ahead bias.
   The model must only learn from past data, exactly as in live trading.
"""

import logging

from src.data.fetcher import fetch_prices, fetch_vix
from src.data.preprocessing import compute_log_returns, compute_simple_returns, split_data
from src.data.regime import label_regimes_rolling_std, label_regimes_vix

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_prices",
    "fetch_vix",
    "compute_log_returns",
    "compute_simple_returns",
    "split_data",
    "label_regimes_vix",
    "label_regimes_rolling_std",
    "load_dataset",
]


def load_dataset(
    tickers: list[str],
    start: str,
    end: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    cache_dir: str | None = None,
) -> dict:
    """Load and preprocess the full dataset for modeling.

    This is the main entry point that orchestrates the entire data pipeline
    described in the module docstring.  Call this once and you get back
    everything needed for model training and evaluation.

    Args:
        tickers: List of ETF ticker symbols.
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        cache_dir: Optional directory for caching raw data.

    Returns:
        Dictionary with keys:
            - "prices": Raw adjusted close prices.
            - "log_returns": Log returns DataFrame.
            - "simple_returns": Simple returns DataFrame.
            - "regime_vix": VIX-based regime labels (Series).
            - "train": Training split of log returns.
            - "val": Validation split of log returns.
            - "test": Test split of log returns.
    """
    logger.info(
        "Loading dataset for %d tickers from %s to %s",
        len(tickers),
        start,
        end,
    )

    # Step 1 & 2: Fetch raw price data and VIX
    prices = fetch_prices(tickers, start=start, end=end, cache_dir=cache_dir)
    vix = fetch_vix(start=start, end=end)
    logger.debug("Fetched prices shape %s, VIX length %d", prices.shape, len(vix))

    # Step 3: Convert prices to returns
    log_returns = compute_log_returns(prices)
    simple_returns = compute_simple_returns(prices)

    # Step 4 & 5: Align VIX to the returns calendar and label regimes.
    # Reindex VIX to the returns calendar so every return row has a
    # regime label.  VIX and the asset universe may trade on slightly
    # different days (e.g., early closes, holidays), so forward-fill
    # carries the last known VIX value into gaps, and back-fill covers
    # any leading NaNs at the start of the series.
    vix_aligned = vix.reindex(log_returns.index).ffill().bfill()
    regime_vix = label_regimes_vix(vix_aligned)

    # Step 6: Temporal (chronological) split -- never random
    train, val, test = split_data(
        log_returns, train_ratio=train_ratio, val_ratio=val_ratio
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )

    return {
        "prices": prices,
        "log_returns": log_returns,
        "simple_returns": simple_returns,
        "regime_vix": regime_vix,
        "train": train,
        "val": val,
        "test": test,
    }
