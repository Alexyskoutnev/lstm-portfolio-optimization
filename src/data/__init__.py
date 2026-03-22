"""Data module for portfolio optimization project.

Provides a single entry point for loading and preprocessing
financial time series data.
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

    Downloads price data, computes returns, labels volatility regimes,
    and creates temporal train/validation/test splits.

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

    prices = fetch_prices(tickers, start=start, end=end, cache_dir=cache_dir)
    vix = fetch_vix(start=start, end=end)
    logger.debug("Fetched prices shape %s, VIX length %d", prices.shape, len(vix))

    log_returns = compute_log_returns(prices)
    simple_returns = compute_simple_returns(prices)

    # Reindex VIX to the returns calendar so every return row has a
    # regime label.  VIX and the asset universe may trade on slightly
    # different days (e.g., early closes, holidays), so forward-fill
    # carries the last known VIX value into gaps, and back-fill covers
    # any leading NaNs at the start of the series.
    vix_aligned = vix.reindex(log_returns.index).ffill().bfill()
    regime_vix = label_regimes_vix(vix_aligned)

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
