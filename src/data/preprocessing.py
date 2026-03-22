"""Preprocessing utilities for financial time series data.

Provides functions for computing returns and splitting data into
temporally-ordered train/validation/test sets.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price DataFrame.

    Log returns are preferred over simple returns for multi-period
    aggregation because they are time-additive:
    ``r(t0, t2) = r(t0, t1) + r(t1, t2)``.

    Args:
        prices: DataFrame with DatetimeIndex and one column per asset,
            containing adjusted close prices.

    Returns:
        DataFrame of log returns with the first row dropped (NaN).
    """
    logger.info(
        "Computing log returns for %d assets over %d periods.",
        prices.shape[1],
        prices.shape[0],
    )

    # shift(1) gives the previous day's price; dividing gives the price
    # ratio, and np.log converts it to a continuously-compounded return.
    log_returns = np.log(prices / prices.shift(1))

    # The first row is always NaN because there is no prior price to
    # compute a return against.  Drop it so downstream code gets a
    # clean, fully-populated DataFrame.
    log_returns = log_returns.dropna()

    logger.debug("Log-return shape after dropping leading NaN row: %s", log_returns.shape)
    return log_returns


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) returns from a price DataFrame.

    Simple returns represent the actual percentage gain/loss and are
    useful for portfolio-weighted return calculations because they
    aggregate linearly across assets.

    Args:
        prices: DataFrame with DatetimeIndex and one column per asset.

    Returns:
        DataFrame of simple returns with the first row dropped.
    """
    logger.info(
        "Computing simple returns for %d assets over %d periods.",
        prices.shape[1],
        prices.shape[0],
    )

    simple_returns = prices.pct_change()

    # Drop the leading NaN row produced by pct_change (no prior price
    # exists for the first observation).
    simple_returns = simple_returns.dropna()

    logger.debug("Simple-return shape after dropping leading NaN row: %s", simple_returns.shape)
    return simple_returns


def _validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    """Check that train/val ratios leave room for a test set.

    Args:
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.

    Raises:
        ValueError: If the ratios sum to 1.0 or more, leaving no data
            for the test set.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into train, validation, and test sets.

    Uses a strict temporal (chronological) split rather than random
    shuffling.  In financial time series, random splits would leak
    future information into the training set, producing overly
    optimistic back-test results that do not generalise to live trading.

    Args:
        df: DataFrame with DatetimeIndex to split.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation. The remainder
            goes to the test set.

    Returns:
        Tuple of (train, validation, test) DataFrames.

    Raises:
        ValueError: If ratios are invalid.
    """
    _validate_split_ratios(train_ratio, val_ratio)

    n = len(df)

    # Integer cutoff indices – floor via int() so rounding never pushes
    # a boundary past the end of the DataFrame.
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        "Temporal split sizes — train: %d, val: %d, test: %d (total: %d).",
        len(train),
        len(val),
        len(test),
        n,
    )

    return train, val, test
