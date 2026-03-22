"""Preprocessing utilities for financial time series data.

Provides functions for computing returns and splitting data into
temporally-ordered train/validation/test sets.

Key concepts for someone new to quantitative finance
-----------------------------------------------------
**Returns, not prices**: ML models should be trained on *returns* (percentage
changes), not raw prices.  Prices are non-stationary (they trend upward over
time), which violates the assumptions of most statistical and ML models.
Returns are approximately stationary and mean-zero, making them suitable
inputs for covariance estimation, regression, and optimization.

**Log returns vs. simple returns**: We compute both.  Each has a distinct
mathematical property that makes it useful in different contexts:

  - **Log returns** ``r_log = ln(P_t / P_{t-1})``:
      * *Time-additive*: the cumulative log return over multiple periods is
        simply the *sum* of daily log returns.  For example, the 5-day log
        return equals ``r1 + r2 + r3 + r4 + r5``.  This makes rolling-window
        calculations trivial and is why log returns are preferred for time
        series modeling.
      * *Closer to normally distributed*: the log transform compresses
        extreme positive returns and stretches extreme negative ones, pulling
        the distribution closer to a Gaussian -- an assumption that underlies
        mean-variance optimization.
      * *Symmetry*: a +10% log return followed by a -10% log return returns
        you exactly to the starting price.  Simple returns do NOT have this
        property (+10% then -10% simple leaves you at 99% of the start).

  - **Simple returns** ``r_simple = (P_t - P_{t-1}) / P_{t-1}``:
      * *Cross-sectionally additive*: a portfolio's return is the weighted
        sum of its constituents' simple returns.  This is essential for
        computing actual portfolio P&L.
      * More intuitive: "the stock went up 3% today."

We use **log returns** as the primary input for modeling and optimization,
and **simple returns** when we need to calculate actual portfolio performance.
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

    Mathematically: ``log_return_t = ln(P_t / P_{t-1})``

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
    # Example: if XLK closed at $100 yesterday and $103 today,
    #   price ratio = 103/100 = 1.03
    #   log return  = ln(1.03) = 0.02956  (~2.96%)
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
    aggregate linearly across assets:
        ``portfolio_return = sum(w_i * r_i)``
    This does NOT hold for log returns, which is why we need both.

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

    **WHY temporal split, not random?**

    In standard ML (e.g., image classification), random train/test splits are
    fine because samples are independent.  Financial time series are *not*
    independent -- today's return is correlated with yesterday's, and market
    regimes persist for months.  A random split would let the model "peek" at
    future data during training, a problem called **look-ahead bias**.

    Concrete example of what goes wrong with random splits:
      Suppose the market crashes on day 500.  With a random split, days 499
      and 501 might land in training while day 500 lands in test.  The model
      effectively learns "a crash is coming" from the surrounding days.  In a
      backtest this looks great -- the model "predicted" the crash -- but in
      live trading it would never have had access to day 501's data when
      making the day-500 decision.  The backtest was unrealistically optimistic.

    A temporal split ensures the model is always trained on *past* data and
    evaluated on *future* data, exactly mimicking the information available
    during real-world portfolio management.

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

    # Integer cutoff indices -- floor via int() so rounding never pushes
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
