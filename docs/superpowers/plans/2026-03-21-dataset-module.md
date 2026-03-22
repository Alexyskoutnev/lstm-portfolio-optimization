# Dataset Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, Google-style data collection and preprocessing pipeline that downloads S&P 500 sector ETF data via yfinance, computes log returns, identifies volatility regimes, and exposes clean DataFrames for downstream MVO and LSTM modules.

**Architecture:** A `src/data/` package with three focused modules: `fetcher.py` (yfinance download + caching), `preprocessing.py` (returns, scaling, train/test splits), and `regime.py` (volatility regime labeling). A thin `config.py` at the project root holds all constants (tickers, date ranges, window sizes). Tests mirror the src structure under `tests/`.

**Tech Stack:** Python 3.11+, yfinance, pandas, numpy, pytest, Google-style docstrings (Napoleon format)

---

## File Structure

```
final_project/
├── src/
│   ├── __init__.py
│   ├── config.py              # All project constants and hyperparameters
│   └── data/
│       ├── __init__.py
│       ├── fetcher.py         # Download and cache ETF data from yfinance
│       ├── preprocessing.py   # Log returns, scaling, train/test split
│       └── regime.py          # Volatility regime identification
├── tests/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── test_fetcher.py
│       ├── test_preprocessing.py
│       └── test_regime.py
├── data/
│   └── raw/                   # Cached CSV files (gitignored)
├── requirements.txt
├── .gitignore
└── README.md                  # (minimal, just setup instructions)
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/data/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
yfinance>=0.2.31
pandas>=2.0.0
numpy>=1.24.0
pytest>=7.4.0
scikit-learn>=1.3.0
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
data/raw/
.env
*.egg-info/
.pytest_cache/
```

- [ ] **Step 3: Create config.py with all project constants**

```python
"""Project-wide configuration constants.

All tunable parameters for data collection, preprocessing, and
volatility regime identification live here.
"""

from typing import Final

# Tickers: S&P 500 sector ETFs
TICKERS: Final[list[str]] = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLY",   # Consumer Discretionary
    "XLV",   # Health Care
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLRE",  # Real Estate
]

# VIX for volatility regime detection
VIX_TICKER: Final[str] = "^VIX"

# Date range
START_DATE: Final[str] = "2010-01-01"
END_DATE: Final[str] = "2025-12-31"

# Preprocessing
ROLLING_WINDOW: Final[int] = 252  # Trading days in a year
TRAIN_RATIO: Final[float] = 0.7
VAL_RATIO: Final[float] = 0.15  # Remaining 0.15 is test

# Volatility regime thresholds (VIX-based)
VIX_LOW_THRESHOLD: Final[float] = 15.0
VIX_HIGH_THRESHOLD: Final[float] = 25.0

# Data paths
RAW_DATA_DIR: Final[str] = "data/raw"
```

- [ ] **Step 4: Create empty __init__.py files**

Create `src/__init__.py`, `src/data/__init__.py`, `tests/__init__.py`, `tests/data/__init__.py` as empty files.

- [ ] **Step 5: Install dependencies and verify**

Run: `pip install -r requirements.txt`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore src/ tests/
git commit -m "chore: scaffold project structure with config and dependencies"
```

---

### Task 2: Data Fetcher Module

**Files:**
- Create: `src/data/fetcher.py`
- Create: `tests/data/test_fetcher.py`

- [ ] **Step 1: Write the failing test for fetch_prices**

```python
"""Tests for the data fetcher module."""

import os

import pandas as pd
import pytest

from src.data.fetcher import fetch_prices, fetch_vix


class TestFetchPrices:
    """Tests for fetch_prices function."""

    def test_returns_dataframe(self):
        """fetch_prices returns a DataFrame with tickers as columns."""
        df = fetch_prices(["XLK", "XLF"], start="2024-01-01", end="2024-01-31")
        assert isinstance(df, pd.DataFrame)
        assert "XLK" in df.columns
        assert "XLF" in df.columns

    def test_index_is_datetime(self):
        """Index should be DatetimeIndex."""
        df = fetch_prices(["XLK"], start="2024-01-01", end="2024-01-31")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_no_missing_values(self):
        """Returned data should have no NaN values."""
        df = fetch_prices(["XLK", "XLF"], start="2024-01-01", end="2024-06-30")
        assert df.isna().sum().sum() == 0

    def test_invalid_ticker_raises(self):
        """Invalid ticker should raise ValueError."""
        with pytest.raises(ValueError, match="No data found"):
            fetch_prices(["INVALIDTICKER123"], start="2024-01-01", end="2024-01-31")

    def test_caching_creates_file(self, tmp_path):
        """When cache_dir is provided, a CSV file should be created."""
        fetch_prices(["XLK"], start="2024-01-01", end="2024-01-31",
                     cache_dir=str(tmp_path))
        cached_files = list(tmp_path.glob("*.csv"))
        assert len(cached_files) == 1


class TestFetchVix:
    """Tests for fetch_vix function."""

    def test_returns_series(self):
        """fetch_vix returns a pandas Series."""
        vix = fetch_vix(start="2024-01-01", end="2024-01-31")
        assert isinstance(vix, pd.Series)
        assert vix.name == "VIX"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/data/test_fetcher.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement fetcher.py**

```python
"""Data fetcher for ETF prices and VIX data via yfinance.

This module handles downloading, validating, and optionally caching
historical price data from Yahoo Finance.
"""

import os
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers.

    Args:
        tickers: List of ticker symbols (e.g., ["XLK", "XLF"]).
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        cache_dir: Optional directory to cache downloaded data as CSV.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker,
        containing adjusted close prices with no missing values.

    Raises:
        ValueError: If no data is found for any of the requested tickers.
    """
    if cache_dir:
        cache_path = Path(cache_dir) / f"prices_{'_'.join(tickers)}_{start}_{end}.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data found for tickers: {tickers}")

    # yf.download returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"]
    else:
        df = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Validate all tickers present
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(f"No data found for tickers: {missing}")

    # Forward-fill then back-fill to handle market holidays across ETFs
    df = df.ffill().bfill()

    if df.isna().any().any():
        raise ValueError("Unable to fill all missing values in price data.")

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)

    return df


def fetch_vix(start: str, end: str) -> pd.Series:
    """Download VIX close prices.

    Args:
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.

    Returns:
        Series with DatetimeIndex containing VIX close values,
        named "VIX".

    Raises:
        ValueError: If no VIX data is found.
    """
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError("No VIX data found for the given date range.")

    vix = raw["Close"].squeeze()
    vix.name = "VIX"
    vix = vix.ffill().bfill()
    return vix
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_fetcher.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/fetcher.py tests/data/test_fetcher.py
git commit -m "feat: add data fetcher module with yfinance download and caching"
```

---

### Task 3: Preprocessing Module

**Files:**
- Create: `src/data/preprocessing.py`
- Create: `tests/data/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    compute_log_returns,
    compute_simple_returns,
    split_data,
)


@pytest.fixture
def sample_prices():
    """Create a small synthetic price DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.random.normal(0.0005, 0.01, 100).cumsum()),
            "B": 50 * np.exp(np.random.normal(0.0003, 0.015, 100).cumsum()),
        },
        index=dates,
    )
    return prices


class TestComputeLogReturns:
    """Tests for compute_log_returns."""

    def test_output_shape(self, sample_prices):
        """Log returns should have one fewer row than prices."""
        returns = compute_log_returns(sample_prices)
        assert returns.shape[0] == sample_prices.shape[0] - 1
        assert returns.shape[1] == sample_prices.shape[1]

    def test_no_nans(self, sample_prices):
        """Log returns should have no NaN values."""
        returns = compute_log_returns(sample_prices)
        assert returns.isna().sum().sum() == 0

    def test_values_correct(self):
        """Log returns should equal ln(P_t / P_{t-1})."""
        prices = pd.DataFrame({"X": [100.0, 110.0, 105.0]})
        returns = compute_log_returns(prices)
        expected = np.log(np.array([110.0 / 100.0, 105.0 / 110.0]))
        np.testing.assert_allclose(returns["X"].values, expected, rtol=1e-10)


class TestComputeSimpleReturns:
    """Tests for compute_simple_returns."""

    def test_values_correct(self):
        """Simple returns should equal (P_t - P_{t-1}) / P_{t-1}."""
        prices = pd.DataFrame({"X": [100.0, 110.0, 105.0]})
        returns = compute_simple_returns(prices)
        expected = np.array([0.1, -0.04545454545])
        np.testing.assert_allclose(returns["X"].values, expected, rtol=1e-6)


class TestSplitData:
    """Tests for split_data."""

    def test_split_sizes(self, sample_prices):
        """Train/val/test splits should sum to original length."""
        train, val, test = split_data(sample_prices, train_ratio=0.7, val_ratio=0.15)
        assert len(train) + len(val) + len(test) == len(sample_prices)

    def test_no_overlap(self, sample_prices):
        """Splits should be temporally ordered with no overlap."""
        train, val, test = split_data(sample_prices, train_ratio=0.7, val_ratio=0.15)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_preserves_columns(self, sample_prices):
        """All splits should have the same columns."""
        train, val, test = split_data(sample_prices, train_ratio=0.7, val_ratio=0.15)
        assert list(train.columns) == list(sample_prices.columns)
        assert list(val.columns) == list(sample_prices.columns)
        assert list(test.columns) == list(sample_prices.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/data/test_preprocessing.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement preprocessing.py**

```python
"""Preprocessing utilities for financial time series data.

Provides functions for computing returns and splitting data into
temporally-ordered train/validation/test sets.
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price DataFrame.

    Args:
        prices: DataFrame with DatetimeIndex and one column per asset,
            containing adjusted close prices.

    Returns:
        DataFrame of log returns with the first row dropped (NaN).
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) returns from a price DataFrame.

    Args:
        prices: DataFrame with DatetimeIndex and one column per asset.

    Returns:
        DataFrame of simple returns with the first row dropped.
    """
    simple_returns = prices.pct_change()
    return simple_returns.dropna()


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into train, validation, and test sets.

    The split is purely temporal (no shuffling) to prevent look-ahead bias.

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
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_preprocessing.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocessing.py tests/data/test_preprocessing.py
git commit -m "feat: add preprocessing module with log returns and temporal splits"
```

---

### Task 4: Volatility Regime Module

**Files:**
- Create: `src/data/regime.py`
- Create: `tests/data/test_regime.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the volatility regime module."""

import numpy as np
import pandas as pd
import pytest

from src.data.regime import label_regimes_vix, label_regimes_rolling_std


@pytest.fixture
def sample_vix():
    """Create synthetic VIX data spanning all three regimes."""
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    # Low: 10-14, Medium: 18-22, High: 28-35
    values = np.concatenate([
        np.random.uniform(10, 14, 30),
        np.random.uniform(18, 22, 30),
        np.random.uniform(28, 35, 30),
    ])
    return pd.Series(values, index=dates, name="VIX")


@pytest.fixture
def sample_returns():
    """Create synthetic returns with varying volatility."""
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    np.random.seed(42)
    # Low vol, then high vol
    low_vol = np.random.normal(0, 0.005, 150)
    high_vol = np.random.normal(0, 0.03, 150)
    returns = pd.Series(np.concatenate([low_vol, high_vol]), index=dates, name="returns")
    return returns


class TestLabelRegimesVix:
    """Tests for VIX-based regime labeling."""

    def test_returns_series(self, sample_vix):
        """Should return a Series of regime labels."""
        labels = label_regimes_vix(sample_vix)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sample_vix)

    def test_label_values(self, sample_vix):
        """Labels should be one of 'low', 'medium', 'high'."""
        labels = label_regimes_vix(sample_vix)
        assert set(labels.unique()).issubset({"low", "medium", "high"})

    def test_low_regime_detected(self, sample_vix):
        """First 30 days (VIX 10-14) should be labeled 'low'."""
        labels = label_regimes_vix(sample_vix, low_threshold=15.0, high_threshold=25.0)
        assert (labels.iloc[:30] == "low").all()

    def test_high_regime_detected(self, sample_vix):
        """Last 30 days (VIX 28-35) should be labeled 'high'."""
        labels = label_regimes_vix(sample_vix, low_threshold=15.0, high_threshold=25.0)
        assert (labels.iloc[60:] == "high").all()


class TestLabelRegimesRollingStd:
    """Tests for rolling-std-based regime labeling."""

    def test_returns_series(self, sample_returns):
        """Should return a Series of regime labels."""
        labels = label_regimes_rolling_std(sample_returns, window=60)
        assert isinstance(labels, pd.Series)

    def test_no_nans_after_warmup(self, sample_returns):
        """After the rolling window warmup, no labels should be NaN."""
        labels = label_regimes_rolling_std(sample_returns, window=60)
        assert labels.iloc[60:].isna().sum() == 0

    def test_label_values(self, sample_returns):
        """Labels should be one of 'low', 'medium', 'high', or NaN."""
        labels = label_regimes_rolling_std(sample_returns, window=60)
        valid_labels = labels.dropna().unique()
        assert set(valid_labels).issubset({"low", "medium", "high"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/data/test_regime.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement regime.py**

```python
"""Volatility regime identification for financial time series.

Provides two approaches for labeling market regimes:
1. VIX-based thresholds (direct fear gauge)
2. Rolling standard deviation of returns (model-free)
"""

import numpy as np
import pandas as pd


def label_regimes_vix(
    vix: pd.Series,
    low_threshold: float = 15.0,
    high_threshold: float = 25.0,
) -> pd.Series:
    """Label volatility regimes based on VIX levels.

    Args:
        vix: Series of VIX close values with DatetimeIndex.
        low_threshold: VIX level below which regime is "low".
        high_threshold: VIX level above which regime is "high".

    Returns:
        Series of regime labels ("low", "medium", "high") aligned
        with the input index.
    """
    conditions = [
        vix < low_threshold,
        vix >= high_threshold,
    ]
    choices = ["low", "high"]
    labels = np.select(conditions, choices, default="medium")
    return pd.Series(labels, index=vix.index, name="regime")


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
        during the warmup period (first `window - 1` rows) are NaN.
    """
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

    low_thresh = rolling_vol.quantile(low_quantile)
    high_thresh = rolling_vol.quantile(high_quantile)

    def _classify(vol):
        if pd.isna(vol):
            return np.nan
        if vol < low_thresh:
            return "low"
        if vol >= high_thresh:
            return "high"
        return "medium"

    labels = rolling_vol.map(_classify)
    return pd.Series(labels, index=returns.index, name="regime")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_regime.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/regime.py tests/data/test_regime.py
git commit -m "feat: add volatility regime labeling (VIX and rolling-std methods)"
```

---

### Task 5: Data Module Public API

**Files:**
- Modify: `src/data/__init__.py`
- Create: `tests/data/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""Integration test for the full data pipeline."""

import pandas as pd
import pytest

from src.data import load_dataset


class TestLoadDataset:
    """End-to-end test for the data loading pipeline."""

    def test_returns_expected_keys(self):
        """load_dataset should return a dict with all expected keys."""
        data = load_dataset(
            tickers=["XLK", "XLF"],
            start="2024-01-01",
            end="2024-06-30",
        )
        assert "prices" in data
        assert "log_returns" in data
        assert "simple_returns" in data
        assert "regime_vix" in data
        assert "train" in data
        assert "val" in data
        assert "test" in data

    def test_shapes_consistent(self):
        """Returns should have one fewer row than prices."""
        data = load_dataset(
            tickers=["XLK", "XLF"],
            start="2024-01-01",
            end="2024-06-30",
        )
        assert data["log_returns"].shape[0] == data["prices"].shape[0] - 1

    def test_splits_cover_returns(self):
        """Train + val + test should cover all returns."""
        data = load_dataset(
            tickers=["XLK", "XLF"],
            start="2024-01-01",
            end="2024-06-30",
        )
        total = len(data["train"]) + len(data["val"]) + len(data["test"])
        assert total == len(data["log_returns"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_integration.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement public API in __init__.py**

```python
"""Data module for portfolio optimization project.

Provides a single entry point for loading and preprocessing
financial time series data.
"""

from src.data.fetcher import fetch_prices, fetch_vix
from src.data.preprocessing import compute_log_returns, compute_simple_returns, split_data
from src.data.regime import label_regimes_vix, label_regimes_rolling_std


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
    prices = fetch_prices(tickers, start=start, end=end, cache_dir=cache_dir)
    vix = fetch_vix(start=start, end=end)

    log_returns = compute_log_returns(prices)
    simple_returns = compute_simple_returns(prices)

    # Align VIX to returns index
    vix_aligned = vix.reindex(log_returns.index).ffill().bfill()
    regime_vix = label_regimes_vix(vix_aligned)

    train, val, test = split_data(log_returns, train_ratio=train_ratio, val_ratio=val_ratio)

    return {
        "prices": prices,
        "log_returns": log_returns,
        "simple_returns": simple_returns,
        "regime_vix": regime_vix,
        "train": train,
        "val": val,
        "test": test,
    }
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (20 total)

- [ ] **Step 5: Commit**

```bash
git add src/data/__init__.py tests/data/test_integration.py
git commit -m "feat: add load_dataset public API with integration tests"
```

---

### Task 6: Full Pipeline Smoke Test

**Files:**
- Create: `scripts/verify_data.py` (optional verification script)

- [ ] **Step 1: Create a quick verification script**

```python
"""Smoke test: download full dataset and print summary statistics."""

from src.config import TICKERS, START_DATE, END_DATE, RAW_DATA_DIR
from src.data import load_dataset


def main():
    """Run the full data pipeline and print diagnostics."""
    data = load_dataset(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        cache_dir=RAW_DATA_DIR,
    )

    print(f"Prices shape: {data['prices'].shape}")
    print(f"Log returns shape: {data['log_returns'].shape}")
    print(f"Train: {len(data['train'])} | Val: {len(data['val'])} | Test: {len(data['test'])}")
    print(f"\nRegime distribution:\n{data['regime_vix'].value_counts()}")
    print(f"\nReturn statistics:\n{data['log_returns'].describe()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the verification script**

Run: `python scripts/verify_data.py`
Expected: Prints shapes, regime counts, and return statistics without errors.

- [ ] **Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_data.py
git commit -m "feat: add data pipeline verification script"
```
