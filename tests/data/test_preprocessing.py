"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    compute_log_returns,
    compute_simple_returns,
    split_data,
)


@pytest.fixture()
def sample_prices() -> pd.DataFrame:
    """Create a small synthetic price DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(rng.normal(0.0005, 0.01, 100).cumsum()),
            "B": 50 * np.exp(rng.normal(0.0003, 0.015, 100).cumsum()),
        },
        index=dates,
    )
    return prices


class TestComputeLogReturns:
    """Tests for compute_log_returns."""

    def test_output_shape(self, sample_prices: pd.DataFrame) -> None:
        """Log returns should have one fewer row than prices."""
        returns = compute_log_returns(sample_prices)
        assert returns.shape[0] == sample_prices.shape[0] - 1
        assert returns.shape[1] == sample_prices.shape[1]

    def test_no_nans(self, sample_prices: pd.DataFrame) -> None:
        """Log returns should have no NaN values."""
        returns = compute_log_returns(sample_prices)
        assert returns.isna().sum().sum() == 0

    def test_values_correct(self) -> None:
        """Log returns should equal ln(P_t / P_{t-1})."""
        prices = pd.DataFrame({"X": [100.0, 110.0, 105.0]})
        returns = compute_log_returns(prices)
        expected = np.log(np.array([110.0 / 100.0, 105.0 / 110.0]))
        np.testing.assert_allclose(returns["X"].values, expected, rtol=1e-10)


class TestComputeSimpleReturns:
    """Tests for compute_simple_returns."""

    def test_values_correct(self) -> None:
        """Simple returns should equal (P_t - P_{t-1}) / P_{t-1}."""
        prices = pd.DataFrame({"X": [100.0, 110.0, 105.0]})
        returns = compute_simple_returns(prices)
        expected = np.array([0.1, -0.04545454545])
        np.testing.assert_allclose(returns["X"].values, expected, rtol=1e-6)


class TestSplitData:
    """Tests for split_data."""

    def test_split_sizes(self, sample_prices: pd.DataFrame) -> None:
        """Train/val/test splits should sum to original length."""
        train, val, test = split_data(sample_prices, train_ratio=0.7, val_ratio=0.15)
        assert len(train) + len(val) + len(test) == len(sample_prices)

    def test_no_overlap(self, sample_prices: pd.DataFrame) -> None:
        """Splits should be temporally ordered with no overlap."""
        train, val, test = split_data(sample_prices, train_ratio=0.7, val_ratio=0.15)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_preserves_columns(self, sample_prices: pd.DataFrame) -> None:
        """All splits should have the same columns."""
        train, val, test = split_data(sample_prices, train_ratio=0.7, val_ratio=0.15)
        assert list(train.columns) == list(sample_prices.columns)
        assert list(val.columns) == list(sample_prices.columns)
        assert list(test.columns) == list(sample_prices.columns)

    def test_invalid_ratios(self, sample_prices: pd.DataFrame) -> None:
        """Should raise ValueError for invalid ratios."""
        with pytest.raises(ValueError, match="must be less than 1.0"):
            split_data(sample_prices, train_ratio=0.8, val_ratio=0.3)

    def test_non_datetime_index_raises(self) -> None:
        """Should raise TypeError if index is not DatetimeIndex."""
        df = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            split_data(df)

    def test_unsorted_index_raises(self) -> None:
        """Should raise ValueError if dates are not ascending."""
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        df = pd.DataFrame({"A": [1, 2, 3]}, index=dates)
        with pytest.raises(ValueError, match="not sorted"):
            split_data(df)
