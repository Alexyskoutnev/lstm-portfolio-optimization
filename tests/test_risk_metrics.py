"""Tests for risk metric computations."""

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    compute_all_metrics,
    conditional_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


@pytest.fixture()
def sample_returns() -> pd.Series:
    """Create synthetic daily returns with positive drift."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    return pd.Series(rng.normal(0.001, 0.008, 252), index=dates)


class TestSharpeRatio:
    """Tests for Sharpe ratio."""

    def test_positive_for_positive_returns(self, sample_returns: pd.Series) -> None:
        """Sharpe should be positive when mean return is positive."""
        assert sharpe_ratio(sample_returns) > 0

    def test_zero_vol_returns_zero(self) -> None:
        """Zero volatility returns should give zero Sharpe."""
        flat = pd.Series([0.0] * 100)
        assert sharpe_ratio(flat) == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio."""

    def test_greater_than_sharpe(self, sample_returns: pd.Series) -> None:
        """Sortino magnitude should be >= Sharpe magnitude for positive returns."""
        s = sortino_ratio(sample_returns)
        sh = sharpe_ratio(sample_returns)
        # For positive mean returns, Sortino >= Sharpe since downside vol <= total vol
        assert s >= sh


class TestMaxDrawdown:
    """Tests for maximum drawdown."""

    def test_negative(self, sample_returns: pd.Series) -> None:
        """Max drawdown should be negative."""
        assert max_drawdown(sample_returns) < 0

    def test_bounded(self, sample_returns: pd.Series) -> None:
        """Max drawdown should be between -1 and 0."""
        mdd = max_drawdown(sample_returns)
        assert -1.0 <= mdd <= 0.0


class TestVaR:
    """Tests for Value-at-Risk."""

    def test_negative(self, sample_returns: pd.Series) -> None:
        """95% VaR should typically be negative."""
        assert value_at_risk(sample_returns, confidence=0.95) < 0

    def test_higher_confidence_more_extreme(self, sample_returns: pd.Series) -> None:
        """99% VaR should be more extreme than 95% VaR."""
        var_95 = value_at_risk(sample_returns, confidence=0.95)
        var_99 = value_at_risk(sample_returns, confidence=0.99)
        assert var_99 <= var_95


class TestCVaR:
    """Tests for Conditional VaR."""

    def test_more_extreme_than_var(self, sample_returns: pd.Series) -> None:
        """CVaR should be more extreme than VaR."""
        var = value_at_risk(sample_returns, confidence=0.95)
        cvar = conditional_var(sample_returns, confidence=0.95)
        assert cvar <= var


class TestComputeAllMetrics:
    """Tests for the combined metrics function."""

    def test_returns_all_keys(self, sample_returns: pd.Series) -> None:
        """Should return all expected metric keys."""
        metrics = compute_all_metrics(sample_returns)
        expected_keys = {
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "var_95",
            "cvar_95",
        }
        assert set(metrics.keys()) == expected_keys
