"""Tests for the Markowitz MVO model."""

import numpy as np
import pandas as pd
import pytest

from src.models.mvo import (
    efficient_frontier,
    estimate_parameters,
    max_sharpe_weights,
    minimum_variance_weights,
    portfolio_performance,
    rolling_backtest,
)


@pytest.fixture()
def synthetic_returns() -> pd.DataFrame:
    """Create synthetic return data with known properties."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.0004, 0.01, 500),  # Higher return, lower vol
            "B": rng.normal(0.0002, 0.02, 500),  # Lower return, higher vol
            "C": rng.normal(0.0003, 0.015, 500),
        },
        index=dates,
    )
    return returns


class TestEstimateParameters:
    """Tests for parameter estimation."""

    def test_shapes(self, synthetic_returns: pd.DataFrame) -> None:
        """Mean returns and cov matrix should have correct shapes."""
        mean_ret, cov_mat = estimate_parameters(synthetic_returns)
        assert mean_ret.shape == (3,)
        assert cov_mat.shape == (3, 3)

    def test_cov_symmetric(self, synthetic_returns: pd.DataFrame) -> None:
        """Covariance matrix should be symmetric."""
        _, cov_mat = estimate_parameters(synthetic_returns)
        np.testing.assert_allclose(cov_mat, cov_mat.T, atol=1e-10)

    def test_cov_positive_semidefinite(self, synthetic_returns: pd.DataFrame) -> None:
        """Covariance matrix eigenvalues should be non-negative."""
        _, cov_mat = estimate_parameters(synthetic_returns)
        eigenvalues = np.linalg.eigvals(cov_mat)
        assert (eigenvalues >= -1e-10).all()


class TestPortfolioPerformance:
    """Tests for portfolio performance computation."""

    def test_equal_weight(self, synthetic_returns: pd.DataFrame) -> None:
        """Equal-weight portfolio should give reasonable values."""
        mean_ret, cov_mat = estimate_parameters(synthetic_returns)
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        ret, vol = portfolio_performance(weights, mean_ret, cov_mat)
        assert isinstance(ret, float)
        assert isinstance(vol, float)
        assert vol > 0


class TestMinimumVarianceWeights:
    """Tests for minimum variance optimization."""

    def test_weights_sum_to_one(self, synthetic_returns: pd.DataFrame) -> None:
        """Weights should sum to 1."""
        _, cov_mat = estimate_parameters(synthetic_returns)
        weights = minimum_variance_weights(cov_mat)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_no_negative_weights(self, synthetic_returns: pd.DataFrame) -> None:
        """Long-only weights should be non-negative."""
        _, cov_mat = estimate_parameters(synthetic_returns)
        weights = minimum_variance_weights(cov_mat, allow_short=False)
        assert (weights >= -1e-6).all()

    def test_lower_vol_than_equal_weight(self, synthetic_returns: pd.DataFrame) -> None:
        """Min variance should have lower volatility than equal weight."""
        mean_ret, cov_mat = estimate_parameters(synthetic_returns)
        mv_weights = minimum_variance_weights(cov_mat)
        eq_weights = np.ones(3) / 3

        _, mv_vol = portfolio_performance(mv_weights, mean_ret, cov_mat)
        _, eq_vol = portfolio_performance(eq_weights, mean_ret, cov_mat)
        assert mv_vol <= eq_vol + 1e-6


class TestMaxSharpeWeights:
    """Tests for maximum Sharpe ratio optimization."""

    def test_weights_sum_to_one(self, synthetic_returns: pd.DataFrame) -> None:
        """Weights should sum to 1."""
        mean_ret, cov_mat = estimate_parameters(synthetic_returns)
        weights = max_sharpe_weights(mean_ret, cov_mat)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_higher_sharpe_than_equal_weight(self, synthetic_returns: pd.DataFrame) -> None:
        """Max Sharpe should have higher Sharpe than equal weight."""
        mean_ret, cov_mat = estimate_parameters(synthetic_returns)
        ms_weights = max_sharpe_weights(mean_ret, cov_mat)
        eq_weights = np.ones(3) / 3

        ms_ret, ms_vol = portfolio_performance(ms_weights, mean_ret, cov_mat)
        eq_ret, eq_vol = portfolio_performance(eq_weights, mean_ret, cov_mat)

        ms_sharpe = ms_ret / ms_vol
        eq_sharpe = eq_ret / eq_vol
        assert ms_sharpe >= eq_sharpe - 1e-6


class TestEfficientFrontier:
    """Tests for efficient frontier computation."""

    def test_returns_dataframe(self, synthetic_returns: pd.DataFrame) -> None:
        """Should return a DataFrame with expected columns."""
        mean_ret, cov_mat = estimate_parameters(synthetic_returns)
        frontier = efficient_frontier(mean_ret, cov_mat, n_points=20)
        assert isinstance(frontier, pd.DataFrame)
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns
        assert "sharpe" in frontier.columns

    def test_monotonic_returns(self, synthetic_returns: pd.DataFrame) -> None:
        """Returns on the frontier should be monotonically increasing."""
        mean_ret, cov_mat = estimate_parameters(synthetic_returns)
        frontier = efficient_frontier(mean_ret, cov_mat, n_points=20)
        assert frontier["return"].is_monotonic_increasing


class TestRollingBacktest:
    """Tests for rolling-window backtesting."""

    def test_output_columns(self, synthetic_returns: pd.DataFrame) -> None:
        """Output should have portfolio_return and cumulative_return."""
        result = rolling_backtest(synthetic_returns, window=100, rebalance_freq=21)
        assert "portfolio_return" in result.columns
        assert "cumulative_return" in result.columns

    def test_output_length(self, synthetic_returns: pd.DataFrame) -> None:
        """Output length should equal total - window."""
        result = rolling_backtest(synthetic_returns, window=100, rebalance_freq=21)
        assert len(result) == len(synthetic_returns) - 100

    def test_weight_columns(self, synthetic_returns: pd.DataFrame) -> None:
        """Output should include weight columns for each asset."""
        result = rolling_backtest(synthetic_returns, window=100, rebalance_freq=21)
        for col in synthetic_returns.columns:
            assert col in result.columns

    def test_both_strategies(self, synthetic_returns: pd.DataFrame) -> None:
        """Both strategies should run without error."""
        r1 = rolling_backtest(synthetic_returns, window=100, strategy="max_sharpe")
        r2 = rolling_backtest(synthetic_returns, window=100, strategy="min_variance")
        assert len(r1) > 0
        assert len(r2) > 0
