"""Markowitz Mean-Variance Optimization (MVO) portfolio model.

Implements classical MVO with support for:
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Efficient frontier computation
- Rolling-window out-of-sample backtesting
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Number of trading days per year, used to annualize daily return statistics.
_TRADING_DAYS_PER_YEAR = 252


def estimate_parameters(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate expected returns and covariance matrix from historical data.

    Annualizes daily sample statistics by scaling the mean by 252 (trading
    days per year) and the covariance likewise, under the standard i.i.d.
    daily-returns assumption.

    Args:
        returns: DataFrame of daily log returns (T x N).

    Returns:
        Tuple of (mean_returns, cov_matrix) where:
            - mean_returns: Annualized expected returns (N,).
            - cov_matrix: Annualized covariance matrix (N x N).
    """
    mean_returns = np.array(returns.mean()) * _TRADING_DAYS_PER_YEAR
    cov_matrix = np.array(returns.cov()) * _TRADING_DAYS_PER_YEAR
    return mean_returns, cov_matrix


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[float, float]:
    """Compute annualized portfolio return and volatility.

    Args:
        weights: Portfolio weights (N,).
        mean_returns: Annualized expected returns (N,).
        cov_matrix: Annualized covariance matrix (N x N).

    Returns:
        Tuple of (portfolio_return, portfolio_volatility).
    """
    port_return = float(weights @ mean_returns)
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
    return port_return, port_vol


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def _build_fully_invested_constraint() -> dict:
    """Return the equality constraint requiring weights to sum to 1."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _build_bounds(n: int, allow_short: bool) -> list[tuple[float, float]] | None:
    """Return weight bounds (long-only or unconstrained).

    Args:
        n: Number of assets.
        allow_short: If True, weights are unbounded (short-selling allowed).

    Returns:
        List of (lower, upper) tuples per asset, or None if unconstrained.
    """
    return None if allow_short else [(0.0, 1.0)] * n


def _solve_portfolio(
    objective: Any,
    n: int,
    allow_short: bool,
    extra_constraints: list[dict] | None = None,
    label: str = "portfolio",
) -> np.ndarray:
    """Run a constrained portfolio optimisation via SLSQP.

    SLSQP (Sequential Least-Squares Quadratic Programming) is used because
    the objective is smooth and we need both equality and bound constraints,
    which SLSQP handles efficiently for moderate-dimensional problems.

    Args:
        objective: Callable(w) -> float to minimise.
        n: Number of assets.
        allow_short: Whether to allow negative weights.
        extra_constraints: Additional constraint dicts beyond full-investment.
        label: Human-readable label for log messages.

    Returns:
        Optimal weight vector (N,).

    Raises:
        RuntimeError: If the solver fails to converge.
    """
    x0 = np.ones(n) / n  # Equal-weight starting point

    constraints = [_build_fully_invested_constraint()]
    if extra_constraints:
        constraints.extend(extra_constraints)

    bounds = _build_bounds(n, allow_short)

    logger.info("Starting %s optimisation (n_assets=%d, allow_short=%s)", label, n, allow_short)

    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        logger.error("%s optimisation failed: %s", label.capitalize(), result.message)
        raise RuntimeError(
            f"{label.capitalize()} optimisation failed: {result.message}"
        )

    logger.info("%s optimisation converged (iterations=%d)", label.capitalize(), result.nit)
    return result.x


# ---------------------------------------------------------------------------
# Public weight-computation functions
# ---------------------------------------------------------------------------

def minimum_variance_weights(
    cov_matrix: np.ndarray,
    allow_short: bool = False,
) -> np.ndarray:
    """Find the minimum variance portfolio weights.

    Args:
        cov_matrix: Annualized covariance matrix (N x N).
        allow_short: If True, allow negative weights.

    Returns:
        Optimal weights array (N,).
    """
    n = cov_matrix.shape[0]
    return _solve_portfolio(
        objective=lambda w: w @ cov_matrix @ w,
        n=n,
        allow_short=allow_short,
        label="min-variance",
    )


def max_sharpe_weights(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> np.ndarray:
    """Find the maximum Sharpe ratio portfolio weights.

    Args:
        mean_returns: Annualized expected returns (N,).
        cov_matrix: Annualized covariance matrix (N x N).
        risk_free_rate: Annualized risk-free rate.
        allow_short: If True, allow negative weights.

    Returns:
        Optimal weights array (N,).
    """
    def neg_sharpe(w: np.ndarray) -> float:
        """Negative Sharpe ratio (minimised to maximise Sharpe)."""
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        return -(ret - risk_free_rate) / vol

    n = len(mean_returns)
    return _solve_portfolio(
        objective=neg_sharpe,
        n=n,
        allow_short=allow_short,
        label="max-sharpe",
    )


def efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 50,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Compute the efficient frontier.

    Traces out the set of optimal portfolios by minimizing variance
    for a range of target returns between the lowest and highest
    single-asset expected returns.

    Args:
        mean_returns: Annualized expected returns (N,).
        cov_matrix: Annualized covariance matrix (N x N).
        n_points: Number of points on the frontier.
        allow_short: If True, allow negative weights.

    Returns:
        DataFrame with columns ["return", "volatility", "sharpe"] and
        one row per frontier point.
    """
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)

    logger.info(
        "Computing efficient frontier (%d points, target range %.4f–%.4f)",
        n_points,
        target_returns[0],
        target_returns[-1],
    )

    results: list[dict] = []
    for target in target_returns:
        point = _solve_frontier_point(mean_returns, cov_matrix, target, allow_short)
        if point is not None:
            results.append(point)

    logger.info("Efficient frontier: %d / %d points converged", len(results), n_points)
    return pd.DataFrame(results)


def _solve_frontier_point(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target: float,
    allow_short: bool,
) -> dict | None:
    """Solve for a single efficient-frontier point.

    Args:
        mean_returns: Annualized expected returns (N,).
        cov_matrix: Annualized covariance matrix (N x N).
        target: Target portfolio return for this point.
        allow_short: Whether to allow negative weights.

    Returns:
        Dict with keys "return", "volatility", "sharpe", or None if the
        optimisation failed to converge at this target.
    """
    n = len(mean_returns)

    # The target-return constraint pins the portfolio's expected return,
    # so the optimiser is free to only minimise variance.
    target_constraint = {
        "type": "eq",
        "fun": lambda w, t=target: w @ mean_returns - t,
    }

    try:
        weights = _solve_portfolio(
            objective=lambda w: w @ cov_matrix @ w,
            n=n,
            allow_short=allow_short,
            extra_constraints=[target_constraint],
            label=f"frontier(target={target:.4f})",
        )
    except RuntimeError:
        # Individual frontier points may be infeasible; skip silently.
        logger.debug("Frontier point at target=%.4f failed; skipping", target)
        return None

    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = ret / vol if vol > 0 else 0.0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = {"max_sharpe", "min_variance"}


def rolling_backtest(
    returns: pd.DataFrame,
    window: int = 252,
    strategy: str = "max_sharpe",
    rebalance_freq: int = 21,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Run a rolling-window out-of-sample backtest.

    At each rebalance date, estimates parameters from the trailing window
    and computes optimal weights. Portfolio return is computed
    out-of-sample for the next period.

    Args:
        returns: DataFrame of daily log returns (T x N).
        window: Lookback window in trading days for parameter estimation.
        strategy: One of "max_sharpe" or "min_variance".
        rebalance_freq: Rebalance every N trading days.
        risk_free_rate: Annualized risk-free rate for Sharpe optimization.

    Returns:
        DataFrame with columns:
            - "portfolio_return": Daily portfolio log return.
            - "cumulative_return": Cumulative portfolio return.
            - Plus one column per asset with the weights.

    Raises:
        ValueError: If *strategy* is not a recognised strategy name.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {sorted(_VALID_STRATEGIES)}."
        )

    n_assets = returns.shape[1]
    asset_names = returns.columns.tolist()
    oos_days = len(returns) - window  # number of out-of-sample days

    logger.info(
        "Starting rolling backtest: strategy=%s, window=%d, rebalance_freq=%d, "
        "oos_days=%d, n_assets=%d",
        strategy,
        window,
        rebalance_freq,
        oos_days,
        n_assets,
    )

    portfolio_returns: list[float] = []
    weight_history: list[np.ndarray] = []
    dates: list = []

    # Begin with equal weights so the portfolio is always fully invested,
    # even before the first optimisation completes or if it fails.
    current_weights = np.ones(n_assets) / n_assets

    for i in range(window, len(returns)):
        # Rebalance at the specified frequency (every rebalance_freq days
        # after the first out-of-sample date).
        if (i - window) % rebalance_freq == 0:
            current_weights = _rebalance(
                returns, i, window, strategy, risk_free_rate, current_weights
            )

        daily_return = _compute_daily_return(returns.iloc[i].values, current_weights)
        portfolio_returns.append(daily_return)
        weight_history.append(current_weights.copy())
        dates.append(returns.index[i])

    return _build_backtest_result(portfolio_returns, weight_history, dates, asset_names)


def _rebalance(
    returns: pd.DataFrame,
    current_idx: int,
    window: int,
    strategy: str,
    risk_free_rate: float,
    fallback_weights: np.ndarray,
) -> np.ndarray:
    """Estimate parameters and compute new portfolio weights.

    If the optimisation fails, the previous (fallback) weights are retained
    so the backtest can continue without interruption.

    Args:
        returns: Full returns DataFrame.
        current_idx: Current row index in *returns*.
        window: Lookback window length.
        strategy: Optimisation strategy name.
        risk_free_rate: Risk-free rate passed to max-Sharpe optimiser.
        fallback_weights: Weights to use if optimisation fails.

    Returns:
        New weight vector, or *fallback_weights* on failure.
    """
    train_slice = returns.iloc[current_idx - window : current_idx]
    mean_ret, cov_mat = estimate_parameters(train_slice)

    rebalance_date = returns.index[current_idx]
    logger.debug("Rebalancing on %s (strategy=%s)", rebalance_date, strategy)

    try:
        if strategy == "max_sharpe":
            return max_sharpe_weights(mean_ret, cov_mat, risk_free_rate=risk_free_rate)
        # strategy == "min_variance" (validated before reaching here)
        return minimum_variance_weights(cov_mat)
    except RuntimeError:
        # Optimisation can fail when the trailing window produces a
        # near-singular covariance matrix.  Fall back to previous weights
        # rather than crashing the entire backtest.
        logger.warning(
            "Optimisation failed on %s; retaining previous weights", rebalance_date
        )
        return fallback_weights


def _compute_daily_return(
    asset_returns: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute a single day's portfolio return.

    Args:
        asset_returns: Per-asset returns for one day (N,).
        weights: Current portfolio weights (N,).

    Returns:
        Weighted portfolio return (scalar).
    """
    return float(asset_returns @ weights)


def _build_backtest_result(
    portfolio_returns: list[float],
    weight_history: list[np.ndarray],
    dates: list,
    asset_names: list[str],
) -> pd.DataFrame:
    """Assemble the backtest output DataFrame.

    Args:
        portfolio_returns: Daily portfolio log returns.
        weight_history: Per-day weight snapshots.
        dates: Corresponding date index.
        asset_names: Column names for weight columns.

    Returns:
        Combined DataFrame of returns and weights.
    """
    result = pd.DataFrame({"portfolio_return": portfolio_returns}, index=dates)

    # Cumulative return: exponentiate the cumulative sum of log returns,
    # then subtract 1 to express as a fractional gain/loss.
    result["cumulative_return"] = result["portfolio_return"].cumsum().apply(np.exp) - 1

    weights_df = pd.DataFrame(weight_history, index=dates, columns=asset_names)
    result = pd.concat([result, weights_df], axis=1)
    return result
