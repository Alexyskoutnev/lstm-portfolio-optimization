"""Markowitz Mean-Variance Optimization (MVO) portfolio model.

Historical context
------------------
In 1952, Harry Markowitz published "Portfolio Selection" in the *Journal of
Finance*, introducing the idea that investors should evaluate portfolios — not
individual stocks — and that **diversification can reduce risk without
proportionally reducing expected return**.  He received the Nobel Prize in
Economics in 1990 for this work.

The key insight
---------------
Suppose you hold two stocks that each fluctuate 20 % per year.  If their
returns are perfectly correlated (they always move together), the combined
portfolio still fluctuates 20 %.  But if the correlation is *less than 1*,
some of their ups and downs cancel out, and the portfolio's overall
fluctuation is *less* than 20 % — even though each stock individually hasn't
changed.  MVO formalises this by treating portfolio selection as a
**constrained optimisation** problem:

    maximise  expected return
    subject to  risk ≤ some limit, weights sum to 1

Or equivalently: for a given target return, find the weights that *minimise*
portfolio variance.  The set of all such optimal (return, risk) pairs traces
out the **efficient frontier** — a curve in return-vs-risk space.

What this module provides
-------------------------
- **Minimum variance portfolio** — lowest-risk portfolio on the frontier.
- **Maximum Sharpe ratio portfolio** — best risk-adjusted return.
- **Efficient frontier** computation (many points on the curve).
- **Rolling-window out-of-sample backtesting** to evaluate how well the
  strategy would have performed historically without look-ahead bias.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Number of trading days per year, used to annualize daily return statistics.
# Stock exchanges in the US are open roughly 252 days per year (365 minus
# weekends minus holidays).  This is the standard industry convention.
_TRADING_DAYS_PER_YEAR = 252


def estimate_parameters(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate expected returns and covariance matrix from historical data.

    Why sample mean and covariance?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    These are the simplest *unbiased* estimators of the true (unknown)
    population parameters.  The sample mean of daily returns approximates the
    expected daily return, and the sample covariance matrix captures both each
    asset's individual variance and the pairwise co-movement (correlation)
    between all asset pairs.

    Annualization math
    ~~~~~~~~~~~~~~~~~~
    Under the standard assumption that daily log-returns are *independent and
    identically distributed* (i.i.d.):

    * **Mean scales linearly with time:**
      E[annual return] = E[daily return] × 252.
      Intuition: if you earn 0.04 % per day on average, over 252 days you
      expect 252 × 0.04 % ≈ 10.08 % per year.

    * **Variance also scales linearly** (a property of sums of independent
      random variables), so Cov_annual = Cov_daily × 252.
      Note: *standard deviation* (volatility) therefore scales by √252 ≈ 15.87.

    Key limitation — estimation error
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sample estimates are *noisy*.  A classic result (Michaud 1989) shows that
    small changes in estimated means or covariances can produce **wildly
    different** optimal weights.  For example, if two stocks have nearly the
    same estimated return (say 10.0 % vs. 10.1 %), the optimiser may put
    almost all weight into one of them — even though the 0.1 % difference is
    well within estimation noise.  This "error maximisation" property of MVO
    is its most important practical drawback and motivates techniques like
    shrinkage estimators, Black-Litterman, or simply using the minimum-
    variance portfolio (which avoids return estimates entirely).

    Example:
    ~~~~~~~
    If you have 3 years of daily data for 5 stocks (≈756 rows × 5 columns),
    this function returns a length-5 vector of annualized expected returns and
    a 5×5 annualized covariance matrix.

    Args:
        returns: DataFrame of daily log returns (T × N), where T is the
            number of trading days and N is the number of assets.

    Returns:
        Tuple of (mean_returns, cov_matrix) where:
            - mean_returns: Annualized expected returns, shape (N,).
            - cov_matrix: Annualized covariance matrix, shape (N × N).
    """
    # Sample mean of each column → daily expected return per asset, then
    # multiply by 252 to annualize.
    mean_returns = np.array(returns.mean()) * _TRADING_DAYS_PER_YEAR

    # pandas .cov() computes the sample covariance matrix (using N-1
    # denominator for unbiasedness).  Multiplying by 252 annualizes it.
    cov_matrix = np.array(returns.cov()) * _TRADING_DAYS_PER_YEAR

    return mean_returns, cov_matrix


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[float, float]:
    """Compute annualized portfolio return and volatility.

    The matrix math, intuitively
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **Portfolio return** — ``w^T μ`` (dot product of weights and expected
    returns) is simply a *weighted average*.  If you put 60 % in an asset
    returning 10 % and 40 % in one returning 5 %, the portfolio return is
    0.6 × 10 % + 0.4 × 5 % = 8 %.

    **Portfolio variance** — ``w^T Σ w`` is more subtle.  When you expand
    the matrix multiplication, you get::

        Var(portfolio) = Σ_i Σ_j  w_i · w_j · Cov(i, j)

    This means the portfolio's risk depends not just on each asset's own
    variance (the diagonal of Σ), but on every *pairwise covariance* (the
    off-diagonal terms).  This is exactly why diversification works: when
    covariances are low or negative, those cross-terms shrink the total
    variance below the weighted sum of individual variances.

    Example with 2 assets
    ~~~~~~~~~~~~~~~~~~~~~
    Suppose w = [0.5, 0.5], each asset has variance 0.04, and Cov(1,2) = 0.01.

        Portfolio variance = 0.5² × 0.04 + 2 × 0.5 × 0.5 × 0.01 + 0.5² × 0.04
                           = 0.01 + 0.005 + 0.01 = 0.025
        Portfolio volatility = √0.025 ≈ 15.8 %

    Each individual asset has volatility √0.04 = 20 %, but the portfolio's
    is only 15.8 % — a free reduction in risk from diversification.

    Args:
        weights: Portfolio weights (N,).  Should sum to 1.
        mean_returns: Annualized expected returns (N,).
        cov_matrix: Annualized covariance matrix (N × N).

    Returns:
        Tuple of (portfolio_return, portfolio_volatility), both annualized.
    """
    # w^T μ  — weighted average return
    port_return = float(weights @ mean_returns)

    # √(w^T Σ w) — portfolio volatility (standard deviation of returns)
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights))

    return port_return, port_vol


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def _build_fully_invested_constraint() -> dict:
    """Return the equality constraint requiring weights to sum to 1.

    This is the "fully invested" or "budget" constraint: every dollar of
    capital must be allocated to *some* asset.  Mathematically: Σ w_i = 1.
    """
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _build_bounds(n: int, allow_short: bool) -> list[tuple[float, float]] | None:
    """Return weight bounds (long-only or unconstrained).

    In a **long-only** portfolio (allow_short=False), each weight is
    restricted to [0, 1]: you cannot bet *against* an asset (short-sell)
    or allocate more than 100 % of capital to a single asset.

    When **short-selling is allowed**, weights are unbounded — a weight of
    −0.3 means you borrowed and sold 30 % of your capital in that asset,
    expecting its price to fall.

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

    What is SLSQP?
    ~~~~~~~~~~~~~~
    SLSQP stands for **Sequential Least-Squares Quadratic Programming**.
    It is a general-purpose numerical optimiser that works well when:

    1. The objective function is smooth (has continuous derivatives).
    2. You need both *equality* constraints (weights sum to 1) and *bound*
       constraints (each weight between 0 and 1).
    3. The problem is moderate-dimensional (up to a few hundred variables).

    At each iteration, SLSQP approximates the objective locally as a
    quadratic function, solves that simpler sub-problem, takes a step, and
    repeats until convergence.  Think of it like gradient descent but smarter
    — it uses curvature information to converge faster.

    Why equal-weight initialisation?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The starting point ``x0 = [1/N, 1/N, ..., 1/N]`` is the *equal-weight*
    or *1/N* portfolio.  This is a sensible default because:

    * It already satisfies the budget constraint (weights sum to 1).
    * It lies in the interior of the feasible region (all weights > 0),
      giving the solver room to move in any direction.
    * Empirically, 1/N is a surprisingly strong benchmark — DeMiguel et al.
      (2009) showed it often outperforms optimised portfolios out-of-sample.

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
    # Start from the equal-weight portfolio — a feasible, diversified
    # starting point (see docstring above for why this is a good choice).
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

    Why use minimum variance?
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    As discussed in ``estimate_parameters``, expected return estimates are
    notoriously noisy.  The minimum-variance portfolio sidesteps this problem
    entirely by **ignoring return estimates** and focusing only on the
    covariance matrix, which is generally estimated more reliably.

    The resulting portfolio is the leftmost point on the efficient frontier:
    the portfolio with the absolute lowest risk, regardless of return.

    Why this is a quadratic program
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The objective ``w^T Σ w`` is a *quadratic function* of the weights w
    (it contains terms like w_i × w_j, which are degree-2 polynomials).
    Combined with *linear* constraints (weights sum to 1, each weight ≥ 0),
    this makes it a **quadratic program** (QP) — one of the best-understood
    classes of optimisation problems, solvable efficiently and reliably.

    Args:
        cov_matrix: Annualized covariance matrix (N × N).
        allow_short: If True, allow negative weights.

    Returns:
        Optimal weights array (N,).
    """
    n = cov_matrix.shape[0]
    return _solve_portfolio(
        # Objective: minimise portfolio variance w^T Σ w.
        # Note we don't take the square root — minimising variance and
        # minimising volatility (std dev) yield the same optimal weights
        # because √ is a monotonically increasing function.
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

    What is the Sharpe ratio?
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    The Sharpe ratio = (portfolio return − risk-free rate) / portfolio
    volatility.  It measures *return per unit of risk*.  A Sharpe ratio of
    1.0 means you earn 1 % of excess return for every 1 % of volatility.
    Higher is better.

    What is the risk-free rate?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This is the return you could earn with *zero* risk — typically the yield
    on short-term US Treasury bills (e.g., ≈4-5 % in 2024).  We subtract it
    because we only want to reward the portfolio for returns *above* what you
    could get risk-free.  A default of 0.0 means we treat all return as
    excess return (common in academic settings or low-rate environments).

    The tangent portfolio
    ~~~~~~~~~~~~~~~~~~~~~
    Geometrically, the max-Sharpe portfolio is the point where a *line drawn
    from the risk-free rate* on the y-axis is tangent to the efficient
    frontier.  This "tangent portfolio" offers the steepest slope (Sharpe
    ratio) of any portfolio on the frontier.  Every rational investor (in
    the Markowitz framework) should hold some combination of this tangent
    portfolio and the risk-free asset.

    Why minimise *negative* Sharpe?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``scipy.optimize.minimize`` can only *minimise* functions.  Since we want
    to *maximise* the Sharpe ratio, we minimise its negative:
    min(−Sharpe) ⟺ max(Sharpe).

    Args:
        mean_returns: Annualized expected returns (N,).
        cov_matrix: Annualized covariance matrix (N × N).
        risk_free_rate: Annualized risk-free rate (e.g., 0.04 for 4 %).
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

    What is the efficient frontier?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Imagine plotting every possible portfolio as a dot on a graph where the
    x-axis is risk (volatility) and the y-axis is expected return.  The
    "cloud" of all possible portfolios forms a region whose *upper-left
    boundary* is the efficient frontier — the set of portfolios that offer
    the **highest return for each level of risk**.

    Any portfolio *below* this curve is "dominated": there exists another
    portfolio with the same risk but higher return (or the same return but
    lower risk).  A rational investor should never hold a dominated portfolio.

    Why is it a curve and not a straight line?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If you mix two assets with imperfect correlation, the resulting
    risk-return combination lies to the *left* of the straight line
    connecting them — because diversification reduces risk.  This "bowing
    out" effect is what makes the frontier a curve (specifically, a
    hyperbola in the long-only case).

    How we compute it
    ~~~~~~~~~~~~~~~~~
    We sweep a target return from the lowest to the highest single-asset
    expected return.  At each target, we solve: "minimise variance subject
    to achieving exactly this return."  Each solution gives one (volatility,
    return) point on the frontier.

    Args:
        mean_returns: Annualized expected returns (N,).
        cov_matrix: Annualized covariance matrix (N × N).
        n_points: Number of points on the frontier (more = smoother curve).
        allow_short: If True, allow negative weights.

    Returns:
        DataFrame with columns ["return", "volatility", "sharpe"] and
        one row per successfully converged frontier point.
    """
    # Sweep target returns from the lowest to the highest individual asset
    # return.  Going beyond this range would require leverage or short-selling
    # in a long-only setting, which is infeasible.
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

    For a given target return, find the weights that minimise portfolio
    variance.  This is the "dual" formulation of the MVO problem: instead of
    maximising return for a risk budget, we minimise risk for a return target.

    Args:
        mean_returns: Annualized expected returns (N,).
        cov_matrix: Annualized covariance matrix (N × N).
        target: Target portfolio return for this point.
        allow_short: Whether to allow negative weights.

    Returns:
        Dict with keys "return", "volatility", "sharpe", or None if the
        optimisation failed to converge at this target.
    """
    n = len(mean_returns)

    # The target-return constraint pins the portfolio's expected return to a
    # specific value, so the optimiser is free to only minimise variance.
    # The ``t=target`` default-argument trick captures the current loop value
    # (otherwise Python closures would bind to the loop variable by reference
    # and all constraints would use the *last* target value).
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
        # This can happen when the target return is unachievable with the
        # given constraints (e.g., no long-only combination of assets can
        # reach a very high target return).
        logger.debug("Frontier point at target=%.4f failed; skipping", target)
        return None

    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = ret / vol if vol > 0 else 0.0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = {"max_sharpe", "min_variance"}


def _default_mu_estimator(
    window_returns: pd.DataFrame,
    current_date: pd.Timestamp,  # noqa: ARG001 — kept for interface symmetry
) -> np.ndarray:
    """Default mu estimator: sample mean of the lookback window, annualized.

    This is the classical Markowitz behavior. Kept as a separate function so
    it satisfies the same callable signature as learned estimators (e.g.,
    ``gbt_mu_estimator``), letting ``rolling_backtest`` treat all mu sources
    uniformly.
    """
    return np.array(window_returns.mean()) * _TRADING_DAYS_PER_YEAR


def rolling_backtest(
    returns: pd.DataFrame,
    window: int = 252,
    strategy: str = "max_sharpe",
    rebalance_freq: int = 21,
    risk_free_rate: float = 0.0,
    mu_estimator: Callable[[pd.DataFrame, pd.Timestamp], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Run a rolling-window out-of-sample backtest.

    What is walk-forward backtesting?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A naïve approach would estimate parameters from *all* available data and
    then evaluate performance on that same data.  This is circular —
    analogous to training a classifier on the test set — and dramatically
    overstates real-world performance.

    Walk-forward (rolling-window) backtesting avoids this by simulating what
    an investor *actually could have done* at each point in time:

    1. On each rebalance date, look back at *only* the most recent ``window``
       days of data (e.g., the past year).
    2. Estimate parameters (mean, covariance) from that window.
    3. Compute optimal weights and hold them for the next ``rebalance_freq``
       days (e.g., one month ≈ 21 trading days).
    4. Record the *out-of-sample* daily returns during that holding period.
    5. Slide the window forward and repeat.

    Why not just train once?  Non-stationarity.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Financial markets are *non-stationary* — the statistical properties of
    returns (means, variances, correlations) change over time.  A covariance
    matrix estimated from 2015 data may be a poor description of 2020 risk
    (e.g., COVID changed correlation structures dramatically).  By
    re-estimating every ``rebalance_freq`` days, the model adapts to the
    most recent market regime.

    What does "rebalancing" mean?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Rebalancing means selling some assets and buying others to return the
    portfolio to its target weights.  Even if you do nothing, market movements
    cause weights to drift (a stock that rises becomes a larger share of your
    portfolio).  Periodic rebalancing keeps the portfolio aligned with the
    optimiser's recommendation.  The default of 21 days (≈1 month) balances
    responsiveness against trading costs.

    Args:
        returns: DataFrame of daily log returns (T × N).
        window: Lookback window in trading days for parameter estimation.
            Default 252 ≈ 1 year.
        strategy: One of "max_sharpe" or "min_variance".
        rebalance_freq: Rebalance every N trading days.  Default 21 ≈ 1 month.
        risk_free_rate: Annualized risk-free rate for Sharpe optimization.
        mu_estimator: Optional callable producing the µ vector at each
            rebalance date. Signature is
            ``(window_returns, current_date) -> np.ndarray``. When None
            (default), the classical sample-mean estimator is used,
            preserving pre-refactor behavior. Pass a learned estimator
            (e.g. ``gbt_mu_estimator``, ``lstm_mu_estimator``) to swap in
            an ML-based µ source.

    Returns:
        DataFrame with columns:
            - "portfolio_return": Daily portfolio log return.
            - "cumulative_return": Cumulative portfolio return.
            - Plus one column per asset with the portfolio weights on that day.

    Raises:
        ValueError: If *strategy* is not a recognised strategy name.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {sorted(_VALID_STRATEGIES)}."
        )

    if mu_estimator is None:
        mu_estimator = _default_mu_estimator

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

    # Begin with equal weights (1/N) so the portfolio is always fully
    # invested, even before the first optimisation completes or if it fails.
    # This is also the starting point for the optimiser itself — see the
    # discussion in _solve_portfolio about why 1/N is a strong default.
    current_weights = np.ones(n_assets) / n_assets

    for i in range(window, len(returns)):
        # Rebalance at the specified frequency (every rebalance_freq days
        # after the first out-of-sample date).
        if (i - window) % rebalance_freq == 0:
            current_weights = _rebalance(
                returns, i, window, strategy, risk_free_rate, current_weights,
                mu_estimator,
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
    mu_estimator: Callable[[pd.DataFrame, pd.Timestamp], np.ndarray],
) -> np.ndarray:
    """Estimate parameters and compute new portfolio weights.

    Why fall back to previous weights on failure?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Optimisation can fail when the trailing window produces a near-singular
    covariance matrix — this happens when assets are nearly collinear (e.g.,
    two ETFs tracking the same index) or when the window is too short
    relative to the number of assets.  Rather than crashing the entire
    backtest, we keep the previous weights.  This mirrors real-world practice:
    if your model breaks on rebalance day, you simply hold your current
    positions.

    Args:
        returns: Full returns DataFrame.
        current_idx: Current row index in *returns*.
        window: Lookback window length.
        strategy: Optimisation strategy name.
        risk_free_rate: Risk-free rate passed to max-Sharpe optimiser.
        fallback_weights: Weights to use if optimisation fails.
        mu_estimator: Callable returning the µ vector for the current
            rebalance date.

    Returns:
        New weight vector, or *fallback_weights* on failure.
    """
    train_slice = returns.iloc[current_idx - window : current_idx]
    rebalance_date = returns.index[current_idx]

    # Σ stays sample-based across all mu sources so the comparison isolates
    # the contribution of the expected-return estimator.
    _, cov_mat = estimate_parameters(train_slice)
    mean_ret = mu_estimator(train_slice, rebalance_date)

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

    This is simply the dot product of asset returns and weights: if asset A
    returned 1 % and you hold 60 % in it, that contributes 0.6 % to your
    portfolio return.  Summing across all assets gives the total.

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

    # Cumulative return from log returns: since log returns are additive,
    # the cumulative log return is just the running sum.  To convert back to
    # a simple (percentage) return, we exponentiate and subtract 1.
    # E.g., if cumulative log return = 0.10, simple return = e^0.10 − 1 ≈ 10.5 %.
    result["cumulative_return"] = result["portfolio_return"].cumsum().apply(np.exp) - 1

    weights_df = pd.DataFrame(weight_history, index=dates, columns=asset_names)
    result = pd.concat([result, weights_df], axis=1)
    return result
