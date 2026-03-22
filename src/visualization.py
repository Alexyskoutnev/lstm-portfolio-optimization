"""Visualization utilities for portfolio analysis.

Provides plotting functions for efficient frontiers, backtest performance,
weight allocations, volatility regimes, and risk metrics.

Why visualization matters for portfolio analysis
-------------------------------------------------
Summary statistics (mean return, Sharpe ratio, volatility) compress an entire
return distribution into a single number.  That compression hides critical
information:

* **Drawdowns** -- A strategy with a great average return might have a -40%
  drawdown that would cause most investors to abandon it.  You cannot see
  this from the mean alone.
* **Regime behavior** -- A portfolio may perform well in calm markets and
  catastrophically in crises.  Histograms split by volatility regime reveal
  this immediately.
* **Weight concentration** -- An optimizer may dump 80% of capital into one
  asset.  A stacked-area chart makes concentration (and its changes over
  time) obvious at a glance.
* **Non-stationarity** -- Financial returns are NOT drawn from a fixed
  distribution.  Rolling Sharpe plots expose periods where a strategy's
  edge disappears entirely.

In short: numbers tell you *what* happened on average; plots tell you *when*,
*how*, and *how badly* things can go wrong.  Always plot before you trust a
backtest.
"""

import logging

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _save_figure(fig: matplotlib.figure.Figure, save_path: str) -> None:
    """Save a figure to disk at high resolution.

    Args:
        fig: The Matplotlib figure to save.
        save_path: Destination file path.
    """
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Figure saved to %s", save_path)


def _style_axis(
    ax: matplotlib.axes.Axes,
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> None:
    """Apply consistent styling to a single axis.

    Args:
        ax: The Matplotlib axes to style.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Axis title.
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    max_sharpe_point: tuple[float, float] | None = None,
    min_var_point: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot the efficient frontier with optional optimal portfolio markers.

    The efficient frontier is the set of portfolios that offer the highest
    expected return for each level of risk (volatility).  Understanding its
    shape tells you a lot:

    * **Curvature (the "bend")** -- The frontier bends to the left because
      diversification reduces portfolio volatility below the weighted average
      of individual asset volatilities.  The more the curve bends, the
      greater the diversification benefit.  If it were a straight line, the
      assets would be perfectly correlated and diversification would be
      useless.
    * **Color = Sharpe ratio** -- Each dot is colored by its Sharpe ratio
      (return per unit of risk).  The brightest point is the "tangency
      portfolio" -- the best risk-adjusted allocation.  Look for WHERE that
      bright spot sits: if it is far from the minimum-variance point, your
      portfolio is very sensitive to return estimates; if they are close,
      the optimizer mostly cares about covariances and is more robust.
    * **Max Sharpe vs Min Variance distance** -- When these two markers are
      far apart on the x-axis, small errors in expected-return forecasts
      will dramatically shift the optimal portfolio.  When they are close,
      the solution is more stable.  This is a quick robustness diagnostic.
    * **Flat top** -- If the frontier flattens at high volatility, adding
      more risk buys almost no extra return -- a clear sign to stay on the
      left (lower risk) side.

    Args:
        frontier: DataFrame with columns ["return", "volatility", "sharpe"].
        max_sharpe_point: (volatility, return) of max Sharpe portfolio.
        min_var_point: (volatility, return) of min variance portfolio.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Plotting efficient frontier with %d points", len(frontier))

    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color-code frontier points by Sharpe ratio for quick visual assessment.
    # Viridis colormap: darker = lower Sharpe, brighter/yellow = higher Sharpe.
    # The brightest cluster marks the region where risk-adjusted return peaks.
    scatter = ax.scatter(
        frontier["volatility"],
        frontier["return"],
        c=frontier["sharpe"],
        cmap="viridis",
        s=10,
        label="Efficient Frontier",
    )
    fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    if max_sharpe_point:
        ax.scatter(
            *max_sharpe_point, color="red", marker="*", s=200, label="Max Sharpe"
        )
    if min_var_point:
        ax.scatter(
            *min_var_point, color="blue", marker="*", s=200, label="Min Variance"
        )

    _style_axis(
        ax,
        xlabel="Annualized Volatility",
        ylabel="Annualized Return",
        title="Efficient Frontier",
    )
    ax.legend()

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_cumulative_returns(
    backtest_results: dict[str, pd.DataFrame],
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot cumulative returns for multiple strategies.

    This is the single most important backtest plot.  Things to look for:

    * **Divergence during crises** -- When markets crash (2008, 2020, etc.),
      the lines will fan apart.  Which strategy falls less?  That one has
      better tail-risk management.  Strategies that look identical in calm
      markets may behave very differently in drawdowns.
    * **Recovery speed** -- After a drop, how quickly does each line return
      to its prior peak?  Faster recovery = lower "time underwater", which
      matters enormously for real investors who may need to withdraw funds.
    * **Which strategy leads, and when** -- If Strategy A leads only in a
      single bull run but lags everywhere else, its good average return is
      misleading -- it is just lucky timing, not a robust edge.
    * **Smoothness vs. jaggedness** -- Smoother curves have lower volatility
      and are more comfortable to hold in practice.  A jagged line with the
      same endpoint represents a rougher ride and higher behavioral risk
      (the investor may panic-sell).
    * **Crossing patterns** -- Frequent crosses between strategy lines
      suggest they have similar long-run performance and the "winner"
      depends heavily on the evaluation window.

    Args:
        backtest_results: Dict mapping strategy name to backtest DataFrame
            (must contain "cumulative_return" column).
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    logger.info(
        "Plotting cumulative returns for %d strategies", len(backtest_results)
    )

    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, result in backtest_results.items():
        ax.plot(result.index, result["cumulative_return"], label=name, linewidth=1.5)

    _style_axis(
        ax,
        xlabel="Date",
        ylabel="Cumulative Return",
        title="Strategy Cumulative Returns Comparison",
    )
    ax.legend()
    # Zero line marks the break-even point.  Anything below this line means
    # the strategy has lost money since inception -- a quick gut-check.
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_weight_allocation(
    backtest_result: pd.DataFrame,
    asset_columns: list[str],
    title: str = "Portfolio Weight Allocation Over Time",
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot stacked area chart of portfolio weight allocation over time.

    This chart reveals HOW the optimizer is allocating capital, not just the
    end result.  Key things to look for:

    * **Concentrated vs. diversified** -- If one color dominates the chart
      (one asset gets 60-80%+ of the weight), the portfolio is concentrated
      and its performance will be driven almost entirely by that single
      asset.  A well-diversified portfolio shows multiple roughly-equal
      color bands.
    * **Weight stability vs. rapid changes** -- Smooth, slowly-evolving
      bands mean the optimizer's recommendations are stable.  Sudden,
      dramatic shifts (the colors "shuffle" rapidly) indicate high
      turnover.  High turnover is costly in practice because every trade
      incurs transaction costs (commissions, bid-ask spread, market
      impact).  A strategy that looks great on paper may be unprofitable
      after costs if it rebalances too aggressively.
    * **Why weights shift** -- Weights change because the optimizer receives
      new data each rebalancing period.  Recent returns, volatilities, and
      correlations shift, so the "optimal" mix shifts too.  During crises,
      correlations spike (everything falls together), and the optimizer may
      flee to the least-correlated asset -- often bonds or cash.
    * **Disappearing assets** -- If an asset's band shrinks to zero, the
      optimizer has decided it adds no value given the current regime.
      This can be a sign of extreme estimation error or genuinely poor
      risk-adjusted prospects for that asset.

    Args:
        backtest_result: DataFrame from rolling_backtest with asset weight columns.
        asset_columns: List of column names containing weights.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Plotting weight allocation for %d assets", len(asset_columns))

    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots(figsize=(12, 6))

    weights = backtest_result[asset_columns]
    ax.stackplot(weights.index, weights.T, labels=asset_columns, alpha=0.8)

    _style_axis(ax, xlabel="Date", ylabel="Weight", title=title)
    # Place legend outside the plot area so it doesn't obscure data.
    # With many assets the legend can cover the stacked area; anchoring it
    # to the right keeps all color bands visible for concentration analysis.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_ylim(0, 1)

    if save_path:
        _save_figure(fig, save_path)
    return fig


def _plot_single_regime_histogram(
    ax: matplotlib.axes.Axes,
    returns: pd.Series,
    regime: str,
    color: str,
    count: int,
) -> None:
    """Plot a histogram of returns for a single volatility regime.

    Each histogram shows the distribution of daily returns that occurred
    while the market was in a particular volatility state.  The vertical
    dashed line at the mean lets you instantly compare central tendency
    across regimes.

    Args:
        ax: The axes to draw on.
        returns: Return values for this regime.
        regime: Regime label (e.g. "low", "medium", "high").
        color: Fill color for the histogram bars.
        count: Number of observations in this regime.
    """
    ax.hist(returns, bins=50, color=color, alpha=0.7, edgecolor="black")
    # Vertical line at the mean highlights central tendency.  Compare this
    # line across the three panels: in high-vol regimes the mean often
    # shifts left (negative), revealing that volatility and negative returns
    # tend to coincide -- the "leverage effect" in finance.
    ax.axvline(returns.mean(), color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"{regime.capitalize()} Volatility\n(n={count})")
    ax.set_xlabel("Daily Return")


def plot_regime_returns(
    returns: pd.Series,
    regimes: pd.Series,
    title: str = "Portfolio Returns by Volatility Regime",
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot return distributions by volatility regime.

    Markets are NOT one homogeneous environment; they cycle between calm
    periods (low volatility) and turbulent periods (high volatility).
    Splitting returns by regime reveals behavior that a single histogram
    would hide.  What to look for:

    * **Width of the distribution** -- High-vol histograms are much wider
      (fatter) than low-vol ones.  This is exactly what "high volatility"
      means: daily returns are more spread out, so both large gains and
      large losses are more likely.
    * **Fat tails in high-vol regimes** -- The high-vol histogram often
      shows returns far from the center that a normal (Gaussian) bell curve
      would consider nearly impossible.  These are "fat tails" or "tail
      risk".  They matter enormously because a single -8% day can wipe out
      months of steady +0.3% days.
    * **Mean line position** -- In the low-vol panel the mean is usually
      slightly positive (calm markets tend to drift up).  In the high-vol
      panel the mean often shifts toward zero or negative, confirming that
      turbulence is associated with losses, not just wider dispersion.
    * **Sample count (n)** -- The subtitle shows how many days fell into
      each regime.  If high-vol has very few days, the histogram is noisy
      and conclusions are tentative.

    Args:
        returns: Series of portfolio returns.
        regimes: Series of regime labels aligned with returns.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Plotting regime return distributions")

    fig: matplotlib.figure.Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    # Explicit cast so type checkers understand axes is an ndarray of Axes.
    # sharey=True forces the same y-axis scale so you can compare bar heights
    # across regimes -- if high-vol bars are shorter, that regime has fewer days.
    axes_arr: list[matplotlib.axes.Axes] = list(axes)

    regime_order = ["low", "medium", "high"]
    colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}

    for ax, regime in zip(axes_arr, regime_order, strict=True):
        mask = regimes == regime
        if mask.sum() > 0:
            regime_returns = pd.Series(returns[mask])
            _plot_single_regime_histogram(
                ax, regime_returns, regime, colors[regime], int(mask.sum())
            )

    axes_arr[0].set_ylabel("Frequency")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    return fig


def _compute_drawdown_series(cumulative: pd.Series) -> pd.Series:
    """Compute a drawdown series from a cumulative equity curve.

    Drawdown at time t = (current value - peak value) / peak value.
    It is always <= 0.  A drawdown of -0.30 means the portfolio is 30% below
    its all-time high at that point.

    Args:
        cumulative: Cumulative portfolio value (1-based).

    Returns:
        Series of drawdown values (negative decimals).
    """
    running_max = cumulative.cummax()
    # Drawdown measures the decline from each running peak as a fraction.
    return (cumulative - running_max) / running_max


def plot_drawdown(
    backtest_result: pd.DataFrame,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot drawdown over time.

    Drawdown is arguably the most important risk metric for real-world
    investing.  Why it matters so much:

    * **Asymmetry of losses** -- A -30% drawdown requires a subsequent
      +42.9% gain just to get back to even (because 0.70 * 1.429 = 1.0).
      A -50% drawdown requires +100%.  This asymmetry means large
      drawdowns are disproportionately destructive.
    * **Depth vs. duration** -- The red-filled area shows both.  A deep
      but brief spike (V-shape) is a sharp crash with fast recovery.  A
      shallow but prolonged trough means the strategy is slowly bleeding --
      psychologically harder to endure because there is no dramatic event,
      just relentless underperformance.
    * **Practical significance** -- Most institutional mandates have a
      maximum drawdown tolerance (e.g., "shut down the fund if drawdown
      exceeds 20%").  Individual investors often capitulate emotionally
      during drawdowns exceeding ~15-20%, locking in losses at the worst
      possible time.
    * **Top panel context** -- The upper plot (cumulative value) shows
      absolute performance, while the lower plot isolates the pain of
      losses relative to peaks.  Together they tell the full story.

    Args:
        backtest_result: DataFrame with "cumulative_return" column.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Plotting drawdown chart")

    # Shift cumulative returns to a 1-based equity curve so that drawdowns
    # are measured as fractions of invested capital (0.0 = no loss, -0.5 = half gone).
    cumulative = backtest_result["cumulative_return"] + 1
    drawdown = _compute_drawdown_series(cumulative)

    fig: matplotlib.figure.Figure
    ax1: matplotlib.axes.Axes
    ax2: matplotlib.axes.Axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(cumulative.index, cumulative, linewidth=1.5, color="#2c3e50")
    _style_axis(ax1, ylabel="Cumulative Value (1 = start)", title="Portfolio Value and Drawdown")

    ax2.fill_between(drawdown.index, drawdown, 0, color="#e74c3c", alpha=0.5)
    _style_axis(ax2, xlabel="Date", ylabel="Drawdown")

    if save_path:
        _save_figure(fig, save_path)
    return fig


def _compute_rolling_sharpe(
    returns: pd.Series,
    window: int,
) -> pd.Series:
    """Compute a rolling annualized Sharpe ratio series.

    The Sharpe ratio (mean return / std of returns, annualized) is the
    standard measure of risk-adjusted performance.  Computing it over a
    rolling window reveals how that ratio evolves -- a strategy with a
    great full-sample Sharpe might have spent half the period below zero.

    Annualization:
    * Mean daily return * 252 (trading days/year) -> annualized return.
    * Daily std * sqrt(252) -> annualized volatility.
    This assumes returns are roughly i.i.d., which is imperfect but
    standard practice.

    Args:
        returns: Daily portfolio returns.
        window: Rolling window size in trading days.

    Returns:
        Series of rolling Sharpe ratios.
    """
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    # Division by zero produces NaN, which is expected for constant-return windows.
    return rolling_mean / rolling_std


def plot_rolling_sharpe(
    backtest_result: pd.DataFrame,
    window: int = 63,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot rolling Sharpe ratio over time.

    A single Sharpe ratio for the full backtest is a convenient summary, but
    it hides the fact that strategy performance is non-stationary -- it
    drifts and shifts as market conditions change.  This plot exposes that
    time-variation.  Key observations:

    * **Why 63 days (~3 months)?** -- This default window is a common
      industry choice that balances responsiveness with noise reduction.
      Shorter windows (e.g., 21 days) are very noisy; longer windows
      (e.g., 252 days) smooth out regime changes you might want to see.
      63 days roughly corresponds to one fiscal quarter, a natural
      evaluation period.
    * **Sustained negative Sharpe** -- If the line stays below zero for
      months, the strategy is persistently losing money on a risk-adjusted
      basis.  This is a red flag: it means the strategy's edge (if any) has
      vanished for an extended period.  A real investor holding through this
      would need extraordinary conviction.
    * **Sharpe > 1 (green line)** -- A rolling Sharpe consistently above 1
      is exceptionally good.  Hedge funds typically target full-period
      Sharpes of 1-2.  Sustained readings above 2 in a backtest often
      indicate look-ahead bias or overfitting rather than genuine alpha.
    * **Variance of the line itself** -- If the rolling Sharpe swings
      wildly between +3 and -2, the strategy's performance is highly
      regime-dependent and hard to rely on.  A "boring" line that hugs 0.5
      is far more investable.

    Args:
        backtest_result: DataFrame with "portfolio_return" column.
        window: Rolling window in trading days (default 63 ~ 3 months).
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Plotting rolling %d-day Sharpe ratio", window)

    rolling_sharpe = _compute_rolling_sharpe(
        pd.Series(backtest_result["portfolio_return"]), window
    )

    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(rolling_sharpe.index, rolling_sharpe, linewidth=1.2, color="#3498db")
    # Reference lines at 0, +1, -1 for quick visual benchmarking.
    # Sharpe = 0 means the strategy is earning no excess return over cash.
    # Sharpe = 1 is a strong result; Sharpe = -1 means severe underperformance.
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.3, label="Sharpe = 1")
    ax.axhline(y=-1, color="red", linestyle="--", alpha=0.3, label="Sharpe = -1")

    _style_axis(
        ax,
        xlabel="Date",
        ylabel="Rolling Sharpe Ratio",
        title=f"Rolling {window}-Day Sharpe Ratio",
    )
    ax.legend()

    if save_path:
        _save_figure(fig, save_path)
    return fig
