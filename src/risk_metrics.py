"""Risk metrics for evaluating portfolio and strategy performance.

Why do we need risk metrics?
----------------------------
In finance, earning a high return is only half the story.  Two strategies can
both average 10% per year, yet one might achieve that with smooth, steady gains
while the other swings wildly -- gaining 50% one year and losing 30% the next.
Risk metrics let us quantify *how* a return was achieved, not just *what* it
was.  They answer questions like:

* "How much pain (volatility, drawdowns) did an investor endure for this
  return?"  (Sharpe ratio, Sortino ratio, max drawdown)
* "What is the worst-case scenario I should plan for?"  (Value-at-Risk,
  Conditional VaR)
* "How expensive is this strategy to operate?"  (Turnover)

Without risk metrics, we could not meaningfully compare strategies, set
position sizes, or satisfy regulatory requirements (e.g., Basel III mandates
banks report VaR and CVaR).

Annualization conventions
~~~~~~~~~~~~~~~~~~~~~~~~~
Daily returns are tiny numbers (often < 0.1%) that are hard to interpret.  We
*annualize* them so every strategy is quoted on the same yearly basis,
regardless of whether we evaluated it over 30 days or 3 years.

* **Returns** are annualized by **multiplying** by 252 (the approximate number
  of trading days in a year).  This follows from the linearity of expectation:
  E[annual] = 252 * E[daily].

* **Volatility** is annualized by multiplying by **sqrt(252)**, not 252.  This
  is because *variance* (not standard deviation) adds over time when returns
  are independent: Var[annual] = 252 * Var[daily].  Taking the square root of
  both sides gives: Std[annual] = sqrt(252) * Std[daily].  This is a direct
  consequence of how standard deviation scales -- it grows with the square root
  of the number of observations, not linearly.
"""

import logging
from typing import cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# There are roughly 252 trading days in a calendar year (weekends and market
# holidays are excluded).  This constant is the universal standard used across
# the finance industry for converting daily statistics to annual ones.
_TRADING_DAYS_PER_YEAR = 252


def _annualize_mean(daily_mean: float) -> float:
    """Scale a daily mean return to an annualized figure.

    Financial intuition:
        If you earn an average of 0.04% per day, your expected annual return is
        0.04% * 252 = ~10.08%.  We simply multiply because expected values are
        linear: E[sum of 252 days] = 252 * E[one day].

    Example:
        daily_mean = 0.0004 (0.04% per day)
        annualized = 0.0004 * 252 = 0.1008 (about 10.1% per year)

    Args:
        daily_mean: Average daily return.

    Returns:
        Annualized return.
    """
    # Linear scaling: expected annual return = daily expected return * trading days.
    return daily_mean * _TRADING_DAYS_PER_YEAR


def _annualize_volatility(daily_std: float) -> float:
    """Scale a daily standard deviation to annualized volatility.

    Financial intuition:
        Volatility measures how "noisy" returns are.  A stock with 1% daily
        volatility has annualized volatility of 1% * sqrt(252) ~ 15.87%, *not*
        1% * 252 = 252%.  The reason is rooted in probability: if daily returns
        are independent, their *variances* add (not their std devs).  So:

            Var_annual = 252 * Var_daily
            => Std_annual = sqrt(252) * Std_daily

        This sqrt-of-time rule is why volatility doesn't "blow up" over long
        periods the way a simple multiplication would suggest.

    Example:
        daily_std = 0.01 (1% daily volatility)
        annualized = 0.01 * sqrt(252) = 0.1587 (about 15.9% per year)

    Limitation:
        This assumes daily returns are independent and identically distributed
        (i.i.d.).  In reality, markets exhibit volatility clustering (calm days
        cluster together, as do turbulent days), which violates i.i.d. and can
        make this estimate optimistic during crises.

    Args:
        daily_std: Daily standard deviation of returns.

    Returns:
        Annualized volatility.
    """
    # Volatility scales with the square root of time under i.i.d. assumption.
    # This is because variance (std^2) is additive for independent variables,
    # so we take sqrt after summing 252 daily variances: sqrt(252 * daily_var).
    return daily_std * np.sqrt(_TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute the annualized Sharpe ratio.

    Financial intuition:
        The Sharpe ratio is the "reward per unit of risk" and is arguably the
        most widely used performance metric in finance.  It asks: "For every
        percentage point of volatility I endured, how much *excess* return
        (above the risk-free rate) did I earn?"

        A Sharpe of 1.0 means you earned 1% of excess return for every 1% of
        volatility -- generally considered good.  A Sharpe above 2.0 is
        excellent; below 0.5 is mediocre.

    Example:
        Suppose a strategy has an annualized excess return of 8% and an
        annualized volatility of 16%.  Its Sharpe ratio = 8% / 16% = 0.5.
        Another strategy with the same 8% return but only 8% volatility has a
        Sharpe of 1.0 -- it delivered the same return with half the risk.

    When to use:
        Use Sharpe to compare any two strategies or assets on a risk-adjusted
        basis.  It's the standard "first metric" everyone reports.

    Limitations:
        - Treats upside and downside volatility equally.  A strategy that
          occasionally surges upward is "penalized" for that positive surprise.
          (The Sortino ratio addresses this; see below.)
        - Assumes returns are normally distributed; can be misleading for
          strategies with fat tails or skew (e.g., option selling).
        - Sensitive to the time period chosen.

    Args:
        returns: Series of daily log returns.
        risk_free_rate: Annualized risk-free rate (e.g., 0.05 for 5%).  This
            represents the return you could earn with zero risk (typically
            short-term government bonds like US T-bills).

    Returns:
        Annualized Sharpe ratio.
    """
    # Compute annualized excess return: how much we beat the risk-free rate by.
    excess = _annualize_mean(cast(float, returns.mean())) - risk_free_rate

    # Compute annualized volatility: the "price" we paid in uncertainty.
    vol = _annualize_volatility(cast(float, returns.std()))

    # Guard against division by zero when returns have no variance.
    # This can happen with a perfectly hedged portfolio or constant returns.
    if vol <= 0:
        logger.debug("Zero volatility encountered; returning Sharpe of 0.0")
        return 0.0

    # The ratio itself: excess return divided by volatility.
    ratio = float(excess / vol)
    logger.info("Sharpe ratio: %.4f (excess=%.4f, vol=%.4f)", ratio, excess, vol)
    return ratio


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute the annualized Sortino ratio.

    Financial intuition:
        The Sortino ratio is a refinement of the Sharpe ratio that only
        penalizes *downside* volatility.  The key insight is that investors
        don't mind upside surprises -- if a stock jumps 5% in a day, that's
        great!  They only dislike *losses*.  Sharpe treats a +5% surprise and
        a -5% surprise as equally risky; Sortino correctly ignores the positive
        one.

        By using only the standard deviation of negative returns in the
        denominator, Sortino gives a fairer score to strategies that have high
        total volatility but whose volatility comes mainly from big gains
        (e.g., trend-following or momentum strategies).

    Example:
        Strategy A: 10% annual return, 15% total volatility, 10% downside vol.
          Sharpe = 10/15 = 0.67, Sortino = 10/10 = 1.0.
        Strategy B: 10% annual return, 15% total volatility, 15% downside vol.
          Sharpe = 10/15 = 0.67, Sortino = 10/15 = 0.67.
        Sortino reveals that A has a better risk profile (its vol comes from
        gains), while Sharpe cannot distinguish them.

    When to use:
        Use Sortino when you suspect a strategy's return distribution is
        asymmetric (skewed).  It's especially informative for strategies like
        momentum, options, or ML-based signals that may have non-normal return
        distributions.

    Limitations:
        - Requires enough negative-return observations to estimate downside
          deviation reliably.  With few data points, the estimate is noisy.
        - Like Sharpe, it's a single-number summary and can't capture the full
          shape of the return distribution.

    Args:
        returns: Series of daily log returns.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Annualized Sortino ratio.
    """
    # Same numerator as Sharpe: annualized excess return over the risk-free rate.
    excess = _annualize_mean(cast(float, returns.mean())) - risk_free_rate

    # Only negative returns contribute to downside risk.  We filter out all
    # gains because the Sortino philosophy is: volatility from profits is
    # desirable, not risky.
    downside = returns[returns < 0]

    # Annualize the downside deviation using the same sqrt(252) rule.
    downside_std = _annualize_volatility(cast(float, downside.std()))

    if downside_std <= 0:
        logger.debug("Zero downside deviation; returning Sortino of 0.0")
        return 0.0

    # Divide excess return by downside-only volatility instead of total volatility.
    ratio = float(excess / downside_std)
    logger.info("Sortino ratio: %.4f (downside_std=%.4f)", ratio, downside_std)
    return ratio


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown from a return series.

    Financial intuition:
        Maximum drawdown (MDD) is the largest peak-to-trough decline in
        portfolio value.  It answers the gut-wrenching question: "If I invested
        at the worst possible time, how much would I have lost before things
        recovered?"

        Emotionally, drawdowns are what cause investors to panic-sell.  A -30%
        drawdown means that if you had $1,000,000 invested, at one point you
        were staring at $700,000 on your screen.  Even if the strategy
        eventually recovered, many people simply cannot stomach that kind of
        loss and will abandon the strategy at the worst moment.  This is why
        MDD is sometimes called the "ulcer metric" -- it measures how much
        pain a strategy can inflict.

    Example:
        Portfolio grows from $100 to $150 (peak), then falls to $120, then
        recovers to $180.  The max drawdown occurred from $150 to $120:
        MDD = (120 - 150) / 150 = -0.20 (-20%).

    When to use:
        Always report MDD alongside Sharpe/Sortino.  A strategy with a great
        Sharpe but a -60% max drawdown may be theoretically attractive but
        psychologically unbearable for most investors.

    Limitations:
        - It's a single worst-case event; it doesn't tell you how *often*
          large drawdowns occur.
        - Highly path-dependent: the same set of returns in a different order
          produces a different MDD.

    Args:
        returns: Series of daily log returns.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.25 for 25% drawdown).
    """
    # Build the equity curve: track the cumulative growth of $1 invested.
    # (1 + r_t) gives the growth factor for each day; cumprod chains them.
    cumulative = (1 + returns).cumprod()

    # Track the running peak -- the highest portfolio value achieved so far.
    # At each point in time, this is the "high-water mark."
    running_max = cumulative.cummax()

    # Drawdown at each point: how far below the peak the portfolio currently is.
    # A value of -0.10 means the portfolio is 10% below its all-time high.
    drawdown = (cumulative - running_max) / running_max

    # The maximum drawdown is the deepest trough (most negative value).
    mdd = float(drawdown.min())

    logger.info("Max drawdown: %.4f", mdd)
    return mdd


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute historical Value-at-Risk (VaR).

    Financial intuition:
        VaR answers: "What is the most I can expect to lose on a *typical* bad
        day?"  More precisely, at 95% confidence, VaR says: "On 95% of days,
        my loss will be no worse than this number."

        For example, a 95% daily VaR of -2% means: "On 19 out of 20 trading
        days, I won't lose more than 2%.  But roughly once a month (5% of
        ~20 trading days), I *could* lose more than 2%."

        VaR was popularized by J.P. Morgan in the 1990s and became a standard
        regulatory requirement under Basel II/III.

    Example:
        With 1000 days of returns sorted from worst to best, the 95% VaR is
        the 50th-worst day (the 5th percentile).  If that value is -0.025,
        then VaR = -2.5%.

    When to use:
        Report VaR for any portfolio where you need to communicate worst-case
        expectations to stakeholders or regulators.

    Limitations:
        - VaR tells you the *boundary* of the bad tail, but says nothing about
          how bad things get *beyond* that boundary.  A VaR of -2% could mean
          the worst day was -2.1% or -20%.  This is precisely why regulators
          also require CVaR (see conditional_var below).
        - Historical VaR assumes the past distribution of returns will repeat,
          which may not hold during unprecedented crises.

    Args:
        returns: Series of daily returns.
        confidence: Confidence level (e.g., 0.95 for 95% VaR).

    Returns:
        VaR as a negative decimal representing the worst expected
        daily loss at the given confidence level.
    """
    # The (1 - confidence) quantile gives the loss threshold.
    # At confidence=0.95, we look at the 5th percentile of returns, meaning
    # only 5% of observed days had a loss worse than this value.
    var = float(returns.quantile(1 - confidence))
    logger.info("VaR (%.0f%%): %.4f", confidence * 100, var)
    return var


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute Conditional VaR (Expected Shortfall / CVaR).

    Financial intuition:
        CVaR (also called Expected Shortfall) answers the question VaR
        deliberately ignores: "When things go *really* bad (beyond VaR), how
        bad do they actually get?"  It is the *average* loss on the very worst
        days -- those in the tail beyond the VaR threshold.

        Why regulators prefer CVaR over VaR:
        - VaR only tells you the door to the danger zone; CVaR tells you what
          happens inside.  A bank could have a VaR of -2% but a CVaR of -15%
          (meaning the rare bad days are catastrophic).  VaR alone would hide
          this tail risk.
        - CVaR is a "coherent" risk measure (it satisfies mathematical
          properties like sub-additivity: diversifying can never make CVaR
          worse), while VaR is not.  This matters when aggregating risks
          across desks/portfolios.
        - The Basel III regulatory framework moved from VaR to CVaR (called
          "Expected Shortfall") precisely because of these properties.

    Example:
        If 95% VaR = -2%, CVaR might be -3.5%.  This means: "On the 5% worst
        days, the average loss is 3.5%."  CVaR is always worse (more negative)
        than VaR because it averages the tail beyond the VaR cutoff.

    When to use:
        Always pair CVaR with VaR for a complete picture of tail risk.  CVaR is
        essential for strategies that might have fat tails (e.g., anything
        involving leverage, concentrated positions, or illiquid assets).

    Limitations:
        - Requires enough tail observations for a reliable estimate.  With only
          100 data points and 95% confidence, you're averaging just 5 values.
        - Like VaR, it's backward-looking and may underestimate risk in regime
          changes.

    Args:
        returns: Series of daily returns.
        confidence: Confidence level.

    Returns:
        CVaR as a negative decimal.
    """
    # First, find the VaR threshold -- this defines where the "danger zone" begins.
    var = value_at_risk(returns, confidence)

    # Average all returns that fall at or below the VaR threshold.
    # These are the truly bad days -- the ones in the far-left tail of the
    # return distribution.  Their average tells us the expected loss given
    # that we are already having a very bad day.
    tail_returns = returns[returns <= var]
    cvar = float(tail_returns.mean())

    logger.info("CVaR (%.0f%%): %.4f", confidence * 100, cvar)
    return cvar


def portfolio_turnover(weights_df: pd.DataFrame) -> float:
    """Compute average portfolio turnover.

    Financial intuition:
        Turnover measures how aggressively a strategy reshuffles its holdings
        at each rebalance.  If you hold 50% stocks and 50% bonds today, then
        rebalance to 70% stocks and 30% bonds tomorrow, the turnover for that
        day is |0.70 - 0.50| + |0.30 - 0.50| = 0.20 + 0.20 = 0.40 (40%).

        Why it matters -- transaction costs:
        Every trade costs money.  You pay bid-ask spreads (the difference
        between buying and selling prices), brokerage commissions, and market
        impact (large orders move the price against you).  A strategy that
        earns 10% per year but has 500% annual turnover could easily spend
        3-5% on transaction costs alone, destroying most of the profit.

        This is why turnover is sometimes called the "silent killer" of
        strategies: a beautiful backtest can fall apart once you account for
        real-world trading frictions.  Lower turnover generally means a more
        implementable and profitable strategy in practice.

    Example:
        A buy-and-hold strategy has ~0% turnover (you rarely trade).
        A daily mean-reversion strategy might have 200%+ turnover (you
        completely flip your portfolio every few days).

    When to use:
        Always report turnover alongside return metrics.  A Sharpe ratio of 2.0
        with 10% turnover is *vastly* more attractive than a Sharpe of 2.5 with
        500% turnover, because the second strategy's real-world returns will be
        eaten by trading costs.

    Limitations:
        - Turnover alone doesn't tell you the *cost*; that depends on the
          specific assets traded (liquid large-cap stocks are cheap to trade;
          illiquid small-caps or emerging-market bonds are expensive).
        - Doesn't account for partial fills or market impact on large orders.

    Args:
        weights_df: DataFrame of portfolio weights over time (T x N), where
            each row is a date and each column is an asset.  Weights in each
            row should sum to 1 (fully invested portfolio).

    Returns:
        Average turnover (sum of absolute weight changes per rebalance).
    """
    # diff() computes the change in each asset's weight from one period to the
    # next.  abs() ensures buys and sells both count as activity (selling 10%
    # of one asset and buying 10% of another is 20% turnover, not 0%).
    # sum(axis=1) totals across all assets for each rebalance date.
    # The first row's diff is NaN (no prior period), which mean() ignores.
    changes = weights_df.diff().abs().sum(axis=1)

    # Average over all rebalance periods to get a "typical" turnover per period.
    turnover = float(changes.mean())

    logger.info("Average portfolio turnover: %.4f", turnover)
    return turnover


def compute_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute all risk metrics for a return series.

    This is a convenience function that runs every metric in one call, making
    it easy to generate a full "report card" for a strategy or portfolio.

    The returned dictionary provides a holistic view:
        - annualized_return: How much did we earn?
        - annualized_volatility: How bumpy was the ride?
        - sharpe_ratio: How much return per unit of total risk?
        - sortino_ratio: How much return per unit of *downside* risk?
        - max_drawdown: What was the worst peak-to-trough loss?
        - var_95: What's the worst "normal" daily loss?
        - cvar_95: How bad are the truly catastrophic days?

    Args:
        returns: Series of daily log returns.
        risk_free_rate: Annualized risk-free rate.
        confidence: Confidence level for VaR/CVaR.

    Returns:
        Dictionary with all computed metrics.
    """
    logger.info("Computing full risk metrics suite for %d observations", len(returns))

    return {
        "annualized_return": float(_annualize_mean(cast(float, returns.mean()))),
        "annualized_volatility": float(_annualize_volatility(cast(float, returns.std()))),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "var_95": value_at_risk(returns, confidence),
        "cvar_95": conditional_var(returns, confidence),
    }
