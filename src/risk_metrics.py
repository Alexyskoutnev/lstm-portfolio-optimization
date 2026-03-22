import logging
from typing import cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard number of trading days per year, used for annualization.
_TRADING_DAYS_PER_YEAR = 252


def _annualize_mean(daily_mean: float) -> float:
    """Scale a daily mean return to an annualized figure.

    Args:
        daily_mean: Average daily return.

    Returns:
        Annualized return.
    """
    return daily_mean * _TRADING_DAYS_PER_YEAR


def _annualize_volatility(daily_std: float) -> float:
    """Scale a daily standard deviation to annualized volatility.

    Args:
        daily_std: Daily standard deviation of returns.

    Returns:
        Annualized volatility.
    """
    # Volatility scales with the square root of time under i.i.d. assumption.
    return daily_std * np.sqrt(_TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute the annualized Sharpe ratio.

    Args:
        returns: Series of daily log returns.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Annualized Sharpe ratio.
    """
    excess = _annualize_mean(cast(float, returns.mean())) - risk_free_rate
    vol = _annualize_volatility(cast(float, returns.std()))

    # Guard against division by zero when returns have no variance.
    if vol <= 0:
        logger.debug("Zero volatility encountered; returning Sharpe of 0.0")
        return 0.0

    ratio = float(excess / vol)
    logger.info("Sharpe ratio: %.4f (excess=%.4f, vol=%.4f)", ratio, excess, vol)
    return ratio


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute the annualized Sortino ratio.

    Uses downside deviation (only negative returns) in the denominator,
    penalizing harmful volatility while ignoring upside variance.

    Args:
        returns: Series of daily log returns.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Annualized Sortino ratio.
    """
    excess = _annualize_mean(cast(float, returns.mean())) - risk_free_rate

    # Only negative returns contribute to downside risk.
    downside = returns[returns < 0]
    downside_std = _annualize_volatility(cast(float, downside.std()))

    if downside_std <= 0:
        logger.debug("Zero downside deviation; returning Sortino of 0.0")
        return 0.0

    ratio = float(excess / downside_std)
    logger.info("Sortino ratio: %.4f (downside_std=%.4f)", ratio, downside_std)
    return ratio


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown from a return series.

    Args:
        returns: Series of daily log returns.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.25 for 25% drawdown).
    """
    # Build equity curve from simple returns, then track the running peak.
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()

    # Drawdown is the percentage decline from each running peak.
    drawdown = (cumulative - running_max) / running_max
    mdd = float(drawdown.min())

    logger.info("Max drawdown: %.4f", mdd)
    return mdd


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute historical Value-at-Risk (VaR).

    Args:
        returns: Series of daily returns.
        confidence: Confidence level (e.g., 0.95 for 95% VaR).

    Returns:
        VaR as a negative decimal representing the worst expected
        daily loss at the given confidence level.
    """
    # The (1 - confidence) quantile gives the loss threshold.
    var = float(returns.quantile(1 - confidence))
    logger.info("VaR (%.0f%%): %.4f", confidence * 100, var)
    return var


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute Conditional VaR (Expected Shortfall / CVaR).

    The average loss in the worst (1 - confidence) fraction of days.

    Args:
        returns: Series of daily returns.
        confidence: Confidence level.

    Returns:
        CVaR as a negative decimal.
    """
    var = value_at_risk(returns, confidence)

    # Average all returns that fall at or below the VaR threshold.
    tail_returns = returns[returns <= var]
    cvar = float(tail_returns.mean())

    logger.info("CVaR (%.0f%%): %.4f", confidence * 100, cvar)
    return cvar


def portfolio_turnover(weights_df: pd.DataFrame) -> float:
    """Compute average portfolio turnover.

    Measures how much the portfolio weights change at each rebalance.

    Args:
        weights_df: DataFrame of portfolio weights over time (T x N).

    Returns:
        Average turnover (sum of absolute weight changes per rebalance).
    """
    # First row diff is NaN (no prior period), which mean() ignores.
    changes = weights_df.diff().abs().sum(axis=1)
    turnover = float(changes.mean())

    logger.info("Average portfolio turnover: %.4f", turnover)
    return turnover


def compute_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute all risk metrics for a return series.

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
