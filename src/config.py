"""Project-wide configuration constants.

All tunable parameters for data collection, preprocessing, and
volatility regime identification live here.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Universe: S&P 500 sector ETFs (all 11 GICS sectors)
# ---------------------------------------------------------------------------
# WHY these ETFs?  Each "XL*" fund tracks one of the 11 GICS (Global Industry
# Classification Standard) sectors of the S&P 500.  Together they span the
# entire large-cap US equity market while being far more manageable than the
# ~500 individual stocks.  Using sector ETFs instead of individual stocks
# gives us:
#   1. Diversification within each sector (reduces idiosyncratic noise).
#   2. High liquidity and tight bid-ask spreads (clean price data).
#   3. A covariance matrix that is 11x11 instead of 500x500, which is much
#      easier to estimate reliably and invert for portfolio optimization.
# Sector-level allocation is also how most institutional investors think about
# portfolio construction -- they first decide sector weights, then pick stocks
# within sectors.
TICKERS: Final[list[str]] = [
    "XLK",   # Technology          – largest sector by market cap (~30%)
    "XLF",   # Financials          – banks, insurance, asset managers
    "XLE",   # Energy              – oil, gas, pipelines; highly cyclical
    "XLY",   # Consumer Discret.   – retail, autos; sensitive to consumer confidence
    "XLV",   # Health Care         – pharma, biotech; defensive during downturns
    "XLP",   # Consumer Staples    – food, household goods; another defensive sector
    "XLI",   # Industrials         – manufacturing, aerospace; tracks GDP growth
    "XLB",   # Materials           – chemicals, mining; commodity-linked
    "XLRE",  # Real Estate         – REITs; interest-rate sensitive
    "XLU",   # Utilities           – regulated power companies; bond-proxy / defensive
    "XLC",   # Communication Svcs  – telecom + media (Meta, Google); created in 2018
]

# ---------------------------------------------------------------------------
# VIX – the market's "fear gauge"
# ---------------------------------------------------------------------------
# The CBOE Volatility Index (^VIX) is derived from the prices of near-term
# S&P 500 index options.  It represents the market's *expectation* of 30-day
# annualized volatility.  A high VIX means option traders are paying more for
# downside protection, which historically coincides with market stress.  We use
# it as an external, forward-looking signal to label volatility regimes.
VIX_TICKER: Final[str] = "^VIX"

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
# WHY 2010-2025?  Starting in 2010 avoids training on the 2008-09 Global
# Financial Crisis, which was a once-in-a-generation structural break.
# Including it would dominate the covariance estimates and distort the
# optimization.  The 2010-2025 window still captures plenty of varied market
# conditions: the post-GFC recovery, the 2015 China-scare, the 2018 vol
# spike, the COVID crash (2020), and the 2022 rate-hiking bear market.
# Fifteen years of daily data gives us roughly 3,780 observations -- enough
# for robust covariance estimation across 11 assets.
START_DATE: Final[str] = "2010-01-01"
END_DATE: Final[str] = "2025-12-31"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# WHY 252?  US equity markets are open roughly 252 days per year (365 calendar
# days minus weekends minus ~10 federal holidays).  A 252-day rolling window
# therefore captures exactly one year of trading history, which is the standard
# horizon for annualizing volatility and estimating covariance in finance.
ROLLING_WINDOW: Final[int] = 252  # Trading days in a year

# WHY 70 / 15 / 15?  This is a standard ML split, but the key constraint is
# that our split must be *temporal* (train on the past, validate/test on the
# future) to avoid look-ahead bias.  70% training gives the optimizer enough
# history to estimate stable covariances.  15% validation lets us tune
# hyperparameters (e.g., regularization strength) on "unseen" future data,
# and the final 15% test set is held out for an unbiased performance estimate.
# With ~3,780 daily observations, test alone has ~567 days -- more than two
# full years of trading, which is enough to judge whether a strategy survives
# different market conditions.
TRAIN_RATIO: Final[float] = 0.7
VAL_RATIO: Final[float] = 0.15  # Remaining 0.15 is test

# ---------------------------------------------------------------------------
# Volatility regime thresholds (VIX-based)
# ---------------------------------------------------------------------------
# WHY 15 and 25?
#   - VIX < 15  ("low vol"):  Markets are calm and complacent.  Historically,
#     VIX spends about a third of its time below 15.  During low-vol regimes,
#     cross-sector correlations are relatively low, so diversification works
#     well and mean-variance optimization tends to produce reliable portfolios.
#   - VIX 15-25 ("medium vol"):  Normal uncertainty.  The long-run median VIX
#     is around 17-18, so this bucket captures "typical" market conditions.
#   - VIX >= 25 ("high vol"):  Fear and stress.  During crises (COVID, 2022
#     rate shock), VIX can spike above 30-40+.  In high-vol regimes, all
#     sectors tend to sell off together -- correlations spike toward 1 --
#     which destroys the diversification benefit that mean-variance
#     optimization relies on.  Knowing the regime lets us adjust portfolio
#     construction accordingly (e.g., shrink toward equal-weight, increase
#     cash allocation, or use a more robust covariance estimator).
# These thresholds are widely used by practitioners and in academic literature
# (e.g., Whaley 2000, "The Investor Fear Gauge").
VIX_LOW_THRESHOLD: Final[float] = 15.0
VIX_HIGH_THRESHOLD: Final[float] = 25.0

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR: Final[str] = "data/raw"
