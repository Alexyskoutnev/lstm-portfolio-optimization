# Risk-Adjusted Portfolio Optimization: ML vs. Markowitz

A modular Python framework for comparing classical Markowitz Mean-Variance Optimization (MVO) with LSTM-based deep learning for portfolio construction. Built for the AIM 5005 Machine Learning course at Yeshiva University.

---

## Project Overview

### The Question

> Can deep learning (LSTM) improve risk-adjusted portfolio performance compared to classical Markowitz optimization, especially during high-volatility market regimes?

### The Approach

We build two portfolio construction pipelines and evaluate them head-to-head:

1. **Markowitz MVO Baseline** — Classical mean-variance optimization using historical returns and covariance
2. **LSTM-Enhanced Portfolio** — Deep learning forecasts of future returns, fed into portfolio optimization

Both are evaluated with a rigorous walk-forward backtesting framework across 15+ years of market data (2010–2025), covering multiple market regimes including the COVID crash, 2022 rate hikes, and calm bull markets.

### Key Metrics

| Metric | What It Measures |
|--------|-----------------|
| Sharpe Ratio | Return per unit of total risk |
| Sortino Ratio | Return per unit of downside risk |
| Maximum Drawdown | Worst peak-to-trough decline |
| VaR / CVaR | Tail risk (worst-case losses) |
| Portfolio Turnover | Trading frequency / transaction cost proxy |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        src/config.py                         │
│              (All constants: tickers, dates, params)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
┌───────────────┐           ┌────────────────┐
│   src/data/   │           │  src/models/   │
│               │           │                │
│ fetcher.py    │──────────►│ mvo.py         │
│  └ yfinance   │  returns  │  └ min-var     │
│  └ caching    │           │  └ max-sharpe  │
│               │           │  └ eff.frontier│
│ preprocessing │           │  └ backtest    │
│  └ log returns│           │                │
│  └ splits     │           │ lstm.py (TODO) │
│               │           │  └ forecasting │
│ regime.py     │           │  └ allocation  │
│  └ VIX-based  │           └────────┬───────┘
│  └ rolling-std│                    │
└───────────────┘                    │
                                     ▼
                    ┌────────────────────────────┐
                    │     src/risk_metrics.py     │
                    │                            │
                    │ sharpe_ratio()             │
                    │ sortino_ratio()            │
                    │ max_drawdown()             │
                    │ compute_all_metrics()      │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │    src/visualization.py     │
                    │                            │
                    │ plot_efficient_frontier()  │
                    │ plot_cumulative_returns()  │
                    │ plot_weight_allocation()   │
                    │ plot_regime_returns()      │
                    │ plot_drawdown()            │
                    │ plot_rolling_sharpe()      │
                    └────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility | Key Functions |
|--------|---------------|---------------|
| `src/config.py` | Single source of truth for all project constants — tickers, date ranges, thresholds, paths | Constants only, no logic |
| `src/data/fetcher.py` | Download and cache ETF/VIX data from Yahoo Finance via yfinance | `fetch_prices()`, `fetch_vix()` |
| `src/data/preprocessing.py` | Transform raw prices into model-ready features | `compute_log_returns()`, `compute_simple_returns()`, `split_data()` |
| `src/data/regime.py` | Label market volatility regimes for conditional analysis | `label_regimes_vix()`, `label_regimes_rolling_std()` |
| `src/data/__init__.py` | Public API combining all data steps into one call | `load_dataset()` |
| `src/models/mvo.py` | Classical Markowitz optimization and backtesting engine | `minimum_variance_weights()`, `max_sharpe_weights()`, `efficient_frontier()`, `rolling_backtest()` |
| `src/models/lstm.py` | *(TODO)* LSTM return forecasting and portfolio construction | — |
| `src/risk_metrics.py` | Compute all risk-adjusted performance metrics | `sharpe_ratio()`, `sortino_ratio()`, `max_drawdown()`, `value_at_risk()`, `conditional_var()` |
| `src/visualization.py` | Generate publication-quality plots for analysis | 6 plot functions |

### Data Flow

```
Yahoo Finance API
       │
       ▼
  fetch_prices() ──► Raw Adj. Close prices (DataFrame: dates × tickers)
       │
       ▼
  compute_log_returns() ──► Daily log returns (DataFrame: dates × tickers)
       │
       ├──► split_data() ──► Train (70%) / Val (15%) / Test (15%)
       │
       ├──► label_regimes_vix() ──► "low" / "medium" / "high" per day
       │
       └──► estimate_parameters() ──► μ (expected returns), Σ (covariance)
                    │
                    ├──► minimum_variance_weights() ──► w*
                    ├──► max_sharpe_weights() ──► w*
                    └──► rolling_backtest() ──► daily portfolio returns
                                │
                                ▼
                        compute_all_metrics() ──► Sharpe, Sortino, MDD, VaR, CVaR
                                │
                                ▼
                        plot_*() functions ──► PNG files in plots/
```

---

## Dataset

### Assets: S&P 500 Sector ETFs

We use all 11 GICS (Global Industry Classification Standard) sector ETFs, providing diversified exposure to the entire US equity market:

| ETF | Sector | Inception |
|-----|--------|-----------|
| XLK | Technology | 1998 |
| XLF | Financials | 1998 |
| XLE | Energy | 1998 |
| XLY | Consumer Discretionary | 1998 |
| XLV | Health Care | 1998 |
| XLP | Consumer Staples | 1998 |
| XLI | Industrials | 1998 |
| XLB | Materials | 1998 |
| XLRE | Real Estate | 2015 |
| XLU | Utilities | 1998 |
| XLC | Communication Services | 2018 |

### Time Period

- **Start**: 2010-01-01
- **End**: 2025-12-31
- **Trading days**: ~4,000

This covers multiple market regimes: post-2008 recovery, 2020 COVID crash, 2021 bull market, 2022 rate hikes.

### Train/Validation/Test Split

Temporal split (no shuffling, to prevent look-ahead bias):
- **Train**: First 70% (~2,815 days, 2010–2020)
- **Validation**: Next 15% (~603 days, 2020–2022)
- **Test**: Final 15% (~604 days, 2022–2025)

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
cd final_project

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Full Pipeline

```bash
python scripts/verify_data.py
```

This will:
1. Download all ETF and VIX data (cached in `data/raw/`)
2. Compute returns and identify volatility regimes
3. Run MVO baseline (min-variance and max-Sharpe strategies)
4. Run rolling backtest across full dataset
5. Compute all risk metrics
6. Generate 6 visualization plots in `plots/`

### Lint and Type Check

```bash
ruff check src/ tests/
pyright src/
```

---

## Project Structure

```
final_project/
├── src/
│   ├── __init__.py
│   ├── config.py               # Project constants and hyperparameters
│   ├── data/
│   │   ├── __init__.py         # load_dataset() public API
│   │   ├── fetcher.py          # yfinance data download + CSV caching
│   │   ├── preprocessing.py    # Returns computation, temporal splits
│   │   └── regime.py           # Volatility regime identification
│   ├── models/
│   │   ├── __init__.py
│   │   └── mvo.py              # Markowitz optimization + backtesting
│   ├── risk_metrics.py         # Sharpe, Sortino, MDD, VaR, CVaR
│   └── visualization.py        # All plotting functions
├── tests/
│   ├── data/
│   │   ├── test_preprocessing.py
│   │   └── test_regime.py
│   ├── models/
│   │   └── test_mvo.py
│   └── test_risk_metrics.py
├── scripts/
│   └── verify_data.py          # Full pipeline runner
├── docs/
│   ├── theory.md               # Comprehensive theory guide
│   └── superpowers/plans/      # Implementation plans
├── data/raw/                   # Cached CSV data (gitignored)
├── plots/                      # Generated visualizations (gitignored)
├── pyproject.toml              # uv + ruff + pyright config
├── requirements.txt
└── .gitignore
```

---

## Team

| Member | Responsibility |
|--------|---------------|
| Alexy Skoutnev | Data collection, preprocessing, MVO baseline |
| Shaun Mukahanana | LSTM model development and tuning |
| Tadiwanashe Chiremba | Backtesting engine, risk metrics, visualization |

---

## Documentation

- **[Theory Guide](docs/theory.md)** — In-depth explanation of all financial and ML concepts
- **Code Documentation** — All functions use Google-style docstrings

---

## References

- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
- Sharpe, W. F. (1994). The Sharpe Ratio. *Journal of Portfolio Management*, 21(1), 49–58.
- Sortino, F. A., & Van der Meer, R. (1991). Downside Risk. *Journal of Portfolio Management*, 17(4), 27–31.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Fischer, T., & Krauss, C. (2018). Deep Learning with LSTM Networks for Financial Market Predictions. *European Journal of Operational Research*, 270(2), 654–669.
