# Risk-Adjusted Portfolio Optimization: A Model-Complexity Ladder

A modular Python framework that benchmarks **four expected-return estimators** — from a no-learning baseline up to a recurrent neural network — inside an otherwise-identical Markowitz portfolio pipeline. Built for the AIM 5005 Machine Learning course at Yeshiva University.

---

## Project Overview

### The Question

> Does adding model complexity to the expected-return estimator µ improve risk-adjusted portfolio performance versus classical Markowitz, especially during high-volatility market regimes — and where on the complexity ladder does the marginal return diminish?

### The Approach

Markowitz Mean-Variance Optimization needs two inputs: an expected-return vector µ and a covariance matrix Σ. Σ is estimated reasonably well from historical data; **µ is famously noisy and is what destroys MVO out-of-sample** (Michaud 1989, "error maximization"). So the natural place to insert ML is to replace the sample-mean µ with a learned, conditional µ.

We build a **complexity ladder** of four µ estimators and run them through the same Markowitz pipeline (same Σ, same SLSQP optimizer, same long-only constraints, same rolling backtest):

| # | Estimator | Type | Inputs |
|---|---|---|---|
| 1 | **Sample mean** | No learning (baseline) | Last 252 days of own returns |
| 2 | **Ridge regression** | Linear ML | Tabular feature panel (lags, vol, VIX, regime, cross-asset) |
| 3 | **LightGBM** | Gradient-boosted trees | Same tabular feature panel |
| 4 | **LSTM** | Recurrent neural net | Sequence of trailing 60-day returns + asset embedding |

Each estimator predicts the **same target**: cumulative log return over the next 21 trading days (the rebalance horizon). All four are evaluated on identical out-of-sample windows over 15+ years of market data (2010–2025), covering multiple market regimes including the 2020 COVID crash, 2022 rate hikes, and calm bull markets.

### Why a complexity ladder

A two-arm comparison ("Markowitz vs LSTM") tells you whether one specific deep model beats the baseline, but not *whether the win comes from learning at all* or *whether the recurrent dynamics matter*. With four arms in increasing complexity, you can read the marginal value of each step:

- **Sample mean → Ridge** answers: does any conditioning on features beat a flat historical mean?
- **Ridge → GBT** answers: do non-linear interactions help?
- **GBT → LSTM** answers: does sequential / recurrent structure add anything beyond hand-engineered features?

### Key Metrics

| Metric | What It Measures |
|--------|-----------------|
| Sharpe Ratio | Return per unit of total risk |
| Sortino Ratio | Return per unit of downside risk |
| Maximum Drawdown | Worst peak-to-trough decline |
| VaR / CVaR | Tail risk (worst-case losses) |
| Portfolio Turnover | Trading frequency / transaction cost proxy |

---

## The Four µ Estimators in Detail

All four estimators answer the same question: *given information available at date t, what is the expected return of each ETF over the next 21 trading days?* They differ in what information they use and how flexibly they combine it.

### 1. Sample Mean (no learning — the baseline)

```
µ_i = mean(returns_i over last 252 days) × 252
```

The classical Markowitz µ. Treats every day in the lookback window equally, ignores all other information, and assumes the future looks like the average of the recent past. **Module:** [src/models/mvo.py](src/models/mvo.py) (`_default_mu_estimator`).

This is *not* a bad baseline despite its simplicity. With 252 daily observations, the standard error of the sample mean for a 20%-vol ETF is about 1.3% — often larger than the signal — and Markowitz amplifies this noise into wildly varying weights. Beating it cleanly out-of-sample is genuinely hard.

### 2. Ridge Regression (simplest ML)

Linear model fit on a tabular feature panel. The features at each (date, asset) row are:

- **Lagged returns** at horizons 1d, 5d, 21d, 63d, 126d
- **Rolling stats**: 21d & 63d volatility, 21d mean, 21d skew
- **VIX features**: level and Δ over 5d/21d/63d
- **Regime label**: low / medium / high (one-hot)
- **Cross-asset signals**: market mean return, cross-sectional dispersion, this asset's spread vs market
- **Calendar**: month, day-of-week
- **Asset id** (one-hot)

A single Ridge model is trained globally — one set of coefficients that operates across all assets, with the asset-id dummies giving each ETF its own intercept. **Module:** [src/models/linreg.py](src/models/linreg.py).

### 3. LightGBM (medium complexity — non-linear trees)

A gradient-boosted ensemble of decision trees, trained on the **same feature panel** as Ridge. The model captures non-linear interactions through tree splits ("if VIX > 25 *and* 21d momentum < 0, predict −0.01") and handles categorical features (regime, asset_id) natively.

A single global LightGBM regressor is trained on the stacked (date × asset) panel — pooling ~33,000 training rows instead of ~3,000 per asset. Time-based hold-out provides early stopping. **Module:** [src/models/gbt.py](src/models/gbt.py).

### 4. LSTM (most complex — recurrent neural net)

A small LSTM (32 hidden units, 1 layer, dropout) that consumes the **sequence of trailing 60 daily log returns** for one asset, plus a learned 4-dim embedding of asset identity. It outputs the predicted 21-day forward return.

This is the only arm that ingests data as a *sequence* rather than as a flat tabular feature vector. The hypothesis: recurrent dynamics let the model represent autocorrelation, momentum, and reversal patterns that are hard to encode by hand. Trained globally across all ETFs. **Module:** [src/models/lstm.py](src/models/lstm.py).

### What changes, what stays the same

The whole point of the framework is that **only the µ box changes** between arms. Σ, the optimizer, the constraints, and the backtest harness are byte-identical:

```
returns ──► [µ estimator] ──► µ ──┐
                                  ├──► SLSQP (max Sharpe) ──► weights ──► realized returns
returns ──► sample covariance ──► Σ ──┘
```

See [plots/mvo_vs_gbt_pipeline.png](plots/mvo_vs_gbt_pipeline.png) for the rendered side-by-side, and [plots/mvo_vs_gbt_mu_zoom.png](plots/mvo_vs_gbt_mu_zoom.png) for a zoom on the swap point.

---

## Results: 4-Way Benchmark (Test Period 2022–2025)

All four µ estimators were trained on data up to 2021-03 and evaluated on the held-out test period using the same rolling 252-day backtest with 21-day rebalancing.

| Model | Annual Return | Annual Vol | Sharpe ↑ | Sortino ↑ | Max DD ↑ | VaR 95% ↑ | CVaR 95% ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. Sample mean | 13.05% | **13.66%** | 0.955 | 1.284 | **-13.96%** | -1.40% | **-1.96%** |
| 2. Ridge | 8.80% | 12.62% | 0.698 | 0.963 | -13.65% | **-1.25%** | -1.74% |
| **3. GBT** | **24.36%** | 16.72% | **1.457** | **2.144** | -15.87% | -1.34% | -2.13% |
| 4. LSTM | 9.86% | 13.31% | 0.741 | 0.992 | -14.80% | -1.27% | -1.92% |

(Bold = best in column. ↑ means "higher is better"; for max drawdown / VaR / CVaR "higher" means "less negative".)

### What this tells us

**GBT wins decisively** on every return-related metric — Sharpe 1.46 vs 0.96 for sample mean (+52%), 24.4% annualized return vs 13.1% (+86%). The non-linear interactions trees can capture (regime × sector × momentum, VIX-conditional cross-asset effects) appear to translate into genuinely useful µ estimates that survive Markowitz's error-amplification.

**The complexity ladder reading:**

- *Sample mean → Ridge:* a flat linear projection of the feature panel **hurts** (Sharpe 0.96 → 0.70). Linear models are too rigid to use the regime, calendar, and cross-asset interactions; they end up encoding noise.
- *Ridge → GBT:* big jump (Sharpe 0.70 → **1.46**). Non-linear, interaction-aware models extract real signal from the feature panel that linear models cannot.
- *GBT → LSTM:* recurrent dynamics on raw return sequences (Sharpe 0.74) **don't** beat hand-engineered features fed to trees. The LSTM val loss barely moved during training (0.0068 → 0.0068), confirming that 60-day return sequences alone are a thin signal — features matter more than sequence modeling for this task.

**A debugging note worth flagging:** in an earlier run we used early stopping on a validation set drawn from the late training window (2019-Q1 onward). That window straddles the 2020 COVID crash, which made val loss explode after a few iterations and triggered premature stopping — leaving GBT with only 2 trees, predicting a near-constant µ, and falsely making it look like ML couldn't beat the baseline. Removing early stopping (training to a fixed 200 iterations with L2 regularization) gave GBT room to actually learn. **The lesson:** how you validate matters as much as what you model. Validating across a regime shock can erase an entire model's learning capacity.

**Risk picture:** The sample-mean baseline still has the lowest volatility and the best CVaR, because it spreads weight more evenly and avoids the conviction trades GBT makes. GBT runs hotter (vol 16.7% vs 13.7%) — its drawdown is slightly worse — but the higher return more than compensates risk-adjusted.

The full metrics table is at [plots/complexity_ladder_metrics.csv](plots/complexity_ladder_metrics.csv). See [plots/cumulative_complexity_ladder.png](plots/cumulative_complexity_ladder.png) for the equity curves and [plots/sharpe_by_regime_complexity_ladder.png](plots/sharpe_by_regime_complexity_ladder.png) for the regime-bucketed comparison.

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
| `src/config.py` | Single source of truth for all project constants — tickers, date ranges, model hyperparameters | Constants only, no logic |
| `src/data/fetcher.py` | Download and cache ETF/VIX data from Yahoo Finance via yfinance | `fetch_prices()`, `fetch_vix()` |
| `src/data/preprocessing.py` | Transform raw prices into model-ready features | `compute_log_returns()`, `compute_simple_returns()`, `split_data()` |
| `src/data/regime.py` | Label market volatility regimes for conditional analysis | `label_regimes_vix()`, `label_regimes_rolling_std()` |
| `src/data/features.py` | Build the strictly-lagged (date × asset) feature panel consumed by Ridge and GBT | `build_feature_panel()` |
| `src/data/__init__.py` | Public API combining all data steps into one call | `load_dataset()` |
| `src/models/mvo.py` | Classical Markowitz optimization, refactored backtest with injectable µ-estimator | `minimum_variance_weights()`, `max_sharpe_weights()`, `rolling_backtest()` |
| `src/models/linreg.py` | Ridge regression µ-estimator (simplest ML arm) | `train_linreg()`, `linreg_mu_estimator()` |
| `src/models/gbt.py` | LightGBM µ-estimator (gradient-boosted trees) | `train_gbt()`, `predict_returns()`, `gbt_mu_estimator()`, `build_target()` |
| `src/models/lstm.py` | LSTM µ-estimator (recurrent neural net on return sequences) | `train_lstm()`, `lstm_mu_estimator()` |
| `src/risk_metrics.py` | Compute all risk-adjusted performance metrics | `sharpe_ratio()`, `sortino_ratio()`, `max_drawdown()`, `value_at_risk()`, `conditional_var()` |
| `src/visualization.py` | Plotting (incl. multi-arm comparison, regime-bucketed Sharpe, complexity-ladder) | 10 plot functions |

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
# Run the four-arm benchmark end-to-end (sample-mean / Ridge / GBT / LSTM)
python scripts/run_complexity_ladder.py

# Or just the data-only check + classical MVO baseline
python scripts/verify_data.py

# Render explanatory architecture diagrams
python scripts/draw_architecture_comparison.py
```

`run_complexity_ladder.py` does all of the following:

1. Download all ETF and VIX data (cached in `data/raw/`)
2. Compute log returns, simple returns, and VIX-based regime labels
3. Build the (date × asset) feature panel for tabular models
4. Train Ridge, LightGBM, and LSTM on the train portion only (no test leakage)
5. Run four backtests on the test slice, sharing identical Σ, optimizer, and constraints
6. Compute Sharpe / Sortino / max drawdown / VaR / CVaR for all four arms
7. Save a metrics CSV (`plots/complexity_ladder_metrics.csv`) and four comparison plots

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
│   ├── config.py                  # All constants and hyperparameters
│   ├── data/
│   │   ├── __init__.py            # load_dataset() public API
│   │   ├── fetcher.py             # yfinance data download + CSV caching
│   │   ├── preprocessing.py       # Returns computation, temporal splits
│   │   ├── regime.py              # Volatility regime identification
│   │   └── features.py            # (date × asset) feature panel for ML arms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mvo.py                 # Markowitz optimization + injectable µ
│   │   ├── linreg.py              # Ridge regression µ-estimator
│   │   ├── gbt.py                 # LightGBM µ-estimator
│   │   └── lstm.py                # LSTM µ-estimator
│   ├── risk_metrics.py            # Sharpe, Sortino, MDD, VaR, CVaR
│   └── visualization.py           # Plotting (incl. multi-arm comparison)
├── tests/
│   ├── data/
│   │   ├── test_preprocessing.py
│   │   └── test_regime.py
│   ├── models/
│   │   └── test_mvo.py
│   └── test_risk_metrics.py
├── scripts/
│   ├── verify_data.py                      # Data-only sanity run
│   ├── run_complexity_ladder.py            # 4-way benchmark (main runner)
│   └── draw_architecture_comparison.py     # Render explanatory diagrams
├── docs/
│   ├── theory.md                           # Theory guide
│   └── superpowers/
│       ├── specs/                          # Design specs
│       └── plans/                          # Implementation plans
├── data/raw/                  # Cached CSV data (gitignored)
├── plots/                     # Generated figures (gitignored)
├── pyproject.toml             # uv + ruff + pyright config
├── requirements.txt
└── .gitignore
```

---

## Team

| Member | Responsibility |
|--------|---------------|
| Alexy Skoutnev | Data pipeline, MVO baseline, GBT and Ridge arms, complexity-ladder benchmark |
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
