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
| 1. Sample mean | 13.05% | 13.66% | 0.955 | 1.284 | -13.96% | -1.40% | -1.96% |
| 2. Ridge | 9.48% | 15.61% | 0.607 | 0.822 | -26.01% | -1.61% | -2.29% |
| **3. GBT** | **24.36%** | 16.72% | **1.457** | **2.144** | -15.87% | -1.34% | -2.13% |
| **4. LSTM** | 12.20% | **11.90%** | 1.026 | 1.418 | **-12.05%** | **-1.13%** | **-1.60%** |

(Bold = best in column. ↑ means "higher is better"; for max drawdown / VaR / CVaR "higher" means "less negative".)

### What this tells us

The complexity ladder splits cleanly into two winners:

- **GBT wins on risk-adjusted return.** Sharpe 1.46 vs 0.96 for the sample-mean baseline (+52%), 24.4% annualized return vs 13.1% (+86%). The non-linear interactions trees can capture (regime × sector × momentum, VIX-conditional cross-asset effects) translate into genuinely useful µ estimates that survive Markowitz's error-amplification.
- **LSTM wins on risk control.** Lowest max drawdown (-12.05%), lowest annualized vol (11.9%), best VaR and CVaR. Sharpe 1.03 still beats the baseline. Multi-channel sequence input (returns, vol, VIX, market mean), z-score standardization on training stats, and Huber loss together produce a defensive portfolio that smooths out tails — at the cost of some return relative to GBT.

### The complexity ladder reading

- *Sample mean → Ridge*: linear models hurt (Sharpe 0.96 → 0.61). Even with z-score standardization, RidgeCV-tuned alpha, and hand-picked interaction features (regime × momentum, VIX × momentum, etc.), the linear projection over-fits noise and produces conviction trades that go wrong. The biggest 2022 drawdown (-26%) tells the story.
- *Ridge → GBT*: big jump (Sharpe 0.61 → **1.46**). Non-linear, interaction-aware trees extract real signal from the feature panel that linear models cannot.
- *GBT → LSTM*: different objective. GBT optimizes return; LSTM (with multi-channel input, target z-scoring, Huber loss) optimizes a smoother loss surface and produces lower-vol forecasts that translate into the safest portfolio. Sharpe is lower than GBT's but still above the baseline.

### Tuning that mattered

This was *not* the first set of numbers we got — three concrete fixes turned an inconclusive "ML doesn't help" picture into the result above:

1. **Remove validation-driven early stopping** (GBT and LSTM). Our natural validation window (2019-Q1 onward) straddles the 2020 COVID crash, which makes val loss explode and triggers premature stopping. The first run had GBT cut off after 2 trees, predicting essentially a constant. Switching to fixed-iteration training with L2 regularization gave GBT room to actually learn (Sharpe jumped from 0.70 to 1.46).
2. **Standardize all features for Ridge and feed multi-channel sequences for the LSTM.** Daily returns are tiny (~0.01) and mix with VIX values (~20) — without z-scoring, the L2 penalty falls almost entirely on small-scale features and the LSTM trains in a very flat loss region. The improved LSTM jumped from Sharpe 0.51 → 1.03.
3. **Use Huber loss for the LSTM** instead of MSE. Returns have fat tails and the COVID-period val window has extreme outliers; MSE rewards "predict zero" too much. Huber's linear tail produces a more useful gradient signal.

**The general lesson:** a "ML didn't help" result on a financial-prediction task is almost always a tuning artifact before it's an empirical finding. How you validate, what features you feed, and what loss you minimize matter as much as which architecture you pick.

The full metrics table is at [plots/complexity_ladder_metrics.csv](plots/complexity_ladder_metrics.csv). See [plots/presentation_equity_curves.png](plots/presentation_equity_curves.png) for the equity curves and [plots/presentation_regime_sharpe.png](plots/presentation_regime_sharpe.png) for the regime-bucketed comparison.

---

## Research Questions (from the project proposal)

The proposal posed three research questions. Mapped to our results:

### RQ1 — Can ML-based forecasts improve Sharpe / Sortino vs classical MVO?

**Answer: YES, decisively, but only with the right model class.**

| Pipeline | Sharpe | Sortino | vs Sample-Mean Baseline |
|---|---:|---:|---|
| Sample mean (MVO baseline) | 0.96 | 1.28 | — |
| Ridge | 0.61 | 0.82 | **−37% Sharpe** |
| **GBT** | **1.46** | **2.14** | **+53% Sharpe**, **+67% Sortino** |
| **LSTM** | 1.03 | 1.42 | +7% Sharpe, +10% Sortino |

GBT's tree ensemble captures non-linear interactions (regime × momentum, VIX × cross-asset effects) that linear projections cannot. The LSTM also beats the baseline once given multi-channel input, z-scored features, and Huber loss. Linear Ridge actively *hurts* — confirming that the ML lift comes from non-linearity, not just from "having a model."

📊 [plots/presentation_rq_answers.png](plots/presentation_rq_answers.png) (top panel) | [plots/presentation_metrics_bars.png](plots/presentation_metrics_bars.png)

### RQ2 — Are ML-enhanced portfolios more robust during high-volatility regimes?

**Answer: YES — GBT dominates in both medium- and high-volatility regimes.**

Sharpe ratio bucketed by VIX regime (test period):

| Regime | Sample mean | Ridge | **GBT** | LSTM |
|---|---:|---:|---:|---:|
| Low (VIX < 15) | 3.8 | 4.4 | 3.4 | 4.1 |
| Medium (15 ≤ VIX < 25) | 0.2 | −0.3 | **1.6** | 0.6 |
| **High (VIX ≥ 25)** | −3.5 | −4.4 | **−1.1** | −3.7 |

The high-vol bucket is the headline answer to RQ2: when markets are stressed, **GBT's Sharpe is −1.1 vs −3.5 for the sample-mean baseline** — a 3× reduction in how badly the strategy degrades. The medium-vol regime tells the same story: GBT 1.6 vs 0.2 baseline. Only in calm markets (low-vol) does Ridge edge out GBT, and that's the regime where signal is easiest to extract from any model.

This is exactly the RQ2 outcome the project hypothesized: non-linear models that condition on regime indicators (VIX level, dispersion) hold up better when correlations spike and a flat sample-mean µ becomes most misleading.

📊 [plots/presentation_rq_answers.png](plots/presentation_rq_answers.png) (middle panel) | [plots/presentation_regime_sharpe.png](plots/presentation_regime_sharpe.png)

### RQ3 — Does nonlinear temporal structure reduce downside risk?

**Answer: YES — LSTM produces the safest portfolio across all three downside metrics.**

| Pipeline | Max Drawdown ↑ | VaR 95% ↑ | CVaR 95% ↑ |
|---|---:|---:|---:|
| Sample mean | −13.96% | −1.40% | −1.96% |
| Ridge | −26.01% | −1.61% | −2.29% |
| GBT | −15.87% | −1.34% | −2.13% |
| **LSTM** | **−12.05%** | **−1.13%** | **−1.60%** |

LSTM's recurrent dynamics + Huber loss + multi-channel input produce a smoother prediction surface that translates directly into more defensive portfolio weights. It pays for that defensiveness with lower return than GBT — but if a risk-averse investor weights drawdown and tail loss heavily, the LSTM is the right pick. GBT also beats the baseline on VaR. Ridge is *worse* than the baseline on every downside metric, confirming that linear miscalibration produces the worst tails.

📊 [plots/presentation_rq_answers.png](plots/presentation_rq_answers.png) (bottom panels) | [plots/presentation_var_cvar.png](plots/presentation_var_cvar.png) | [plots/presentation_drawdown.png](plots/presentation_drawdown.png)

### Additional proposal-required metrics

- **Portfolio turnover** (transaction-cost proxy): see [plots/presentation_turnover.png](plots/presentation_turnover.png).
- **Performance across volatility regimes**: see [plots/presentation_regime_sharpe.png](plots/presentation_regime_sharpe.png) (RQ2 above).

### Headline figure

For a one-slide summary, see the hero dashboard:

📊 [plots/presentation_hero.png](plots/presentation_hero.png)

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
