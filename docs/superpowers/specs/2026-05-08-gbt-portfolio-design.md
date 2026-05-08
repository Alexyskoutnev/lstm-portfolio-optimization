# GBT-Based Portfolio Optimization — Design Spec

**Date:** 2026-05-08
**Author:** Alexy Skoutnev
**Project:** AIM 5005 ML Final Project — Risk-Adjusted Portfolio Optimization

---

## 1. Motivation

The project's central question is:

> Can ML improve risk-adjusted portfolio performance vs. classical Markowitz, especially during high-volatility market regimes?

The existing pipeline implements two of three planned arms:

- **Classical MVO** ([src/models/mvo.py](../../../src/models/mvo.py)) — sample-mean µ + sample covariance Σ → SLSQP optimizer.
- **LSTM-MVO** — planned (`src/models/lstm.py` not yet implemented).

This spec defines a third arm: **GBT-MVO**, where a gradient-boosting tree forecasts each ETF's expected return and feeds it into the same Markowitz optimizer.

The ML sub-problem inside the portfolio question is to estimate µ better than the sample mean. Σ is estimated reliably from history; µ is famously noisy and is what destroys MVO out-of-sample (Michaud 1989, "error maximization"). Replacing the sample-mean µ with a learned, conditional µ is the cleanest place to insert ML — and the only thing that should differ across the three arms.

---

## 2. Scope

### In scope

- New module `src/models/gbt.py`: training, prediction, µ-estimator interface.
- New module `src/data/features.py`: shared feature engineering (consumed by GBT now, available to LSTM later).
- Refactor `rolling_backtest` in `src/models/mvo.py` to accept an injectable µ-estimator. Default behavior unchanged.
- `pyproject.toml` adds `lightgbm>=4.0.0` and `optuna>=3.0.0`.
- Tests for feature engineering (no look-ahead leakage) and GBT integration.
- Comparison plots and regime-conditional metrics.

### Out of scope

- LSTM implementation (separate effort, separate teammate).
- End-to-end weight prediction (rejected during brainstorming — no clean labels for trees, breaks apples-to-apples comparison with LSTM, throws away Markowitz inductive bias).
- Alternative covariance estimators (Ledoit-Wolf shrinkage etc.) — Σ stays sample-based across all three arms to isolate the µ contribution.
- Transaction-cost modeling beyond turnover as a proxy.

---

## 3. Architecture

```
src/
├── data/
│   ├── fetcher.py           # unchanged
│   ├── preprocessing.py     # unchanged
│   ├── regime.py            # unchanged
│   └── features.py          # NEW — feature engineering
├── models/
│   ├── mvo.py               # rolling_backtest extended (backward compatible)
│   └── gbt.py               # NEW — GBT model + µ-estimator
├── config.py                # extended — GBT/feature hyperparameter defaults
├── risk_metrics.py          # unchanged
└── visualization.py         # extended — three-way comparison plots
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `src/data/features.py` | Build the (date × asset) feature panel from returns + VIX + regime labels. Returns a DataFrame indexed by (date, asset) with columns per feature group. Strictly lagged — every feature at time `t` depends only on data ≤ `t`. |
| `src/models/gbt.py` | Train a single global LightGBM regressor on the stacked panel, predict 21-day forward returns per asset, and expose a µ-estimator function compatible with the refactored `rolling_backtest`. |
| `src/models/mvo.py` (refactor) | Add a `mu_estimator` parameter to `rolling_backtest`. Default = current sample-mean behavior. Accepted callable signature: `mu_estimator(returns_window: pd.DataFrame, current_date: pd.Timestamp) -> np.ndarray`. |

### Backtest integration

`rolling_backtest` currently does:
```python
mean_ret, cov_mat = estimate_parameters(train_slice)
weights = max_sharpe_weights(mean_ret, cov_mat, ...)
```

After refactor:
```python
mean_ret = mu_estimator(train_slice, current_date)   # default: sample mean
_, cov_mat = estimate_parameters(train_slice)        # Σ unchanged
weights = max_sharpe_weights(mean_ret, cov_mat, ...)
```

The GBT µ-estimator on a rebalance day:
1. Build features for `current_date` for each of the 11 ETFs.
2. Predict ŷ for each asset using the trained model.
3. Annualize: `µ_i = ŷ_i × (252 / 21)`.
4. Return µ vector (length 11) to MVO.

---

## 4. ML Problem Definition

### Task

Supervised regression. Given a feature vector for `(date t, asset i)`, predict the realized 21-day forward log return:

```
y_{t,i} = sum_{k=1}^{21} log_return_{t+k, i}
```

21 trading days matches the rebalance frequency, so each prediction directly corresponds to "expected return over the next holding period."

### Model

**Single global LightGBM regressor** trained on the stacked panel. Reasons:

- Pools ~33,000 training rows (3,000 days × 11 assets) instead of ~3,000 per asset.
- Handles the 11-level `asset_id` natively as a categorical feature — no one-hot encoding.
- Allows the model to learn cross-sectional patterns (e.g., "when dispersion is high, defensive sectors outperform").

### Features

All features are strictly lagged. The full feature panel is built once over the entire date range (no leakage because each feature only references data at-or-before its row's timestamp).

| Group | Features |
|---|---|
| Own-asset returns | lag 1d, 5d, 21d, 63d, 126d |
| Own-asset stats | rolling vol (21d, 63d), rolling mean (21d), rolling skew (21d) |
| VIX | level, Δ5d, Δ21d, Δ63d |
| Regime | categorical (low / medium / high) |
| Cross-asset | market mean return (21d avg log return across all 11 sectors), cross-sectional dispersion (std across sectors' 21d returns), this asset's 21d return minus the market mean |
| Calendar | month-of-year, day-of-week |
| Identity | asset_id (categorical, 11 levels) |

### Target horizon rationale

21 days (≈ 1 month) matches the rebalance frequency. Shorter horizons (1d, 5d) yield more samples but have a much lower signal-to-noise ratio in equity returns. Longer horizons (63d+) are noisier and have fewer non-overlapping samples. 21d is the standard practitioner choice for monthly-rebalanced strategies.

### Annualization

Predictions are 21-day cumulative log returns. To match the existing MVO interface (which expects annualized µ), multiply by `252 / 21 ≈ 12`. Σ remains annualized via the existing `× 252` factor in `estimate_parameters`.

---

## 5. Training Methodology

### Walk-forward training

Mirrors the rolling backtest structure:

- **Initial fit:** trained on the first 70% of data (train split). Hyperparameter tuning happens here.
- **Refits during backtest:** every 63 trading days (~quarterly), retrain using all data available up to the refit date. Hyperparameters frozen between refits to keep cost reasonable; only refit annually if needed.
- **Strict no-look-ahead:** at any prediction date `t`, the training set contains only `(date, asset)` rows where `date + 21 ≤ t` (the target window must be fully realized before `t`).

### Hyperparameter search

- **Tool:** Optuna with `TimeSeriesSplit` cross-validation on the train portion only.
- **Search space:** `num_leaves` ∈ {15, 31, 63, 127}, `learning_rate` ∈ [0.01, 0.1], `min_data_in_leaf` ∈ {20, 50, 100}, `feature_fraction` ∈ [0.6, 1.0], `lambda_l2` ∈ [0, 10].
- **Objective:** validation MSE on held-out fold.
- **Trials:** ~50 (configurable in `config.py`).
- Run once at the start; freeze between refits.

### Loss

- **Primary:** L2 (regression_l2 / MSE).
- **Ablation:** quantile loss at 0.1, 0.5, 0.9 — useful for tail-risk-aware portfolios. The 0.5 quantile (median) becomes the µ; the 0.1/0.9 estimates can later inform a CVaR-aware variant. Reported in the writeup as a comparison; not part of the primary GBT-MVO arm.

### Early stopping

Validation MSE on the last fold; patience = 50 rounds.

---

## 6. Evaluation

### Three-way comparison

All three pipelines run on identical out-of-sample windows with identical Σ, optimizer, weight constraints, and risk-metric calculations. Only the µ-estimator differs.

| Pipeline | µ source | Σ source | Optimizer |
|---|---|---|---|
| Classical MVO | sample mean over 252-day window | sample cov | SLSQP, long-only |
| **GBT-MVO** (new) | LightGBM 21-day forecast × 12 | sample cov | SLSQP, long-only |
| LSTM-MVO (later) | LSTM forecast | sample cov | SLSQP, long-only |

### Metrics

Reuse `src/risk_metrics.py`:

- Sharpe ratio, Sortino ratio, maximum drawdown, VaR, CVaR.

Add:

- **Portfolio turnover** — average L1 weight change per rebalance, as a transaction-cost proxy.
- **Regime-conditional metrics** — break out each metric by low / medium / high VIX regime. This is the direct test of the central project question.

### Plots

Reuse `src/visualization.py` and add:

- **Three-way cumulative returns** (one line per pipeline).
- **Prediction-vs-actual scatter** for GBT (sanity-check the forecaster — is it actually forecasting or just predicting the mean?).
- **Feature importance** (LightGBM `gain` importance) — for the writeup.
- **Regime-bucketed Sharpe bars** (3 regimes × 3 pipelines).

---

## 7. Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    ...
    "lightgbm>=4.0.0",
    "optuna>=3.0.0",
]
```

LightGBM is preferred over XGBoost for this use case: faster training (matters for hyperparameter search), native categorical handling for `asset_id`, built-in quantile loss.

---

## 8. Testing

| Test file | Coverage |
|---|---|
| `tests/data/test_features.py` | Feature shapes correct; no look-ahead (assert features at time `t` use only data ≤ `t` via fixture with synthetic data); regime / calendar features encode correctly. |
| `tests/models/test_gbt.py` | Model fits on small synthetic panel; predicts correct shape; integrates with refactored `rolling_backtest`; µ-estimator returns annualized vector of length 11. |
| `tests/models/test_mvo.py` (extension) | Backward compatibility: `rolling_backtest` with default args produces identical output to current behavior (frozen-output regression test on small fixture). |

---

## 9. File-Level Plan

| File | Action | Notes |
|---|---|---|
| `src/data/features.py` | NEW | `build_feature_panel(returns, vix, regimes) -> pd.DataFrame` |
| `src/models/gbt.py` | NEW | `train_gbt`, `predict_returns`, `gbt_mu_estimator` |
| `src/models/mvo.py` | REFACTOR | Add `mu_estimator` param to `rolling_backtest`; default preserves current behavior |
| `src/config.py` | EXTEND | GBT hyperparameters, feature lookback windows, refit frequency |
| `src/visualization.py` | EXTEND | Three-way comparison plot, feature-importance plot, regime-bucketed bars |
| `pyproject.toml` | EXTEND | Add lightgbm, optuna |
| `tests/data/test_features.py` | NEW | Shape + leakage tests |
| `tests/models/test_gbt.py` | NEW | Fit / predict / integration tests |
| `tests/models/test_mvo.py` | EXTEND | Regression test for default behavior |
| `scripts/run_gbt_backtest.py` | NEW (optional) | Standalone runner mirroring `verify_data.py` for the GBT arm |

---

## 10. Open Questions / Risks

- **Hyperparameter stability across refits:** if the optimal hyperparameters drift over time, freezing them at initial fit may underperform. Mitigation: re-tune every N refits if validation performance degrades.
- **21-day target overlap:** consecutive daily observations have overlapping target windows (rows at `t` and `t+1` share 20 of 21 forward days). This induces autocorrelation in residuals. LightGBM handles this fine for point predictions, but standard error estimates would be biased — not relevant for our use case (we only care about predictions, not confidence intervals).
- **Quantile-loss ablation may not pay off** — included as a stretch goal, not blocking.
- **Optuna runtime:** 50 trials × CV folds × LightGBM training could take 5–20 minutes. Acceptable for a course project; cache best params in `config.py` after the first run.
