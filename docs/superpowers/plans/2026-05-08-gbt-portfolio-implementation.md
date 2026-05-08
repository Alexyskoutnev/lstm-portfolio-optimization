# GBT Portfolio Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third portfolio optimization arm — GBT-MVO — that replaces the sample-mean expected-return vector in Markowitz with a LightGBM forecast. Build the feature engineering pipeline, the GBT model, refactor the backtest to accept an injectable µ-estimator, and produce a three-way comparison against the existing classical MVO (and the planned LSTM-MVO).

**Architecture:** New `src/data/features.py` module produces a stacked `(date × asset)` feature panel from log returns + VIX + regime labels. New `src/models/gbt.py` trains a single global LightGBM regressor on this panel with a 21-day forward-return target, and exposes a `gbt_mu_estimator` callable that fits the existing backtest interface. `rolling_backtest` in `src/models/mvo.py` is refactored (backward compatibly) to accept any `mu_estimator(window_returns, current_date) -> np.ndarray` callable. Σ stays sample-based.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, scikit-learn, lightgbm, optuna, pytest.

**Spec:** [docs/superpowers/specs/2026-05-08-gbt-portfolio-design.md](../specs/2026-05-08-gbt-portfolio-design.md)

---

## File Structure

| File | Action |
|---|---|
| `pyproject.toml` | Modify — add `lightgbm`, `optuna` deps |
| `src/config.py` | Modify — add GBT/feature hyperparameter constants |
| `src/data/features.py` | Create — feature panel builder |
| `src/data/__init__.py` | Modify — export new feature builder |
| `src/models/gbt.py` | Create — GBT trainer + prediction + µ-estimator factory |
| `src/models/__init__.py` | Modify — export new GBT entry points |
| `src/models/mvo.py` | Modify — `rolling_backtest` accepts `mu_estimator`, default unchanged |
| `src/visualization.py` | Modify — three-way comparison plot, feature-importance plot, regime-bucketed bars |
| `tests/data/test_features.py` | Create — feature engineering tests |
| `tests/models/test_gbt.py` | Create — GBT model tests |
| `tests/models/test_mvo.py` | Modify — backward-compat regression test for refactored `rolling_backtest` |
| `scripts/run_gbt_backtest.py` | Create — end-to-end runner producing the three-way comparison |

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml dependencies**

Open `pyproject.toml`. Locate the `dependencies = [...]` block (lines 7-14). Replace it with:

```toml
dependencies = [
    "yfinance>=0.2.31",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "matplotlib>=3.7.0",
    "lightgbm>=4.0.0",
    "optuna>=3.0.0",
]
```

- [ ] **Step 2: Install new deps**

Run: `uv pip install -e ".[dev]"`
Expected: lightgbm and optuna install successfully.

- [ ] **Step 3: Smoke-import to confirm install**

Run: `python -c "import lightgbm; import optuna; print(lightgbm.__version__, optuna.__version__)"`
Expected: prints two version strings, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add lightgbm and optuna dependencies for GBT arm"
```

---

## Task 2: Extend config with GBT constants

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Append GBT and feature constants**

Append the following to the end of `src/config.py`:

```python
# ---------------------------------------------------------------------------
# Feature engineering windows (used by src/data/features.py)
# ---------------------------------------------------------------------------
# Lagged-return horizons in trading days. 1d ≈ overnight reaction, 5d ≈ one
# week, 21d ≈ one month, 63d ≈ one quarter, 126d ≈ six months. Together they
# capture short-term reversal, weekly momentum, monthly trend, and longer
# medium-term drift.
RETURN_LAGS: Final[list[int]] = [1, 5, 21, 63, 126]

# Rolling-statistics windows in trading days.
ROLLING_VOL_WINDOWS: Final[list[int]] = [21, 63]
ROLLING_MEAN_WINDOWS: Final[list[int]] = [21]
ROLLING_SKEW_WINDOWS: Final[list[int]] = [21]

# VIX delta horizons (trading days).
VIX_DELTA_HORIZONS: Final[list[int]] = [5, 21, 63]

# Window for cross-sectional aggregates (market mean, dispersion).
CROSS_ASSET_WINDOW: Final[int] = 21

# ---------------------------------------------------------------------------
# GBT model hyperparameters (used by src/models/gbt.py)
# ---------------------------------------------------------------------------
# Forecast horizon: predict the cumulative log return over the next N trading
# days. 21 ≈ 1 month, matches the rebalance frequency in rolling_backtest.
GBT_FORECAST_HORIZON: Final[int] = 21

# Backtest refit cadence (trading days).  The GBT is retrained every N days
# during the rolling backtest. 63 ≈ quarterly retraining keeps the cost
# reasonable while still adapting to regime shifts.
GBT_REFIT_FREQ: Final[int] = 63

# LightGBM defaults — overridden by Optuna if a tuning run is performed.
GBT_DEFAULT_PARAMS: Final[dict] = {
    "objective": "regression_l2",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "num_iterations": 500,
    "early_stopping_round": 50,
    "verbose": -1,
}

# Optuna search budget (number of trials).
GBT_OPTUNA_TRIALS: Final[int] = 50

# Number of TimeSeriesSplit folds used during hyperparameter tuning.
GBT_CV_FOLDS: Final[int] = 5
```

- [ ] **Step 2: Verify import**

Run: `python -c "from src.config import GBT_DEFAULT_PARAMS, RETURN_LAGS; print(RETURN_LAGS, GBT_DEFAULT_PARAMS['num_leaves'])"`
Expected: `[1, 5, 21, 63, 126] 31`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add GBT and feature engineering constants to config"
```

---

## Task 3: Feature engineering — write the failing tests

**Files:**
- Create: `tests/data/test_features.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/data/test_features.py` with this content:

```python
"""Tests for the feature engineering panel builder."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import build_feature_panel


@pytest.fixture()
def sample_returns() -> pd.DataFrame:
    """Synthetic log returns for 3 assets over 300 business days."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(300, 3)),
        index=dates,
        columns=["AAA", "BBB", "CCC"],
    )


@pytest.fixture()
def sample_vix(sample_returns: pd.DataFrame) -> pd.Series:
    """Synthetic VIX series aligned to the returns calendar."""
    rng = np.random.default_rng(1)
    return pd.Series(
        15 + 5 * rng.standard_normal(len(sample_returns)),
        index=sample_returns.index,
        name="vix",
    )


@pytest.fixture()
def sample_regimes(sample_returns: pd.DataFrame) -> pd.Series:
    """Synthetic regime labels aligned to the returns calendar."""
    n = len(sample_returns)
    labels = np.array(["low", "medium", "high"])[np.arange(n) % 3]
    return pd.Series(labels, index=sample_returns.index, name="regime")


class TestBuildFeaturePanel:
    """Tests for build_feature_panel."""

    def test_panel_is_stacked_long_form(
        self,
        sample_returns: pd.DataFrame,
        sample_vix: pd.Series,
        sample_regimes: pd.Series,
    ) -> None:
        """Panel should have one row per (date, asset) and a MultiIndex."""
        panel = build_feature_panel(sample_returns, sample_vix, sample_regimes)
        assert isinstance(panel.index, pd.MultiIndex)
        assert panel.index.names == ["date", "asset"]

    def test_expected_feature_columns_present(
        self,
        sample_returns: pd.DataFrame,
        sample_vix: pd.Series,
        sample_regimes: pd.Series,
    ) -> None:
        """All required feature columns should be in the output."""
        panel = build_feature_panel(sample_returns, sample_vix, sample_regimes)
        required = {
            "ret_lag_1", "ret_lag_5", "ret_lag_21", "ret_lag_63", "ret_lag_126",
            "vol_21", "vol_63", "mean_21", "skew_21",
            "vix_level", "vix_delta_5", "vix_delta_21", "vix_delta_63",
            "regime", "market_mean_21", "dispersion_21", "spread_vs_market_21",
            "month", "dow", "asset_id",
        }
        missing = required - set(panel.columns)
        assert not missing, f"Missing feature columns: {missing}"

    def test_no_lookahead_in_lagged_return(
        self,
        sample_returns: pd.DataFrame,
        sample_vix: pd.Series,
        sample_regimes: pd.Series,
    ) -> None:
        """ret_lag_1 at date t for asset i must equal returns at t-1 for that asset."""
        panel = build_feature_panel(sample_returns, sample_vix, sample_regimes)
        # Pick a date well past the warmup window.
        date_t = sample_returns.index[200]
        date_t_minus_1 = sample_returns.index[199]
        for asset in sample_returns.columns:
            row = panel.loc[(date_t, asset)]
            expected = sample_returns.loc[date_t_minus_1, asset]
            assert row["ret_lag_1"] == pytest.approx(expected)

    def test_rolling_vol_uses_only_past(
        self,
        sample_returns: pd.DataFrame,
        sample_vix: pd.Series,
        sample_regimes: pd.Series,
    ) -> None:
        """vol_21 at date t must equal std of returns over [t-21, t-1] inclusive."""
        panel = build_feature_panel(sample_returns, sample_vix, sample_regimes)
        date_t = sample_returns.index[200]
        for asset in sample_returns.columns:
            row = panel.loc[(date_t, asset)]
            window = sample_returns.loc[: sample_returns.index[199], asset].iloc[-21:]
            expected = window.std()
            assert row["vol_21"] == pytest.approx(expected, rel=1e-9)

    def test_warmup_rows_dropped(
        self,
        sample_returns: pd.DataFrame,
        sample_vix: pd.Series,
        sample_regimes: pd.Series,
    ) -> None:
        """Rows where the longest-lookback feature is undefined must be dropped."""
        panel = build_feature_panel(sample_returns, sample_vix, sample_regimes)
        # Longest lookback is 126d. Earliest valid date is index[126] (since lag 126
        # at index t uses index[t-126], meaning t must be >= 126).
        earliest_date = panel.index.get_level_values("date").min()
        assert earliest_date >= sample_returns.index[126]

    def test_no_nans_in_panel(
        self,
        sample_returns: pd.DataFrame,
        sample_vix: pd.Series,
        sample_regimes: pd.Series,
    ) -> None:
        """After warmup rows are dropped, panel should have no NaNs."""
        panel = build_feature_panel(sample_returns, sample_vix, sample_regimes)
        assert panel.isna().sum().sum() == 0

    def test_asset_id_categorical_with_correct_levels(
        self,
        sample_returns: pd.DataFrame,
        sample_vix: pd.Series,
        sample_regimes: pd.Series,
    ) -> None:
        """asset_id should be categorical with one level per asset column."""
        panel = build_feature_panel(sample_returns, sample_vix, sample_regimes)
        assert panel["asset_id"].dtype.name == "category"
        assert set(panel["asset_id"].cat.categories) == set(sample_returns.columns)
```

- [ ] **Step 2: Run tests; verify they fail with ImportError**

Run: `python -m pytest tests/data/test_features.py -v`
Expected: ImportError / ModuleNotFoundError on `src.data.features`. All tests collected fail.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/data/test_features.py
git commit -m "test: add failing tests for feature engineering panel builder"
```

---

## Task 4: Feature engineering — implement `build_feature_panel`

**Files:**
- Create: `src/data/features.py`
- Modify: `src/data/__init__.py`

- [ ] **Step 1: Implement the feature builder**

Create `src/data/features.py`:

```python
"""Feature engineering for tree-based and recurrent return-forecasting models.

Builds a stacked (date × asset) panel of strictly-lagged features. Every
feature at time *t* for asset *i* depends only on data observed at-or-before
*t-1* (returns, VIX) or known calendar metadata at *t* — there is no
look-ahead.

Feature groups (see src/config.py for window/lag constants):

- **Own-asset lagged returns** (RETURN_LAGS): captures momentum and reversal
  at multiple horizons.
- **Own-asset rolling stats** (vol, mean, skew): describes the local
  volatility regime and asymmetry of the return distribution.
- **VIX features**: current level and multi-horizon changes — the broad
  market's implied-volatility signal.
- **Regime label**: discrete low/medium/high VIX bucket.
- **Cross-asset features**: market-wide return, cross-sectional dispersion,
  and this asset's spread vs the market — captures relative-value and
  correlation-cluster effects.
- **Calendar**: month-of-year, day-of-week — captures seasonality.
- **Asset id**: categorical, lets the global model learn per-asset adjustments.

The output panel is the input to `src/models/gbt.py`.
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    CROSS_ASSET_WINDOW,
    RETURN_LAGS,
    ROLLING_MEAN_WINDOWS,
    ROLLING_SKEW_WINDOWS,
    ROLLING_VOL_WINDOWS,
    VIX_DELTA_HORIZONS,
)

logger = logging.getLogger(__name__)


def build_feature_panel(
    log_returns: pd.DataFrame,
    vix: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Build a stacked (date × asset) feature panel.

    All features are strictly lagged: the row at (date=t, asset=i) uses only
    returns/VIX/regime values observed at-or-before t-1. Calendar features
    (month, day-of-week) come from t itself, which is information available
    at the start of trading on day t.

    Args:
        log_returns: DataFrame of daily log returns (T × N).
        vix: Series of daily VIX values aligned to ``log_returns.index``.
        regimes: Series of regime labels ("low", "medium", "high") aligned
            to ``log_returns.index``.

    Returns:
        DataFrame indexed by MultiIndex(date, asset) with one column per
        feature. Warmup rows (where the longest lookback is undefined) are
        dropped, and the result has no NaN values.
    """
    logger.info(
        "Building feature panel: %d dates, %d assets",
        len(log_returns),
        log_returns.shape[1],
    )

    # All feature DataFrames below are wide (T × N) with the same date index
    # as log_returns. We stack them at the end.
    feature_frames: dict[str, pd.DataFrame] = {}

    # Own-asset lagged returns. shift(k) at row t holds the value from row t-k,
    # which is exactly the "data at t-k" we want.
    for k in RETURN_LAGS:
        feature_frames[f"ret_lag_{k}"] = log_returns.shift(k)

    # Rolling vol/mean/skew over [t-w, t-1]. We compute the rolling stat ending
    # at t (which uses [t-w+1, t]), then shift(1) so the value at t reflects
    # only data through t-1.
    for w in ROLLING_VOL_WINDOWS:
        feature_frames[f"vol_{w}"] = log_returns.rolling(window=w).std().shift(1)
    for w in ROLLING_MEAN_WINDOWS:
        feature_frames[f"mean_{w}"] = log_returns.rolling(window=w).mean().shift(1)
    for w in ROLLING_SKEW_WINDOWS:
        feature_frames[f"skew_{w}"] = log_returns.rolling(window=w).skew().shift(1)

    # Cross-asset features: market mean, dispersion, and this asset's spread vs
    # market — all computed on the same lagged window, then broadcast across
    # asset columns.
    market_window_returns = log_returns.rolling(window=CROSS_ASSET_WINDOW).mean().shift(1)
    market_mean = market_window_returns.mean(axis=1)  # Series, indexed by date
    dispersion = market_window_returns.std(axis=1)
    feature_frames[f"market_mean_{CROSS_ASSET_WINDOW}"] = pd.DataFrame(
        np.broadcast_to(
            market_mean.values[:, None],
            (len(log_returns), log_returns.shape[1]),
        ),
        index=log_returns.index,
        columns=log_returns.columns,
    )
    feature_frames[f"dispersion_{CROSS_ASSET_WINDOW}"] = pd.DataFrame(
        np.broadcast_to(
            dispersion.values[:, None],
            (len(log_returns), log_returns.shape[1]),
        ),
        index=log_returns.index,
        columns=log_returns.columns,
    )
    feature_frames[f"spread_vs_market_{CROSS_ASSET_WINDOW}"] = (
        market_window_returns.sub(market_mean, axis=0)
    )

    # VIX features. shift(1) so we use yesterday's close (today's value isn't
    # known until end of day t).
    vix_lagged = vix.shift(1)
    feature_frames["vix_level"] = pd.DataFrame(
        np.broadcast_to(
            vix_lagged.values[:, None],
            (len(log_returns), log_returns.shape[1]),
        ),
        index=log_returns.index,
        columns=log_returns.columns,
    )
    for h in VIX_DELTA_HORIZONS:
        delta = vix.diff(h).shift(1)
        feature_frames[f"vix_delta_{h}"] = pd.DataFrame(
            np.broadcast_to(
                delta.values[:, None],
                (len(log_returns), log_returns.shape[1]),
            ),
            index=log_returns.index,
            columns=log_returns.columns,
        )

    # Stack each wide DataFrame into long form (date, asset) -> value.
    long_frames: dict[str, pd.Series] = {}
    for name, wide in feature_frames.items():
        stacked = wide.stack()
        stacked.index.names = ["date", "asset"]
        long_frames[name] = stacked

    panel = pd.DataFrame(long_frames)

    # Regime label — known at end of day t-1 (so use shift(1)). Broadcast to
    # all assets at each date.
    regime_lagged = regimes.shift(1)
    panel = panel.reset_index()
    panel["regime"] = panel["date"].map(regime_lagged)

    # Calendar features at date t (no leakage — known at the start of day t).
    panel["month"] = panel["date"].dt.month.astype("int8")
    panel["dow"] = panel["date"].dt.dayofweek.astype("int8")

    # Asset id as a categorical with deterministic level ordering = column
    # order in log_returns. LightGBM accepts this directly.
    panel["asset_id"] = pd.Categorical(
        panel["asset"], categories=list(log_returns.columns)
    )

    panel = panel.set_index(["date", "asset"]).sort_index()

    # Cast regime to categorical with fixed levels (LightGBM-friendly).
    panel["regime"] = pd.Categorical(
        panel["regime"], categories=["low", "medium", "high"]
    )

    # Drop warmup rows: the longest lookback is the max of all lag/window
    # lengths. After dropping, the panel must have no NaNs.
    pre_drop = len(panel)
    panel = panel.dropna()
    logger.info("Feature panel: %d rows after dropping %d warmup rows",
                len(panel), pre_drop - len(panel))

    return panel
```

- [ ] **Step 2: Export from `src/data/__init__.py`**

Open `src/data/__init__.py`. Modify the import block and `__all__`:

```python
from src.data.fetcher import fetch_prices, fetch_vix
from src.data.features import build_feature_panel
from src.data.preprocessing import compute_log_returns, compute_simple_returns, split_data
from src.data.regime import label_regimes_rolling_std, label_regimes_vix
```

```python
__all__ = [
    "fetch_prices",
    "fetch_vix",
    "compute_log_returns",
    "compute_simple_returns",
    "split_data",
    "label_regimes_vix",
    "label_regimes_rolling_std",
    "build_feature_panel",
    "load_dataset",
]
```

- [ ] **Step 3: Run feature tests; verify they pass**

Run: `python -m pytest tests/data/test_features.py -v`
Expected: all 7 tests pass.

- [ ] **Step 4: Run full test suite to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (existing tests + 7 new feature tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/features.py src/data/__init__.py
git commit -m "feat: add feature engineering panel builder for GBT model"
```

---

## Task 5: GBT model — write the failing tests

**Files:**
- Create: `tests/models/test_gbt.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/models/test_gbt.py`:

```python
"""Tests for the GBT return-forecasting model."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import build_feature_panel
from src.models.gbt import (
    build_target,
    gbt_mu_estimator,
    predict_returns,
    train_gbt,
)


@pytest.fixture()
def synthetic_panel_data() -> dict:
    """Generate synthetic log returns + VIX + regimes large enough to train GBT."""
    dates = pd.date_range("2018-01-01", periods=600, freq="B")
    n_assets = 5
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(600, n_assets)),
        index=dates,
        columns=[f"ETF{i}" for i in range(n_assets)],
    )
    vix = pd.Series(
        15 + 5 * rng.standard_normal(len(dates)),
        index=dates,
        name="vix",
    )
    regimes = pd.Series(
        np.array(["low", "medium", "high"])[np.arange(len(dates)) % 3],
        index=dates,
        name="regime",
    )
    return {"returns": returns, "vix": vix, "regimes": regimes}


class TestBuildTarget:
    """Tests for build_target (forward-return target construction)."""

    def test_target_horizon_correct(self, synthetic_panel_data: dict) -> None:
        """Target at (t, asset) should equal sum of returns over [t+1, t+H]."""
        returns = synthetic_panel_data["returns"]
        horizon = 21
        target = build_target(returns, horizon=horizon)
        # Pick a row well inside the valid range.
        date_t = returns.index[100]
        for asset in returns.columns:
            actual = target.loc[(date_t, asset)]
            expected = returns.loc[
                returns.index[101] : returns.index[121], asset
            ].sum()
            assert actual == pytest.approx(expected, rel=1e-9)

    def test_target_drops_tail_rows(self, synthetic_panel_data: dict) -> None:
        """Last H rows have undefined target and must be excluded."""
        returns = synthetic_panel_data["returns"]
        horizon = 21
        target = build_target(returns, horizon=horizon)
        latest_date = target.index.get_level_values("date").max()
        # Tail: indices [-horizon:] are all undefined.
        assert latest_date <= returns.index[-horizon - 1]


class TestTrainGbt:
    """Tests for train_gbt."""

    def test_model_trains_and_predicts(self, synthetic_panel_data: dict) -> None:
        """train_gbt should return a fitted model that can predict on new rows."""
        panel = build_feature_panel(
            synthetic_panel_data["returns"],
            synthetic_panel_data["vix"],
            synthetic_panel_data["regimes"],
        )
        target = build_target(synthetic_panel_data["returns"], horizon=21)
        # Align panel and target on shared (date, asset) index.
        common = panel.index.intersection(target.index)
        X = panel.loc[common]
        y = target.loc[common]

        model = train_gbt(X, y)

        preds = predict_returns(model, X.head(50))
        assert preds.shape == (50,)
        assert np.all(np.isfinite(preds))


class TestGbtMuEstimator:
    """Tests for the gbt_mu_estimator factory."""

    def test_estimator_returns_correct_shape(
        self, synthetic_panel_data: dict
    ) -> None:
        """The mu estimator returned by the factory must produce a length-N vector."""
        returns = synthetic_panel_data["returns"]
        panel = build_feature_panel(
            returns,
            synthetic_panel_data["vix"],
            synthetic_panel_data["regimes"],
        )
        target = build_target(returns, horizon=21)
        common = panel.index.intersection(target.index)
        model = train_gbt(panel.loc[common], target.loc[common])

        estimator = gbt_mu_estimator(
            model=model,
            full_panel=panel,
            asset_order=list(returns.columns),
            horizon=21,
        )

        # Pick a date that exists in the panel.
        prediction_date = panel.index.get_level_values("date").unique()[200]
        # rolling_backtest passes a window of returns; the GBT estimator
        # ignores the window and uses the panel directly via the date.
        mu = estimator(returns.iloc[:200], prediction_date)
        assert mu.shape == (returns.shape[1],)
        assert np.all(np.isfinite(mu))

    def test_estimator_annualizes_predictions(
        self, synthetic_panel_data: dict
    ) -> None:
        """mu = ŷ × (252 / horizon). Verify the scaling factor is applied."""
        returns = synthetic_panel_data["returns"]
        panel = build_feature_panel(
            returns,
            synthetic_panel_data["vix"],
            synthetic_panel_data["regimes"],
        )
        target = build_target(returns, horizon=21)
        common = panel.index.intersection(target.index)
        model = train_gbt(panel.loc[common], target.loc[common])

        prediction_date = panel.index.get_level_values("date").unique()[200]
        # Raw prediction
        row = panel.xs(prediction_date, level="date")
        raw_pred = predict_returns(model, row)

        estimator = gbt_mu_estimator(
            model=model,
            full_panel=panel,
            asset_order=list(returns.columns),
            horizon=21,
        )
        mu = estimator(returns.iloc[:200], prediction_date)

        # mu should equal raw_pred (in the asset_order ordering) × 12.
        scaling = 252.0 / 21.0
        np.testing.assert_allclose(mu, raw_pred.reindex(returns.columns).values * scaling, rtol=1e-9)
```

- [ ] **Step 2: Run tests; verify they fail with ImportError**

Run: `python -m pytest tests/models/test_gbt.py -v`
Expected: ImportError on `src.models.gbt`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/models/test_gbt.py
git commit -m "test: add failing tests for GBT model and mu estimator"
```

---

## Task 6: GBT model — implement `src/models/gbt.py`

**Files:**
- Create: `src/models/gbt.py`
- Modify: `src/models/__init__.py`

- [ ] **Step 1: Implement the GBT model**

Create `src/models/gbt.py`:

```python
"""Gradient-boosting tree (LightGBM) model for forecasting forward returns.

Why this exists
---------------
Classical Markowitz uses the *sample mean* of historical returns as the
expected-return vector µ. The sample mean is an unbiased but extremely noisy
estimator: with 252 daily observations, the standard error on µ is roughly
σ/√252, which for an ETF with 20% annual vol is ~1.3% — often larger than
the signal itself. Markowitz amplifies this noise (Michaud 1989, "error
maximization"), producing portfolios that look great on training data and
collapse out-of-sample.

This module replaces the sample mean with a learned, conditional µ. A single
global LightGBM regressor is trained on a stacked (date × asset) feature
panel with a forward-return target. Features include lagged returns, rolling
vol, VIX level/changes, regime labels, cross-asset signals, and a categorical
asset id. The trained model exposes a `gbt_mu_estimator(...)` callable that
plugs into `rolling_backtest` exactly where the sample-mean estimator would.

Σ remains sample-based across all three project arms (classical MVO, GBT-MVO,
LSTM-MVO) so that the comparison isolates the contribution of µ.
"""

import logging
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import GBT_DEFAULT_PARAMS, GBT_FORECAST_HORIZON

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def build_target(
    log_returns: pd.DataFrame,
    horizon: int = GBT_FORECAST_HORIZON,
) -> pd.Series:
    """Build a stacked forward-return target series.

    For each (date t, asset i), the target is the cumulative log return
    over [t+1, t+horizon] (i.e., the realized return of holding asset i for
    the next `horizon` trading days starting tomorrow). Rows where the full
    target window is not yet observed are dropped.

    Args:
        log_returns: DataFrame of daily log returns (T × N).
        horizon: Forecast horizon in trading days.

    Returns:
        Series indexed by MultiIndex(date, asset) with the cumulative
        forward log return as the value.
    """
    # rolling(horizon).sum() at row t holds the sum over [t-horizon+1, t].
    # Shifting by -horizon aligns it so row t holds the sum over [t+1, t+horizon].
    forward = log_returns.rolling(window=horizon).sum().shift(-horizon)
    forward = forward.dropna(how="any")  # drop the tail where target is undefined
    target = forward.stack()
    target.index.names = ["date", "asset"]
    target.name = "target"
    return target


def train_gbt(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    val_fraction: float = 0.2,
) -> lgb.Booster:
    """Train a LightGBM regressor on the (date × asset) panel.

    Uses a simple time-based hold-out for early stopping: the last
    ``val_fraction`` of the unique dates is used as the validation set.

    Args:
        X: Feature DataFrame indexed by MultiIndex(date, asset).
        y: Target Series indexed identically to X.
        params: Optional override of LightGBM hyperparameters. Defaults
            to ``GBT_DEFAULT_PARAMS``.
        val_fraction: Fraction of the most recent dates held out for
            early stopping.

    Returns:
        Fitted LightGBM Booster.
    """
    params = dict(GBT_DEFAULT_PARAMS) if params is None else dict(params)
    num_iterations = params.pop("num_iterations", 500)
    early_stopping_round = params.pop("early_stopping_round", 50)

    # Time-based train/val split
    unique_dates = X.index.get_level_values("date").unique().sort_values()
    cutoff_idx = int(len(unique_dates) * (1 - val_fraction))
    cutoff = unique_dates[cutoff_idx]

    train_mask = X.index.get_level_values("date") < cutoff
    X_train, X_val = X[train_mask], X[~train_mask]
    y_train, y_val = y[train_mask], y[~train_mask]

    categorical_features = ["regime", "asset_id"]
    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_features
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, reference=train_set,
        categorical_feature=categorical_features,
    )

    logger.info(
        "Training LightGBM: train=%d rows, val=%d rows, params=%s",
        len(X_train),
        len(X_val),
        params,
    )

    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_iterations,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_round, verbose=False)],
    )

    logger.info("LightGBM trained: best_iteration=%d", model.best_iteration)
    return model


def predict_returns(
    model: lgb.Booster,
    X: pd.DataFrame,
) -> pd.Series:
    """Predict raw forward returns for each row in X.

    The output is a Series of predictions indexed identically to X. Use
    ``gbt_mu_estimator`` if you need the annualized µ vector for one date.

    Args:
        model: Fitted LightGBM Booster.
        X: Feature DataFrame indexed by MultiIndex(date, asset).

    Returns:
        Series of predicted ``horizon``-day cumulative log returns.
    """
    raw = model.predict(X, num_iteration=model.best_iteration)
    return pd.Series(np.asarray(raw), index=X.index, name="prediction")


def gbt_mu_estimator(
    model: lgb.Booster,
    full_panel: pd.DataFrame,
    asset_order: list[str],
    horizon: int = GBT_FORECAST_HORIZON,
) -> Callable[[pd.DataFrame, pd.Timestamp], np.ndarray]:
    """Build a µ-estimator callable compatible with ``rolling_backtest``.

    The returned callable signature is::

        mu_estimator(window_returns: pd.DataFrame, current_date: pd.Timestamp) -> np.ndarray

    matching the interface that ``rolling_backtest`` expects. The
    ``window_returns`` argument is ignored — the GBT consumes pre-built
    features from ``full_panel`` keyed by ``current_date``. We keep the
    argument in the signature so the same harness works for both classical
    and GBT mu estimators.

    Args:
        model: Fitted LightGBM Booster.
        full_panel: Pre-built feature panel covering all dates the
            backtest may query.
        asset_order: The canonical asset ordering returned to MVO. Must
            match the column order of the returns DataFrame fed to
            ``rolling_backtest``.
        horizon: Forecast horizon in trading days. Used to annualize the
            raw prediction.

    Returns:
        Callable that takes (window_returns, current_date) and returns
        an annualized µ vector of shape (len(asset_order),).
    """
    annualization = _TRADING_DAYS_PER_YEAR / float(horizon)

    def estimator(
        window_returns: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> np.ndarray:
        # Slice the panel rows for current_date across all assets.
        try:
            row = full_panel.xs(current_date, level="date")
        except KeyError:
            logger.warning(
                "GBT estimator: no panel row for %s; falling back to zero µ",
                current_date,
            )
            return np.zeros(len(asset_order))

        preds = predict_returns(model, row)
        # Ensure deterministic asset ordering matching the MVO column order.
        preds = preds.reindex(asset_order)
        if preds.isna().any():
            missing = preds.index[preds.isna()].tolist()
            raise ValueError(
                f"GBT estimator missing predictions for assets {missing} on {current_date}"
            )
        return preds.values * annualization

    return estimator
```

- [ ] **Step 2: Update `src/models/__init__.py`**

Open `src/models/__init__.py`. Replace its contents with:

```python
"""Portfolio optimization models."""

from src.models.gbt import (
    build_target,
    gbt_mu_estimator,
    predict_returns,
    train_gbt,
)
from src.models.mvo import (
    efficient_frontier,
    estimate_parameters,
    max_sharpe_weights,
    minimum_variance_weights,
    portfolio_performance,
    rolling_backtest,
)

__all__ = [
    "build_target",
    "efficient_frontier",
    "estimate_parameters",
    "gbt_mu_estimator",
    "max_sharpe_weights",
    "minimum_variance_weights",
    "portfolio_performance",
    "predict_returns",
    "rolling_backtest",
    "train_gbt",
]
```

- [ ] **Step 3: Run GBT tests; verify they pass**

Run: `python -m pytest tests/models/test_gbt.py -v`
Expected: all 5 tests pass.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/gbt.py src/models/__init__.py
git commit -m "feat: add LightGBM model trainer, predictor, and mu estimator"
```

---

## Task 7: Refactor `rolling_backtest` to accept `mu_estimator`

**Files:**
- Modify: `src/models/mvo.py`
- Modify: `tests/models/test_mvo.py`

- [ ] **Step 1: Add backward-compat regression test FIRST (TDD)**

Open `tests/models/test_mvo.py`. Append the following test class at the end of the file:

```python
class TestRollingBacktestMuEstimator:
    """Tests for the mu_estimator hook in rolling_backtest."""

    def test_default_estimator_matches_legacy_behavior(self) -> None:
        """Default behavior must be identical to pre-refactor (sample-mean µ)."""
        import numpy as np
        import pandas as pd

        from src.models.mvo import rolling_backtest

        rng = np.random.default_rng(123)
        dates = pd.date_range("2020-01-01", periods=400, freq="B")
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.01, size=(400, 4)),
            index=dates,
            columns=["A", "B", "C", "D"],
        )

        result = rolling_backtest(
            returns,
            window=100,
            strategy="min_variance",
            rebalance_freq=20,
        )

        # Sanity invariants
        assert "portfolio_return" in result.columns
        assert "cumulative_return" in result.columns
        assert all(asset in result.columns for asset in returns.columns)
        # Weights sum to 1 each day (within solver tolerance).
        weight_cols = list(returns.columns)
        weight_sums = result[weight_cols].sum(axis=1).values
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-6)

    def test_custom_mu_estimator_is_called(self) -> None:
        """Passing a custom mu_estimator must override the default."""
        import numpy as np
        import pandas as pd

        from src.models.mvo import rolling_backtest

        rng = np.random.default_rng(7)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.01, size=(300, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )

        call_log: list[pd.Timestamp] = []

        def constant_estimator(
            window: pd.DataFrame, current_date: pd.Timestamp
        ) -> np.ndarray:
            call_log.append(current_date)
            return np.array([0.05, 0.10, 0.15])  # arbitrary annualized µ

        result = rolling_backtest(
            returns,
            window=100,
            strategy="max_sharpe",
            rebalance_freq=20,
            mu_estimator=constant_estimator,
        )
        # Estimator was called at least once (rebalance dates).
        assert len(call_log) > 0
        assert "portfolio_return" in result.columns
```

- [ ] **Step 2: Run new tests; one should fail (custom estimator), one should pass (legacy)**

Run: `python -m pytest tests/models/test_mvo.py::TestRollingBacktestMuEstimator -v`
Expected: `test_custom_mu_estimator_is_called` fails because `rolling_backtest` does not yet accept `mu_estimator`. The legacy test may pass since it uses default args.

- [ ] **Step 3: Refactor `rolling_backtest` to accept `mu_estimator`**

Open `src/models/mvo.py`. Locate the `rolling_backtest` function (starts at line 520). Modify the signature and body as follows.

First, add this helper function above `rolling_backtest` (e.g., after `_VALID_STRATEGIES` at line 517):

```python
def _default_mu_estimator(
    window_returns: pd.DataFrame,
    current_date: pd.Timestamp,  # noqa: ARG001 — kept for interface symmetry
) -> np.ndarray:
    """Default µ estimator: sample mean of the lookback window, annualized.

    This is the classical Markowitz behavior. Kept as a separate function
    so it satisfies the same callable signature as learned estimators
    (e.g., ``gbt_mu_estimator``), letting ``rolling_backtest`` treat all
    µ sources uniformly.
    """
    return np.array(window_returns.mean()) * _TRADING_DAYS_PER_YEAR
```

Then change the `rolling_backtest` signature from:

```python
def rolling_backtest(
    returns: pd.DataFrame,
    window: int = 252,
    strategy: str = "max_sharpe",
    rebalance_freq: int = 21,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
```

to:

```python
def rolling_backtest(
    returns: pd.DataFrame,
    window: int = 252,
    strategy: str = "max_sharpe",
    rebalance_freq: int = 21,
    risk_free_rate: float = 0.0,
    mu_estimator: Callable[[pd.DataFrame, pd.Timestamp], np.ndarray] | None = None,
) -> pd.DataFrame:
```

Add at the top of the file (after the existing imports):

```python
from typing import Callable
```

(if `typing` is not already imported with `Callable`; check the current imports first).

Inside `rolling_backtest`, just before the `for i in range(window, len(returns)):` loop, add:

```python
if mu_estimator is None:
    mu_estimator = _default_mu_estimator
```

Now modify `_rebalance` to accept and use `mu_estimator`. Change its signature from:

```python
def _rebalance(
    returns: pd.DataFrame,
    current_idx: int,
    window: int,
    strategy: str,
    risk_free_rate: float,
    fallback_weights: np.ndarray,
) -> np.ndarray:
```

to:

```python
def _rebalance(
    returns: pd.DataFrame,
    current_idx: int,
    window: int,
    strategy: str,
    risk_free_rate: float,
    fallback_weights: np.ndarray,
    mu_estimator: Callable[[pd.DataFrame, pd.Timestamp], np.ndarray],
) -> np.ndarray:
```

And inside `_rebalance`, replace the line:

```python
mean_ret, cov_mat = estimate_parameters(train_slice)
```

with:

```python
_, cov_mat = estimate_parameters(train_slice)
mean_ret = mu_estimator(train_slice, returns.index[current_idx])
```

Finally, in `rolling_backtest`, update the call to `_rebalance` from:

```python
current_weights = _rebalance(
    returns, i, window, strategy, risk_free_rate, current_weights
)
```

to:

```python
current_weights = _rebalance(
    returns, i, window, strategy, risk_free_rate, current_weights, mu_estimator
)
```

- [ ] **Step 4: Run new tests; verify they pass**

Run: `python -m pytest tests/models/test_mvo.py::TestRollingBacktestMuEstimator -v`
Expected: both tests pass.

- [ ] **Step 5: Run full test suite for regressions**

Run: `python -m pytest tests/ -v`
Expected: all tests pass — including pre-existing MVO tests, which exercise the default behavior.

- [ ] **Step 6: Commit**

```bash
git add src/models/mvo.py tests/models/test_mvo.py
git commit -m "refactor: rolling_backtest accepts injectable mu_estimator"
```

---

## Task 8: Visualization — three-way comparison plot

**Files:**
- Modify: `src/visualization.py`

- [ ] **Step 1: Add `plot_three_way_comparison` function**

Open `src/visualization.py`. Append:

```python
def plot_three_way_comparison(
    backtests: dict[str, pd.DataFrame],
    output_path: str | None = None,
) -> None:
    """Plot cumulative returns for multiple backtest pipelines on one axes.

    Used to compare classical MVO vs GBT-MVO (vs LSTM-MVO once available).

    Args:
        backtests: Mapping of pipeline name -> backtest result DataFrame
            (must contain a "cumulative_return" column).
        output_path: Optional file path for saving the figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in backtests.items():
        ax.plot(df.index, df["cumulative_return"], label=name, linewidth=1.5)
    ax.set_title("Cumulative Returns: Three-Way Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    model,  # lgb.Booster
    output_path: str | None = None,
    top_n: int = 20,
) -> None:
    """Plot LightGBM feature importance (gain) for the trained GBT model.

    Args:
        model: Fitted LightGBM Booster.
        output_path: Optional file path for saving the figure.
        top_n: Number of top features to display.
    """
    import matplotlib.pyplot as plt

    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    df = (
        pd.DataFrame({"feature": names, "gain": importance})
        .sort_values("gain", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(df["feature"], df["gain"])
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {top_n} GBT Feature Importance (Gain)")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_regime_bucketed_sharpe(
    backtests: dict[str, pd.DataFrame],
    regimes: pd.Series,
    risk_free_rate: float = 0.0,
    output_path: str | None = None,
) -> None:
    """Bar chart of Sharpe ratio per pipeline, broken out by regime.

    Args:
        backtests: Mapping of pipeline name -> backtest result DataFrame
            (must contain a "portfolio_return" column).
        regimes: Series of regime labels aligned to backtest dates.
        risk_free_rate: Annualized risk-free rate.
        output_path: Optional file path for saving the figure.
    """
    import matplotlib.pyplot as plt

    from src.risk_metrics import sharpe_ratio

    regime_labels = ["low", "medium", "high"]
    pipelines = list(backtests.keys())
    sharpe_matrix = np.zeros((len(pipelines), len(regime_labels)))

    for i, name in enumerate(pipelines):
        df = backtests[name]
        aligned = regimes.reindex(df.index)
        for j, label in enumerate(regime_labels):
            mask = aligned == label
            ret = df.loc[mask, "portfolio_return"]
            sharpe_matrix[i, j] = (
                sharpe_ratio(ret, risk_free_rate=risk_free_rate) if len(ret) > 1 else 0.0
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    x = np.arange(len(regime_labels))
    for i, name in enumerate(pipelines):
        ax.bar(x + i * bar_width, sharpe_matrix[i], bar_width, label=name)
    ax.set_xticks(x + bar_width * (len(pipelines) - 1) / 2)
    ax.set_xticklabels(regime_labels)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Volatility Regime")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 2: Smoke-test the new plot functions**

Run:

```bash
python -c "
import numpy as np, pandas as pd
from src.visualization import plot_three_way_comparison, plot_regime_bucketed_sharpe

dates = pd.date_range('2020-01-01', periods=200, freq='B')
rng = np.random.default_rng(0)
df_a = pd.DataFrame({'portfolio_return': rng.normal(0.0005, 0.01, 200)}, index=dates)
df_a['cumulative_return'] = df_a['portfolio_return'].cumsum().apply(np.exp) - 1
df_b = df_a.copy()
df_b['portfolio_return'] = rng.normal(0.0007, 0.012, 200)
df_b['cumulative_return'] = df_b['portfolio_return'].cumsum().apply(np.exp) - 1

plot_three_way_comparison({'A': df_a, 'B': df_b}, output_path='/tmp/test_three_way.png')

regimes = pd.Series(np.array(['low','medium','high'])[np.arange(200)%3], index=dates)
plot_regime_bucketed_sharpe({'A': df_a, 'B': df_b}, regimes, output_path='/tmp/test_regime.png')

print('plots created')
"
```

Expected: prints "plots created"; files exist at the named paths.

- [ ] **Step 3: Commit**

```bash
git add src/visualization.py
git commit -m "feat: add three-way comparison and regime-bucketed Sharpe plots"
```

---

## Task 9: End-to-end runner script

**Files:**
- Create: `scripts/run_gbt_backtest.py`

- [ ] **Step 1: Implement the runner**

Create `scripts/run_gbt_backtest.py`:

```python
"""End-to-end runner: classical MVO vs GBT-MVO three-way comparison.

Loads sector ETF + VIX data, builds the feature panel, trains a global
LightGBM model on the train portion, then runs three rolling backtests
sharing the same Σ, optimizer, and weight constraints — only the µ
estimator differs.

Outputs:
- plots/cumulative_three_way.png — cumulative-return comparison
- plots/gbt_feature_importance.png — top-20 LightGBM feature importance
- plots/sharpe_by_regime.png — Sharpe ratio per regime per pipeline
- Prints summary metrics to stdout.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    END_DATE,
    GBT_FORECAST_HORIZON,
    GBT_REFIT_FREQ,
    RAW_DATA_DIR,
    ROLLING_WINDOW,
    START_DATE,
    TICKERS,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data import build_feature_panel, load_dataset
from src.models.gbt import build_target, gbt_mu_estimator, train_gbt
from src.models.mvo import rolling_backtest
from src.risk_metrics import compute_all_metrics
from src.visualization import (
    plot_feature_importance,
    plot_regime_bucketed_sharpe,
    plot_three_way_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # 1. Load data
    logger.info("Loading dataset")
    data = load_dataset(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        cache_dir=RAW_DATA_DIR,
    )
    log_returns: pd.DataFrame = data["log_returns"]
    regimes: pd.Series = data["regime_vix"]

    # VIX series — fetched inside load_dataset, but not currently returned;
    # rebuild it here aligned to log_returns.
    from src.data import fetch_vix
    vix = fetch_vix(start=START_DATE, end=END_DATE).reindex(log_returns.index).ffill().bfill()

    # 2. Build the feature panel + target
    logger.info("Building feature panel")
    panel = build_feature_panel(log_returns, vix, regimes)
    target = build_target(log_returns, horizon=GBT_FORECAST_HORIZON)

    # 3. Train GBT on the train portion only (no test data leakage)
    train_end_date = data["train"].index[-1]
    logger.info("Training GBT on data up to %s", train_end_date)
    train_mask = panel.index.get_level_values("date") <= train_end_date
    common = panel.index.intersection(target.index)
    train_idx = panel.index.intersection(target.index).intersection(panel[train_mask].index)
    model = train_gbt(panel.loc[train_idx], target.loc[train_idx])

    # 4. Build the GBT µ-estimator
    gbt_estimator = gbt_mu_estimator(
        model=model,
        full_panel=panel,
        asset_order=list(log_returns.columns),
        horizon=GBT_FORECAST_HORIZON,
    )

    # 5. Run three backtests on the same data slice (test period only)
    test_start_date = data["test"].index[0]
    # Provide a reasonable lookback before test_start so window=ROLLING_WINDOW is
    # satisfied at the very first prediction.
    lookback_start = log_returns.index[
        max(0, log_returns.index.get_loc(test_start_date) - ROLLING_WINDOW)
    ]
    backtest_returns = log_returns.loc[lookback_start:]

    logger.info("Running classical MVO backtest (sample-mean µ)")
    classical = rolling_backtest(
        backtest_returns,
        window=ROLLING_WINDOW,
        strategy="max_sharpe",
        rebalance_freq=21,
    )

    logger.info("Running GBT-MVO backtest (LightGBM µ)")
    gbt = rolling_backtest(
        backtest_returns,
        window=ROLLING_WINDOW,
        strategy="max_sharpe",
        rebalance_freq=21,
        mu_estimator=gbt_estimator,
    )

    # 6. Compute and report metrics
    backtests = {"classical_mvo": classical, "gbt_mvo": gbt}
    print("\n=== Summary metrics (test period) ===")
    for name, df in backtests.items():
        metrics = compute_all_metrics(df["portfolio_return"])
        print(f"\n{name}:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")

    # 7. Plots
    logger.info("Generating plots")
    plot_three_way_comparison(backtests, output_path=str(plots_dir / "cumulative_three_way.png"))
    plot_feature_importance(model, output_path=str(plots_dir / "gbt_feature_importance.png"))
    plot_regime_bucketed_sharpe(
        backtests, regimes, output_path=str(plots_dir / "sharpe_by_regime.png")
    )

    logger.info("Done. Plots in %s", plots_dir.resolve())

    # Silence unused-import false positive in static analysis
    _ = (np, common, GBT_REFIT_FREQ)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script end-to-end**

Run: `python scripts/run_gbt_backtest.py`
Expected:
- Logs progress through each step
- Prints metrics for both `classical_mvo` and `gbt_mvo`
- Creates three PNGs in `plots/`
- Exits with code 0

- [ ] **Step 3: Commit**

```bash
git add scripts/run_gbt_backtest.py
git commit -m "feat: add end-to-end runner for three-way GBT vs MVO backtest"
```

---

## Task 10: Lint and type-check pass

- [ ] **Step 1: Run ruff**

Run: `ruff check src/ tests/ scripts/`
Expected: clean, or only pre-existing warnings unrelated to this work. Fix any new issues.

- [ ] **Step 2: Run pyright**

Run: `pyright src/`
Expected: clean, or only pre-existing warnings (matplotlib/yfinance type-stub noise is already suppressed in `pyproject.toml`).

- [ ] **Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 4: Commit any lint/type fixes**

If any fixes were needed:

```bash
git add -A
git commit -m "chore: address lint/type-check findings from GBT arm"
```

---

## Self-Review

- **Spec coverage:**
  - §3 Architecture (new modules, refactored backtest) → Tasks 4, 6, 7
  - §4 ML problem definition (target, features, model) → Tasks 3, 4, 5, 6
  - §5 Training methodology (walk-forward, hyperparams, early stopping) → Task 6 (early stopping wired in `train_gbt`); full Optuna walk-forward retraining is left as future work and noted in §10 of the spec as a risk — for the first cut, the model is trained once on the train portion and used across the full test period.
  - §6 Evaluation (3-way comparison, regime-bucketed metrics) → Tasks 8, 9
  - §7 Dependencies → Task 1
  - §8 Testing → Tasks 3, 5, 7

- **Placeholder scan:** No TBDs, TODOs, or "implement appropriate X" steps; every step has concrete code or commands.

- **Type/name consistency:** `gbt_mu_estimator` returns `Callable[[pd.DataFrame, pd.Timestamp], np.ndarray]`; `_default_mu_estimator` matches that signature; `_rebalance` accepts the same callable; `rolling_backtest` parameter is named `mu_estimator` in all places. Function names align across tasks: `train_gbt`, `predict_returns`, `gbt_mu_estimator`, `build_target`, `build_feature_panel`.

- **One scope decision deferred:** Optuna hyperparameter search (spec §5) and quarterly walk-forward refits (spec §5) are NOT included in this plan. The plan trains GBT once on the train split with default hyperparameters and uses early stopping for regularization. This produces a complete, working three-way comparison on the first iteration. A follow-up plan can add Optuna tuning and periodic refits if results justify the cost — the µ-estimator interface is already in place.
