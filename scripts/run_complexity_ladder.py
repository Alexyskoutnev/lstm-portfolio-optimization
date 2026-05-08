"""End-to-end 4-way benchmark: model-complexity ladder for portfolio mu.

Runs four backtests on identical data, with identical Sigma, optimizer,
and weight constraints. Only the mu estimator differs:

    1. Sample mean   (no learning)
    2. Linear/Ridge  (simplest ML)
    3. GBT (LightGBM) (medium complexity, tree ensemble)
    4. LSTM          (most complex, recurrent)

Outputs (to plots/):
    cumulative_complexity_ladder.png
    sharpe_by_regime_complexity_ladder.png
    gbt_feature_importance.png
    complexity_ladder_metrics.png

And prints a summary metrics table to stdout.
"""

# IMPORTANT: macOS / OpenMP fix. LightGBM and PyTorch each ship their own
# libomp.dylib. When both are loaded into the same process the runtime gets
# confused and segfaults during the first LSTM forward pass. These env vars
# *must* be set before any import that pulls torch or lightgbm.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    END_DATE,
    GBT_FORECAST_HORIZON,
    LSTM_SEQUENCE_LENGTH,
    RAW_DATA_DIR,
    ROLLING_WINDOW,
    START_DATE,
    TICKERS,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data import build_feature_panel, fetch_vix, load_dataset
from src.models.gbt import build_target, gbt_mu_estimator, train_gbt
from src.models.linreg import linreg_mu_estimator, train_linreg
from src.models.lstm import lstm_mu_estimator, train_lstm
from src.models.mvo import rolling_backtest
from src.risk_metrics import compute_all_metrics
from src.visualization import (
    plot_complexity_ladder,
    plot_feature_importance,
    plot_multi_arm_cumulative,
    plot_regime_bucketed_sharpe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # ---- 1. Load data ----
    logger.info("Loading dataset for %d tickers", len(TICKERS))
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
    vix = fetch_vix(start=START_DATE, end=END_DATE).reindex(log_returns.index).ffill().bfill()

    # ---- 2. Build feature panel + target ----
    logger.info("Building feature panel and target")
    panel = build_feature_panel(log_returns, vix, regimes)
    target = build_target(log_returns, horizon=GBT_FORECAST_HORIZON)
    common = panel.index.intersection(target.index)

    # ---- 3. Train each model on the train portion only ----
    train_end_date = data["train"].index[-1]
    train_dates_mask = common.get_level_values("date") <= train_end_date
    train_idx = common[train_dates_mask]
    logger.info(
        "Train rows: %d (dates ≤ %s)", len(train_idx), train_end_date.date()
    )

    logger.info("[1/3] Training Ridge regression")
    ridge_model, ridge_scaler, ridge_numeric, ridge_cols = train_linreg(
        panel.loc[train_idx], target.loc[train_idx]
    )

    logger.info("[2/3] Training LightGBM")
    gbt_model = train_gbt(panel.loc[train_idx], target.loc[train_idx])

    logger.info("[3/3] Training LSTM")
    train_log_returns = log_returns.loc[: train_end_date]
    train_vix = vix.loc[: train_end_date]
    lstm_model, lstm_asset_order, lstm_channel_stats, lstm_target_stats = train_lstm(
        train_log_returns,
        train_vix,
        sequence_length=LSTM_SEQUENCE_LENGTH,
        horizon=GBT_FORECAST_HORIZON,
    )

    # ---- 4. Build mu estimators ----
    asset_order = list(log_returns.columns)
    estimators = {
        "1_sample_mean":  None,  # default behavior
        "2_linreg":       linreg_mu_estimator(
            ridge_model, ridge_scaler, ridge_numeric, ridge_cols,
            panel, asset_order, GBT_FORECAST_HORIZON,
        ),
        "3_gbt":          gbt_mu_estimator(
            gbt_model, panel, asset_order, GBT_FORECAST_HORIZON,
        ),
        "4_lstm":         lstm_mu_estimator(
            lstm_model, log_returns, vix, lstm_asset_order,
            lstm_channel_stats, lstm_target_stats,
            LSTM_SEQUENCE_LENGTH, GBT_FORECAST_HORIZON,
        ),
    }

    # ---- 5. Run 4 backtests on the test slice (with lookback before) ----
    test_start_date = data["test"].index[0]
    lookback_idx = max(0, log_returns.index.get_loc(test_start_date) - ROLLING_WINDOW)
    backtest_returns = log_returns.iloc[lookback_idx:]

    backtests: dict[str, pd.DataFrame] = {}
    for name, est in estimators.items():
        logger.info("Running backtest: %s", name)
        backtests[name] = rolling_backtest(
            backtest_returns,
            window=ROLLING_WINDOW,
            strategy="max_sharpe",
            rebalance_freq=21,
            mu_estimator=est,
        )

    # ---- 6. Compute and print metrics ----
    metrics_rows = {}
    print("\n" + "=" * 78)
    print("4-WAY BENCHMARK — model-complexity ladder")
    print("=" * 78)
    for name, df in backtests.items():
        m = compute_all_metrics(df["portfolio_return"])
        metrics_rows[name] = m
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k:<22s} {v:+.4f}")

    metrics_table = pd.DataFrame(metrics_rows).T
    print("\n" + "-" * 78)
    print("Summary table:")
    print("-" * 78)
    print(metrics_table.to_string())
    print("=" * 78 + "\n")

    metrics_table.to_csv(plots_dir / "complexity_ladder_metrics.csv")
    logger.info("Wrote %s", plots_dir / "complexity_ladder_metrics.csv")

    # ---- 7. Plots ----
    plot_multi_arm_cumulative(
        backtests, save_path=str(plots_dir / "cumulative_complexity_ladder.png"),
    )
    plot_regime_bucketed_sharpe(
        backtests, regimes,
        save_path=str(plots_dir / "sharpe_by_regime_complexity_ladder.png"),
    )
    plot_feature_importance(
        gbt_model, save_path=str(plots_dir / "gbt_feature_importance.png"),
    )
    plot_complexity_ladder(
        metrics_table, save_path=str(plots_dir / "complexity_ladder_metrics.png"),
    )

    logger.info("Done. Plots in %s", plots_dir.resolve())


if __name__ == "__main__":
    main()
