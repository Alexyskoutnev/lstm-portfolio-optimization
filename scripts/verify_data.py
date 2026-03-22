"""Smoke test: download full dataset and print summary statistics.

Also generates visualization plots to visually verify the data pipeline.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from src.config import END_DATE, RAW_DATA_DIR, START_DATE, TICKERS  # noqa: E402
from src.data import load_dataset
from src.models.mvo import (
    efficient_frontier,
    estimate_parameters,
    max_sharpe_weights,
    minimum_variance_weights,
    portfolio_performance,
    rolling_backtest,
)
from src.risk_metrics import compute_all_metrics
from src.visualization import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_efficient_frontier,
    plot_regime_returns,
    plot_rolling_sharpe,
    plot_weight_allocation,
)

PLOTS_DIR = Path("plots")


def main() -> None:
    """Run the full data pipeline, MVO backtest, and generate plots."""
    PLOTS_DIR.mkdir(exist_ok=True)

    # --- Data Pipeline ---
    print("=" * 60)
    print("STEP 1: Loading dataset...")
    print("=" * 60)
    data = load_dataset(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        cache_dir=RAW_DATA_DIR,
    )

    print(f"Prices shape: {data['prices'].shape}")
    print(f"Log returns shape: {data['log_returns'].shape}")
    print(f"Train: {len(data['train'])} | Val: {len(data['val'])} | Test: {len(data['test'])}")
    print(f"\nRegime distribution:\n{data['regime_vix'].value_counts()}")
    print(f"\nReturn statistics:\n{data['log_returns'].describe().round(6)}")

    # --- MVO Baseline ---
    print("\n" + "=" * 60)
    print("STEP 2: Running MVO Baseline...")
    print("=" * 60)

    train_returns = data["train"]
    mean_ret, cov_mat = estimate_parameters(train_returns)

    # Optimal portfolios
    mv_weights = minimum_variance_weights(cov_mat)
    ms_weights = max_sharpe_weights(mean_ret, cov_mat)

    mv_ret, mv_vol = portfolio_performance(mv_weights, mean_ret, cov_mat)
    ms_ret, ms_vol = portfolio_performance(ms_weights, mean_ret, cov_mat)

    print(f"\nMin Variance Portfolio: Return={mv_ret:.4f}, Vol={mv_vol:.4f}, "
          f"Sharpe={mv_ret / mv_vol:.4f}")
    print(f"Max Sharpe Portfolio:   Return={ms_ret:.4f}, Vol={ms_vol:.4f}, "
          f"Sharpe={ms_ret / ms_vol:.4f}")

    print("\nWeights:")
    for i, ticker in enumerate(TICKERS):
        print(f"  {ticker}: MinVar={mv_weights[i]:.4f}, MaxSharpe={ms_weights[i]:.4f}")

    # --- Backtesting ---
    print("\n" + "=" * 60)
    print("STEP 3: Running Rolling Backtest...")
    print("=" * 60)

    bt_sharpe = rolling_backtest(
        data["log_returns"], window=252, strategy="max_sharpe", rebalance_freq=21
    )
    bt_minvar = rolling_backtest(
        data["log_returns"], window=252, strategy="min_variance", rebalance_freq=21
    )

    # Risk metrics
    for name, bt in [("Max Sharpe", bt_sharpe), ("Min Variance", bt_minvar)]:
        metrics = compute_all_metrics(bt["portfolio_return"])
        print(f"\n{name} Strategy Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # --- Visualizations ---
    print("\n" + "=" * 60)
    print("STEP 4: Generating Visualizations...")
    print("=" * 60)

    # Efficient frontier
    frontier = efficient_frontier(mean_ret, cov_mat, n_points=100)
    plot_efficient_frontier(
        frontier,
        max_sharpe_point=(ms_vol, ms_ret),
        min_var_point=(mv_vol, mv_ret),
        save_path=str(PLOTS_DIR / "efficient_frontier.png"),
    )
    print("  Saved: plots/efficient_frontier.png")

    # Cumulative returns comparison
    plot_cumulative_returns(
        {"Max Sharpe": bt_sharpe, "Min Variance": bt_minvar},
        save_path=str(PLOTS_DIR / "cumulative_returns.png"),
    )
    print("  Saved: plots/cumulative_returns.png")

    # Weight allocation
    plot_weight_allocation(
        bt_sharpe,
        asset_columns=TICKERS,
        title="Max Sharpe Weight Allocation Over Time",
        save_path=str(PLOTS_DIR / "weight_allocation_sharpe.png"),
    )
    print("  Saved: plots/weight_allocation_sharpe.png")

    # Regime analysis
    regime_aligned = data["regime_vix"].reindex(bt_sharpe.index).ffill()
    plot_regime_returns(
        bt_sharpe["portfolio_return"],
        regime_aligned,
        save_path=str(PLOTS_DIR / "regime_returns.png"),
    )
    print("  Saved: plots/regime_returns.png")

    # Drawdown
    plot_drawdown(bt_sharpe, save_path=str(PLOTS_DIR / "drawdown.png"))
    print("  Saved: plots/drawdown.png")

    # Rolling Sharpe
    plot_rolling_sharpe(bt_sharpe, save_path=str(PLOTS_DIR / "rolling_sharpe.png"))
    print("  Saved: plots/rolling_sharpe.png")

    plt.close("all")
    print("\nDone! All plots saved to plots/ directory.")


if __name__ == "__main__":
    main()
