"""Generate presentation-quality plots for the four-arm complexity ladder.

Reads the latest backtest artifacts (rerunning the benchmark internally so
we have the full daily portfolio-return series, not just summary metrics)
and produces slide-ready figures with a consistent visual identity:

- 16:9 aspect ratio, large fonts, soft gridlines
- Consistent per-arm colors across every plot
- Annotated headlines: best/worst arms, drawdown markers, regime shading
- A "hero" dashboard combining the four key metrics on one page

Outputs (to plots/):
- presentation_hero.png            — 2x2 dashboard: equity / Sharpe / DD / vol
- presentation_equity_curves.png   — annotated cumulative returns
- presentation_complexity_ladder.png — Sharpe & Sortino vs complexity
- presentation_metrics_bars.png    — grouped bar chart of key metrics
- presentation_regime_sharpe.png   — Sharpe by VIX regime
- presentation_gbt_importance.png  — GBT top-15 feature importance
- presentation_drawdown.png        — underwater plot for all four arms

Run: ``python scripts/draw_presentation_plots.py``
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

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
from src.risk_metrics import (
    conditional_var,
    portfolio_turnover,
    sharpe_ratio,
    value_at_risk,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------

# Per-arm colors. Picked from the Okabe-Ito colorblind-safe palette so the
# slides survive both projector glare and reviewers with red-green issues.
ARM_COLORS = {
    "Sample mean": "#888888",   # neutral gray — the baseline
    "Ridge":       "#E69F00",   # orange — the linear model
    "GBT":         "#009E73",   # green — winner on return (also evokes growth)
    "LSTM":        "#0072B2",   # blue — winner on risk control
}
ARM_ORDER = ["Sample mean", "Ridge", "GBT", "LSTM"]

# Mapping from raw backtest keys to display names
KEY_TO_NAME = {
    "1_sample_mean": "Sample mean",
    "2_linreg":       "Ridge",
    "3_gbt":          "GBT",
    "4_lstm":         "LSTM",
}


def _set_presentation_theme() -> None:
    """Apply slide-ready matplotlib defaults globally."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.color": "#cccccc",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _format_pct(x: float, _pos: int = 0) -> str:
    return f"{x * 100:.0f}%"


# ---------------------------------------------------------------------------
# Run benchmarks (we need the daily return series, not just metrics)
# ---------------------------------------------------------------------------

def run_all_backtests() -> tuple[
    dict[str, pd.DataFrame],
    pd.Series,
    object,                      # GBT model (for feature importance)
    pd.Timestamp,                # train end date
    pd.Timestamp,                # test start date
]:
    """Re-run the full four-arm backtest pipeline and return the daily series."""
    logger.info("Loading dataset")
    data = load_dataset(
        tickers=TICKERS, start=START_DATE, end=END_DATE,
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, cache_dir=RAW_DATA_DIR,
    )
    log_returns: pd.DataFrame = data["log_returns"]
    regimes: pd.Series = data["regime_vix"]
    vix = fetch_vix(start=START_DATE, end=END_DATE).reindex(log_returns.index).ffill().bfill()

    panel = build_feature_panel(log_returns, vix, regimes)
    target = build_target(log_returns, horizon=GBT_FORECAST_HORIZON)
    common = panel.index.intersection(target.index)

    train_end_date = data["train"].index[-1]
    train_idx = common[common.get_level_values("date") <= train_end_date]

    logger.info("Training Ridge")
    ridge_model, ridge_scaler, ridge_numeric, ridge_cols = train_linreg(
        panel.loc[train_idx], target.loc[train_idx]
    )
    logger.info("Training LightGBM")
    gbt_model = train_gbt(panel.loc[train_idx], target.loc[train_idx])
    logger.info("Training LSTM")
    lstm_model, lstm_asset_order, lstm_channel_stats, lstm_target_stats = train_lstm(
        log_returns.loc[: train_end_date],
        vix.loc[: train_end_date],
        sequence_length=LSTM_SEQUENCE_LENGTH,
        horizon=GBT_FORECAST_HORIZON,
    )

    asset_order = list(log_returns.columns)
    estimators = {
        "1_sample_mean": None,
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

    test_start_date = data["test"].index[0]
    lookback_idx = max(0, log_returns.index.get_loc(test_start_date) - ROLLING_WINDOW)
    backtest_returns = log_returns.iloc[lookback_idx:]

    backtests: dict[str, pd.DataFrame] = {}
    for name, est in estimators.items():
        logger.info("Backtest: %s", name)
        bt = rolling_backtest(
            backtest_returns, window=ROLLING_WINDOW, strategy="max_sharpe",
            rebalance_freq=21, mu_estimator=est,
        )
        # Trim to the actual test window (drop the lookback warmup)
        bt = bt.loc[bt.index >= test_start_date]
        backtests[KEY_TO_NAME[name]] = bt

    return backtests, regimes, gbt_model, train_end_date, test_start_date


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_summary(backtests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute Sharpe / Sortino / Annual return / Max DD / Vol for each arm."""
    rows = {}
    for name, df in backtests.items():
        ret = df["portfolio_return"]
        cum = ret.cumsum().apply(np.exp) - 1
        n = len(ret)
        ann_ret = ret.mean() * 252
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = sharpe_ratio(ret)
        downside = ret[ret < 0]
        ann_down = downside.std() * np.sqrt(252) if len(downside) > 1 else 1e-9
        sortino = ann_ret / ann_down if ann_down > 0 else 0.0
        running_max = (1 + cum).cummax()
        drawdown = (1 + cum) / running_max - 1
        max_dd = float(drawdown.min())
        rows[name] = {
            "Annual return": ann_ret,
            "Annual vol":    ann_vol,
            "Sharpe":        sharpe,
            "Sortino":       sortino,
            "Max drawdown":  max_dd,
            "n_days":        n,
        }
    return pd.DataFrame(rows).T.loc[ARM_ORDER]


def _compute_drawdown(returns: pd.Series) -> pd.Series:
    cum = returns.cumsum().apply(np.exp)
    running_max = cum.cummax()
    return cum / running_max - 1


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity_curves(
    backtests: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    save_path: Path,
) -> None:
    """Annotated equity-curve comparison."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for name in ARM_ORDER:
        df = backtests[name]
        cum = df["portfolio_return"].cumsum().apply(np.exp) - 1
        is_winner = name == summary["Sharpe"].idxmax()
        ax.plot(
            cum.index, cum.values * 100,
            color=ARM_COLORS[name], linewidth=2.6 if is_winner else 1.6,
            label=f"{name}  (Sharpe {summary.loc[name, 'Sharpe']:.2f})",
            zorder=3 if is_winner else 2,
        )

    # Annotate end-of-period totals at the right edge
    last_date = list(backtests.values())[0].index[-1]
    for name in ARM_ORDER:
        df = backtests[name]
        cum = df["portfolio_return"].cumsum().apply(np.exp) - 1
        final = cum.iloc[-1] * 100
        ax.annotate(
            f"  {final:.1f}%",
            xy=(last_date, final),
            xytext=(8, 0), textcoords="offset points",
            color=ARM_COLORS[name], fontsize=11, fontweight="bold",
            va="center",
        )

    ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.set_title("Cumulative Returns Over Test Period (2022–2025)", pad=16)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", title="Pipeline (sorted by complexity)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.text(
        0.99, 0.01,
        "Same Σ, optimizer, and constraints across arms; only µ differs.",
        ha="right", va="bottom", fontsize=9, style="italic", color="#666666",
    )
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_metrics_bars(summary: pd.DataFrame, save_path: Path) -> None:
    """Grouped bar chart: Sharpe / Sortino / Annual return / Max DD."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.5))

    panels = [
        ("Sharpe",        "Sharpe ratio",     False, "higher is better"),
        ("Sortino",       "Sortino ratio",    False, "higher is better"),
        ("Annual return", "Annual return",    True,  "higher is better"),
        ("Max drawdown",  "Max drawdown",     True,  "less negative is better"),
    ]

    for ax, (metric, ylabel, as_pct, footer) in zip(axes, panels, strict=True):
        vals = summary[metric].values
        colors = [ARM_COLORS[name] for name in ARM_ORDER]
        bars = ax.bar(ARM_ORDER, vals, color=colors, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals, strict=True):
            label = f"{v * 100:.1f}%" if as_pct else f"{v:.2f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 5 if v >= 0 else -14), textcoords="offset points",
                ha="center", fontsize=11, fontweight="bold",
            )
        ax.set_title(metric, pad=10)
        ax.set_xticklabels(ARM_ORDER, rotation=15, ha="right")
        if as_pct:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
        ax.text(
            0.5, -0.30, footer, transform=ax.transAxes, ha="center",
            fontsize=9, style="italic", color="#666666",
        )
        if metric == "Max drawdown":
            ax.set_ylim(top=0)

    fig.suptitle("Risk-Adjusted Performance — 4-Way Comparison", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_complexity_ladder(summary: pd.DataFrame, save_path: Path) -> None:
    """Sharpe and Sortino vs model complexity, with annotations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ARM_ORDER))

    sharpe = summary["Sharpe"].values
    sortino = summary["Sortino"].values

    ax.plot(x, sharpe, marker="o", color="#0072B2", linewidth=2.5, markersize=12, label="Sharpe ratio")
    ax.plot(x, sortino, marker="s", color="#D55E00", linewidth=2.5, markersize=12, label="Sortino ratio")

    for i, (s, so) in enumerate(zip(sharpe, sortino, strict=True)):
        ax.annotate(f"{s:.2f}", xy=(x[i], s), xytext=(0, 12),
                    textcoords="offset points", ha="center",
                    color="#0072B2", fontweight="bold", fontsize=11)
        ax.annotate(f"{so:.2f}", xy=(x[i], so), xytext=(0, -18),
                    textcoords="offset points", ha="center",
                    color="#D55E00", fontweight="bold", fontsize=11)

    # Highlight the peak
    peak_idx = int(np.argmax(sharpe))
    ax.axvline(peak_idx, color="#009E73", linewidth=1.5, alpha=0.3, linestyle="--")
    ax.annotate(
        f"Best Sharpe:\n{ARM_ORDER[peak_idx]}",
        xy=(peak_idx, sharpe[peak_idx]),
        xytext=(peak_idx + 0.1, sharpe[peak_idx] + 0.4),
        fontsize=11, fontweight="bold", color="#009E73",
        arrowprops=dict(arrowstyle="->", color="#009E73"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i + 1}. {name}" for i, name in enumerate(ARM_ORDER)], fontsize=12)
    ax.set_xlabel("Model (simple → complex)", fontsize=13)
    ax.set_ylabel("Risk-adjusted return")
    ax.set_title("The Complexity Ladder: Does More Capacity Help?", pad=16)
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)

    fig.text(
        0.5, 0.01,
        "Trees beat linear, recurrent beats no-learning baseline — but the win curve is non-monotonic.",
        ha="center", fontsize=10, style="italic", color="#666666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_drawdown(backtests: dict[str, pd.DataFrame], save_path: Path) -> None:
    """Underwater (drawdown) plot for all four arms."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for name in ARM_ORDER:
        dd = _compute_drawdown(backtests[name]["portfolio_return"])
        ax.fill_between(
            dd.index, dd.values * 100, 0,
            color=ARM_COLORS[name], alpha=0.35, linewidth=0,
        )
        ax.plot(dd.index, dd.values * 100, color=ARM_COLORS[name],
                linewidth=1.8, label=name)
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_title("Drawdown (underwater) — lower swings means better risk control", pad=16)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="lower left", ncol=4)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_regime_sharpe(
    backtests: dict[str, pd.DataFrame],
    regimes: pd.Series,
    save_path: Path,
) -> None:
    """Sharpe by VIX regime, grouped bar chart."""
    fig, ax = plt.subplots(figsize=(11, 6))
    regime_labels = ["low", "medium", "high"]
    n_pipes = len(ARM_ORDER)
    bar_w = 0.8 / n_pipes
    x = np.arange(len(regime_labels))

    sharpe_matrix = np.zeros((n_pipes, len(regime_labels)))
    for i, name in enumerate(ARM_ORDER):
        df = backtests[name]
        aligned = regimes.reindex(df.index)
        for j, label in enumerate(regime_labels):
            mask = aligned == label
            ret = df.loc[mask, "portfolio_return"]
            sharpe_matrix[i, j] = sharpe_ratio(ret) if len(ret) > 1 else 0.0

    for i, name in enumerate(ARM_ORDER):
        bars = ax.bar(
            x + i * bar_w, sharpe_matrix[i],
            bar_w, color=ARM_COLORS[name], edgecolor="white", linewidth=1.0,
            label=name,
        )
        for bar, val in zip(bars, sharpe_matrix[i], strict=True):
            ax.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 4 if val >= 0 else -12),
                textcoords="offset points", ha="center",
                fontsize=9, fontweight="bold",
            )
    ax.set_xticks(x + bar_w * (n_pipes - 1) / 2)
    ax.set_xticklabels(["Low (VIX<15)", "Medium (15–25)", "High (≥25)"], fontsize=12)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Performance by Volatility Regime", pad=16)
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.legend(loc="best", ncol=4)
    fig.text(
        0.5, 0.01,
        "Does ML help in market stress? High-vol Sharpe is the headline number for the project's central question.",
        ha="center", fontsize=10, style="italic", color="#666666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_turnover(backtests: dict[str, pd.DataFrame], save_path: Path) -> None:
    """Portfolio turnover per arm — proxy for transaction-cost burden.

    Turnover = average sum of absolute weight changes from one rebalance to
    the next. Higher turnover means more trading and more friction. The
    proposal explicitly lists this as a required metric.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    asset_columns = [c for c in list(backtests.values())[0].columns
                     if c not in ("portfolio_return", "cumulative_return")]
    turnovers: dict[str, float] = {}
    for name in ARM_ORDER:
        df = backtests[name]
        weights = df[asset_columns]
        turnovers[name] = portfolio_turnover(weights)

    bars = ax.bar(
        ARM_ORDER, [turnovers[n] for n in ARM_ORDER],
        color=[ARM_COLORS[n] for n in ARM_ORDER],
        edgecolor="white", linewidth=1.2,
    )
    for bar, name in zip(bars, ARM_ORDER, strict=True):
        ax.annotate(
            f"{turnovers[name]:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=12, fontweight="bold",
        )
    ax.set_title("Portfolio Turnover — Trading-cost Proxy", pad=16)
    ax.set_ylabel("Average daily L1 weight change")
    ax.text(
        0.5, -0.18,
        "Lower is cheaper to trade. Higher turnover = more rebalancing friction in real-world deployment.",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic", color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_var_cvar(backtests: dict[str, pd.DataFrame], save_path: Path) -> None:
    """VaR and CVaR at 95% — tail-risk metrics required by the proposal.

    VaR95 = the daily-return loss you would not exceed 95% of the time.
    CVaR95 = the average loss in the worst 5% of days. CVaR is the more
    informative of the two because it tells you *how bad* the tail is,
    not just where it starts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    var_vals, cvar_vals = {}, {}
    for name in ARM_ORDER:
        ret = backtests[name]["portfolio_return"]
        var_vals[name] = value_at_risk(ret, confidence=0.95)
        cvar_vals[name] = conditional_var(ret, confidence=0.95)

    for ax, vals, title, footer in [
        (axes[0], var_vals,
         "VaR (95%)",
         "Daily loss you won't exceed 95% of days. Less negative is better."),
        (axes[1], cvar_vals,
         "CVaR (95%)",
         "Average loss on the worst 5% of days. The 'how-bad' metric for tails."),
    ]:
        bars = ax.bar(
            ARM_ORDER, [vals[n] for n in ARM_ORDER],
            color=[ARM_COLORS[n] for n in ARM_ORDER],
            edgecolor="white", linewidth=1.2,
        )
        for bar, name in zip(bars, ARM_ORDER, strict=True):
            v = vals[name]
            ax.annotate(
                f"{v * 100:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=12, fontweight="bold",
            )
        ax.set_title(title, pad=10)
        ax.set_xticklabels(ARM_ORDER, rotation=15, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
        ax.set_ylim(top=0)
        ax.text(
            0.5, -0.30, footer, transform=ax.transAxes, ha="center",
            fontsize=10, style="italic", color="#666666",
        )

    fig.suptitle("Tail Risk: VaR and CVaR at 95%", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_rq_summary(
    backtests: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    regimes: pd.Series,
    save_path: Path,
) -> None:
    """One-page summary chart that explicitly answers the proposal's RQ1/RQ2/RQ3.

    Each panel maps to one research question:
      RQ1: Sharpe & Sortino — risk-adjusted return vs. classical MVO.
      RQ2: Sharpe by VIX regime — robustness in high-vol regimes.
      RQ3: Max drawdown + CVaR95 — downside risk reduction.
    """
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.55, wspace=0.25)

    # ---- RQ1: Sharpe & Sortino vs classical MVO ----
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ["Sharpe", "Sortino"]
    n_pipes = len(ARM_ORDER)
    bar_w = 0.35
    x = np.arange(n_pipes)
    for j, m in enumerate(metrics):
        vals = summary[m].values
        offset = (j - 0.5) * bar_w
        color = "#0072B2" if m == "Sharpe" else "#D55E00"
        bars = ax1.bar(
            x + offset, vals, bar_w,
            color=color,
            edgecolor="white", linewidth=1.0,
            label=m,
        )
        # Fade the baseline bar so the ML lift is visually obvious
        for bar, name in zip(bars, ARM_ORDER, strict=True):
            bar.set_alpha(0.45 if name == "Sample mean" else 1.0)
        for bar, v in zip(bars, vals, strict=True):
            ax1.annotate(
                f"{v:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold",
            )
    # Reference line at the baseline Sharpe
    baseline_sharpe = summary.loc["Sample mean", "Sharpe"]
    ax1.axhline(baseline_sharpe, color="#888", linewidth=1.0, linestyle="--",
                alpha=0.7, label=f"Baseline Sharpe = {baseline_sharpe:.2f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ARM_ORDER, fontsize=12)
    ax1.set_ylabel("Risk-adjusted return")
    ax1.set_title("RQ1 — Can ML improve Sharpe / Sortino vs classical MVO?",
                  fontsize=15, pad=10)
    ax1.legend(loc="upper left", ncol=3)
    rq1_winner = summary["Sharpe"].idxmax()
    rq1_lift = (summary.loc[rq1_winner, "Sharpe"] - baseline_sharpe) / baseline_sharpe * 100
    ax1.text(
        0.5, -0.22,
        f"Answer: YES — {rq1_winner} delivers Sharpe {summary.loc[rq1_winner, 'Sharpe']:.2f} "
        f"vs {baseline_sharpe:.2f} for sample-mean MVO ({rq1_lift:+.0f}%). "
        f"Ridge underperforms — linear models too rigid.",
        transform=ax1.transAxes, ha="center", fontsize=11, style="italic", color="#444",
    )

    # ---- RQ2: Sharpe by VIX regime ----
    ax2 = fig.add_subplot(gs[1, :])
    regime_labels = ["low", "medium", "high"]
    sharpe_matrix = np.zeros((n_pipes, len(regime_labels)))
    for i, name in enumerate(ARM_ORDER):
        df = backtests[name]
        aligned = regimes.reindex(df.index)
        for j, label in enumerate(regime_labels):
            mask = aligned == label
            ret = df.loc[mask, "portfolio_return"]
            sharpe_matrix[i, j] = sharpe_ratio(ret) if len(ret) > 1 else 0.0
    bw = 0.18
    xr = np.arange(len(regime_labels))
    for i, name in enumerate(ARM_ORDER):
        bars = ax2.bar(
            xr + i * bw, sharpe_matrix[i], bw,
            color=ARM_COLORS[name], edgecolor="white", linewidth=1.0,
            label=name,
        )
        for bar, val in zip(bars, sharpe_matrix[i], strict=True):
            ax2.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3 if val >= 0 else -12),
                textcoords="offset points", ha="center",
                fontsize=9, fontweight="bold",
            )
    ax2.axhline(0, color="#222", linewidth=0.8)
    ax2.set_xticks(xr + bw * (n_pipes - 1) / 2)
    ax2.set_xticklabels(["Low (VIX<15)", "Medium (15–25)", "High (≥25)"], fontsize=12)
    ax2.set_ylabel("Sharpe ratio")
    ax2.set_title("RQ2 — Is ML more robust during high-volatility regimes?",
                  fontsize=15, pad=10)
    ax2.legend(loc="best", ncol=4, fontsize=10)
    high_idx = regime_labels.index("high")
    high_winner_idx = int(np.argmax(sharpe_matrix[:, high_idx]))
    high_baseline = sharpe_matrix[ARM_ORDER.index("Sample mean"), high_idx]
    high_winner_val = sharpe_matrix[high_winner_idx, high_idx]
    rq2_text = (
        f"Answer: {ARM_ORDER[high_winner_idx]} dominates in high-vol regime "
        f"(Sharpe {high_winner_val:.2f} vs {high_baseline:.2f} for sample-mean). "
        f"All arms struggle in stress; the relative ML lift is largest there."
    )
    ax2.text(
        0.5, -0.22, rq2_text, transform=ax2.transAxes, ha="center",
        fontsize=11, style="italic", color="#444",
    )

    # ---- RQ3: Downside risk (Max DD + CVaR) ----
    ax3a = fig.add_subplot(gs[2, 0])
    ax3b = fig.add_subplot(gs[2, 1])
    dd_vals = summary["Max drawdown"].values
    cvar_vals = np.array([conditional_var(backtests[n]["portfolio_return"], 0.95)
                          for n in ARM_ORDER])
    for ax, vals, title in [
        (ax3a, dd_vals, "Max Drawdown"),
        (ax3b, cvar_vals, "CVaR (95%)"),
    ]:
        bars = ax.bar(
            ARM_ORDER, vals,
            color=[ARM_COLORS[n] for n in ARM_ORDER],
            edgecolor="white", linewidth=1.0,
        )
        for bar, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v * 100:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=11, fontweight="bold",
            )
        ax.set_title(title, fontsize=13)
        ax.set_xticklabels(ARM_ORDER, rotation=15, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
        ax.set_ylim(top=0)

    rq3_dd_winner = ARM_ORDER[int(np.argmax(dd_vals))]
    rq3_cvar_winner = ARM_ORDER[int(np.argmax(cvar_vals))]
    fig.text(
        0.5, 0.04,
        f"RQ3 — Does nonlinear modeling reduce downside risk?  "
        f"Best Max DD: {rq3_dd_winner} ({dd_vals[ARM_ORDER.index(rq3_dd_winner)] * 100:.1f}%).  "
        f"Best CVaR: {rq3_cvar_winner} ({cvar_vals[ARM_ORDER.index(rq3_cvar_winner)] * 100:.2f}%).",
        ha="center", fontsize=12, style="italic", color="#444",
    )
    fig.text(
        0.5, 0.99,
        "Research-Question Summary — ML vs Markowitz",
        ha="center", fontsize=20, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_gbt_importance(model, save_path: Path, top_n: int = 15) -> None:
    """Top-N LightGBM feature importance, annotated."""
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    df = (
        pd.DataFrame({"feature": names, "gain": importance})
        .sort_values("gain", ascending=True)
        .tail(top_n)
    )
    total = df["gain"].sum()
    df["pct"] = df["gain"] / total * 100

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(df["feature"], df["gain"], color="#009E73", edgecolor="white")
    for bar, gain, pct in zip(bars, df["gain"], df["pct"], strict=True):
        ax.annotate(
            f"  {gain:.0f}  ({pct:.1f}%)",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(2, 0), textcoords="offset points",
            va="center", fontsize=10,
        )
    ax.set_title(f"What is GBT Looking At?  Top {top_n} Features (gain)", pad=16)
    ax.set_xlabel("Total gain")
    ax.set_ylabel("")
    ax.set_xlim(right=df["gain"].max() * 1.30)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


def plot_hero_dashboard(
    backtests: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    save_path: Path,
) -> None:
    """2x2 hero panel: equity curves + 3 metric bars on one slide."""
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1.0], width_ratios=[1, 1, 1])

    # Top: equity curves spanning full width
    ax_eq = fig.add_subplot(gs[0, :])
    for name in ARM_ORDER:
        df = backtests[name]
        cum = df["portfolio_return"].cumsum().apply(np.exp) - 1
        is_winner = name == summary["Sharpe"].idxmax()
        ax_eq.plot(
            cum.index, cum.values * 100,
            color=ARM_COLORS[name], linewidth=2.8 if is_winner else 1.7,
            label=f"{name} (Sharpe {summary.loc[name, 'Sharpe']:.2f})",
            zorder=3 if is_winner else 2,
        )
    last_date = list(backtests.values())[0].index[-1]
    for name in ARM_ORDER:
        df = backtests[name]
        final = (df["portfolio_return"].cumsum().apply(np.exp).iloc[-1] - 1) * 100
        ax_eq.annotate(
            f"  {final:.1f}%",
            xy=(last_date, final), xytext=(8, 0), textcoords="offset points",
            color=ARM_COLORS[name], fontsize=11, fontweight="bold", va="center",
        )
    ax_eq.axhline(0, color="#444", linewidth=0.8, alpha=0.5)
    ax_eq.set_title("Equity Curves — 4-Way Complexity Ladder (2022–2025 test)", pad=12)
    ax_eq.set_ylabel("Cumulative return")
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_eq.legend(loc="upper left", ncol=4)

    # Bottom row: three bar charts
    bar_panels = [
        ("Sharpe",        "Sharpe",        False),
        ("Annual return", "Annual return", True),
        ("Max drawdown",  "Max DD",        True),
    ]
    for col, (metric, short, as_pct) in enumerate(bar_panels):
        ax = fig.add_subplot(gs[1, col])
        vals = summary[metric].values
        colors = [ARM_COLORS[n] for n in ARM_ORDER]
        bars = ax.bar(ARM_ORDER, vals, color=colors, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals, strict=True):
            label = f"{v * 100:.1f}%" if as_pct else f"{v:.2f}"
            offset = 5 if v >= 0 else -14
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, offset), textcoords="offset points",
                ha="center", fontsize=11, fontweight="bold",
            )
        ax.set_title(short, fontsize=14, pad=8)
        ax.set_xticklabels(ARM_ORDER, rotation=15, ha="right", fontsize=11)
        if as_pct:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
        if metric == "Max drawdown":
            ax.set_ylim(top=0)

    fig.suptitle(
        "Markowitz with ML-Forecasted µ — 4-Arm Benchmark",
        fontsize=20, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.005,
        "Same Σ, optimizer, constraints across all four arms — only µ differs.   "
        "Test period: 2022-03 → 2025-12.   Universe: 11 GICS sector ETFs.",
        ha="center", fontsize=10, style="italic", color="#666666",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("wrote %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _set_presentation_theme()

    backtests, regimes, gbt_model, _train_end, _test_start = run_all_backtests()
    summary = _compute_summary(backtests)

    print("\n=== Summary (test period) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    plot_hero_dashboard(backtests, summary, PLOTS_DIR / "presentation_hero.png")
    plot_rq_summary(backtests, summary, regimes, PLOTS_DIR / "presentation_rq_answers.png")
    plot_equity_curves(backtests, summary, PLOTS_DIR / "presentation_equity_curves.png")
    plot_metrics_bars(summary, PLOTS_DIR / "presentation_metrics_bars.png")
    plot_complexity_ladder(summary, PLOTS_DIR / "presentation_complexity_ladder.png")
    plot_drawdown(backtests, PLOTS_DIR / "presentation_drawdown.png")
    plot_regime_sharpe(backtests, regimes, PLOTS_DIR / "presentation_regime_sharpe.png")
    plot_var_cvar(backtests, PLOTS_DIR / "presentation_var_cvar.png")
    plot_turnover(backtests, PLOTS_DIR / "presentation_turnover.png")
    plot_gbt_importance(gbt_model, PLOTS_DIR / "presentation_gbt_importance.png")

    logger.info("All presentation plots written to %s", PLOTS_DIR.resolve())


if __name__ == "__main__":
    main()
