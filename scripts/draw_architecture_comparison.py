"""Generate side-by-side architecture diagrams comparing classical MVO and GBT-MVO.

Outputs (in plots/):
- mvo_vs_gbt_pipeline.png — vertical flow comparison, side by side
- mvo_vs_gbt_mu_zoom.png — zoom on the only step that differs (the mu estimator)
- mvo_vs_gbt_data_flow.png — data flow into both pipelines

These are explanatory diagrams (not runtime backtest plots). Useful for
the writeup, slides, or README. Re-run any time the architecture changes.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Color palette — semantically meaningful so the eye tracks "shared" vs "different"
COLOR_SHARED = "#cfe2f3"   # light blue: identical between pipelines
COLOR_DIFF = "#f4cccc"     # light red: the only step that differs
COLOR_INPUT = "#d9ead3"    # light green: raw inputs
COLOR_OUTPUT = "#fce5cd"   # light orange: outputs / metrics
EDGE = "#333333"


def _box(ax, x, y, w, h, text, face, fontsize=9, fontweight="normal"):
    """Draw a rounded rectangle with centered text."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2, edgecolor=EDGE, facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.3, color=EDGE),
    )


# ---------------------------------------------------------------------------
# 1. Side-by-side full-pipeline comparison
# ---------------------------------------------------------------------------
def draw_pipeline_comparison() -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Title
    ax.text(6.5, 10.5, "Classical MVO  vs  GBT-MVO",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(6.5, 10.05,
            "Identical pipeline, except the µ (expected-return) estimator",
            ha="center", fontsize=10, style="italic", color="#555555")

    # Column headers
    ax.text(2.75, 9.4, "Classical MVO", ha="center", fontsize=13, fontweight="bold")
    ax.text(10.25, 9.4, "GBT-MVO", ha="center", fontsize=13, fontweight="bold")

    # ---- Classical MVO column (left) ----
    box_w, box_h = 4.5, 0.7
    x_left = 0.5

    steps_left = [
        ("Adj. close prices (11 sector ETFs)", COLOR_INPUT),
        ("Daily log returns", COLOR_SHARED),
        ("Lookback window (252 days)", COLOR_SHARED),
        ("µ = sample mean × 252", COLOR_DIFF),
        ("Σ = sample covariance × 252", COLOR_SHARED),
        ("SLSQP: max Sharpe / min variance", COLOR_SHARED),
        ("Portfolio weights w (sum to 1, ≥ 0)", COLOR_SHARED),
        ("Risk metrics (Sharpe, Sortino, MDD, …)", COLOR_OUTPUT),
    ]

    y_top = 8.5
    y_step = 1.0
    for i, (text, color) in enumerate(steps_left):
        y = y_top - i * y_step
        _box(ax, x_left, y, box_w, box_h, text, color)
        if i < len(steps_left) - 1:
            _arrow(ax, x_left + box_w / 2, y, x_left + box_w / 2, y - (y_step - box_h))

    # ---- GBT-MVO column (right) ----
    x_right = 8.0
    steps_right = [
        ("Adj. close prices (11 sector ETFs) + VIX", COLOR_INPUT),
        ("Daily log returns + regime labels", COLOR_SHARED),
        ("Feature panel (lags, vol, VIX, regime, …)", COLOR_DIFF),
        ("µ = LightGBM forecast × (252 / 21)", COLOR_DIFF),
        ("Σ = sample covariance × 252", COLOR_SHARED),
        ("SLSQP: max Sharpe / min variance", COLOR_SHARED),
        ("Portfolio weights w (sum to 1, ≥ 0)", COLOR_SHARED),
        ("Risk metrics (Sharpe, Sortino, MDD, …)", COLOR_OUTPUT),
    ]

    for i, (text, color) in enumerate(steps_right):
        y = y_top - i * y_step
        _box(ax, x_right, y, box_w, box_h, text, color)
        if i < len(steps_right) - 1:
            _arrow(ax, x_right + box_w / 2, y, x_right + box_w / 2, y - (y_step - box_h))

    # Legend — two rows of two items each
    legend_items = [
        ("Shared (identical across pipelines)", COLOR_SHARED),
        ("Differs (µ + features)", COLOR_DIFF),
        ("Raw input", COLOR_INPUT),
        ("Output", COLOR_OUTPUT),
    ]
    for idx, (label, color) in enumerate(legend_items):
        row = idx // 2
        col = idx % 2
        legend_x = 0.5 + col * 6.5
        legend_y = 0.6 - row * 0.4
        rect = patches.FancyBboxPatch(
            (legend_x, legend_y), 0.35, 0.22,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.0, edgecolor=EDGE, facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(legend_x + 0.5, legend_y + 0.11, label, va="center", fontsize=9)

    fig.savefig(PLOTS_DIR / "mvo_vs_gbt_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS_DIR / 'mvo_vs_gbt_pipeline.png'}")


# ---------------------------------------------------------------------------
# 2. Zoom on the only step that differs — the µ estimator
# ---------------------------------------------------------------------------
def draw_mu_zoom() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6, 5.5, "The only thing that changes:  µ", ha="center",
            fontsize=15, fontweight="bold")
    ax.text(6, 5.05,
            "Σ, the optimizer, the constraints, and the backtest harness all stay identical.",
            ha="center", fontsize=10, style="italic", color="#555555")

    # Classical
    ax.text(2.5, 4.5, "Classical MVO", ha="center", fontsize=12, fontweight="bold")
    _box(ax, 0.5, 3.4, 4.0, 0.7, "Last 252 days of returns", COLOR_INPUT)
    _arrow(ax, 2.5, 3.4, 2.5, 3.0)
    _box(ax, 0.5, 2.3, 4.0, 0.7, "µ = mean of those 252 days × 252", COLOR_DIFF, fontweight="bold")
    ax.text(2.5, 1.7, "One number per asset.", ha="center", fontsize=9, style="italic")
    ax.text(2.5, 1.4, "No conditioning, no learning,", ha="center", fontsize=9, style="italic")
    ax.text(2.5, 1.1, "treats all 252 days equally.", ha="center", fontsize=9, style="italic")

    # GBT
    ax.text(9.0, 4.5, "GBT-MVO", ha="center", fontsize=12, fontweight="bold")
    _box(ax, 7.0, 3.4, 4.0, 0.7, "Feature vector at date t (per asset)", COLOR_INPUT)
    _arrow(ax, 9.0, 3.4, 9.0, 3.0)
    _box(ax, 7.0, 2.3, 4.0, 0.7, "µ = LightGBM(features) × (252 / 21)", COLOR_DIFF, fontweight="bold")
    ax.text(9.0, 1.7, "Conditional on today's state:", ha="center", fontsize=9, style="italic")
    ax.text(9.0, 1.4, "lags, vol, VIX, regime, cross-asset.", ha="center", fontsize=9, style="italic")
    ax.text(9.0, 1.1, "Trained to predict next 21-day return.", ha="center", fontsize=9, style="italic")

    # Big vs.
    ax.text(6.0, 3.0, "vs.", ha="center", fontsize=22, fontweight="bold", color="#888888")

    fig.savefig(PLOTS_DIR / "mvo_vs_gbt_mu_zoom.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS_DIR / 'mvo_vs_gbt_mu_zoom.png'}")


# ---------------------------------------------------------------------------
# 3. Data-flow diagram showing where features feed the GBT
# ---------------------------------------------------------------------------
def draw_data_flow() -> None:
    """Left-to-right data-flow diagram for GBT-MVO with two clear lanes.

    Top lane (µ path):  inputs -> feature panel -> LightGBM -> predicted µ -> MVO
    Bottom lane (Σ path): returns -> sample covariance -> MVO
    Bottom-left aside: training-target construction (labels for LightGBM)
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7.5, 7.5, "GBT-MVO: data flow", ha="center", fontsize=16, fontweight="bold")
    ax.text(7.5, 7.1,
            "Two lanes: µ goes through the GBT, Σ stays sample-based.",
            ha="center", fontsize=10, style="italic", color="#555555")

    # ---- Inputs (left column, vertically stacked) ----
    inputs = [
        (5.6, "Sector-ETF prices"),
        (4.7, "VIX"),
        (3.8, "Regime labels"),
        (2.9, "Calendar features"),
    ]
    for y, text in inputs:
        _box(ax, 0.3, y, 2.4, 0.6, text, COLOR_INPUT)

    # ---- Top lane: µ path ----
    # Feature panel
    _box(ax, 3.5, 3.7, 2.6, 1.2,
         "Feature panel\n(date × asset)",
         COLOR_DIFF, fontsize=10, fontweight="bold")
    for y, _ in inputs:
        _arrow(ax, 2.7, y + 0.3, 3.5, 4.3)

    # GBT
    _box(ax, 7.0, 3.9, 2.2, 0.8, "LightGBM\n(global)",
         COLOR_DIFF, fontsize=11, fontweight="bold")
    _arrow(ax, 6.1, 4.3, 7.0, 4.3)

    # Predicted mu
    _box(ax, 10.0, 3.9, 1.8, 0.8, "Predicted µ\n(11 × 1)",
         COLOR_OUTPUT, fontsize=10)
    _arrow(ax, 9.2, 4.3, 10.0, 4.3)

    # ---- Bottom lane: Σ path ----
    _box(ax, 3.5, 1.7, 2.6, 0.7, "Daily log returns", COLOR_SHARED, fontsize=10)
    _arrow(ax, 1.5, 2.9, 3.5, 2.05)  # from prices
    _box(ax, 7.0, 1.7, 2.2, 0.7, "Σ = sample cov", COLOR_SHARED, fontsize=10)
    _arrow(ax, 6.1, 2.05, 7.0, 2.05)

    # ---- Convergence: MVO ----
    _box(ax, 12.5, 2.8, 1.8, 1.1, "MVO\n(SLSQP)\nmax Sharpe",
         COLOR_SHARED, fontsize=10, fontweight="bold")
    _arrow(ax, 11.8, 4.3, 12.5, 3.6)   # µ → MVO
    _arrow(ax, 9.2, 2.05, 12.5, 3.1)    # Σ → MVO

    # ---- Output: weights ----
    _box(ax, 12.5, 1.0, 1.8, 0.8, "Weights w\n(11 × 1)", COLOR_OUTPUT, fontsize=10)
    _arrow(ax, 13.4, 2.8, 13.4, 1.8)

    # ---- Training-label aside (below feature panel) ----
    _box(ax, 3.5, 0.3, 4.0, 0.7,
         "Training labels: y = sum of next 21 daily log returns",
         COLOR_INPUT, fontsize=9)
    # arrow up to LightGBM (training-time only)
    ax.annotate(
        "", xy=(8.1, 3.9), xytext=(5.5, 1.0),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#888888",
                        linestyle=(0, (4, 2))),
    )
    ax.text(6.7, 1.7, "(training only)", fontsize=8, style="italic", color="#888888")

    fig.savefig(PLOTS_DIR / "mvo_vs_gbt_data_flow.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS_DIR / 'mvo_vs_gbt_data_flow.png'}")


if __name__ == "__main__":
    draw_pipeline_comparison()
    draw_mu_zoom()
    draw_data_flow()
