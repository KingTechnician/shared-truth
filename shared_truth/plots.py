"""Figure style and the three standard per-pair figures.

All four notebooks re-declared the palette and _save; the rebuttal figures
notebook re-declared them again under a comment saying "reproduced from the
experiment notebook".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .naming import short_name

__all__ = [
    "BLUE", "RED", "GREEN", "GREY", "ORANGE", "PLOT_RC", "use_style", "save_fig",
    "plot_alpha_sweep", "plot_score_distributions", "plot_geometry_collapse",
    "plot_all",
]

BLUE, RED, GREEN, GREY, ORANGE = "#2166AC", "#D6604D", "#4DAC26", "#888888", "#F4A582"

PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
}

COLLAPSE_DELTA_THRESHOLD = 0.15  # delta_mean below this marks the collapse zone


def use_style():
    plt.rcParams.update(PLOT_RC)


def save_fig(fig, out_dir, stem):
    """Write both .pdf (for LaTeX) and .png (for previews)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")


def plot_alpha_sweep(payload, out_dir):
    use_style()
    sweep = payload["alpha_sweep"]
    alphas = [r["alpha"] for r in sweep]
    accs = [r["accuracy"] * 100 for r in sweep]
    aurocs = [r["auroc"] for r in sweep]

    src_name = short_name(payload["source_model"])
    tgt_name = short_name(payload["target_model"])
    src_acc = payload["ceiling"]["source"]["accuracy"] * 100
    tgt_acc = payload["ceiling"]["target"]["accuracy"] * 100
    src_auroc = payload["ceiling"]["source"]["auroc"]
    tgt_auroc = payload["ceiling"]["target"]["auroc"]

    peak_idx = int(np.argmax(accs))
    peak_alpha, peak_acc = alphas[peak_idx], accs[peak_idx]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    l1, = ax1.plot(alphas, accs, color=BLUE, marker="o", ms=5, lw=2,
                   label="Accuracy (%)")
    l2, = ax2.plot(alphas, aurocs, color=RED, marker="s", ms=5, lw=2, ls="--",
                   label="AUROC")

    ax1.axhline(tgt_acc, color=BLUE, lw=1, ls=":", alpha=0.7)
    ax1.axhline(src_acc, color=GREEN, lw=1, ls=":", alpha=0.7)
    ax2.axhline(tgt_auroc, color=RED, lw=1, ls=":", alpha=0.7)

    ax1.text(1.01, tgt_acc, f"{tgt_name}\nnative ({tgt_acc:.1f}%)",
             transform=ax1.get_yaxis_transform(), va="center", fontsize=7,
             color=BLUE, alpha=0.8)
    ax1.text(1.01, src_acc, f"{src_name}\nnative ({src_acc:.1f}%)",
             transform=ax1.get_yaxis_transform(), va="center", fontsize=7,
             color=GREEN, alpha=0.8)
    ax2.text(-0.02, tgt_auroc, f"{tgt_auroc:.4f}",
             transform=ax2.get_yaxis_transform(), ha="right", va="center",
             fontsize=7, color=RED, alpha=0.8)

    ax1.annotate(f"peak {peak_acc:.1f}%\n(\u03b1={peak_alpha})",
                 xy=(peak_alpha, peak_acc),
                 xytext=(peak_alpha - 0.18, peak_acc - 6),
                 fontsize=8, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))

    ax1.set_xlabel("Injection strength \u03b1\n"
                   "(0 = native target, 1 = full adapter output)", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", color=BLUE, fontsize=10)
    ax2.set_ylabel("AUROC", color=RED, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_xticks(alphas)
    ax1.set_xlim(-0.03, 1.03)

    ax1.legend(handles=[
        l1, l2,
        Line2D([0], [0], color=BLUE, lw=1, ls=":",
               label=f"{tgt_name} ceiling ({tgt_acc:.1f}% / {tgt_auroc:.4f})"),
        Line2D([0], [0], color=GREEN, lw=1, ls=":",
               label=f"{src_name} native ({src_acc:.1f}% / {src_auroc:.4f})"),
    ], fontsize=8, loc="lower left")

    fig.suptitle("Truth Preservation vs. Injection Strength\n"
                 f"{payload['display_label']}", fontsize=10, y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig1_alpha_sweep")
    plt.close(fig)
    return peak_alpha


def plot_score_distributions(payload, peak_alpha, out_dir):
    use_style()
    by_alpha = {round(r["alpha"], 1): r for r in payload["alpha_sweep"]}
    hist_alphas = [0.0, round(peak_alpha, 1), 1.0]
    hist_labels = ["\u03b1=0.0  (native target)",
                   f"\u03b1={peak_alpha}  (peak accuracy)",
                   "\u03b1=1.0  (full injection)"]
    bins = np.linspace(0, 1, 21)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, alpha, label in zip(axes, hist_alphas, hist_labels):
        row = by_alpha[alpha]
        ax.hist(row["false_scores"], bins=bins, alpha=0.55, color=RED,
                label="False", density=True)
        ax.hist(row["true_scores"], bins=bins, alpha=0.55, color=GREEN,
                label="True", density=True)
        delta = np.mean(row["true_scores"]) - np.mean(row["false_scores"])
        ax.set_title(f"{label}\n\u0394 Mean = {delta:.4f}", fontsize=9)
        ax.set_xlabel("Probe score (P(true))", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle("Probe Score Distributions at Key \u03b1 Values\n"
                 f"{payload['display_label']}", fontsize=10, y=1.03)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_score_distributions")
    plt.close(fig)


def plot_geometry_collapse(payload, out_dir):
    use_style()
    sweep = payload["alpha_sweep"]
    alphas = [r["alpha"] for r in sweep]
    deltas = [r["delta_mean"] for r in sweep]
    pct_pos = [r["pct_pred_positive"] * 100 for r in sweep]

    fig, ax3 = plt.subplots(figsize=(7, 4.0))
    ax4 = ax3.twinx()
    ax4.spines["right"].set_visible(True)

    l3, = ax3.plot(alphas, deltas, color=BLUE, marker="o", ms=5, lw=2,
                   label="\u0394 Mean (True \u2212 False score separation)")
    l4, = ax4.plot(alphas, pct_pos, color=RED, marker="s", ms=5, lw=2, ls="--",
                   label="% predicted positive")

    ax4.axhline(50, color=GREY, lw=1, ls=":", alpha=0.6)
    ax4.text(1.01, 50, "50%", transform=ax4.get_yaxis_transform(),
             va="center", fontsize=7, color=GREY)

    collapse_alpha = next(
        (a for a, d in zip(alphas, deltas) if d < COLLAPSE_DELTA_THRESHOLD), None)
    if collapse_alpha is not None:
        ax3.axvspan(collapse_alpha, 1.0, alpha=0.06, color=RED)

    ax3.set_xlabel("Injection strength \u03b1", fontsize=10)
    ax3.set_ylabel("\u0394 Mean score (True \u2212 False)", color=BLUE, fontsize=10)
    ax4.set_ylabel("% examples predicted positive", color=RED, fontsize=10)
    ax3.tick_params(axis="y", labelcolor=BLUE)
    ax4.tick_params(axis="y", labelcolor=RED)
    ax3.set_xticks(alphas)
    ax3.set_xlim(-0.03, 1.03)
    ax4.set_ylim(0, 110)

    ax3.legend(handles=[
        l3, l4,
        Line2D([0], [0], color=RED, alpha=0.3, lw=8, label="Geometry collapse zone"),
    ], fontsize=8, loc="center left")

    fig.suptitle("Geometry Separation and Threshold Shift vs. Injection Strength\n"
                 f"{payload['display_label']}", fontsize=10, y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_geometry_collapse")
    plt.close(fig)


def plot_all(payload, out_dir):
    peak = plot_alpha_sweep(payload, out_dir)
    plot_score_distributions(payload, peak, out_dir)
    plot_geometry_collapse(payload, out_dir)
