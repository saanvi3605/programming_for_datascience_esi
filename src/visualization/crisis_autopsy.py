"""
src/visualization/crisis_autopsy.py
--------------------------------------
Crisis autopsy charts showing engine breakdown per historical episode.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RESULTS_FIG, ENGINE_COLORS

CRISES = {
    "Oil Embargo\n(1973-11)":        "1973-11-01",
    "Volcker Shock\n(1980-03)":      "1980-03-01",
    "Recession trough\n(1982-12)":   "1982-12-01",
    "Dot-com / 9-11\n(2001-10)":     "2001-10-01",
    "GFC Peak\n(2008-10)":           "2008-10-01",
    "GFC Trough\n(2009-06)":         "2009-06-01",
    "COVID Shock\n(2020-04)":        "2020-04-01",
    "Inflation Peak\n(2022-06)":     "2022-06-01",
}


def plot_crisis_autopsies(feat, esi, save_dir):
    """Spider / bar chart for each crisis showing engine scores."""
    engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "processed", "engine_scores.csv")

    if not os.path.exists(engine_path):
        print("  [Autopsy] engine_scores.csv not found — skipping")
        return

    eng_df = pd.read_csv(engine_path, index_col=0, parse_dates=True)
    eng_cols = [c for c in eng_df.columns if c not in ("ESI", "RECESSION")]

    n_crises = len(CRISES)
    n_cols   = 4
    n_rows   = (n_crises + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4.5))
    axes_flat = axes.flatten()

    for idx, (label, date_str) in enumerate(CRISES.items()):
        ax = axes_flat[idx]
        try:
            target  = pd.Timestamp(date_str)
            loc     = eng_df.index.get_indexer([target], method="nearest")[0]
            row     = eng_df.iloc[loc]
            esi_val = esi.iloc[esi.index.get_indexer([target], method="nearest")[0]]

            vals   = [row.get(e, 0) for e in eng_cols]
            colors = [ENGINE_COLORS.get(e, "#888888") for e in eng_cols]
            bars   = ax.barh(eng_cols, vals, color=colors, alpha=0.85)
            ax.axvline(50, color="grey", lw=0.8, ls="--", alpha=0.5)
            ax.axvline(80, color="red",  lw=0.7, ls=":", alpha=0.4)
            ax.set_xlim(0, 105)
            ax.set_xlabel("Score (0-100)", fontsize=8)
            ax.set_title(f"{label}\nESI={esi_val:.0f}", fontweight="bold", fontsize=9)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(min(val + 1.5, 102), bar.get_y() + bar.get_height()/2,
                            f"{val:.0f}", va="center", fontsize=8, fontweight="bold")
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n{e}", transform=ax.transAxes, ha="center")
            ax.set_title(label, fontsize=9)

    for extra_ax in axes_flat[n_crises:]:
        extra_ax.set_visible(False)

    fig.suptitle("Crisis Autopsies — Engine Score Breakdown at Historical Stress Peaks\n"
                 "(Each bar = how stressed that engine was vs all history up to that date)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "13_crisis_autopsies.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_annual_heatmap(feat, esi, save_dir):
    """Fallback annual heatmap if engine_scores not available."""
    feature_cols = [c for c in feat.columns if c != "RECESSION"]
    annual = feat[feature_cols].copy(); annual["year"] = annual.index.year
    import seaborn as sns
    annual_med = annual.groupby("year").median()[feature_cols[:10]]
    scaler_val = annual_med.apply(lambda x: (x-x.min())/(x.max()-x.min())*100)
    fig, ax = plt.subplots(figsize=(max(14, len(scaler_val)*0.45), 6))
    sns.heatmap(scaler_val.T, ax=ax, cmap="RdYlGn_r", vmin=0, vmax=100,
                linewidths=0.3, cbar_kws={"shrink":0.6})
    ax.set_title("Annual Feature Heatmap", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "14_stress_heatmap_annual.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")
