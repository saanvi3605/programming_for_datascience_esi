"""
src/analysis/stress_index.py — v4 wrapper
Builds final ESI from engine scores and generates main ESI charts.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_PROC, RESULTS_FIG, RESULTS_TAB, STRESS_LABELS, ENGINE_COLORS

SHADE_COLOR = "#fee0d2"
plt.rcParams.update({"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False,
                      "axes.grid":True,"grid.alpha":0.3,"font.size":10})

def _shade(ax, recession, alpha=0.6):
    in_rec, start = False, None
    for dt, v in recession.items():
        if v==1 and not in_rec: start, in_rec=dt,True
        elif v==0 and in_rec:
            ax.axvspan(start,dt,color=SHADE_COLOR,alpha=alpha,zorder=0); in_rec=False
    if in_rec: ax.axvspan(start,recession.index[-1],color=SHADE_COLOR,alpha=alpha,zorder=0)


def plot_main_esi(engine_scores, esi, recession, save_dir):
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
    rec = recession.reindex(esi.index).fillna(0)

    # Top: ESI
    ax = axes[0]
    _shade(ax, rec)
    ax.fill_between(esi.index, esi.values, alpha=0.22, color="#c0392b")
    ax.plot(esi.index, esi.values, lw=2.2, color="#c0392b", label="ESI (equal-weight of 5 engines)")
    for lbl, (lo, hi) in STRESS_LABELS.items():
        ax.axhline(lo, color="grey", lw=0.6, ls=":", alpha=0.7)
        if lo > 0: ax.text(esi.index[3], lo+1.5, lbl, fontsize=7, color="grey")
    ax.set_ylim(0, 105); ax.set_ylabel("ESI (0-100)")
    ax.set_title("Economic Stress Index — 5-Engine Framework\n"
                 "(Equal weight of Inflation + Labour + Financial + Monetary + Real engines; "
                 "shading = NBER recessions)", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")

    # Bottom: stacked engine contributions
    ax2 = axes[1]
    _shade(ax2, rec)
    engine_cols = [c for c in engine_scores.columns if c in ENGINE_COLORS]
    eng_aligned = engine_scores[engine_cols].reindex(esi.index)
    n_eng = len(engine_cols)
    bottom = pd.Series(0.0, index=esi.index)
    for eng in engine_cols:
        s = eng_aligned[eng].fillna(0) / n_eng
        ax2.fill_between(esi.index, bottom, bottom+s,
                         color=ENGINE_COLORS.get(eng,"#888"), alpha=0.72, label=eng)
        bottom += s
    ax2.set_ylim(0, 25.5)
    ax2.set_yticks([0,5,10,15,20])
    ax2.set_yticklabels(["0","20","40","60","80"])
    ax2.set_ylabel("Engine Contribution (rescaled)")
    ax2.set_title("ESI Decomposition by Engine — Which Dimension Drives Stress?", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9, ncol=5)

    fig.tight_layout()
    path = os.path.join(save_dir, "08_stress_index.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_stress_heatmap(engine_scores, esi, save_dir):
    annual = engine_scores.copy(); annual["year"] = annual.index.year
    annual_med = annual.groupby("year").median()
    fig, ax = plt.subplots(figsize=(max(14, len(annual_med)*0.45), 5))
    sns.heatmap(annual_med.T, ax=ax, cmap="RdYlGn_r", vmin=0, vmax=100,
                linewidths=0.3, annot=True, fmt=".0f", annot_kws={"size":6},
                cbar_kws={"label":"Median Engine Score","shrink":0.6})
    ax.set_title("Annual Engine Stress Heatmap", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "14_stress_heatmap_annual.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def main():
    feat_path   = os.path.join(DATA_PROC, "fred_features.csv")
    engine_path = os.path.join(DATA_PROC, "engine_scores.csv")
    for p in [feat_path, engine_path]:
        if not os.path.exists(p): raise FileNotFoundError(f"Run earlier steps: {p}")
    os.makedirs(RESULTS_FIG, exist_ok=True); os.makedirs(RESULTS_TAB, exist_ok=True)

    feat        = pd.read_csv(feat_path,   index_col=0, parse_dates=True)
    eng_raw     = pd.read_csv(engine_path, index_col=0, parse_dates=True)
    recession   = feat["RECESSION"]
    engine_cols = [c for c in eng_raw.columns if c not in ("ESI","RECESSION")]
    engine_scores = eng_raw[engine_cols]
    esi           = eng_raw["ESI"] if "ESI" in eng_raw.columns else engine_scores.mean(axis=1)

    print(f"\n{'='*64}")
    print("  Stress Index Summary Charts")
    print(f"{'='*64}\n")

    # Save ESI table
    esi_df = pd.DataFrame({"ESI": esi, "RECESSION": recession})
    esi_df["stress_level"] = pd.cut(esi_df["ESI"], bins=[0,30,60,80,100],
                                     labels=["Low","Moderate","High","Extreme"], right=True)
    for eng in engine_cols:
        esi_df[eng] = engine_scores[eng]
    esi_df.to_csv(os.path.join(RESULTS_TAB, "04_stress_index.csv"))

    print(f"  ESI at key events:")
    events = {"1973-11":"Oil Embargo","1980-03":"Volcker Shock","1982-12":"GFC-era",
              "2001-10":"Dot-com","2008-10":"GFC peak","2009-06":"GFC trough",
              "2020-04":"COVID","2022-06":"Inflation peak","2022-09":"CPI >8%","2024-01":"Post-tightening"}
    for d, ev in events.items():
        try:
            i = esi.index.get_indexer([pd.Timestamp(d)],method="nearest")[0]
            eng_str = "  ".join(f"{engine_scores[e].iloc[i]:>5.0f}" for e in engine_cols)
            print(f"  {d}  ESI={esi.iloc[i]:>5.1f}  [{eng_str}]  {ev}")
        except: pass

    plot_main_esi(engine_scores, esi, recession, RESULTS_FIG)
    plot_stress_heatmap(engine_scores, esi, RESULTS_FIG)
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
