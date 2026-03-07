"""
src/analysis/eda.py
--------------------
Comprehensive Exploratory Data Analysis for the Economic Stress Index project.

Outputs
-------
results/figures/01_raw_time_series.png
results/figures/02_correlation_matrix.png
results/figures/03_recession_distributions.png
results/figures/04_lag_correlations.png
results/figures/05_phillips_curve.png
results/tables/01_summary_statistics.csv
results/tables/02_recession_vs_normal.csv

Usage:
    python src/analysis/eda.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_PROC, RESULTS_FIG, RESULTS_TAB, STRESS_FEATURES

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"     : 150,
    "font.family"    : "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid"      : True,
    "grid.alpha"     : 0.3,
    "axes.labelsize" : 11,
    "axes.titlesize" : 12,
})
RECESSION_COLOR = "#d62728"
SHADE_COLOR     = "#fee0d2"


def shade_recessions(ax, recession: pd.Series):
    """Add grey shaded bands for recession periods."""
    in_rec = False
    start  = None
    for date, val in recession.items():
        if val == 1 and not in_rec:
            start  = date
            in_rec = True
        elif val == 0 and in_rec:
            ax.axvspan(start, date, color=SHADE_COLOR, alpha=0.6, zorder=0)
            in_rec = False
    if in_rec:
        ax.axvspan(start, recession.index[-1], color=SHADE_COLOR, alpha=0.6, zorder=0)


# ── Plot 1: Raw time series ───────────────────────────────────────────────────
def plot_raw_series(df: pd.DataFrame, save_dir: str):
    features = [f for f in STRESS_FEATURES if f in df.columns]
    n        = len(features)
    ncols    = 2
    nrows    = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes      = axes.flatten()
    recession = df["RECESSION"]

    labels = {
        "inflation_yoy"      : "CPI Inflation YoY (%)",
        "core_inflation_yoy" : "Core CPI Inflation YoY (%)",
        "infl_exp"           : "5Y Breakeven Inflation (%)",
        "unemployment_change": "Unemployment Rate Δ12m (pp)",
        "real_rate"          : "Real Interest Rate (%)",
        "yield_spread"       : "Yield Spread Negated (10Y−2Y × −1)",
        "credit_spread"      : "BAA Credit Spread (%)",
        "fsi"                : "KC Financial Stress Index",
        "indpro_growth"      : "Industrial Production Decline (−YoY %)",
        "m2_growth"          : "M2 Money Supply YoY (%)",
        "payroll_growth"     : "Payroll Employment Decline (−YoY %)",
        "jobless_claims_ma"  : "Initial Claims (log, detrended)",
    }

    for i, feat in enumerate(features):
        ax = axes[i]
        s  = df[feat].dropna()
        ax.plot(s.index, s.values, lw=1.2, color="#2166ac")
        shade_recessions(ax, recession.reindex(s.index).fillna(0))
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.set_title(labels.get(feat, feat))
        ax.set_xlabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    rec_patch = mpatches.Patch(color=SHADE_COLOR, label="NBER Recession")
    fig.legend(handles=[rec_patch], loc="lower right", fontsize=10)
    fig.suptitle("Macroeconomic Stress Indicators (1971 – Present)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, "01_raw_time_series.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: Correlation matrix ────────────────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame, save_dir: str):
    features = [f for f in STRESS_FEATURES if f in df.columns]
    corr     = df[features].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr,
        mask      = mask,
        annot     = True,
        fmt       = ".2f",
        cmap      = "RdYlGn_r",
        center    = 0,
        vmin      = -1,
        vmax      = 1,
        linewidths= 0.5,
        ax        = ax,
        annot_kws = {"size": 8},
    )
    ax.set_title("Pearson Correlation Matrix – Stress Features",
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    path = os.path.join(save_dir, "02_correlation_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: Distribution during recession vs normal ──────────────────────────
def plot_recession_distributions(df: pd.DataFrame, save_dir: str):
    features = [
        "inflation_yoy", "credit_spread", "yield_spread",
        "fsi", "unemployment_change", "indpro_growth"
    ]
    features = [f for f in features if f in df.columns]
    recession = df["RECESSION"]

    ncols = 3
    nrows = (len(features) + 2) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    nice = {
        "inflation_yoy"      : "CPI Inflation YoY (%)",
        "credit_spread"      : "BAA Credit Spread",
        "yield_spread"       : "Yield Spread (negated)",
        "fsi"                : "KC FSI",
        "unemployment_change": "Unemployment Δ12m",
        "indpro_growth"      : "Indus. Prod. Decline",
    }

    for i, feat in enumerate(features):
        ax     = axes[i]
        normal = df.loc[recession == 0, feat].dropna()
        rec    = df.loc[recession == 1, feat].dropna()

        ax.hist(normal, bins=40, alpha=0.6, color="#2166ac", density=True,
                label=f"Normal  (n={len(normal)})")
        ax.hist(rec,    bins=40, alpha=0.6, color=RECESSION_COLOR, density=True,
                label=f"Recession (n={len(rec)})")

        # KDE overlay
        for series, color in [(normal, "#2166ac"), (rec, RECESSION_COLOR)]:
            if len(series) > 5:
                try:
                    kde = stats.gaussian_kde(series)
                    xs  = np.linspace(series.min(), series.max(), 200)
                    ax.plot(xs, kde(xs), color=color, lw=2)
                except Exception:
                    pass

        # T-test
        t_stat, p_val = stats.ttest_ind(normal, rec, equal_var=False)
        ax.set_title(f"{nice.get(feat, feat)}\np-value = {p_val:.4f}", fontsize=10)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Feature Distributions: Recession vs Normal Periods",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "03_recession_distributions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4: Lead-lag correlation with recessions ──────────────────────────────
def plot_lag_correlations(df: pd.DataFrame, save_dir: str):
    features = [f for f in STRESS_FEATURES if f in df.columns]
    recession = df["RECESSION"].reindex(df.index).fillna(0)
    lags      = range(-12, 25)   # −12 (indicator lags recession) to +24 (leads)

    results = {}
    for feat in features:
        s    = df[feat].dropna()
        idx  = s.index.intersection(recession.index)
        s, r = s[idx], recession[idx]
        cors = []
        for lag in lags:
            shifted = s.shift(lag)   # positive lag: indicator leads recession
            aligned = pd.concat([shifted, r], axis=1).dropna()
            if len(aligned) > 20:
                c = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            else:
                c = np.nan
            cors.append(c)
        results[feat] = cors

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab10
    for i, (feat, cors) in enumerate(results.items()):
        ax.plot(list(lags), cors, label=feat, lw=1.5, color=cmap(i / len(results)))

    ax.axvline(0, color="black", lw=1.2, ls="--", label="Contemporaneous")
    ax.axhline(0, color="grey",  lw=0.7)
    ax.set_xlabel("Lead (+) / Lag (−) in months\n"
                  "(positive = indicator leads recession)")
    ax.set_ylabel("Pearson Correlation with RECESSION")
    ax.set_title("Lead-Lag Correlations of Stress Indicators with Recessions",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    path = os.path.join(save_dir, "04_lag_correlations.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 5: Phillips curve ────────────────────────────────────────────────────
def plot_phillips_curve(df: pd.DataFrame, save_dir: str):
    """Scatter: unemployment change vs inflation with decade colouring."""
    if "inflation_yoy" not in df.columns or "unemployment_change" not in df.columns:
        return

    d = df[["inflation_yoy", "unemployment_change", "RECESSION"]].dropna()
    d["decade"] = (d.index.year // 10) * 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: coloured by decade
    ax = axes[0]
    decades    = sorted(d["decade"].unique())
    cmap_dec   = plt.cm.plasma
    for k, dec in enumerate(decades):
        sub = d[d["decade"] == dec]
        ax.scatter(sub["unemployment_change"], sub["inflation_yoy"],
                   s=15, alpha=0.7, label=str(dec),
                   color=cmap_dec(k / max(len(decades) - 1, 1)))
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.axvline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel("Unemployment Rate Δ12m (pp)")
    ax.set_ylabel("CPI Inflation YoY (%)")
    ax.set_title("Phillips Curve (coloured by decade)")
    ax.legend(title="Decade", fontsize=7, ncol=2)

    # Right: coloured by recession
    ax = axes[1]
    colors = d["RECESSION"].map({0: "#2166ac", 1: RECESSION_COLOR})
    ax.scatter(d["unemployment_change"], d["inflation_yoy"],
               c=colors, s=15, alpha=0.7)
    # Regression line
    slope, intercept, r, p, _ = stats.linregress(
        d["unemployment_change"], d["inflation_yoy"]
    )
    xs = np.linspace(d["unemployment_change"].min(), d["unemployment_change"].max(), 100)
    ax.plot(xs, slope * xs + intercept, "k--", lw=1.5,
            label=f"OLS  r={r:.2f}  p={p:.4f}")
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.axvline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel("Unemployment Rate Δ12m (pp)")
    ax.set_ylabel("CPI Inflation YoY (%)")
    ax.set_title("Phillips Curve (recession highlighted)")
    normal_p = mpatches.Patch(color="#2166ac",   label="Normal")
    rec_p    = mpatches.Patch(color=RECESSION_COLOR, label="Recession")
    ax.legend(handles=[normal_p, rec_p, ax.get_lines()[0]], fontsize=8)

    fig.suptitle("Phillips Curve: Inflation vs Unemployment",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "05_phillips_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Table 1: Summary statistics ───────────────────────────────────────────────
def compute_summary_statistics(df: pd.DataFrame, save_dir: str):
    features = [f for f in STRESS_FEATURES if f in df.columns]
    stats_df  = df[features].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    stats_df["skewness"] = df[features].skew()
    stats_df["kurtosis"] = df[features].kurt()
    stats_df = stats_df.round(3)
    path = os.path.join(save_dir, "01_summary_statistics.csv")
    stats_df.to_csv(path)
    print(f"  Saved: {path}")
    return stats_df


# ── Table 2: Recession vs Normal means ───────────────────────────────────────
def compute_recession_comparison(df: pd.DataFrame, save_dir: str):
    features  = [f for f in STRESS_FEATURES if f in df.columns]
    recession = df["RECESSION"]

    rows = []
    for feat in features:
        normal = df.loc[recession == 0, feat].dropna()
        rec    = df.loc[recession == 1, feat].dropna()
        t, p   = stats.ttest_ind(normal, rec, equal_var=False)
        rows.append({
            "feature"        : feat,
            "mean_normal"    : normal.mean(),
            "mean_recession" : rec.mean(),
            "difference"     : rec.mean() - normal.mean(),
            "t_statistic"    : t,
            "p_value"        : p,
            "significant"    : "Yes" if p < 0.05 else "No",
        })

    comp = pd.DataFrame(rows).set_index("feature").round(3)
    path = os.path.join(save_dir, "02_recession_vs_normal.csv")
    comp.to_csv(path)
    print(f"  Saved: {path}")
    return comp


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    feat_path = os.path.join(DATA_PROC, "fred_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Features not found at {feat_path}.\n"
            "Run  python src/features/engineer_features.py  first."
        )

    os.makedirs(RESULTS_FIG, exist_ok=True)
    os.makedirs(RESULTS_TAB, exist_ok=True)

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    print(f"\n{'='*60}")
    print("  Running Exploratory Data Analysis")
    print(f"{'='*60}\n")

    plot_raw_series(df, RESULTS_FIG)
    plot_correlation_matrix(df, RESULTS_FIG)
    plot_recession_distributions(df, RESULTS_FIG)
    plot_lag_correlations(df, RESULTS_FIG)
    plot_phillips_curve(df, RESULTS_FIG)
    stats_df = compute_summary_statistics(df, RESULTS_TAB)
    comp_df  = compute_recession_comparison(df, RESULTS_TAB)

    print(f"\n{'='*60}")
    print("  EDA complete. Key findings:")
    print(f"\n  Recession vs Normal — top 5 features by |difference|:")
    top5 = comp_df.reindex(comp_df["difference"].abs().sort_values(ascending=False).index)
    print(top5[["mean_normal","mean_recession","difference","p_value","significant"]].head(5).to_string())
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


# Alias for pipeline compatibility
def run_eda(df, results_fig, results_tab):
    """
    Run full EDA on a pre-loaded features DataFrame.
    Called by run_pipeline.py step4_eda.
    """
    import os
    os.makedirs(results_fig, exist_ok=True)
    os.makedirs(results_tab, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Running Exploratory Data Analysis")
    print(f"{'='*60}\n")

    plot_raw_series(df, results_fig)
    plot_correlation_matrix(df, results_fig)
    plot_recession_distributions(df, results_fig)
    plot_lag_correlations(df, results_fig)
    plot_phillips_curve(df, results_fig)
    stats_df = compute_summary_statistics(df, results_tab)
    comp_df  = compute_recession_comparison(df, results_tab)

    print(f"\n{'='*60}")
    print("  EDA complete. Key findings:")
    print(f"\n  Recession vs Normal — top 5 features by |difference|:")
    top5 = comp_df.reindex(comp_df["difference"].abs().sort_values(ascending=False).index)
    print(top5[["mean_normal","mean_recession","difference","p_value","significant"]].head(5).to_string())
    print(f"{'='*60}\n")
