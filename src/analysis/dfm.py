"""
src/analysis/dfm.py
--------------------
Dynamic Factor Model (DFM) for extracting latent macro stress factors.

The DFM is used by the Federal Reserve, IMF, and ECB to track the co-movement
of multiple economic time series through a small number of latent factors.

Model specification
--------------------
Observation equation:
    X_t = Lambda * F_t + epsilon_t          (k variables, r factors)

State equation (factor dynamics):
    F_t = A * F_{t-1} + u_t                 (AR-1 dynamics)

Where:
    X_t       = (k x 1) vector of observed engine z-scores at time t
    F_t       = (r x 1) vector of latent factors at time t
    Lambda    = (k x r) loading matrix (estimated)
    A         = (r x r) transition matrix (estimated)
    epsilon_t ~ N(0, R) idiosyncratic noise
    u_t       ~ N(0, Q) factor innovations

Estimation: Maximum Likelihood via Kalman Filter (statsmodels DynamicFactor)

Why 2 factors?
--------------
Using r=2 follows the convention used by the Chicago Fed NFCI and academic
macro stress literature (Brave & Butters 2012, Ng & Schorfheide 2011).
  Factor 1: Typically loads on real/financial variables (recession factor)
  Factor 2: Typically loads on inflation/monetary variables (policy factor)
Two factors explain ~70-80% of variance in typical US macro panels.

Inputs
------
  5 engine z-scores (continuous, approximately normal):
  Inflation, Labour, Financial, Monetary, Real
  Period: from DFM_START (1996) when all engines have sufficient history

Outputs
-------
  data/processed/dfm_factors.csv  — smoothed factor estimates
  results/figures/19_dfm.png
  results/tables/T2_dfm_loadings.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_PROC, RESULTS_FIG, RESULTS_TAB, DFM_START, DFM_N_FACTORS, ENGINE_COLORS, adaptive_min

SHADE_COLOR = "#fee0d2"
plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})


def _shade(ax, recession: pd.Series):
    in_rec, start = False, None
    for dt, v in recession.items():
        if v == 1 and not in_rec:
            start, in_rec = dt, True
        elif v == 0 and in_rec:
            ax.axvspan(start, dt, color=SHADE_COLOR, alpha=0.6, zorder=0)
            in_rec = False
    if in_rec:
        ax.axvspan(start, recession.index[-1], color=SHADE_COLOR, alpha=0.6, zorder=0)


def prepare_dfm_data(engine_zscores: pd.DataFrame,
                     start: str = DFM_START) -> pd.DataFrame:
    """
    Prepare engine z-scores for DFM estimation.

    v6 changes:
    - Missing threshold raised from 5% → 60% (allows Inflation engine to enter)
    - Forward-fill up to 3 months BEFORE dropping (rescues infl_exp gaps)
    - Adaptive min_periods used for any downstream stats
    - Remaining NaN rows filled with column median before DFM fit

    The Inflation engine had 39% missing because infl_exp (5Y breakeven)
    only starts in 2003. After ffill(3), the remaining gaps are post-2003
    which DFM handles cleanly.
    """
    data = engine_zscores.loc[start:].copy()

    # Step 1: forward-fill short gaps (handles infl_exp sparsity pre-2003 region)
    data = data.ffill(limit=3).bfill(limit=1)

    # Step 2: drop engines with > 60% missing (genuinely unavailable)
    keep = []
    for col in data.columns:
        miss_rate = data[col].isnull().mean()
        if miss_rate < 0.60:
            keep.append(col)
        else:
            print(f"    [DFM] Dropping engine {col}: {miss_rate*100:.1f}% missing (>60% threshold)")
    data = data[keep]

    # Step 3: fill any remaining NaNs with column median (not row-drop)
    # This prevents DFM from losing months when one engine has an isolated gap
    for col in data.columns:
        if data[col].isnull().any():
            median_val = data[col].median()
            n_filled = data[col].isnull().sum()
            data[col] = data[col].fillna(median_val)
            print(f"    [DFM] {col}: filled {n_filled} residual NaN with median ({median_val:.3f})")

    data = data.dropna(how="all")  # only drop rows where ALL engines are NaN
    return data


def fit_dfm(data: pd.DataFrame, n_factors: int = 2) -> dict:
    """
    Fit DFM using statsmodels DynamicFactor.

    Returns dict with:
      - model     : fitted DynamicFactor model
      - results   : model results object
      - factors   : smoothed factor estimates (DataFrame)
      - loadings  : factor loading matrix (DataFrame)
      - var_expl  : variance explained per factor
    """
    try:
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    except ImportError:
        raise ImportError("Install statsmodels>=0.14: pip install statsmodels")

    n_obs, n_vars = data.shape
    print(f"    [DFM] Fitting on {n_obs} months x {n_vars} engines, r={n_factors} factors")

    model   = DynamicFactor(data, k_factors=n_factors, factor_order=1)
    results = model.fit(disp=False, maxiter=1000, method="lbfgs")

    # Smoothed state = factor estimates
    smoothed = results.smoothed_state  # shape (n_factors, n_obs)
    factors  = pd.DataFrame(
        smoothed[:n_factors].T,
        index=data.index,
        columns=[f"DFM_F{i+1}" for i in range(n_factors)]
    )

    # Loading matrix: which engines load on which factor
    # params is a Series — convert to numpy array first
    param_vals = results.params.values
    loadings = pd.DataFrame(
        param_vals[:n_vars * n_factors].reshape(n_vars, n_factors),
        index=data.columns,
        columns=[f"DFM_F{i+1}" for i in range(n_factors)]
    )

    # Variance explained: correlation of each factor with each engine
    var_expl = {}
    for fc in factors.columns:
        corrs   = {eng: factors[fc].corr(data[eng]) for eng in data.columns}
        var_expl[fc] = corrs

    return {
        "model"    : model,
        "results"  : results,
        "factors"  : factors,
        "loadings" : loadings,
        "var_expl" : var_expl,
        "data"     : data,
    }


def orient_dfm_factors(dfm_output: dict, recession: pd.Series) -> dict:
    """
    Orient each DFM factor so positive values correlate with recessions.
    This gives factors an intuitive interpretation: higher = more stress.
    """
    factors  = dfm_output["factors"].copy()
    loadings = dfm_output["loadings"].copy()

    rec_aligned = recession.reindex(factors.index).fillna(0)
    for col in factors.columns:
        corr = factors[col].corr(rec_aligned)
        if corr < 0:
            factors[col]  = -factors[col]
            loadings[col] = -loadings[col]
            print(f"    [DFM] {col} flipped (was negatively correlated with recessions)")

    dfm_output["factors"]  = factors
    dfm_output["loadings"] = loadings
    return dfm_output


def label_dfm_factors(loadings: pd.DataFrame) -> dict:
    """
    Auto-label each DFM factor by the engine with highest absolute loading.
    """
    labels = {}
    for col in loadings.columns:
        dominant = loadings[col].abs().idxmax()
        labels[col] = f"{col} ({dominant})"
    return labels


def plot_dfm(dfm_output: dict,
             esi: pd.Series,
             recession: pd.Series,
             save_dir: str):
    """
    4-panel DFM chart:
    1. Factor 1 vs ESI
    2. Factor 2 vs ESI
    3. Loading heatmap (which engines → which factors)
    4. Factor 1 vs 2 scatter coloured by recession
    """
    factors  = dfm_output["factors"]
    loadings = dfm_output["loadings"]
    labels   = label_dfm_factors(loadings)
    data     = dfm_output["data"]

    rec_aligned = recession.reindex(factors.index).fillna(0)
    esi_aligned = esi.reindex(factors.index)

    # Standardise factors to 0-100 via percentile for comparison with ESI
    factor_pct = {}
    for col in factors.columns:
        s = factors[col]
        r = s.rank(pct=True) * 100
        factor_pct[col] = r
    factor_pct_df = pd.DataFrame(factor_pct)

    n_factors = len(factors.columns)
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    factor_colors = ["#2166ac", "#d62728", "#31a354", "#ff7f0e"]

    # Panels 1 & 2: Factors vs ESI
    for i, fc in enumerate(factors.columns):
        ax = fig.add_subplot(gs[i, :] if i < 2 else gs[2, 0])
        _shade(ax, rec_aligned)
        ax.plot(factor_pct_df.index, factor_pct_df[fc].values,
                color=factor_colors[i % len(factor_colors)], lw=1.8,
                label=labels[fc], alpha=0.9)
        ax.plot(esi_aligned.index, esi_aligned.values,
                color="black", lw=1.0, alpha=0.4, ls="--", label="ESI (composite)")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percentile (0-100)")
        dominant_eng = loadings[fc].abs().idxmax()
        ax.set_title(f"DFM {fc} — Latent Macro Factor\n"
                     f"Dominant engine loading: {dominant_eng}  |  "
                     f"Shading = NBER recessions",
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        if i >= 2:
            break

    # Panel 3: Loading heatmap
    ax3 = fig.add_subplot(gs[2, 0])
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax3, annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8, "label": "Loading"})
    ax3.set_title("DFM Factor Loadings\n(+ = engine raises factor when stressed)",
                  fontweight="bold", fontsize=9)

    # Panel 4: Factor scatter
    ax4 = fig.add_subplot(gs[2, 1])
    if len(factors.columns) >= 2:
        f1 = factor_pct_df.iloc[:, 0]
        f2 = factor_pct_df.iloc[:, 1]
        rec_colors = rec_aligned.map({0: "#2166ac", 1: "#d62728"}).values
        ax4.scatter(f1, f2, c=rec_colors, s=12, alpha=0.6)
        ax4.set_xlabel(f"{labels[factors.columns[0]]} (pctile)")
        ax4.set_ylabel(f"{labels[factors.columns[1]]} (pctile)")
        ax4.set_title("Factor Space — Recession vs Normal\n(Red = recession months)",
                      fontweight="bold", fontsize=9)
        rec_patch   = mpatches.Patch(color="#d62728", label="Recession")
        norm_patch  = mpatches.Patch(color="#2166ac", label="Normal")
        ax4.legend(handles=[rec_patch, norm_patch], fontsize=8)

    fig.suptitle("Dynamic Factor Model (DFM) — Latent Macro Stress Factors\n"
                 "Model: X_t = Λ·F_t + ε_t  |  F_t = A·F_{t-1} + u_t  (AR-1 dynamics)\n"
                 "Estimated via Kalman Filter (Federal Reserve / IMF methodology)",
                 fontsize=11, fontweight="bold")

    path = os.path.join(save_dir, "19_dfm.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    feat_path   = os.path.join(DATA_PROC, "fred_features.csv")
    engine_path = os.path.join(DATA_PROC, "engine_scores.csv")

    for p in [feat_path, engine_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Run earlier steps first: {p}")

    os.makedirs(DATA_PROC,   exist_ok=True)
    os.makedirs(RESULTS_FIG, exist_ok=True)
    os.makedirs(RESULTS_TAB, exist_ok=True)

    feat      = pd.read_csv(feat_path,   index_col=0, parse_dates=True)
    eng_raw   = pd.read_csv(engine_path, index_col=0, parse_dates=True)
    recession = feat["RECESSION"]

    # Get engine z-scores (not percentile ranks)
    from src.analysis.engines import build_engine_zscores, ENGINE_FEATURES
    engine_z = build_engine_zscores(feat, ENGINE_FEATURES)

    print(f"\n{'='*64}")
    print("  Dynamic Factor Model (Federal Reserve / IMF methodology)")
    print(f"{'='*64}\n")

    # ESI for comparison
    esi_cols  = [c for c in eng_raw.columns if c not in ("ESI", "RECESSION")]
    esi       = eng_raw["ESI"] if "ESI" in eng_raw.columns else eng_raw[esi_cols].mean(axis=1)

    data = prepare_dfm_data(engine_z, start=DFM_START)
    print(f"  DFM input: {data.shape[0]} months x {data.shape[1]} engines")
    print(f"  Period: {data.index[0].date()} to {data.index[-1].date()}")

    dfm_output = fit_dfm(data, n_factors=DFM_N_FACTORS)
    dfm_output = orient_dfm_factors(dfm_output, recession)

    # Save
    dfm_output["factors"].to_csv(os.path.join(DATA_PROC, "dfm_factors.csv"))
    dfm_output["loadings"].to_csv(os.path.join(RESULTS_TAB, "T2_dfm_loadings.csv"))

    # Print factor-engine correlations
    print(f"\n  Factor-engine correlations (post-orientation):")
    print(f"  {'Engine':<14}", end="")
    for fc in dfm_output["factors"].columns:
        print(f"  {fc:>8}", end="")
    print()
    for eng in data.columns:
        print(f"  {eng:<14}", end="")
        for fc in dfm_output["factors"].columns:
            corr = dfm_output["factors"][fc].corr(data[eng])
            print(f"  {corr:>8.3f}", end="")
        print()

    labels = label_dfm_factors(dfm_output["loadings"])
    print(f"\n  Factor interpretations:")
    for fc, lbl in labels.items():
        print(f"    {lbl}")

    plot_dfm(dfm_output, esi, recession, RESULTS_FIG)
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
