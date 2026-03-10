"""
src/analysis/regime_detection.py
----------------------------------
Detects stress regimes using Gaussian Mixture Models on engine scores.

What is a stress regime?
-------------------------
A regime is a distinct macroeconomic "state" characterised by a specific
pattern of engine scores. Regimes are not just "stressed vs calm" —
they reveal WHAT TYPE of stress is occurring:

  Normal Growth    : all engines low/moderate
  Inflation Shock  : Inflation engine dominant, others moderate
  Financial Crisis : Financial + Real engines dominant
  Labour Recession : Labour engine dominant, Financial elevated
  Monetary Squeeze : Monetary engine dominant (tightening cycle)
  Stagflation      : Inflation + Labour both elevated simultaneously

Methodology
-----------
1. Gaussian Mixture Model (GMM) on the 5 engine percentile scores
   - GMM is preferred over k-means because regimes have elliptical shapes
     in 5-dimensional engine space (not spherical)
   - Each regime is a multivariate Gaussian with its own mean and covariance
2. Auto-label each regime by identifying the dominant engine in its centroid
3. Compute regime transition probabilities (Markov-style)
4. Use smoothed posterior probabilities (soft assignment) for charts

Why GMM over HMM?
-----------------
Hidden Markov Models add temporal constraints (regime transitions must be
Markovian). GMM is simpler and more robust when regimes don't necessarily
follow strict AR-1 transitions. For a first pass, GMM is the standard
used by Ang & Bekaert (2002) and later Guidolin & Timmermann (2007).

Outputs
-------
  data/processed/regimes.csv          — regime assignment + probabilities
  results/figures/20_regimes.png      — regime timeline
  results/figures/21_regime_scatter.png
  results/tables/T3_regime_summary.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_PROC, RESULTS_FIG, RESULTS_TAB, N_REGIMES, ENGINE_COLORS

SHADE_COLOR = "#fee0d2"
REGIME_NARRATIVES = {
    "Normal Growth"       : "Output expanding, inflation contained, labour healthy. Classic mid-cycle.",
    "Inflation Shock"     : "Demand-pull or supply-push inflation dominates. Labour still healthy → not yet recession.",
    "Financial Crisis"    : "Credit spreads blow out, funding markets freeze. Historically precedes deep recessions.",
    "Labour Recession"    : "Jobs disappearing, claims rising. Labour market in contraction. Classic NBER recession signal.",
    "Monetary Squeeze"    : "Central bank tightening: real rates high, yield curve flat/inverted. Growth slowing.",
    "Stagflation"         : "Inflation + labour stress simultaneously. Worst policy environment — no easy levers.",
    "Multi-Engine Crisis" : "Three or more engines elevated simultaneously. Systemic macro shock.",
    "Mild Stress"         : "One or two engines slightly elevated. Below recession threshold.",
    "Real Contraction"    : "Industrial output declining, retail falling. Demand-side weakness.",
}


def get_regime_narrative(label: str) -> str:
    """Return the economic narrative for a given regime label."""
    for key, narrative in REGIME_NARRATIVES.items():
        if key.lower() in label.lower():
            return narrative
    return "Mixed signals across engines. Monitor for regime clarification."


REGIME_PALETTE = [
    "#2ecc71",  # green   — Normal Growth
    "#f39c12",  # orange  — Mild Stress
    "#e74c3c",  # red     — Financial Crisis
    "#8e44ad",  # purple  — Labour Recession
    "#3498db",  # blue    — Monetary Squeeze
    "#c0392b",  # dark red — Stagflation
]

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})


# ── GMM fitting ───────────────────────────────────────────────────────────────

def fit_gmm(engine_scores: pd.DataFrame,
            n_regimes: int = N_REGIMES,
            n_init: int = 20,
            random_state: int = 42) -> tuple:
    """
    Fit a Gaussian Mixture Model to engine scores.

    Uses n_init random initialisations to avoid local optima (GMM is
    sensitive to initialisation in high dimensions).

    Returns
    -------
    gmm        : fitted GaussianMixture object
    labels     : array of hard cluster assignments
    probs      : (n_obs, n_regimes) soft probabilities
    clean_data : DataFrame used for fitting
    """
    # Use data where all engines are available
    clean = engine_scores.dropna()
    X     = clean.values.astype(float)

    # Standardise (GMM operates in score space; percentile scores 0-100
    # are already comparable, but standardising helps numerical stability)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM with BIC model selection check
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        n_init=n_init,
        random_state=random_state,
        max_iter=500,
    )
    gmm.fit(X_scaled)

    labels = gmm.predict(X_scaled)
    probs  = gmm.predict_proba(X_scaled)

    print(f"    [GMM] Converged: {gmm.converged_}  |  "
          f"Log-likelihood: {gmm.lower_bound_:.3f}")

    return gmm, labels, probs, clean, scaler


def auto_label_regimes(gmm, engine_names: list,
                        scaler, n_regimes: int) -> dict:
    """
    Auto-label each GMM component by its centroid characteristics.

    Labels are assigned by finding the engine with the highest mean score
    in each regime centroid. Special rules:
      - If ALL engines < 35: "Normal Growth"
      - If two engines both > 60: "Stagflation" (if Inflation+Labour) or combined
      - Otherwise: "{dominant engine} Stress/Crisis"
    """
    # Un-scale the centroids
    centroids = scaler.inverse_transform(gmm.means_)  # (n_regimes, n_engines)
    centroid_df = pd.DataFrame(centroids, columns=engine_names)

    regime_labels = {}
    crisis_words = {
        "Inflation": "Inflation Shock",
        "Labour"   : "Labour Recession",
        "Financial": "Financial Crisis",
        "Monetary" : "Monetary Squeeze",
        "Real"     : "Real Contraction",
    }

    for i in range(n_regimes):
        row = centroid_df.iloc[i]
        max_val = row.max()
        max_eng = row.idxmax()

        # Count engines above 60 (High stress threshold)
        elevated = (row > 55).sum()

        if max_val < 35:
            label = "Normal Growth"
        elif elevated >= 3:
            label = "Multi-Engine Crisis"
        elif elevated == 2:
            dominant_two = row.nlargest(2).index.tolist()
            if "Inflation" in dominant_two and "Labour" in dominant_two:
                label = "Stagflation"
            elif "Financial" in dominant_two and "Real" in dominant_two:
                label = "Financial Crisis"
            elif "Monetary" in dominant_two:
                label = f"Monetary-{dominant_two[0] if dominant_two[0]!='Monetary' else dominant_two[1]} Stress"
            else:
                label = f"{dominant_two[0]}-{dominant_two[1]} Stress"
        else:
            label = crisis_words.get(max_eng, f"{max_eng} Stress")

        regime_labels[i] = label

    # Deduplicate labels
    seen = {}
    for i, lbl in regime_labels.items():
        if lbl in seen:
            seen[lbl] += 1
            regime_labels[i] = f"{lbl} ({seen[lbl]})"
        else:
            seen[lbl] = 1

    return regime_labels, centroid_df


def build_regime_df(labels: np.ndarray,
                     probs: np.ndarray,
                     clean_data: pd.DataFrame,
                     regime_labels: dict) -> pd.DataFrame:
    """
    Build a DataFrame with regime assignments and probabilities.
    """
    out = pd.DataFrame(index=clean_data.index)
    out["regime_id"]   = labels
    out["regime_name"] = [regime_labels[l] for l in labels]
    for i, lbl in regime_labels.items():
        col_name = f"prob_{lbl[:10].replace(' ', '_')}"
        out[col_name] = probs[:, i]
    return out


def compute_transition_matrix(labels: np.ndarray,
                                regime_labels: dict) -> pd.DataFrame:
    """
    Compute empirical Markov transition matrix between regimes.
    T[i, j] = P(go to regime j | currently in regime i)
    """
    n = len(regime_labels)
    T = np.zeros((n, n))
    for t in range(len(labels) - 1):
        T[labels[t], labels[t + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    T = np.divide(T, row_sums, where=row_sums > 0)

    names = [regime_labels[i] for i in range(n)]
    return pd.DataFrame(T, index=names, columns=names)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _shade(ax, recession: pd.Series):
    in_rec, start = False, None
    for dt, v in recession.items():
        if v == 1 and not in_rec:
            start, in_rec = dt, True
        elif v == 0 and in_rec:
            ax.axvspan(start, dt, color=SHADE_COLOR, alpha=0.55, zorder=0)
            in_rec = False
    if in_rec:
        ax.axvspan(start, recession.index[-1], color=SHADE_COLOR, alpha=0.55, zorder=0)


def plot_regime_timeline(regime_df: pd.DataFrame,
                          engine_scores: pd.DataFrame,
                          esi: pd.Series,
                          recession: pd.Series,
                          regime_labels: dict,
                          save_dir: str):
    """
    Timeline chart showing:
    1. ESI with regime colour-coded background
    2. Engine scores with regime overlay
    """
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    rec_aligned = recession.reindex(regime_df.index).fillna(0)

    unique_regimes = sorted(regime_df["regime_id"].unique())
    regime_colors  = {r: REGIME_PALETTE[i % len(REGIME_PALETTE)]
                      for i, r in enumerate(unique_regimes)}

    # Panel 1: ESI with regime background
    ax = axes[0]
    esi_aligned = esi.reindex(regime_df.index)

    # Draw regime background bands
    prev_regime, band_start = None, None
    for dt, row in regime_df.iterrows():
        rid = row["regime_id"]
        if rid != prev_regime:
            if prev_regime is not None and band_start is not None:
                ax.axvspan(band_start, dt, color=regime_colors[prev_regime],
                           alpha=0.12, zorder=0)
            band_start   = dt
            prev_regime  = rid
    if prev_regime is not None and band_start is not None:
        ax.axvspan(band_start, regime_df.index[-1],
                   color=regime_colors[prev_regime], alpha=0.12, zorder=0)

    _shade(ax, rec_aligned)
    ax.fill_between(esi_aligned.index, esi_aligned.values, alpha=0.25, color="#c0392b")
    ax.plot(esi_aligned.index, esi_aligned.values, lw=2.0, color="#c0392b", label="ESI")

    for rid in unique_regimes:
        lbl = regime_labels[rid]
        ax.fill_between([], [], color=regime_colors[rid], alpha=0.5, label=lbl)

    ax.set_ylim(0, 105)
    ax.set_ylabel("ESI (0-100)")
    ax.set_title("Economic Stress Index with Detected Stress Regimes\n"
                 "(Background colour = current regime; red shading = NBER recession)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.8)

    # Panel 2: Regime probability stacked area
    ax2 = axes[1]
    _shade(ax2, rec_aligned)

    prob_cols = [c for c in regime_df.columns if c.startswith("prob_")]
    bottom    = pd.Series(0.0, index=regime_df.index)
    for i, (prob_col, rid) in enumerate(zip(prob_cols, unique_regimes)):
        lbl   = regime_labels.get(rid, f"Regime {rid}")
        color = regime_colors[rid]
        p     = regime_df[prob_col].fillna(0)
        ax2.fill_between(regime_df.index, bottom, bottom + p,
                         color=color, alpha=0.75, label=lbl[:20])
        bottom += p

    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Regime Probability")
    ax2.set_title("Regime Posterior Probabilities (GMM soft assignment)\n"
                  "(Height of each band = probability of being in that regime)",
                  fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.8)

    fig.tight_layout()
    path = os.path.join(save_dir, "20_regimes.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_regime_profiles(centroid_df: pd.DataFrame,
                          regime_labels: dict,
                          transition_matrix: pd.DataFrame,
                          save_dir: str):
    """
    2-panel chart:
    1. Regime centroid radar (engine profile per regime)
    2. Transition matrix heatmap
    """
    n_regimes = len(regime_labels)
    engines   = centroid_df.columns.tolist()
    n_eng     = len(engines)
    angles    = np.linspace(0, 2 * np.pi, n_eng, endpoint=False).tolist()
    angles   += angles[:1]

    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(2, n_regimes, hspace=0.5, wspace=0.4)

    # Row 1: Radar charts
    for i in range(n_regimes):
        ax = fig.add_subplot(gs[0, i], polar=True)
        row    = centroid_df.iloc[i]
        values = row.tolist() + [row.iloc[0]]
        color  = REGIME_PALETTE[i % len(REGIME_PALETTE)]
        ax.plot(angles, values, lw=2, color=color)
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([e[:4] for e in engines], fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75])
        ax.set_yticklabels(["", "", ""], fontsize=0)
        lbl = regime_labels.get(i, f"Regime {i}")
        ax.set_title(lbl[:20], fontsize=7, fontweight="bold", pad=10, color=color)

    # Row 2: Transition matrix
    ax_trans = fig.add_subplot(gs[1, :])
    names_short = [regime_labels.get(i, f"R{i}")[:12] for i in range(n_regimes)]
    T_plot = transition_matrix.copy()
    T_plot.index   = names_short
    T_plot.columns = names_short
    sns.heatmap(T_plot, ax=ax_trans, cmap="Blues", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.3, cbar_kws={"shrink": 0.6, "label": "Transition P"})
    ax_trans.set_title("Regime Transition Matrix  (rows = from, cols = to)\n"
                       "Diagonal = persistence probability",
                       fontweight="bold", fontsize=10)
    ax_trans.tick_params(axis="x", rotation=30)
    ax_trans.tick_params(axis="y", rotation=0)

    fig.suptitle("Stress Regime Profiles — Engine Fingerprints and Transitions\n"
                 "(Gaussian Mixture Model on 5 engine percentile scores)",
                 fontsize=12, fontweight="bold")
    path = os.path.join(save_dir, "21_regime_profiles.png")
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

    feat        = pd.read_csv(feat_path,   index_col=0, parse_dates=True)
    eng_raw     = pd.read_csv(engine_path, index_col=0, parse_dates=True)
    recession   = feat["RECESSION"]
    # CRITICAL: GMM must receive only the 5 pure economic engine scores.
    # Feeding ESI_expanding and ESI_ML (derived from engines) causes the
    # clustering to find patterns in the index values, not the economic mechanisms.
    # The 5 pure engines are: Inflation, Labour, Financial, Monetary, Real
    PURE_ENGINES = ["Inflation", "Labour", "Financial", "Monetary", "Real"]
    engine_cols   = [c for c in PURE_ENGINES if c in eng_raw.columns]
    if not engine_cols:
        # Fallback: any column that isn't a derived metric
        engine_cols = [c for c in eng_raw.columns
                       if c not in ("ESI","ESI_expanding","ESI_ML","RECESSION")
                       and not c.startswith("ESI")]
    engine_scores = eng_raw[engine_cols]
    esi           = eng_raw["ESI"] if "ESI" in eng_raw.columns else engine_scores.mean(axis=1)

    print(f"\n{'='*64}")
    print("  Stress Regime Detection (Gaussian Mixture Model)")
    print(f"{'='*64}\n")
    print(f"  Input: {len(engine_scores.dropna())} months x {len(engine_cols)} engines")

    # Fit GMM
    gmm, labels, probs, clean_data, scaler = fit_gmm(
        engine_scores, n_regimes=N_REGIMES
    )
    regime_labels, centroid_df = auto_label_regimes(
        gmm, engine_cols, scaler, N_REGIMES
    )

    # Build output DataFrame
    regime_df      = build_regime_df(labels, probs, clean_data, regime_labels)
    transition_mat = compute_transition_matrix(labels, regime_labels)

    # Print regime summary
    print(f"\n  Detected regimes (centroids):")
    print(f"  {'Regime':<25}", end="")
    for eng in engine_cols:
        print(f"  {eng[:6]:>6}", end="")
    print(f"  {'N months':>10}  {'%Recession':>10}")
    print(f"  {'-'*80}")

    rec_aligned = recession.reindex(clean_data.index).fillna(0)
    for i in sorted(regime_labels):
        lbl   = regime_labels[i]
        mask  = (regime_df["regime_id"] == i)
        n_m   = mask.sum()
        rec_r = rec_aligned.loc[mask.index[mask]].mean() * 100
        row   = centroid_df.iloc[i]
        print(f"  {lbl[:25]:<25}", end="")
        for eng in engine_cols:
            print(f"  {row.get(eng, np.nan):>6.0f}", end="")
        print(f"  {n_m:>10}  {rec_r:>9.1f}%")

    # Current regime
    latest = regime_df.iloc[-1]
    latest_label = latest['regime_name']
    latest_narrative = get_regime_narrative(latest_label)
    print(f"\n  Current regime (latest month):  "
          f"{latest_label}  (ID={latest['regime_id']})")
    print(f"  Economic narrative:  {latest_narrative}")

    # Save
    regime_df.to_csv(os.path.join(DATA_PROC, "regimes.csv"))
    transition_mat.to_csv(os.path.join(RESULTS_TAB, "T3_regime_transitions.csv"))

    # Regime summary table
    summary_rows = []
    for i in sorted(regime_labels):
        lbl  = regime_labels[i]
        mask = (regime_df["regime_id"] == i)
        n_m  = mask.sum()
        rec_r = rec_aligned.loc[mask.index[mask]].mean() * 100
        row   = centroid_df.iloc[i]
        r     = {"Regime": lbl, "N_months": n_m, "Recession_%": round(rec_r, 1)}
        for eng in engine_cols:
            r[eng] = round(row.get(eng, np.nan), 1)
        summary_rows.append(r)
    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_TAB, "T3_regime_summary.csv"),
                                       index=False)

    plot_regime_timeline(regime_df, engine_scores, esi, recession, regime_labels, RESULTS_FIG)
    plot_regime_profiles(centroid_df, regime_labels, transition_mat, RESULTS_FIG)
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
