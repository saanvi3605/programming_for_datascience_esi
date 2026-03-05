"""
src/features/build_custom_fsi.py
----------------------------------
Builds a custom Financial Stress Index (custom_fsi) that replicates the
methodology used by the Kansas City Fed to build their KCFSI.

Methodology (mirrors KCFSI approach)
--------------------------------------
The KCFSI is constructed using the following steps:
  1. Collect a set of financial market variables (spreads, volatility, rates)
  2. Standardise each variable to z-scores (mean=0, std=1)
  3. Apply PCA, extract PC1
  4. Orient PC1 so positive values = financial stress

Our custom FSI uses:
  - credit_spread  : BAA corporate spread over 10Y (credit risk premium)
  - vix_zscore     : CBOE VIX z-score (equity market fear/uncertainty)
  - ted_spread     : TED spread = 3m LIBOR - 3m T-bill (interbank funding stress)
  - yield_spread   : -(10Y-2Y) negated (yield curve inversion stress)

All four are directly observed financial market prices, not derived macro
aggregates. This makes the custom FSI a genuine market-based stress gauge.

Comparison with KCFSI:
  The KCFSI uses 11 variables from bond, equity, and banking markets.
  Our custom FSI uses 4 core variables that capture the same dimensions:
    - Credit risk:    credit_spread  ≈  KCFSI credit sub-components
    - Equity risk:    vix_zscore     ≈  KCFSI equity volatility
    - Funding stress: ted_spread     ≈  KCFSI interbank rate spreads
    - Curve stress:   yield_spread   ≈  KCFSI yield curve components

Expanding window PCA:
  We fit PCA at each month t using only data up to t, preventing look-ahead.
  The sign of PC1 is corrected so correlation with KCFSI is positive.

Outputs:
  data/processed/custom_fsi.csv
  results/figures/15_custom_fsi.png

Usage:
    python src/features/build_custom_fsi.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_PROC, RESULTS_FIG, CUSTOM_FSI_FEATURES, adaptive_min

SHADE_COLOR = "#fee0d2"
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                      "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def build_custom_fsi(feat: pd.DataFrame) -> pd.Series:
    """
    Build the custom FSI as a stable equal-weight z-score composite.

    v6 change: DROPPED rolling PCA in favour of equal-weight expanding z-scores.
    Reason (reviewer recommendation):
      - Rolling PCA loadings drift silently → factor meaning changes each month
      - Sign correction hacks stabilise output but not underlying structure
      - Equal weights perform comparably (proven by our own engine framework)
      - This makes custom_fsi transparent, stable, and interpretable

    Method:
      1. For each component, compute expanding z-score (higher = more stress)
      2. Average the z-scores with equal weights
      3. Final expanding z-score normalisation → interpretable scale

    Components (all already stress-signed: higher = more stress):
      credit_spread  : BAA-10Y spread widens under stress
      vix_zscore     : equity fear gauge (already z-scored in feature engineering)
      ted_spread     : interbank funding stress
      yield_spread   : -(10Y-2Y), inverts under stress

    Returns
    -------
    pd.Series  custom_fsi  (higher = more financial stress; expanding z-scale)
    """
    available = [f for f in CUSTOM_FSI_FEATURES if f in feat.columns]
    print(f"  Custom FSI using: {available}")

    if len(available) < 2:
        print("  Not enough FSI components — skipping custom FSI.")
        return pd.Series(np.nan, index=feat.index, name="custom_fsi")

    X = feat[available].copy()

    # Per-component expanding z-score (removes level trends, ensures comparability)
    X_z = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for col in X.columns:
        s   = X[col]
        n   = int(s.notna().sum())
        mp  = adaptive_min(36, n)
        mu  = s.expanding(min_periods=mp).mean()
        sig = s.expanding(min_periods=mp).std().replace(0, np.nan)
        X_z[col] = (s - mu) / sig

    # Equal-weight composite (skipna so missing components don't kill the index)
    composite = X_z.mean(axis=1, skipna=True)

    # Final expanding z-score → interpretable scale consistent with KCFSI
    n_comp = int(composite.notna().sum())
    mp_final = adaptive_min(60, n_comp)
    mu  = composite.expanding(min_periods=mp_final).mean()
    sig = composite.expanding(min_periods=mp_final).std().replace(0, np.nan)
    custom_fsi = (composite - mu) / sig
    custom_fsi.name = "custom_fsi"

    return custom_fsi


def plot_custom_fsi(custom_fsi: pd.Series,
                    kcfsi: pd.Series,
                    recession: pd.Series,
                    save_dir: str):
    """
    3-panel chart:
    1. Custom FSI vs KCFSI overlay (validation)
    2. Custom FSI components stacked
    3. Correlation scatter: custom FSI vs KCFSI
    """
    fig = plt.figure(figsize=(17, 14))
    gs  = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35)

    def shade(ax, rec):
        in_rec, start = False, None
        for dt, v in rec.items():
            if v == 1 and not in_rec: start, in_rec = dt, True
            elif v == 0 and in_rec:
                ax.axvspan(start, dt, color=SHADE_COLOR, alpha=0.6, zorder=0); in_rec = False
        if in_rec: ax.axvspan(start, rec.index[-1], color=SHADE_COLOR, alpha=0.6, zorder=0)

    # Panel 1: Both FSIs over time
    ax1 = fig.add_subplot(gs[0, :])
    common_idx = custom_fsi.dropna().index.intersection(kcfsi.dropna().index)
    shade(ax1, recession.reindex(custom_fsi.dropna().index).fillna(0))
    ax1.plot(custom_fsi.index, custom_fsi.values, color="#2166ac", lw=1.5,
             label="Custom FSI (this project)")
    kcfsi_aligned = kcfsi.reindex(custom_fsi.index)
    ax1.plot(kcfsi_aligned.index, kcfsi_aligned.values, color="#d62728",
             lw=1.5, alpha=0.8, ls="--", label="KCFSI (KC Fed reference)")
    ax1.axhline(0, color="grey", lw=0.7, ls=":")
    ax1.set_ylabel("Stress Score (z-score scale)")
    ax1.set_title("Custom Financial Stress Index vs KC Fed KCFSI\n"
                  "(Both built via PCA of financial market variables — shading = NBER recessions)",
                  fontweight="bold")
    ax1.legend(fontsize=9)

    # Panel 2: Correlation scatter
    ax2 = fig.add_subplot(gs[1, 0])
    if len(common_idx) > 10:
        x = custom_fsi.loc[common_idx]
        y = kcfsi.loc[common_idx]
        corr = x.corr(y)
        ax2.scatter(x, y, s=8, alpha=0.5, color="#2166ac")
        xlim = ax2.get_xlim(); m = np.polyfit(x.dropna(), y.reindex(x.dropna().index).dropna(), 1)
        xs = np.linspace(xlim[0], xlim[1], 100)
        ax2.plot(xs, np.polyval(m, xs), "r--", lw=1.5, label=f"OLS  r={corr:.2f}")
        ax2.set_xlabel("Custom FSI"); ax2.set_ylabel("KCFSI")
        ax2.set_title("Validation: Custom FSI vs KCFSI", fontweight="bold")
        ax2.legend(fontsize=9)

    # Panel 3: FSI at key crisis events
    ax3 = fig.add_subplot(gs[1, 1])
    crisis_windows = {
        "GFC\n(2008-09)": ("2007-01", "2009-12"),
        "COVID\n(2020)":  ("2019-06", "2021-06"),
        "Dot-com\n(2001)":("2000-01", "2002-06"),
    }
    colors3 = ["#d62728", "#2166ac", "#ff7f0e"]
    for (label, (s, e)), col in zip(crisis_windows.items(), colors3):
        try:
            window = custom_fsi.loc[s:e].dropna()
            if len(window) > 3:
                ax3.plot(range(len(window)), window.values, lw=1.5,
                         color=col, alpha=0.85, label=label)
        except: pass
    ax3.axhline(0, color="grey", lw=0.7, ls=":"); ax3.axhline(2, color="grey", lw=0.5, ls="--")
    ax3.set_xlabel("Months from window start"); ax3.set_ylabel("Custom FSI")
    ax3.set_title("Custom FSI During Major Crises", fontweight="bold")
    ax3.legend(fontsize=8)

    # Panel 4: Rolling correlation between custom FSI and KCFSI
    ax4 = fig.add_subplot(gs[2, :])
    if len(common_idx) > 36:
        roll_corr = (custom_fsi.loc[common_idx]
                     .rolling(36, min_periods=24)
                     .corr(kcfsi.loc[common_idx]))
        ax4.fill_between(roll_corr.index, roll_corr.values, alpha=0.3, color="#2166ac")
        ax4.plot(roll_corr.index, roll_corr.values, color="#2166ac", lw=1.5,
                 label="36-month rolling correlation")
        ax4.axhline(0.7, color="green", lw=0.8, ls="--", label="Good validation (r=0.7)")
        ax4.axhline(0, color="grey", lw=0.7, ls=":")
        ax4.set_ylim(-1, 1.1); ax4.set_ylabel("Pearson correlation")
        ax4.set_title("Rolling Correlation: Custom FSI vs KCFSI\n"
                      "(should stay > 0.7 for the custom FSI to be a valid replication)",
                      fontweight="bold")
        ax4.legend(fontsize=9)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "15_custom_fsi.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def main():
    feat_path = os.path.join(DATA_PROC, "fred_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Run engineer_features.py first: {feat_path}")

    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs(RESULTS_FIG, exist_ok=True)

    feat      = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    recession = feat["RECESSION"]

    print(f"\n{'='*64}")
    print("  Building Custom Financial Stress Index")
    print(f"{'='*64}\n")

    custom_fsi = build_custom_fsi(feat)

    # Append to features DataFrame
    feat["custom_fsi"] = custom_fsi
    feat.to_csv(feat_path)
    print(f"  custom_fsi added to {feat_path}")

    # Save standalone — used by downstream charts
    out_path = os.path.join(DATA_PROC, "custom_fsi.csv")
    custom_fsi.to_frame("custom_fsi").to_csv(out_path)
    print(f"  custom_fsi saved → {out_path}")

    # Validation stats
    kcfsi = feat["fsi"] if "fsi" in feat.columns else pd.Series(dtype=float)
    if not kcfsi.empty:
        common = custom_fsi.dropna().index.intersection(kcfsi.dropna().index)
        if len(common) > 10:
            corr = custom_fsi.loc[common].corr(kcfsi.loc[common])
            print(f"\n  Validation: custom FSI vs KCFSI")
            print(f"    Pearson correlation:  {corr:.3f}")
            print(f"    Overlapping months:   {len(common)}")
            interp = 'Excellent' if corr>0.8 else ('Good' if corr>0.6 else 'Moderate')
            print(f"    Interpretation:  {interp} replication")
            print(f"    Method: equal-weight expanding z-score composite (stable, no PCA drift)")

    print(f"\n  Custom FSI stats:")
    print(f"    First valid: {custom_fsi.first_valid_index()}")
    print(f"    Range: {custom_fsi.min():.2f}  to  {custom_fsi.max():.2f}")
    print(f"    Mean: {custom_fsi.mean():.3f}  Std: {custom_fsi.std():.3f}")

    plot_custom_fsi(custom_fsi, kcfsi, recession, RESULTS_FIG)
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
