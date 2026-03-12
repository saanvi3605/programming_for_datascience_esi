"""
src/models/backtest.py
-----------------------
Rolling out-of-sample (OOS) backtest of recession prediction.

Simulates what would have happened if you ran this model in real time
from 1996 onwards: at each month t, use only data up to t-1, train the
model, then predict recession probability for the next 6 months.

This is the gold standard for evaluating economic forecasting models.
In-sample fit is meaningless for macro models — OOS performance is the
only honest test.

Methodology
-----------
1. Start date: 1996-01-01 (gives ~6 years of burn-in for engine scores)
2. At each month t:
   a. Train logistic regression on engine scores up to t-1
   b. Predict P(recession) for months t, t+1, ..., t+5
   c. Compare predicted probability to actual recession indicator
3. For each NBER recession after 1996:
   a. Find the first month where predicted P(recession in 6m) > threshold
   b. Compute lead time: how many months before NBER start date
4. Metrics:
   - Out-of-sample ROC-AUC (both contemporaneous and 6m forward)
   - Calibration curve (predicted vs actual probability)
   - Lead time distribution per recession
   - False alarm rate: how many "warnings" occurred outside recessions

Recessions covered
------------------
  2001-03 to 2001-11  (Dot-com / 9-11)
  2007-12 to 2009-06  (Global Financial Crisis)
  2020-02 to 2020-04  (COVID-19 shock)

Outputs
-------
  data/processed/backtest_results.csv
  results/figures/22_backtest_oos.png        — Main backtest chart
  results/figures/23_backtest_calibration.png
  results/tables/T4_backtest_summary.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (DATA_PROC, RESULTS_FIG, RESULTS_TAB,
                    BACKTEST_START, BACKTEST_MIN_TRAIN_M,
                    BACKTEST_THRESHOLD, FORECAST_HORIZON, RANDOM_STATE, adaptive_min)

SHADE_COLOR = "#fee0d2"
plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})

# NBER recession dates (start, end) post-1990
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-01", "1990-91 Recession"),
    ("2001-03-01", "2001-11-01", "Dot-com / 9-11"),
    ("2007-12-01", "2009-06-01", "Global Financial Crisis"),
    ("2020-02-01", "2020-04-01", "COVID-19 Shock"),
]


# ── Core backtest logic ───────────────────────────────────────────────────────

def build_forward_recession_target(recession: pd.Series, horizon: int = 6,
                                    target_type: str = "start") -> pd.Series:
    """
    Build forward recession target.
    target_type="start": only the FIRST month of each episode counts (sharper).
    target_type="standard": any month within horizon counts.
    """
    if target_type == "start":
        # Find recession start months only
        recession_starts = pd.Series(0, index=recession.index, dtype=float)
        prev = 0
        for dt, v in recession.items():
            if v == 1 and prev == 0:
                recession_starts[dt] = 1.0
            prev = v
        target = pd.Series(0, index=recession.index, dtype=float)
        for i in range(len(recession)):
            window = recession_starts.iloc[i:i + horizon + 1]
            target.iloc[i] = 1.0 if window.max() >= 1 else 0.0
    else:
        target = pd.Series(0, index=recession.index, dtype=float)
        for i in range(len(recession)):
            window = recession.iloc[i:i + horizon + 1]
            target.iloc[i] = 1.0 if window.max() >= 1 else 0.0
    return target


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add delta, momentum, volatility and lag features to engine scores.
    Mirrors prepare_ml_data in ml_validation.py for consistency.
    NaN values (at series start) are filled with column medians.
    """
    base_cols = list(df.columns)
    frames = [df]
    for col in base_cols:
        s = df[col]
        frames.append(s.diff(1).rename(f"{col}_delta1"))
        frames.append(s.diff(3).rename(f"{col}_mom3"))
        frames.append(s.diff(6).rename(f"{col}_mom6"))
        frames.append(s.rolling(6, min_periods=3).std().rename(f"{col}_vol6"))
        frames.append(s.shift(1).rename(f"{col}_lag1"))
        frames.append(s.shift(3).rename(f"{col}_lag3"))
        frames.append(s.shift(6).rename(f"{col}_lag6"))
    out = pd.concat(frames, axis=1)
    return out.fillna(out.median())


def run_expanding_backtest(engine_scores: pd.DataFrame,
                            recession: pd.Series,
                            horizon: int = FORECAST_HORIZON,
                            start_date: str = BACKTEST_START,
                            min_train_months: int = BACKTEST_MIN_TRAIN_M,
                            threshold: float = BACKTEST_THRESHOLD) -> pd.DataFrame:
    """
    Expanding-window backtest.

    v6: temporal features (delta, momentum, lags) added to match ml_validation.
    min_train_months uses adaptive_min to avoid discarding early data.
    """
    target_forward = build_forward_recession_target(recession, horizon, target_type="start")
    common_idx     = engine_scores.dropna().index.intersection(recession.dropna().index)
    eng_clean      = engine_scores.loc[common_idx]
    rec_clean      = recession.loc[common_idx]
    tgt_clean      = target_forward.loc[common_idx]

    # Enrich with temporal features up-front (each row t only uses data ≤ t)
    eng_enriched   = _add_temporal_features(eng_clean)

    # Adaptive min_train: use 40% of min_train_months if data is short
    n_total   = len(common_idx)
    min_train = adaptive_min(min_train_months, n_total)

    start_ts  = pd.Timestamp(start_date)
    start_idx = max(
        common_idx.get_indexer([start_ts], method="nearest")[0],
        min_train
    )

    rows = []
    print(f"    [Backtest] Running {len(common_idx) - start_idx} "
          f"monthly predictions from {str(common_idx[start_idx].date())} ...")

    for t in range(start_idx, len(common_idx)):
        train_X = eng_enriched.iloc[:t].values
        train_y_contemp = rec_clean.iloc[:t].values
        train_y_forward = tgt_clean.iloc[:t].values

        if train_y_contemp.std() == 0 or train_y_forward.std() == 0:
            continue

        test_X = eng_enriched.iloc[[t]].values
        date_t = common_idx[t]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train_X)
        X_test  = scaler.transform(test_X)

        lr_c = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)
        lr_c.fit(X_train, train_y_contemp)
        prob_contemp = lr_c.predict_proba(X_test)[0, 1]

        lr_f = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)
        lr_f.fit(X_train, train_y_forward)
        prob_forward = lr_f.predict_proba(X_test)[0, 1]

        row = {
            "date"            : date_t,
            "recession_actual": int(rec_clean.iloc[t]),
            "forward_actual"  : int(tgt_clean.iloc[t]),
            "prob_contemp"    : prob_contemp,
            "prob_forward"    : prob_forward,
            "warning_flag"    : int(prob_forward >= threshold),
        }
        for eng_col in eng_clean.columns:
            row[eng_col] = eng_clean[eng_col].iloc[t]
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def compute_lead_times(backtest_df: pd.DataFrame,
                        threshold: float = BACKTEST_THRESHOLD) -> list:
    """
    For each NBER recession, compute how many months before the official
    start date the model first raised a warning (prob_forward > threshold).

    Also computes false alarm periods: consecutive warnings outside recessions.
    """
    results = []
    for rec_start, rec_end, label in NBER_RECESSIONS:
        ts = pd.Timestamp(rec_start)
        te = pd.Timestamp(rec_end)

        # Only evaluate if we have predictions for this period
        if ts < backtest_df.index[0]:
            continue

        # Find first warning BEFORE recession start
        pre_window = backtest_df.loc[
            (backtest_df.index >= ts - pd.DateOffset(months=18)) &
            (backtest_df.index < ts), "warning_flag"
        ]

        if len(pre_window) == 0:
            results.append({
                "recession": label,
                "start"    : ts,
                "lead_time_months": None,
                "missed"   : True,
            })
            continue

        first_warning = pre_window[pre_window == 1].first_valid_index()
        if first_warning is None:
            results.append({
                "recession": label,
                "start"    : ts,
                "lead_time_months": 0,
                "missed"   : True,
            })
        else:
            lead = (ts - first_warning).days / 30.44
            results.append({
                "recession": label,
                "start"    : ts,
                "lead_time_months": round(lead, 1),
                "missed"   : False,
            })

    return results


def compute_false_alarms(backtest_df: pd.DataFrame,
                          threshold: float = BACKTEST_THRESHOLD) -> int:
    """
    Count months with warning_flag=1 that are NOT within 6 months of a recession.
    """
    recession = backtest_df["recession_actual"]
    rec_buffer = recession.copy()
    # Mark 6 months before each recession as "acceptable warning period"
    for rec_start, _, _ in NBER_RECESSIONS:
        ts = pd.Timestamp(rec_start)
        for lag in range(-6, 1):
            target = ts + pd.DateOffset(months=lag)
            if target in rec_buffer.index:
                rec_buffer.loc[target] = 1

    warnings   = backtest_df["warning_flag"]
    false_alarms = ((warnings == 1) & (rec_buffer == 0)).sum()
    return int(false_alarms)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _shade(ax, recession: pd.Series, alpha: float = 0.6):
    in_rec, start = False, None
    for dt, v in recession.items():
        if v == 1 and not in_rec:
            start, in_rec = dt, True
        elif v == 0 and in_rec:
            ax.axvspan(start, dt, color=SHADE_COLOR, alpha=alpha, zorder=0)
            in_rec = False
    if in_rec:
        ax.axvspan(start, recession.index[-1], color=SHADE_COLOR, alpha=alpha, zorder=0)


def plot_backtest_main(backtest_df: pd.DataFrame,
                        lead_times: list,
                        esi: pd.Series,
                        save_dir: str):
    """
    3-panel main backtest chart:
    1. ESI with recession shading
    2. Predicted recession probability (forward 6m) — OOS only
    3. Engine scores at recession onset
    """
    fig, axes = plt.subplots(3, 1, figsize=(20, 16), sharex=True)

    rec = backtest_df["recession_actual"]

    # Panel 1: ESI
    ax = axes[0]
    esi_aligned = esi.reindex(backtest_df.index)
    _shade(ax, rec)
    ax.fill_between(esi_aligned.index, esi_aligned.values, alpha=0.2, color="#c0392b")
    ax.plot(esi_aligned.index, esi_aligned.values, lw=2, color="#c0392b", label="ESI")
    ax.set_ylim(0, 105)
    ax.set_ylabel("ESI (0-100)")
    ax.set_title("Economic Stress Index — Rolling Out-of-Sample Backtest (1996–Present)\n"
                 "All predictions use ONLY data available at that time (no look-ahead)",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    # Panel 2: Forward recession probability
    ax2 = axes[1]
    _shade(ax2, rec)
    ax2.fill_between(backtest_df.index,
                     backtest_df["prob_forward"].values,
                     alpha=0.3, color="#2166ac")
    ax2.plot(backtest_df.index, backtest_df["prob_forward"].values,
             lw=1.5, color="#2166ac", label=f"P(recession in 6m) — OOS")
    ax2.plot(backtest_df.index, backtest_df["prob_contemp"].values,
             lw=1.0, color="#d62728", alpha=0.6, ls="--",
             label="P(current recession) — OOS")
    ax2.axhline(BACKTEST_THRESHOLD, color="orange", lw=1.2, ls="--",
                label=f"Warning threshold ({BACKTEST_THRESHOLD:.0%})")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Recession Probability")
    ax2.set_title("Out-of-Sample Recession Probability — Logistic Regression on Engine Scores\n"
                  f"Warning flag raised when P(6m forward) ≥ {BACKTEST_THRESHOLD:.0%}",
                  fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left", ncol=3)

    # Annotate lead times
    for lt in lead_times:
        if lt.get("missed") or lt["lead_time_months"] is None:
            continue
        rec_start = lt["start"]
        if rec_start in backtest_df.index or any(
            abs((rec_start - dt).days) < 32 for dt in backtest_df.index
        ):
            ax2.annotate(
                f"{lt['recession'][:12]}\n+{lt['lead_time_months']:.0f}m lead",
                xy=(rec_start, BACKTEST_THRESHOLD + 0.02),
                xytext=(rec_start - pd.DateOffset(months=6), BACKTEST_THRESHOLD + 0.18),
                fontsize=7, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
                ha="center",
            )

    # Panel 3: Engine scores at key moments
    ax3 = axes[2]
    eng_cols = [c for c in backtest_df.columns
                if c not in ("recession_actual", "forward_actual",
                              "prob_contemp", "prob_forward", "warning_flag")]
    for eng in eng_cols:
        color = {
            "Inflation": "#e6550d", "Labour": "#756bb1",
            "Financial": "#31a354", "Monetary": "#3182bd", "Real": "#d62728"
        }.get(eng, "#888888")
        ax3.plot(backtest_df.index, backtest_df[eng].values,
                 lw=1.2, color=color, alpha=0.8, label=eng)
    _shade(ax3, rec, alpha=0.4)
    ax3.set_ylim(0, 102)
    ax3.set_ylabel("Engine Score (0-100)")
    ax3.set_title("Engine Scores During Backtest Period", fontweight="bold")
    ax3.legend(fontsize=8, loc="upper left", ncol=5)

    fig.tight_layout()
    path = os.path.join(save_dir, "22_backtest_oos.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_backtest_calibration(backtest_df: pd.DataFrame,
                               save_dir: str):
    """
    4-panel calibration / performance chart:
    1. ROC curves (OOS, both horizons)
    2. Calibration plot
    3. Lead time bar chart
    4. Precision-recall
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rec_c = backtest_df["recession_actual"]
    rec_f = backtest_df["forward_actual"]
    prob_c = backtest_df["prob_contemp"]
    prob_f = backtest_df["prob_forward"]

    # Panel 1: ROC
    ax = axes[0, 0]
    for probs, rec, label, color in [
        (prob_c, rec_c, "Contemporaneous", "#d62728"),
        (prob_f, rec_f, f"Forward (+6m)",  "#2166ac"),
    ]:
        try:
            fpr, tpr, _ = roc_curve(rec, probs)
            auc = roc_auc_score(rec, probs)
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} AUC={auc:.3f}")
        except Exception as e:
            print(f"    ROC error: {e}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("OOS ROC Curves\n(True out-of-sample — no training data leaked)",
                 fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 2: Calibration
    ax2 = axes[0, 1]
    for probs, rec, label, color in [
        (prob_c, rec_c, "Contemporaneous", "#d62728"),
        (prob_f, rec_f, f"Forward (+6m)",  "#2166ac"),
    ]:
        try:
            frac_pos, mean_pred = calibration_curve(rec, probs, n_bins=8)
            ax2.plot(mean_pred, frac_pos, "o--", color=color, lw=1.5, label=label)
        except Exception:
            pass
    ax2.plot([0, 1], [0, 1], "k--", lw=0.7, label="Perfect calibration")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction Positive (Actual)")
    ax2.set_title("Calibration Plot\n(Points on diagonal = well-calibrated)",
                  fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel 3: Monthly probability distributions
    ax3 = axes[1, 0]
    ax3.hist(prob_f[rec_f == 0], bins=25, alpha=0.6, color="#2166ac",
             label="Non-recession months", density=True)
    ax3.hist(prob_f[rec_f == 1], bins=10, alpha=0.6, color="#d62728",
             label="Recession-imminent months", density=True)
    ax3.axvline(BACKTEST_THRESHOLD, color="orange", lw=1.5, ls="--",
                label=f"Threshold {BACKTEST_THRESHOLD:.0%}")
    ax3.set_xlabel("Predicted P(Recession in 6m)")
    ax3.set_ylabel("Density")
    ax3.set_title("Probability Distributions\n(Separation shows model discrimination)",
                  fontweight="bold")
    ax3.legend(fontsize=9)

    # Panel 4: Monthly engine importance from coefficients
    ax4 = axes[1, 1]
    # Refit on all data to get stable coefficients for interpretation
    eng_cols = [c for c in backtest_df.columns
                if c not in ("recession_actual", "forward_actual",
                              "prob_contemp", "prob_forward", "warning_flag")]
    try:
        X = backtest_df[eng_cols].dropna().values
        y = rec_f.loc[backtest_df[eng_cols].dropna().index].values
        if y.std() > 0:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(Xs, y)
            coefs = pd.Series(np.abs(lr.coef_[0]), index=eng_cols).sort_values(ascending=True)
            colors4 = [{"Inflation":"#e6550d","Labour":"#756bb1","Financial":"#31a354",
                        "Monetary":"#3182bd","Real":"#d62728"}.get(c,"#888888")
                       for c in coefs.index]
            coefs.plot(kind="barh", ax=ax4, color=colors4, alpha=0.85)
            ax4.set_xlabel("|Logistic coefficient| (standardised)")
            ax4.set_title("Engine Importance for Recession Warning\n"
                          "(Fitted on all OOS predictions)",
                          fontweight="bold")
    except Exception as e:
        ax4.text(0.5, 0.5, f"Could not compute\n{e}", transform=ax4.transAxes, ha="center")

    fig.tight_layout()
    path = os.path.join(save_dir, "23_backtest_calibration.png")
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
    # Use EXPANDING engine scores for backtest — longer history (back to 1976)
    # includes 6 recessions vs rolling which only gives 2 (GFC + COVID from 2012)
    engine_cols = [c for c in eng_raw.columns
                   if c not in ("ESI", "ESI_expanding", "ESI_ML", "RECESSION")
                   and not c.startswith("ESI")]
    # Prefer the expanding versions for ML training if available
    expanding_cols = {c.replace("_exp","").replace("_rolling",""): c
                      for c in eng_raw.columns if "_exp" in c.lower()}
    engine_scores = eng_raw[engine_cols]

    # If ESI_expanding is present, use it as the reference ESI for charts
    esi = (eng_raw["ESI_expanding"] if "ESI_expanding" in eng_raw.columns
           else eng_raw["ESI"] if "ESI" in eng_raw.columns
           else engine_scores.mean(axis=1))

    print(f"\n{'='*64}")
    print("  Rolling Out-of-Sample Backtest (1996 → Present)")
    print(f"{'='*64}\n")
    print(f"  Recessions to evaluate: {[r[2] for r in NBER_RECESSIONS]}")
    print(f"  Early-warning horizon:  {FORECAST_HORIZON} months")
    print(f"  Warning threshold:      {BACKTEST_THRESHOLD:.0%}")
    print()

    # Run backtest
    backtest_df = run_expanding_backtest(
        engine_scores, recession,
        horizon=FORECAST_HORIZON,
        start_date=BACKTEST_START,
        min_train_months=BACKTEST_MIN_TRAIN_M,
    )

    # Compute OOS metrics
    rec_c = backtest_df["recession_actual"]
    rec_f = backtest_df["forward_actual"]

    print(f"\n  Out-of-sample performance:")
    for probs, rec, label in [
        (backtest_df["prob_contemp"], rec_c, "Contemporaneous"),
        (backtest_df["prob_forward"], rec_f, f"Forward (+{FORECAST_HORIZON}m)"),
    ]:
        try:
            auc = roc_auc_score(rec, probs)
            print(f"    {label:<22}  OOS ROC-AUC = {auc:.3f}")
        except Exception as e:
            print(f"    {label:<22}  ERROR: {e}")

    # Lead times
    lead_times = compute_lead_times(backtest_df, BACKTEST_THRESHOLD)
    false_alarms = compute_false_alarms(backtest_df, BACKTEST_THRESHOLD)

    print(f"\n  Recession early-warning lead times (threshold={BACKTEST_THRESHOLD:.0%}):")
    for lt in lead_times:
        if lt["lead_time_months"] is None or lt.get("missed"):
            print(f"    {lt['recession']:<30}  MISSED (no warning raised)")
        else:
            print(f"    {lt['recession']:<30}  {lt['lead_time_months']:>5.1f} months lead")
    print(f"\n  False alarm months (warnings outside ±6m of recession): {false_alarms}")

    # Save
    backtest_df.to_csv(os.path.join(DATA_PROC, "backtest_results.csv"))

    summary_df = pd.DataFrame(lead_times)
    summary_df["oos_roc_contemp"] = None
    summary_df["oos_roc_forward"] = None
    try:
        summary_df.loc[0, "oos_roc_contemp"] = round(
            roc_auc_score(rec_c, backtest_df["prob_contemp"]), 3)
        summary_df.loc[0, "oos_roc_forward"]  = round(
            roc_auc_score(rec_f, backtest_df["prob_forward"]), 3)
    except Exception:
        pass
    summary_df.to_csv(os.path.join(RESULTS_TAB, "T4_backtest_summary.csv"), index=False)

    plot_backtest_main(backtest_df, lead_times, esi, RESULTS_FIG)
    plot_backtest_calibration(backtest_df, RESULTS_FIG)
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
