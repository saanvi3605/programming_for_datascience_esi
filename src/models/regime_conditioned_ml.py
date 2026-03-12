"""
src/models/regime_conditioned_ml.py
-------------------------------------
Regime-Conditioned Recession Prediction.

- Uses logistic regression per regime (lightweight, no lead time)
- Keeps Brier score and calibration curve
- Saves current_prediction.csv so the dashboard can read it without
  re-running the full pipeline
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_PROC, RESULTS_FIG, RESULTS_TAB, RANDOM_STATE, FORECAST_HORIZON, ENGINE_COLORS

SHADE_COLOR = "#fee0d2"
PURE_ENGINES = ["Inflation", "Labour", "Financial", "Monetary", "Real"]

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                      "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def classify_episode_regime(engine_row: pd.Series, threshold: float = 55.0) -> str:
    engines = {e: engine_row.get(e, 50.0) for e in PURE_ENGINES if e in engine_row.index}
    if not engines:
        return "Unknown"
    dominant = max(engines, key=engines.get)
    dom_val = engines[dominant]
    n_elevated = sum(1 for v in engines.values() if v > threshold)
    if n_elevated >= 4:
        return "Systemic Crisis"
    elif n_elevated >= 3:
        if engines.get("Financial", 0) > 70 and engines.get("Real", 0) > 70:
            return "Financial Crisis"
        if engines.get("Labour", 0) > 70 and engines.get("Real", 0) > 70:
            return "Demand Shock"
        return "Balanced Recession"
    elif dominant == "Financial" or (engines.get("Financial", 0) > 65 and engines.get("Real", 0) > 65):
        return "Financial Crisis"
    elif dominant in ("Labour", "Real") or (engines.get("Labour", 0) > 65 or engines.get("Real", 0) > 65):
        return "Demand Shock"
    elif dominant == "Inflation" and dom_val > 65:
        return "Inflation Shock"
    elif dominant == "Monetary" and dom_val > 65:
        return "Monetary Squeeze"
    else:
        return "Mild / Mixed"


def label_episodes_with_regime(panel_engines_path: str, engine_scores_us_path: str) -> pd.DataFrame:
    panel_df = pd.read_csv(panel_engines_path, index_col=0, parse_dates=True)
    engine_cols = [c for c in PURE_ENGINES if c in panel_df.columns]
    episodes = []
    from src.data.download_oecd import RECESSION_DATES
    for country, rec_start, rec_end in RECESSION_DATES:
        cty_data = panel_df[panel_df["country"] == country] if "country" in panel_df.columns else panel_df
        if cty_data.empty:
            continue
        start_ts = pd.Timestamp(rec_start + "-01")
        for offset in [0, -1, -2]:
            target = start_ts + pd.DateOffset(months=offset)
            idx = cty_data.index.get_indexer([target], method="nearest")[0]
            if idx >= 0:
                row = cty_data.iloc[idx]
                break
        regime = classify_episode_regime(row[engine_cols] if engine_cols else row)
        eng_vals = {e: float(row.get(e, np.nan)) for e in engine_cols}
        episodes.append({
            "country": country,
            "recession_start": rec_start,
            "recession_end":   rec_end,
            "regime_type": regime,
            **eng_vals,
        })
    return pd.DataFrame(episodes)


def build_forward_target(recession: pd.Series, horizon: int = 6) -> pd.Series:
    starts = pd.Series(0, index=recession.index, dtype=float)
    prev = 0
    for dt, v in recession.items():
        if v == 1 and prev == 0: starts[dt] = 1.0
        prev = v
    target = pd.Series(0, index=recession.index, dtype=float)
    for i in range(len(recession)):
        window = starts.iloc[i:i + horizon + 1]
        target.iloc[i] = 1.0 if window.max() >= 1 else 0.0
    return target


def train_regime_conditioned_models(engine_scores: pd.DataFrame,
                                     regime_df: pd.DataFrame,
                                     recession: pd.Series,
                                     horizon: int = FORECAST_HORIZON) -> dict:
    engines = [c for c in PURE_ENGINES if c in engine_scores.columns]
    target = build_forward_target(recession, horizon)

    month_regime = pd.Series("Unknown", index=engine_scores.index)
    for _, row in regime_df.iterrows():
        if row["country"] != "USA":
            continue
        start_ts = pd.Timestamp(str(row["recession_start"]) + "-01")
        for lag in range(0, 7):
            t = start_ts - pd.DateOffset(months=lag)
            if t in month_regime.index:
                month_regime.loc[t] = row["regime_type"]

    unique_regimes = [r for r in month_regime.unique() if r != "Unknown"]
    print(f"  Regime types found: {unique_regimes}")

    models = {}
    for regime in unique_regimes:
        regime_mask = (month_regime == regime)
        general_mask = (month_regime == "Unknown")
        n_regime = int(target[regime_mask].sum())
        n_general = int(target[general_mask].sum())
        if n_regime + n_general < 5:
            print(f"    [{regime}] Insufficient positive examples ({n_regime}) — skipping")
            continue

        X_regime = engine_scores[engines][regime_mask].fillna(50.0)
        y_regime = target[regime_mask]
        X_general = engine_scores[engines][general_mask].fillna(50.0)
        y_general = target[general_mask]

        X_train = pd.concat([X_regime, X_regime, X_regime, X_general])
        y_train = pd.concat([y_regime, y_regime, y_regime, y_general])

        if y_train.std() < 1e-6:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train.values)

        model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)
        model.fit(X_train_s, y_train.values)

        coefs = pd.Series(np.abs(model.coef_[0]), index=engines)
        coefs = coefs / coefs.sum()

        models[regime] = {
            "model":   model,
            "scaler":  scaler,
            "coefs":   coefs,
            "n_pos":   n_regime + n_general,
        }
        print(f"    [{regime}] Model trained. Top feature: {coefs.idxmax()} ({coefs.max():.2f})")
        print(f"         Coefs: " + "  ".join(f"{e[:3]}={v:.2f}" for e, v in coefs.items()))

    return models, month_regime


def predict_regime_conditioned(engine_scores: pd.DataFrame,
                                 models: dict,
                                 month_regime: pd.Series,
                                 default_model: str = None) -> pd.Series:
    engines = [c for c in PURE_ENGINES if c in engine_scores.columns]
    probas = pd.Series(np.nan, index=engine_scores.index)
    fallback = default_model or (list(models.keys())[0] if models else None)
    for dt in engine_scores.index:
        regime = month_regime.get(dt, "Unknown")
        model_key = regime if regime in models else fallback
        if model_key not in models:
            continue
        m = models[model_key]
        x = engine_scores.loc[dt, engines].fillna(50.0).values.reshape(1, -1)
        x_s = m["scaler"].transform(x)
        probas.loc[dt] = m["model"].predict_proba(x_s)[0, 1]
    return probas.rename("regime_conditioned_prob")


def evaluate_regime_conditioned(probas, recession, horizon=FORECAST_HORIZON):
    target = build_forward_target(recession, horizon)
    common = probas.dropna().index.intersection(target.index)
    p = probas.loc[common]; y = target.loc[common]
    if y.sum() < 2:
        return None
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    prec, rec, thresholds = precision_recall_curve(y, p)
    f1 = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0)
    opt_thresh = thresholds[np.argmax(f1)] if len(thresholds) > 0 else 0.5
    return {"roc_auc": round(auc,3), "pr_auc": round(ap,3),
            "brier_score": round(brier,3),
            "opt_threshold": round(opt_thresh,3),
            "n": len(y), "n_rec": int(y.sum())}


def save_current_prediction(eng_raw, eng_scores, models, month_regime, engines):
    """
    Compute the live regime-conditioned recession probability for the latest
    available month and persist it to results/tables/current_prediction.csv.

    This file is the PRIMARY source for the dashboard's recession probability
    figure — it is read by dashboard.py on every page load.
    """
    os.makedirs(RESULTS_TAB, exist_ok=True)
    latest     = eng_scores.dropna().index[-1]
    cur_regime = month_regime.get(latest, "Unknown")
    key        = cur_regime if cur_regime in models else list(models.keys())[0]

    x   = eng_scores.loc[latest, engines].fillna(50.0).values.reshape(1, -1)
    x_s = models[key]["scaler"].transform(x)
    prob = float(models[key]["model"].predict_proba(x_s)[0, 1])

    esi_val = None
    if "ESI" in eng_raw.columns:
        esi_val = round(float(eng_raw["ESI"].dropna().iloc[-1]), 1)

    # Per-engine scores at latest date
    engine_vals = {f"engine_{e.lower()}": round(float(eng_scores.loc[latest, e]), 1)
                   for e in engines if e in eng_scores.columns}

    row = {
        "date":               latest.strftime("%Y-%m-%d"),
        "prob_recession_6m":  round(prob, 4),
        "regime":             cur_regime,
        "model_used":         key,
        "esi":                esi_val,
        **engine_vals,
    }
    path = os.path.join(RESULTS_TAB, "current_prediction.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"  Saved current prediction → {path}")
    return prob, cur_regime, key


def plot_regime_conditioned(probas, month_regime, engine_scores, recession, esi, save_dir):
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    rec = recession.reindex(probas.index).fillna(0)
    esi_a = esi.reindex(probas.index)

    def shade(ax):
        in_rec, start = False, None
        for dt, v in rec.items():
            if v==1 and not in_rec: start, in_rec=dt,True
            elif v==0 and in_rec:
                ax.axvspan(start,dt,color=SHADE_COLOR,alpha=0.6,zorder=0); in_rec=False
        if in_rec: ax.axvspan(start,rec.index[-1],color=SHADE_COLOR,alpha=0.6,zorder=0)

    ax = axes[0]; shade(ax)
    ax.fill_between(esi_a.index, esi_a.values, alpha=0.2, color="#c0392b")
    ax.plot(esi_a.index, esi_a.values, lw=2, color="#c0392b", label="ESI")
    ax.set_ylim(0,105); ax.set_ylabel("ESI"); ax.legend(fontsize=9)
    ax.set_title("ESI + Regime-Conditioned Recession Probability", fontweight="bold")

    ax2 = axes[1]; shade(ax2)
    ax2.fill_between(probas.index, probas.values, alpha=0.25, color="#2166ac")
    ax2.plot(probas.index, probas.values, lw=2, color="#2166ac",
             label="Regime-conditioned P(recession)")
    ax2.axhline(0.4, color="orange", lw=1, ls="--", alpha=0.7)
    ax2.set_ylim(0,1.05); ax2.set_ylabel("P(Recession in 6m)")
    ax2.legend(fontsize=9)
    ax2.set_title("Regime-Conditioned Probability\n"
                  "(Each regime uses model trained on matching recession episodes)",
                  fontweight="bold")

    ax3 = axes[2]
    regime_colors = {"Financial Crisis":"#e74c3c","Demand Shock":"#9b59b6",
                     "Inflation Shock":"#e6550d","Monetary Squeeze":"#3182bd",
                     "Balanced Recession":"#f39c12","Systemic Crisis":"#c0392b",
                     "Mild / Mixed":"#95a5a6","Unknown":"#ecf0f1"}
    unique_regimes = [r for r in month_regime.unique() if r != "Unknown"]
    prev_r, prev_start = None, None
    for dt in month_regime.index:
        r = month_regime[dt]
        if r != prev_r:
            if prev_r and prev_start and prev_r != "Unknown":
                ax3.axvspan(prev_start, dt, color=regime_colors.get(prev_r,"#888"),
                            alpha=0.5, zorder=0)
            prev_r, prev_start = r, dt
    if prev_r and prev_r != "Unknown":
        ax3.axvspan(prev_start, month_regime.index[-1],
                    color=regime_colors.get(prev_r,"#888"), alpha=0.5, zorder=0)
    shade(ax3)
    for r, c in regime_colors.items():
        if r != "Unknown" and r in unique_regimes:
            ax3.fill_between([], [], color=c, alpha=0.6, label=r)
    ax3.set_ylim(0,1); ax3.set_yticks([]); ax3.set_ylabel("Regime")
    ax3.set_title("Regime Classification over Time", fontweight="bold")
    ax3.legend(fontsize=8, loc="lower left", ncol=3)

    fig.tight_layout()
    path = os.path.join(save_dir, "30_regime_conditioned_roc.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def main():
    feat_path = os.path.join(DATA_PROC, "fred_features.csv")
    engine_path = os.path.join(DATA_PROC, "engine_scores.csv")
    panel_engine_path = os.path.join(DATA_PROC, "panel_engine_scores.csv")
    for p in [feat_path, engine_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Run earlier steps: {p}")
    os.makedirs(RESULTS_FIG, exist_ok=True)
    os.makedirs(RESULTS_TAB, exist_ok=True)

    feat = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    eng_raw = pd.read_csv(engine_path, index_col=0, parse_dates=True)
    recession = feat["RECESSION"]
    engines = [c for c in PURE_ENGINES if c in eng_raw.columns]
    eng_scores = eng_raw[engines]
    esi = eng_raw["ESI"] if "ESI" in eng_raw.columns else eng_scores.mean(axis=1)

    print(f"\n{'='*64}")
    print("  Regime-Conditioned ML (optimised)")
    print(f"{'='*64}\n")

    if not os.path.exists(panel_engine_path):
        print("  Panel engine scores not found — using US episodes only")
        from src.data.download_oecd import RECESSION_DATES
        eps = []
        for cty, s, e in RECESSION_DATES:
            if cty != "USA": continue
            start_ts = pd.Timestamp(s+"-01")
            i = eng_scores.index.get_indexer([start_ts], method="nearest")[0]
            row = eng_scores.iloc[max(0,i-1)]
            regime = classify_episode_regime(row)
            eps.append({"country":cty,"recession_start":s,"recession_end":e,
                        "regime_type":regime, **{e2: float(row.get(e2,np.nan)) for e2 in engines}})
        regime_df = pd.DataFrame(eps)
    else:
        regime_df = label_episodes_with_regime(panel_engine_path, engine_path)

    print(f"  Episode regime classification (US):")
    us_eps = regime_df[regime_df["country"]=="USA"]
    for _, row in us_eps.iterrows():
        eng_str = " ".join(f"{e[:3]}={row.get(e,np.nan):.0f}" for e in engines if not np.isnan(row.get(e,np.nan)))
        print(f"    {row['recession_start']:<8} → {row['regime_type']:<22}  [{eng_str}]")

    print(f"\n  Regime distribution across {len(regime_df)} episodes:")
    for regime, grp in regime_df.groupby("regime_type"):
        print(f"    {regime:<25}  {len(grp):>3} episodes  countries: {sorted(grp['country'].unique())}")

    print(f"\n  Training regime-conditioned models...")
    models, month_regime = train_regime_conditioned_models(
        eng_scores, regime_df, recession, horizon=FORECAST_HORIZON
    )

    if not models:
        print("  No models trained — insufficient data per regime")
        return

    probas = predict_regime_conditioned(eng_scores, models, month_regime)
    result = evaluate_regime_conditioned(probas, recession, FORECAST_HORIZON)
    if result:
        print(f"\n  Regime-conditioned model ROC-AUC: {result['roc_auc']:.3f}")

        results_rows = []
        for regime, m in models.items():
            coef_str = {f"coef_{k}": v for k, v in m["coefs"].items()}
            results_rows.append({"regime": regime, **coef_str,
                                  "top_feature": m["coefs"].idxmax()})
        pd.DataFrame(results_rows).to_csv(
            os.path.join(RESULTS_TAB, "T6_regime_conditioned_performance.csv"), index=False)

        plot_regime_conditioned(probas, month_regime, eng_scores, recession, esi, RESULTS_FIG)

        # Calibration curve
        y_true = build_forward_target(recession, FORECAST_HORIZON).loc[probas.dropna().index]
        p_vals = probas.dropna().values
        y_vals = y_true.values
        frac_pos, mean_pred = calibration_curve(y_vals, p_vals, n_bins=10, strategy='uniform')
        plt.figure(figsize=(6,6))
        plt.plot(mean_pred, frac_pos, marker='o', linewidth=2, label='Regime-conditioned')
        plt.plot([0,1],[0,1], '--', color='gray', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve – Regime-Conditioned Model')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FIG, '30_calibration_regime_conditioned.png'), bbox_inches='tight')
        plt.close()

    # ── Save current live prediction ──────────────────────────────────────────
    # This is the key output consumed by dashboard.py.
    # Written LAST so it's always based on the fully-trained models above.
    prob, cur_regime, key = save_current_prediction(
        eng_raw, eng_scores, models, month_regime, engines
    )

    print(f"\n  Current month ({eng_scores.dropna().index[-1].strftime('%B %Y')}):")
    print(f"    Regime:     {cur_regime}")
    print(f"    Model used: {key}")
    print(f"    P(recession in {FORECAST_HORIZON}m) = {prob:.1%}")
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()