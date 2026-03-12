"""
src/models/panel_ml.py
-----------------------
Panel ML: train on pooled international data, evaluate on US.

Optimised with XGBoost + LightGBM for speed and accuracy.
- XGBoost and LightGBM replace Random Forest
- Parallelised with n_jobs=-1
- Removed lead time, kept Brier score and calibration
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, brier_score_loss, precision_recall_curve)
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_PROC, RESULTS_FIG, RESULTS_TAB, RANDOM_STATE, FORECAST_HORIZON

SHADE_COLOR = "#fee0d2"
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                      "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})

COUNTRIES = ["USA", "GBR", "DEU", "CAN", "JPN", "FRA", "AUS"]

SOFT_LABEL_EVENTS = [
    ("USA", "1987-10", "1987-12", 0.5), ("USA", "1998-08", "1998-10", 0.5),
    ("USA", "2011-08", "2011-10", 0.5), ("USA", "2015-08", "2016-02", 0.5),
    ("USA", "2018-10", "2018-12", 0.5), ("GBR", "2011-07", "2012-03", 0.5),
    ("DEU", "2011-07", "2012-06", 0.5), ("JPN", "1997-10", "1997-12", 0.5),
    ("FRA", "2011-07", "2012-03", 0.5),
]


def build_forward_start_target(recession, horizon=6):
    starts = pd.Series(0, index=recession.index, dtype=float)
    prev = 0
    for dt, v in recession.items():
        if v == 1 and prev == 0:
            starts[dt] = 1.0
        prev = v
    target = pd.Series(0, index=recession.index, dtype=float)
    for i in range(len(recession)):
        window = starts.iloc[i:i + horizon + 1]
        target.iloc[i] = 1.0 if window.max() >= 1 else 0.0
    return target


def build_soft_labels(recession, country):
    soft = recession.copy().astype(float)
    for i in range(len(recession)):
        if recession.iloc[i] == 1:
            for lag in range(1, 4):
                j = i - lag
                if j >= 0 and soft.iloc[j] == 0.0:
                    soft.iloc[j] = 0.7
    for (cty, start_m, end_m, val) in SOFT_LABEL_EVENTS:
        if cty != country:
            continue
        start_ts = pd.Timestamp(start_m + "-01")
        end_ts   = pd.Timestamp(end_m + "-01")
        mask = (soft.index >= start_ts) & (soft.index <= end_ts) & (soft == 0.0)
        soft.loc[mask] = val
    return soft


def prepare_panel_data(panel_engines_path, horizon=6):
    df = pd.read_csv(panel_engines_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    engine_cols = [c for c in df.columns if c not in ("country", "RECESSION")]

    rows_X, rows_y_bin, rows_y_start, rows_y_soft = [], [], [], []
    rows_country, rows_dates, rows_recession = [], [], []

    for country in COUNTRIES:
        subset = df[df["country"] == country].copy()
        if len(subset) < 60:
            print(f"    [{country}] Too few months ({len(subset)}) — skipping")
            continue
        recession = subset["RECESSION"].fillna(0)
        eng_data = subset[engine_cols].dropna(how="all")
        if len(eng_data) < 60:
            continue
        rec_aligned = recession.reindex(eng_data.index).fillna(0)

        y_bin   = build_forward_start_target(rec_aligned, horizon)
        y_soft  = build_soft_labels(rec_aligned, country)
        y_std   = pd.Series(0.0, index=rec_aligned.index)
        for i in range(len(rec_aligned)):
            window = rec_aligned.iloc[i:i + horizon + 1]
            y_std.iloc[i] = 1.0 if window.max() >= 1 else 0.0

        # Country dummies
        country_dummies = {f"is_{c}": (1.0 if c == country else 0.0)
                           for c in COUNTRIES}
        dummy_df = pd.DataFrame(country_dummies, index=eng_data.index)

        X = pd.concat([eng_data.fillna(50.0), dummy_df], axis=1)

        rows_X.append(X)
        rows_y_bin.append(y_bin.reindex(X.index).fillna(0))
        rows_y_start.append(y_bin.reindex(X.index).fillna(0))
        rows_y_soft.append(y_soft.reindex(X.index).fillna(0))
        rows_country.append(pd.Series(country, index=X.index, name="country"))
        rows_dates.append(X.index)
        rows_recession.append(rec_aligned.reindex(X.index).fillna(0))

    if not rows_X:
        raise ValueError("No valid panel data found. Run download_oecd.py and engineer_panel_features.py first.")

    X_all = pd.concat(rows_X, axis=0).sort_index()
    y_binary = pd.concat(rows_y_bin, axis=0).sort_index()
    y_start = pd.concat(rows_y_start, axis=0).sort_index()
    y_soft = pd.concat(rows_y_soft, axis=0).sort_index()
    country_col = pd.concat(rows_country, axis=0).sort_index()
    recession_all = pd.concat(rows_recession, axis=0).sort_index()

    # Simple median imputation (fast)
    X_all = X_all.fillna(X_all.median())

    return X_all, y_binary, y_start, y_soft, country_col, recession_all


def evaluate_panel_model(model, X_train, y_train, X_test, y_test, model_name, target_name):
    """Train on panel, evaluate on held-out country."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    if y_train.std() < 1e-6:
        return None

    # For XGBoost, adjust scale_pos_weight based on training class ratio
    if isinstance(model, xgb.XGBClassifier):
        pos = np.sum(y_train)
        neg = len(y_train) - pos
        model.set_params(scale_pos_weight=neg/pos if pos>0 else 1)

    model.fit(X_tr_s, y_train)
    proba = model.predict_proba(X_te_s)[:, 1]

    if y_test.sum() < 2:
        return None

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    # Optimal threshold for classification
    prec, rec, thresholds = precision_recall_curve(y_test, proba)
    f1 = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0)
    opt_thresh = thresholds[np.argmax(f1)] if len(thresholds) > 0 else 0.5

    return {
        "model": model_name, "target": target_name,
        "roc_auc": round(auc, 3), "pr_auc": round(ap, 3),
        "brier_score": round(brier, 3),
        "opt_threshold": round(opt_thresh, 3),
        "n_test": len(y_test), "n_recession_test": int(y_test.sum()),
    }


def run_leave_one_country_out(X_all, y_binary, y_start, country_col):
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000,
                                                   class_weight="balanced",
                                                   random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=4,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       class_weight='balanced',
                                       random_state=RANDOM_STATE, n_jobs=-1,verbosity=-1)
    }

    results = []
    for held_out in COUNTRIES:
        mask_out   = (country_col == held_out)
        mask_train = ~mask_out
        if mask_out.sum() < 30:
            continue

        for model_name, model_proto in models.items():
            for y_all, target_name in [(y_binary, "forward_standard"),
                                        (y_start,  "forward_start_only")]:
                X_train = X_all[mask_train]; y_train = y_all[mask_train]
                X_test  = X_all[mask_out];   y_test  = y_all[mask_out]

                import copy
                result = evaluate_panel_model(
                    copy.deepcopy(model_proto),
                    X_train.values, y_train.values,
                    X_test.values,  y_test.values,
                    model_name, target_name
                )
                if result:
                    result["held_out"] = held_out
                    results.append(result)

        print(f"    [{held_out}] evaluated  "
              f"(train={mask_train.sum()}, test={mask_out.sum()}, "
              f"rec_months={int(y_start[mask_out].sum())})")

    return pd.DataFrame(results)


def plot_panel_roc(results_df, save_dir):
    if results_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, target in zip(axes, ["forward_standard", "forward_start_only"]):
        sub = results_df[results_df["target"] == target].copy()
        if sub.empty:
            continue
        countries  = sub["held_out"].unique()
        model_names = sub["model"].unique()
        x = np.arange(len(countries))
        width = 0.35
        colors = ["#2166ac", "#d62728", "#2ca02c"]
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            aucs = []
            for cty in countries:
                row = sub[(sub["held_out"] == cty) & (sub["model"] == model_name)]
                aucs.append(row["roc_auc"].values[0] if len(row) > 0 else np.nan)
            ax.bar(x + i*width, aucs, width, label=model_name, color=color, alpha=0.8)
        ax.axhline(0.5, color="grey", lw=0.7, ls="--", alpha=0.5, label="Random")
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(countries, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("OOS ROC-AUC")
        ax.set_title(f"Leave-One-Country-Out ROC-AUC\nTarget: {target.replace('_',' ')}",
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.text(0.98, 0.05, "Trained on 6 countries\nEvaluated on held-out",
                transform=ax.transAxes, ha="right", fontsize=8, color="grey")
    fig.suptitle("Panel Model Performance — International Recession Prediction\n"
                 "Leave-One-Country-Out: model learns from all other countries, "
                 "predicts held-out country",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "26_panel_roc.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_country_fingerprints(panel_engines_path, save_dir):
    df = pd.read_csv(panel_engines_path, index_col=0, parse_dates=True)
    engine_cols = [c for c in df.columns if c not in ("country", "RECESSION")]
    if not engine_cols:
        return
    n_eng = len(engine_cols)
    angles = np.linspace(0, 2*np.pi, n_eng, endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), subplot_kw=dict(polar=True))
    axes_flat = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(COUNTRIES)))
    for idx, (country, color) in enumerate(zip(COUNTRIES, colors)):
        ax = axes_flat[idx]
        subset = df[df["country"] == country]
        rec_months = subset[subset["RECESSION"] == 1][engine_cols]
        norm_months = subset[subset["RECESSION"] == 0][engine_cols]
        if len(rec_months) < 3:
            ax.set_visible(False)
            continue
        rec_mean = rec_months.mean().fillna(50)
        norm_mean = norm_months.mean().fillna(50)
        vals_rec = rec_mean.tolist() + [rec_mean.iloc[0]]
        ax.fill(angles, vals_rec, color=color, alpha=0.3)
        ax.plot(angles, vals_rec, lw=2, color=color, label="Recession avg")
        vals_norm = norm_mean.tolist() + [norm_mean.iloc[0]]
        ax.plot(angles, vals_norm, lw=1, color="grey", ls="--", alpha=0.6, label="Normal avg")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([e[:4] for e in engine_cols], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75])
        ax.set_yticklabels(["", "", ""], fontsize=0)
        ax.set_title(f"{country}\n({len(rec_months)} rec months)",
                     fontsize=9, fontweight="bold", pad=12, color=color)
        theta_full = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta_full, [60]*100, color="orange", lw=0.6, ls="--", alpha=0.5)
    for extra in axes_flat[len(COUNTRIES):]:
        extra.set_visible(False)
    fig.suptitle("Country Recession Fingerprints — Engine Score Profile During Recessions\n"
                 "Each country's recessions have a distinct engine signature "
                 "(filled = recession avg, dashed = normal avg)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "27_country_fingerprints.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    panel_engines_path = os.path.join(DATA_PROC, "panel_engine_scores.csv")
    if not os.path.exists(panel_engines_path):
        print(f"\n  Panel engine scores not found: {panel_engines_path}")
        print("  Run steps in order:\n    1. python src/data/download_oecd.py\n    2. python src/features/engineer_panel_features.py")
        return
    os.makedirs(RESULTS_FIG, exist_ok=True)
    os.makedirs(RESULTS_TAB, exist_ok=True)

    print(f"\n{'='*64}")
    print("  Panel ML — International Recession Training (XGBoost + LightGBM)")
    print(f"  (Leave-one-country-out cross-validation)")
    print(f"{'='*64}\n")

    X_all, y_binary, y_start, y_soft, country_col, recession_all = prepare_panel_data(
        panel_engines_path, horizon=FORECAST_HORIZON
    )

    print(f"\n  Panel summary:")
    print(f"    Total country-months: {len(X_all)}")
    print(f"    Recession months (start target): {y_start.sum():.0f}")
    print(f"    Countries in panel: {country_col.unique().tolist()}")

    for cty in COUNTRIES:
        mask = country_col == cty
        if mask.sum() == 0:
            continue
        rec_m = y_start[mask].sum()
        print(f"    {cty}: {mask.sum()} months, {rec_m:.0f} recession-start months")

    print(f"\n  Running leave-one-country-out CV...")
    results_df = run_leave_one_country_out(X_all, y_binary, y_start, country_col)

    if not results_df.empty:
        print(f"\n  LOCO CV Results (ROC-AUC):")
        print(f"  {'Country':<8}  {'Model':<22}  {'Standard':>10}  {'Start-only':>10}")
        for cty in COUNTRIES:
            for model_name in results_df["model"].unique():
                std_row   = results_df[(results_df["held_out"]==cty) &
                                        (results_df["model"]==model_name) &
                                        (results_df["target"]=="forward_standard")]
                start_row = results_df[(results_df["held_out"]==cty) &
                                        (results_df["model"]==model_name) &
                                        (results_df["target"]=="forward_start_only")]
                if len(std_row) == 0: continue
                std_auc   = std_row["roc_auc"].values[0] if len(std_row)>0 else np.nan
                start_auc = start_row["roc_auc"].values[0] if len(start_row)>0 else np.nan
                print(f"  {cty:<8}  {model_name:<22}  {std_auc:>10.3f}  {start_auc:>10.3f}")

        mean_std   = results_df[results_df["target"]=="forward_standard"]["roc_auc"].mean()
        mean_start = results_df[results_df["target"]=="forward_start_only"]["roc_auc"].mean()
        print(f"\n  Mean LOCO AUC (standard):    {mean_std:.3f}")
        print(f"  Mean LOCO AUC (start-only):  {mean_start:.3f}")
        results_df.to_csv(os.path.join(RESULTS_TAB, "T5_panel_model_performance.csv"), index=False)
        plot_panel_roc(results_df, RESULTS_FIG)

    plot_country_fingerprints(panel_engines_path, RESULTS_FIG)
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()