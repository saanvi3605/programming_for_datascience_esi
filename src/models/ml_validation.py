"""
src/models/ml_validation.py  v9
---------------------------------
Optimised for speed and accuracy using XGBoost + LightGBM.
- XGBoost and LightGBM replace slower tree models
- Parallelised with n_jobs=-1
- Logistic regression kept as baseline
- Removed lead time, kept Brier score and calibration
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, brier_score_loss, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (DATA_PROC, RESULTS_FIG, RESULTS_TAB, RANDOM_STATE,
                    CV_FOLDS, FORECAST_HORIZON, ENGINE_COLORS)

SHADE_COLOR = "#fee0d2"
plt.rcParams.update({"figure.dpi":150, "axes.spines.top":False, "axes.spines.right":False,
                      "axes.grid":True, "grid.alpha":0.3, "font.size":10})


def _shade(ax, recession, alpha=0.6):
    in_rec, start = False, None
    for dt, v in recession.items():
        if v==1 and not in_rec: start, in_rec=dt,True
        elif v==0 and in_rec:
            ax.axvspan(start,dt,color=SHADE_COLOR,alpha=alpha,zorder=0); in_rec=False
    if in_rec: ax.axvspan(start,recession.index[-1],color=SHADE_COLOR,alpha=alpha,zorder=0)


def build_recession_start_target(recession, horizon=6):
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
    return target


def build_forward_recession_target(recession, horizon=6):
    target = pd.Series(0, index=recession.index, dtype=float)
    for i in range(len(recession)):
        window = recession.iloc[i:i + horizon + 1]
        target.iloc[i] = 1.0 if window.max() >= 1 else 0.0
    return target


def prepare_ml_data(engine_scores, recession, horizon=0, target_type="standard",
                     dfm_factors=None):
    if target_type == "recession_start":
        target = build_recession_start_target(recession, horizon)
    elif horizon > 0:
        target = build_forward_recession_target(recession, horizon)
    else:
        target = recession.copy().astype(float)

    X_df = engine_scores.copy()

    if dfm_factors is not None:
        dfm_aligned = dfm_factors.reindex(X_df.index).ffill(limit=3)
        X_df = pd.concat([X_df, dfm_aligned], axis=1)

    # Temporal features
    base_cols = list(X_df.columns)
    temporal_frames = []
    for col in base_cols:
        s = X_df[col]
        temporal_frames.append(s.diff(1).rename(f"{col}_delta1"))
        temporal_frames.append(s.diff(3).rename(f"{col}_mom3"))
        temporal_frames.append(s.diff(6).rename(f"{col}_mom6"))
        temporal_frames.append(s.rolling(6, min_periods=3).std().rename(f"{col}_vol6"))
        temporal_frames.append(s.shift(1).rename(f"{col}_lag1"))
        temporal_frames.append(s.shift(3).rename(f"{col}_lag3"))
        temporal_frames.append(s.shift(6).rename(f"{col}_lag6"))
    if temporal_frames:
        X_df = pd.concat([X_df] + temporal_frames, axis=1)

    common = X_df.dropna(how="all").index.intersection(target.dropna().index)
    X_out  = X_df.loc[common]
    y_out  = target.loc[common]

    # Simple median imputation (fast)
    X_out = X_out.fillna(X_out.median())

    return X_out, y_out


def get_models():
    """Return fast models: Logistic Regression, XGBoost, LightGBM."""
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000,
                                                   class_weight="balanced",
                                                   random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=4,
                                     learning_rate=0.05, subsample=0.8,
                                     scale_pos_weight=1,  # will be set per fold if needed
                                     random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       class_weight='balanced',
                                       random_state=RANDOM_STATE, n_jobs=-1,verbosity=-1)
    }


def cv_predict_proba(model, X, y, n_splits=CV_FOLDS):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    probas = np.full(len(y), np.nan)
    scaler = StandardScaler()
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te, y_tr = X[train_idx], X[test_idx], y[train_idx]
        X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
        if len(np.unique(y_tr)) < 2:
            probas[test_idx] = y_tr.mean() if y_tr.mean()>0 else 0.083
        else:
            # For XGBoost, adjust scale_pos_weight based on training class ratio
            if isinstance(model, xgb.XGBClassifier):
                pos = np.sum(y_tr)
                neg = len(y_tr) - pos
                model.set_params(scale_pos_weight=neg/pos if pos>0 else 1)
            m = type(model)(**model.get_params())
            m.fit(X_tr_s, y_tr)
            probas[test_idx] = m.predict_proba(X_te_s)[:,1]
    return probas


def evaluate_models(X_df, y, label, feature_names=None):
    X, y_arr = X_df.values, y.values
    n_rec = int(y_arr.sum())
    print(f"\n  {label}  (n={len(y_arr)}, {y_arr.mean()*100:.1f}% positive  "
          f"[{n_rec} episodes]):")
    rows = []
    for name, model in get_models().items():
        try:
            probas = cv_predict_proba(model, X, y_arr)
            valid  = ~np.isnan(probas)
            if valid.sum() < 10: continue

            auc  = roc_auc_score(y_arr[valid], probas[valid])
            ap   = average_precision_score(y_arr[valid], probas[valid])
            brier = brier_score_loss(y_arr[valid], probas[valid])

            # Find optimal threshold for classification (optional)
            prec, rec, thresholds = precision_recall_curve(y_arr[valid], probas[valid])
            f1 = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0)
            opt_thresh = thresholds[np.argmax(f1)] if len(thresholds) > 0 else 0.5

            print(f"    {name:<26}  AUC-ROC={auc:.3f}  PR-AUC={ap:.3f}  "
                  f"Brier={brier:.3f}  thresh={opt_thresh:.2f}")
            rows.append({"model":name, "horizon":label, "roc_auc":round(auc,3),
                         "pr_auc":round(ap,3), "brier_score":round(brier,3),
                         "optimal_threshold":round(opt_thresh,3)})
        except Exception as e:
            print(f"    {name:<26}  ERROR: {e}")
    return pd.DataFrame(rows)


def get_feature_importance(X_df, y, feature_names):
    """Fast importance using XGBoost on full data."""
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                              learning_rate=0.05, random_state=RANDOM_STATE,
                              n_jobs=-1)
    model.fit(X_df.fillna(X_df.median()), y)
    return pd.Series(model.feature_importances_, index=feature_names,
                     name="importance").sort_values(ascending=False)


def plot_calibration_curve(y_true, probas, model_name, save_dir):
    probas = np.array(probas)
    y_true = np.array(y_true)
    frac_pos, mean_pred = calibration_curve(y_true, probas, n_bins=10, strategy='uniform')
    plt.figure(figsize=(5,5))
    plt.plot(mean_pred, frac_pos, marker='o', linewidth=2, label=model_name)
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration curve – {model_name}')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"calibration_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved calibration: {path}")


def plot_current_state(engine_scores, esi, fi, recession, save_dir):
    latest_idx = engine_scores.dropna().index[-1]
    latest_eng = engine_scores.loc[latest_idx]
    latest_esi = esi.loc[latest_idx] if latest_idx in esi.index else 50.0
    X_df, y = prepare_ml_data(engine_scores, recession, horizon=FORECAST_HORIZON,
                               target_type="recession_start")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_df.values)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE,
                            class_weight='balanced')
    lr.fit(X_s[:-1], y.values[:-1])
    current_prob = lr.predict_proba(X_s[[-1]])[0, 1]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0,:2])
    esi_recent = esi.loc[esi.index >= esi.index[-1] - pd.DateOffset(years=10)]
    _shade(ax1, recession.reindex(esi_recent.index).fillna(0))
    ax1.fill_between(esi_recent.index, esi_recent.values, alpha=0.2, color="#c0392b")
    ax1.plot(esi_recent.index, esi_recent.values, lw=2, color="#c0392b", label="ESI")
    ax1.scatter([latest_idx],[latest_esi],s=80,color="#c0392b",zorder=5,
                label=f"Current: {latest_esi:.0f}")
    ax1.set_ylim(0,105); ax1.set_ylabel("ESI (0-100)")
    ax1.set_title(f"ESI — Last 10 Years | Current: {latest_esi:.1f}  "
                  f"({latest_idx.strftime('%B %Y')})", fontweight="bold")
    ax1.legend(fontsize=9)
    ax2 = fig.add_subplot(gs[0,2])
    eng_vals = [latest_eng.get(e, np.nan) for e in engine_scores.columns]
    eng_names = list(engine_scores.columns)
    colors2 = [ENGINE_COLORS.get(e,"#888") for e in eng_names]
    bars = ax2.barh(eng_names, eng_vals, color=colors2, alpha=0.85)
    ax2.axvline(50,color="grey",lw=0.8,ls="--",alpha=0.6)
    ax2.axvline(80,color="red",lw=0.7,ls=":",alpha=0.5)
    ax2.set_xlim(0,110); ax2.set_xlabel("Score (0-100)")
    ax2.set_title(f"Engine Scores\n{latest_idx.strftime('%B %Y')}",fontweight="bold")
    for bar, val in zip(bars, eng_vals):
        if not np.isnan(val):
            ax2.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                     f"{val:.0f}", va="center", fontsize=9, fontweight="bold")
    ax3 = fig.add_subplot(gs[1,0])
    colors3 = ["#2ecc71","#f39c12","#e74c3c"]
    wedges = [0.33,0.34,0.33]
    theta  = np.linspace(0,np.pi,100)
    r_out, r_in = 1.0, 0.55
    prev_t = 0.0
    for w, col in zip(wedges, colors3):
        t_start=prev_t*np.pi; t_end=(prev_t+w)*np.pi; t_seg=np.linspace(t_start,t_end,50)
        xs=np.concatenate([[r_in*np.cos(t_start)],r_out*np.cos(t_seg),[r_in*np.cos(t_end)]])
        ys=np.concatenate([[r_in*np.sin(t_start)],r_out*np.sin(t_seg),[r_in*np.sin(t_end)]])
        ax3.fill(xs,ys,color=col,alpha=0.4); prev_t+=w
    needle_angle = current_prob*np.pi
    ax3.annotate("",xy=(0.75*np.cos(needle_angle),0.75*np.sin(needle_angle)),
                 xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color="black",lw=2.5))
    ax3.text(0,-0.15,f"{current_prob:.0%}",ha="center",fontsize=18,fontweight="bold",
             color="#c0392b" if current_prob>0.4 else "#2c3e50")
    ax3.text(0,-0.32,f"P(Rec START in {FORECAST_HORIZON}m)",ha="center",fontsize=9)
    ax3.set_xlim(-1.2,1.2); ax3.set_ylim(-0.4,1.2); ax3.set_aspect("equal"); ax3.axis("off")
    ax3.set_title("Recession START Probability\n(sharper target: first month only)",fontweight="bold")
    ax4 = fig.add_subplot(gs[1,1])
    fi_s  = fi.sort_values(ascending=True); colors4=[ENGINE_COLORS.get(e,"#888") for e in fi_s.index]
    fi_s.plot(kind="barh",ax=ax4,color=colors4,alpha=0.85)
    ax4.set_xlabel("RF Importance"); ax4.set_title(f"Feature Importance (horizon={FORECAST_HORIZON}m)",fontweight="bold")
    for i, val in enumerate(fi_s.values): ax4.text(val+0.003,i,f"{val:.3f}",va="center",fontsize=8)
    ax5 = fig.add_subplot(gs[1,2]); ax5.axis("off")
    stress_map={(0,30):("LOW","#2ecc71"),(30,60):("MODERATE","#f39c12"),
                (60,80):("HIGH","#e74c3c"),(80,101):("EXTREME","#c0392b")}
    def get_level(v):
        for (lo,hi),(l,c) in stress_map.items():
            if lo<=v<hi: return l,c
        return "EXTREME","#c0392b"
    y_p=0.95
    ax5.text(0.05,y_p,"Current State  (v5 rolling normalisation)",fontsize=10,fontweight="bold",transform=ax5.transAxes)
    ax5.text(0.05,y_p-0.08,latest_idx.strftime('%B %Y'),fontsize=9,color="grey",transform=ax5.transAxes)
    y_p-=0.22
    for eng, val in latest_eng.items():
        if np.isnan(val): continue
        lbl,col=get_level(val); sym="▲" if val>60 else ("▼" if val<30 else "■")
        ax5.text(0.05,y_p,f"{sym} {eng}: {lbl} ({val:.0f})",fontsize=9,color=col,
                 fontweight="bold" if val>60 else "normal",transform=ax5.transAxes); y_p-=0.13
    y_p-=0.05
    rec_lbl="HIGH" if current_prob>0.4 else ("MODERATE" if current_prob>0.2 else "LOW")
    rec_col="#c0392b" if current_prob>0.4 else ("#f39c12" if current_prob>0.2 else "#2ecc71")
    ax5.text(0.05,y_p,f"Recession Start Risk: {rec_lbl}\nP(start in {FORECAST_HORIZON}m) = {current_prob:.1%}",
             fontsize=10,fontweight="bold",color=rec_col,transform=ax5.transAxes)
    fig.suptitle(f"Current Economic State — {latest_idx.strftime('%B %Y')}\n"
                 "Rolling 20Y normalisation  |  Leading/Coincident engine split  |  Sharper recession-start target",
                 fontsize=12,fontweight="bold")
    path = os.path.join(save_dir,"24_current_state.png")
    fig.savefig(path,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def plot_roc_curves(X_df, y, horizon, save_dir, label_prefix=""):
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    for ax, (lbl, target_type) in zip(axes,[
        ("Standard forward target", "standard"),
        ("Recession START target (sharp)", "recession_start"),
    ]):
        X, y_arr = X_df.values, y.values
        for (name,model),color in zip(get_models().items(),
                                      ["#d62728","#2166ac","#31a354"]):
            try:
                probas = cv_predict_proba(model, X, y_arr)
                valid  = ~np.isnan(probas)
                if valid.sum()>10:
                    fpr,tpr,_ = roc_curve(y_arr[valid], probas[valid])
                    auc = roc_auc_score(y_arr[valid], probas[valid])
                    ax.plot(fpr,tpr,lw=2,color=color,label=f"{name}  AUC={auc:.3f}")
            except Exception: pass
        ax.plot([0,1],[0,1],"k--",lw=0.7,alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {lbl}",fontweight="bold"); ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(save_dir,"10_roc_curves.png")
    fig.savefig(path,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def plot_feature_importance(fi, save_dir):
    fig, ax = plt.subplots(figsize=(9,max(4,len(fi)*0.6)))
    fi_s = fi.sort_values(ascending=True)
    colors = [ENGINE_COLORS.get(e.split("_")[0] if "_" not in e else e,"#888") for e in fi_s.index]
    fi_s.plot(kind="barh",ax=ax,color=colors[:len(fi_s)],alpha=0.85)
    ax.set_xlabel("RF Importance"); ax.set_title(f"Feature Importance (engines + DFM factors, horizon={FORECAST_HORIZON}m)",fontweight="bold")
    for i,val in enumerate(fi_s.values): ax.text(val+0.003,i,f"{val:.3f}",va="center",fontsize=8)
    fig.tight_layout()
    path = os.path.join(save_dir,"11_feature_importance.png")
    fig.savefig(path,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def plot_early_warning(X_df, y, esi, recession_series, save_dir):
    # Use XGBoost for final probability
    model = xgb.XGBClassifier(n_estimators=150, max_depth=4,
                              learning_rate=0.05, subsample=0.8,
                              random_state=RANDOM_STATE, n_jobs=-1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_df.values)
    model.fit(X_s, y.values)
    proba = pd.Series(model.predict_proba(X_s)[:,1], index=X_df.index)

    # Calibration curve for this model
    plot_calibration_curve(y, proba, "XGBoost", save_dir)

    # Find optimal threshold
    prec, rec, thresholds = precision_recall_curve(y, proba)
    f1 = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0)
    opt_thresh = thresholds[np.argmax(f1)] if len(thresholds) > 0 else 0.5

    fig, ax1 = plt.subplots(figsize=(18,6)); ax2 = ax1.twinx()
    rec = y.copy(); _shade(ax1, rec, alpha=0.4)
    esi_a = esi.reindex(proba.index)
    ax1.fill_between(esi_a.index,esi_a.values,alpha=0.15,color="#c0392b")
    ax1.plot(esi_a.index,esi_a.values,lw=1.5,color="#c0392b",alpha=0.7,label="ESI (left)")
    ax2.plot(proba.index,proba.values,lw=2,color="#2166ac",label=f"P(rec start in {FORECAST_HORIZON}m) — XGB")
    ax2.axhline(opt_thresh,color="orange",lw=1.2,ls="--",alpha=0.8,label=f"Optimal threshold {opt_thresh:.2f}")
    ax2.axhline(0.4,color="grey",lw=0.8,ls=":",alpha=0.5,label="0.40 reference")
    ax1.set_ylabel("ESI (0-100)",color="#c0392b"); ax2.set_ylabel("P(Recession Start)",color="#2166ac")
    ax1.set_ylim(0,110); ax2.set_ylim(0,1.1)
    ax1.set_title(f"Early Warning — {FORECAST_HORIZON}m Ahead Recession Start Probability\n"
                  f"(Sharper target + optimal threshold = {opt_thresh:.2f})",fontweight="bold")
    l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,lb1+lb2,fontsize=9,loc="upper left")
    fig.tight_layout()
    path = os.path.join(save_dir,"12_early_warning.png")
    fig.savefig(path,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def main():
    feat_path   = os.path.join(DATA_PROC,"fred_features.csv")
    engine_path = os.path.join(DATA_PROC,"engine_scores.csv")
    dfm_path    = os.path.join(DATA_PROC,"dfm_factors.csv")
    for p in [feat_path, engine_path]:
        if not os.path.exists(p): raise FileNotFoundError(f"Run earlier steps: {p}")
    os.makedirs(RESULTS_FIG,exist_ok=True); os.makedirs(RESULTS_TAB,exist_ok=True)

    feat        = pd.read_csv(feat_path,index_col=0,parse_dates=True)
    eng_raw     = pd.read_csv(engine_path,index_col=0,parse_dates=True)
    recession   = feat["RECESSION"]
    engine_cols = [c for c in eng_raw.columns if c not in ("ESI","ESI_expanding","ESI_ML","RECESSION")]
    engine_scores = eng_raw[engine_cols]
    esi = eng_raw["ESI"] if "ESI" in eng_raw.columns else engine_scores.mean(axis=1)

    dfm_factors = None
    if os.path.exists(dfm_path):
        dfm_factors = pd.read_csv(dfm_path,index_col=0,parse_dates=True)
        print(f"  DFM factors loaded: {list(dfm_factors.columns)}")

    print(f"\n{'='*64}\n  ML Validation v9 (XGBoost + LightGBM)\n{'='*64}")
    print(f"  Engines: {engine_cols}")

    perf_rows = []
    for horizon, target_type, lbl in [
        (0,              "standard",       "Contemporaneous (standard)"),
        (FORECAST_HORIZON,"standard",      f"Forward +{FORECAST_HORIZON}m (standard)"),
        (FORECAST_HORIZON,"recession_start",f"Forward +{FORECAST_HORIZON}m (start only — sharp)"),
    ]:
        X_df, y = prepare_ml_data(engine_scores, recession, horizon=horizon,
                                   target_type=target_type, dfm_factors=dfm_factors)
        perf_rows.append(evaluate_models(X_df, y, lbl, feature_names=list(X_df.columns)))

    pd.concat(perf_rows,ignore_index=True).to_csv(
        os.path.join(RESULTS_TAB,"05_model_performance.csv"), index=False)

    X_df_fi, y_fi = prepare_ml_data(engine_scores, recession, horizon=FORECAST_HORIZON,
                                     target_type="recession_start", dfm_factors=dfm_factors)
    fi = get_feature_importance(X_df_fi, y_fi, list(X_df_fi.columns))
    fi.to_frame("importance").to_csv(os.path.join(RESULTS_TAB,"06_feature_importance.csv"))

    print(f"\n  Feature importance (top 8):")
    for feat_name, val in fi.head(8).items():
        print(f"    {feat_name:<20}  {val:.3f}")

    X_df_fw, y_fw = prepare_ml_data(engine_scores, recession, horizon=FORECAST_HORIZON,
                                     target_type="standard")
    plot_roc_curves(X_df_fw, y_fw, FORECAST_HORIZON, RESULTS_FIG)
    plot_feature_importance(fi, RESULTS_FIG)
    plot_early_warning(X_df_fi, y_fi, esi, recession, RESULTS_FIG)
    plot_current_state(engine_scores, esi, fi, recession, RESULTS_FIG)
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()