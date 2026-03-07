"""
src/analysis/engines.py  v5
Four structural improvements:
1. Rolling normalisation (10Y z-score, 20Y percentile)
2. Leading/coincident engine split (0.6/0.4)
3. Dispersion-based ESI: mean + 0.5*std
4. Dual-track output: expanding + rolling + ML-weighted
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
from config import (ENGINE_FEATURES, ENGINE_COLORS, DATA_PROC, RESULTS_FIG, RESULTS_TAB,
                    ROLLING_ZSCORE_WINDOW, ROLLING_PCTILE_WINDOW,
                    ENGINE_LEAD_WEIGHTS, ENGINE_VARIABLE_TYPES, ESI_DISPERSION_LAMBDA,
                    adaptive_min)
SHADE_COLOR = "#fee0d2"
plt.rcParams.update({"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False,
                      "axes.grid":True,"grid.alpha":0.3,"font.size":10})


def expanding_zscore(s, min_periods=None):
    n = s.notna().sum()
    mp = adaptive_min(36, n) if min_periods is None else min_periods
    mu = s.expanding(min_periods=mp).mean()
    sig = s.expanding(min_periods=mp).std().replace(0, np.nan)
    return (s - mu) / sig


def rolling_zscore(s, window=ROLLING_ZSCORE_WINDOW, min_periods=None):
    n = s.notna().sum()
    mp = adaptive_min(window, n) if min_periods is None else min_periods
    mu = s.rolling(window=window, min_periods=mp).mean()
    sig = s.rolling(window=window, min_periods=mp).std().replace(0, np.nan)
    return (s - mu) / sig


def expanding_percentile_rank(s, min_periods=None):
    n = s.notna().sum()
    mp = adaptive_min(60, n) if min_periods is None else min_periods
    vals = s.values.copy().astype(float)
    ranks = np.full(len(vals), np.nan)
    for i in range(mp, len(vals)):
        if np.isnan(vals[i]): continue
        hist = vals[:i][~np.isnan(vals[:i])]
        if len(hist) < 10: continue
        ranks[i] = float((hist < vals[i]).sum()) / len(hist) * 100.0
    return pd.Series(ranks, index=s.index, name=s.name)


def rolling_percentile_rank(s, window=ROLLING_PCTILE_WINDOW, min_periods=None):
    n = s.notna().sum()
    mp = adaptive_min(window, n) if min_periods is None else min_periods
    vals = s.values.copy().astype(float)
    ranks = np.full(len(vals), np.nan)
    for i in range(mp, len(vals)):
        if np.isnan(vals[i]): continue
        start = max(0, i - window)
        hist = vals[start:i][~np.isnan(vals[start:i])]
        if len(hist) < 10: continue
        ranks[i] = float((hist < vals[i]).sum()) / len(hist) * 100.0
    return pd.Series(ranks, index=s.index, name=s.name)


def expanding_impute_median(s):
    vals = s.values.copy().astype(float)
    out = vals.copy()
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            hist = vals[:i][~np.isnan(vals[:i])]
            if len(hist) > 0: out[i] = float(np.median(hist))
    return pd.Series(out, index=s.index, name=s.name)


def score_engine_with_lead_lag(df, feature_list, engine_name, mode="rolling",
                                z_min=None, pct_min=None):
    available = [f for f in feature_list if f in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, name=engine_name)
    z_scores = {}
    for col in available:
        n = df[col].notna().sum()
        _z_min  = adaptive_min(ROLLING_ZSCORE_WINDOW, n) if z_min is None else z_min
        _pct_min = adaptive_min(ROLLING_PCTILE_WINDOW, n) if pct_min is None else pct_min
        z = rolling_zscore(df[col], min_periods=_z_min) if mode == "rolling" \
            else expanding_zscore(df[col], min_periods=_z_min)
        z_scores[col] = expanding_impute_median(z)
    z_df = pd.DataFrame(z_scores)
    leading    = [c for c in available if ENGINE_VARIABLE_TYPES.get(c) == "leading"]
    coincident = [c for c in available if ENGINE_VARIABLE_TYPES.get(c) != "leading"]
    if leading and coincident:
        composite = (ENGINE_LEAD_WEIGHTS["leading"]   * z_df[leading].mean(axis=1) +
                     ENGINE_LEAD_WEIGHTS["coincident"] * z_df[coincident].mean(axis=1))
    elif leading:
        composite = z_df[leading].mean(axis=1)
    else:
        composite = z_df[available].mean(axis=1)
    n_comp = composite.notna().sum()
    _pct_min = adaptive_min(ROLLING_PCTILE_WINDOW, n_comp) if pct_min is None else pct_min
    pct = (rolling_percentile_rank(composite, min_periods=_pct_min) if mode == "rolling"
           else expanding_percentile_rank(composite, min_periods=_pct_min))
    pct.name = engine_name
    return pct


def build_all_engines(df, engine_defs=None, mode="rolling"):
    if engine_defs is None: engine_defs = ENGINE_FEATURES
    print(f"\n  Building engines [mode={mode}]:")
    engines = {}
    for name, feats in engine_defs.items():
        available = [f for f in feats if f in df.columns]
        score = score_engine_with_lead_lag(df, feats, name, mode=mode)
        fv = score.first_valid_index()
        L = len([c for c in available if ENGINE_VARIABLE_TYPES.get(c) == "leading"])
        C = len(available) - L
        print(f"    {name:<12}  {score.notna().sum():>4}mo  "
              f"from {str(fv.date()) if fv else 'N/A':>12}  L={L} C={C}")
        engines[name] = score
    return pd.DataFrame(engines)


def build_engine_zscores(df, engine_defs=None, mode="rolling"):
    if engine_defs is None: engine_defs = ENGINE_FEATURES
    z_engines = {}
    for name, feats in engine_defs.items():
        available = [f for f in feats if f in df.columns]
        if not available: continue
        z_dict = {}
        for col in available:
            z = rolling_zscore(df[col]) if mode=="rolling" else expanding_zscore(df[col])
            z_dict[col] = expanding_impute_median(z)
        z_df = pd.DataFrame(z_dict)
        leading = [c for c in available if ENGINE_VARIABLE_TYPES.get(c) == "leading"]
        coinc   = [c for c in available if ENGINE_VARIABLE_TYPES.get(c) != "leading"]
        if leading and coinc:
            z_engines[name] = 0.6*z_df[leading].mean(axis=1) + 0.4*z_df[coinc].mean(axis=1)
        else:
            z_engines[name] = z_df[available].mean(axis=1)
    return pd.DataFrame(z_engines)


def build_esi(engine_scores, lambda_disp=ESI_DISPERSION_LAMBDA):
    """
    ESI = mean(engines) + λ * std(engines)
    Replaces 0.70*mean + 0.30*max — std is more stable than max.
    High std in 2022 (one extreme engine, others calm) correctly scores as moderate-high.
    """
    esi = (engine_scores.mean(axis=1) +
           lambda_disp * engine_scores.std(axis=1).fillna(0))
    return esi.clip(0, 100).rename("ESI")


def build_esi_ml_weighted(engine_scores, recession, min_periods=None):
    """Expanding-window logistic regression coefficients as engine weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    engines_clean = engine_scores.dropna()
    rec_aligned   = recession.reindex(engines_clean.index).fillna(0)
    X = engines_clean.values; y = rec_aligned.values
    n = len(engines_clean)
    mp = adaptive_min(60, n) if min_periods is None else min_periods
    weights_series = pd.DataFrame(0.0, index=engines_clean.index,
                                   columns=engines_clean.columns)
    for t in range(mp, len(engines_clean)):
        if y[:t].std() < 1e-6: continue
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[:t])
        lr     = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        lr.fit(X_tr, y[:t])
        coefs  = np.abs(lr.coef_[0]); coefs /= coefs.sum()
        weights_series.iloc[t] = coefs
    weighted_raw = (engines_clean * weights_series).sum(axis=1)
    return (weighted_raw.rank(pct=True) * 100).rename("ESI_ML")


def build_esi(engine_scores, lambda_disp=ESI_DISPERSION_LAMBDA):
    """
    ESI = mean(engines) + λ * std(engines)
    Replaces 0.70*mean + 0.30*max — std is more stable than max.
    High std in 2022 (one extreme engine, others calm) correctly scores as moderate-high.
    """
    esi = (engine_scores.mean(axis=1) +
           lambda_disp * engine_scores.std(axis=1).fillna(0))
    return esi.clip(0, 100).rename("ESI")


def build_esi_ml_weighted(engine_scores, recession, min_periods=60):
    """Expanding-window logistic regression coefficients as engine weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    engines_clean = engine_scores.dropna()
    rec_aligned   = recession.reindex(engines_clean.index).fillna(0)
    X = engines_clean.values; y = rec_aligned.values
    weights_series = pd.DataFrame(0.0, index=engines_clean.index,
                                   columns=engines_clean.columns)
    for t in range(min_periods, len(engines_clean)):
        if y[:t].std() < 1e-6: continue
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[:t])
        lr     = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        lr.fit(X_tr, y[:t])
        coefs  = np.abs(lr.coef_[0]); coefs /= coefs.sum()
        weights_series.iloc[t] = coefs
    weighted_raw = (engines_clean * weights_series).sum(axis=1)
    return (weighted_raw.rank(pct=True) * 100).rename("ESI_ML")


def _shade(ax, recession, alpha=0.6):
    in_rec, start = False, None
    for dt, v in recession.items():
        if v==1 and not in_rec: start, in_rec = dt, True
        elif v==0 and in_rec:
            ax.axvspan(start, dt, color=SHADE_COLOR, alpha=alpha, zorder=0); in_rec=False
    if in_rec: ax.axvspan(start, recession.index[-1], color=SHADE_COLOR, alpha=alpha, zorder=0)


def plot_engine_panel(engine_scores, esi, recession, save_dir):
    engines = list(engine_scores.columns); n = len(engines)
    fig, axes = plt.subplots(n, 1, figsize=(18, 3.8*n), sharex=True)
    if n==1: axes=[axes]
    rec = recession.reindex(engine_scores.index).fillna(0)
    esi_a = esi.reindex(engine_scores.index)
    for ax, eng in zip(axes, engines):
        color = ENGINE_COLORS.get(eng,"#888")
        s = engine_scores[eng]
        _shade(ax, rec)
        ax.fill_between(s.index, s.values, alpha=0.22, color=color)
        ax.plot(s.index, s.values, lw=1.8, color=color, label=f"{eng}")
        ax.plot(esi_a.index, esi_a.values, color="black", lw=0.8, alpha=0.35, ls="--", label="ESI")
        ax.axhline(60, color="orange", lw=0.7, ls=":", alpha=0.8)
        ax.axhline(80, color="red",    lw=0.7, ls=":", alpha=0.8)
        ax.set_ylim(0,102); ax.set_ylabel("Score (0-100)")
        ax.set_title(f"{eng} Engine  [rolling 20Y percentile, L/C split]", fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Engine Scores v5", fontsize=13, fontweight="bold", y=1.002)
    fig.tight_layout()
    path = os.path.join(save_dir,"16_engine_panel.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def plot_esi_comparison(esi_exp, esi_roll, esi_ml, recession, save_dir):
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    rec = recession.reindex(esi_exp.index).fillna(0)
    for ax,(esi,label,color) in zip(axes,[
        (esi_exp, "ESI — expanding window, equal weights (v4)", "#c0392b"),
        (esi_roll,"ESI — rolling 20Y window, equal weights  (v5 structural)", "#2166ac"),
        (esi_ml,  "ESI_ML — rolling 20Y, ML-weighted         (v5 predictive)", "#31a354"),
    ]):
        _shade(ax, rec)
        ax.fill_between(esi.index, esi.values, alpha=0.2, color=color)
        ax.plot(esi.index, esi.values, lw=2.0, color=color, label=label)
        ax.set_ylim(0,105); ax.set_ylabel("ESI (0-100)")
        ax.set_title(label, fontweight="bold"); ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("ESI Variant Comparison: the effect of rolling normalisation and ML weighting",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir,"25_esi_comparison.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def plot_engine_radar(engine_scores, recession, save_dir):
    key_dates = {"1980-03\nVolcker":"1980-03-01","2001-10\nDot-com":"2001-10-01",
                 "2008-10\nGFC Peak":"2008-10-01","2009-06\nGFC Trough":"2009-06-01",
                 "2020-04\nCOVID":"2020-04-01","2022-06\nInflation":"2022-06-01","Latest":None}
    engines = list(engine_scores.columns); n_eng = len(engines)
    angles  = np.linspace(0,2*np.pi,n_eng,endpoint=False).tolist(); angles+=angles[:1]
    fig, axes = plt.subplots(2, 4, figsize=(18,9), subplot_kw=dict(polar=True))
    axes_flat = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0,1,len(key_dates)))
    for idx,(label,date_str) in enumerate(key_dates.items()):
        ax = axes_flat[idx]
        if date_str is None:
            row=engine_scores.dropna(how="all").iloc[-1]
            label=f"Latest\n({engine_scores.dropna(how='all').index[-1].strftime('%Y-%m')})"
        else:
            try:
                t=pd.Timestamp(date_str); loc=engine_scores.index.get_indexer([t],method="nearest")[0]
                row=engine_scores.iloc[loc]
            except: continue
        values=[row.get(e,50) for e in engines]; vplot=values+values[:1]
        ax.plot(angles,vplot,lw=2,color=colors[idx],alpha=0.9)
        ax.fill(angles,vplot,color=colors[idx],alpha=0.25)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(engines,fontsize=8)
        ax.set_ylim(0,100); ax.set_yticks([25,50,75]); ax.set_yticklabels(["","",""],fontsize=0)
        ax.set_title(label,fontsize=9,fontweight="bold",pad=12)
        theta_full=np.linspace(0,2*np.pi,100); ax.plot(theta_full,[60]*100,color="orange",lw=0.6,ls="--",alpha=0.5)
    for extra in axes_flat[len(key_dates):]: extra.set_visible(False)
    fig.suptitle("Engine Fingerprints (rolling v5)",fontsize=12,fontweight="bold")
    fig.tight_layout()
    path=os.path.join(save_dir,"17_engine_radar.png")
    fig.savefig(path,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def plot_engine_heatmap_annual(engine_scores, esi, save_dir):
    annual=engine_scores.copy(); annual["year"]=annual.index.year
    am=annual.groupby("year").median()
    fig,ax=plt.subplots(figsize=(max(14,len(am)*0.45),5))
    sns.heatmap(am.T,ax=ax,cmap="RdYlGn_r",vmin=0,vmax=100,linewidths=0.3,
                annot=True,fmt=".0f",annot_kws={"size":6},cbar_kws={"shrink":0.6})
    ax.set_title("Annual Engine Heatmap (rolling v5)",fontweight="bold")
    fig.tight_layout()
    path=os.path.join(save_dir,"18_engine_heatmap_annual.png")
    fig.savefig(path,bbox_inches="tight"); plt.close(fig); print(f"  Saved: {path}")


def main():
    feat_path = os.path.join(DATA_PROC, "fred_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Run engineer_features.py first: {feat_path}")
    os.makedirs(DATA_PROC,exist_ok=True); os.makedirs(RESULTS_FIG,exist_ok=True); os.makedirs(RESULTS_TAB,exist_ok=True)
    df = pd.read_csv(feat_path,index_col=0,parse_dates=True); recession=df["RECESSION"]
    print(f"\n{'='*64}\n  Economic Engine Scoring v5  (rolling normalisation + L/C split)\n{'='*64}")
    print("\n  [Expanding — for comparison]")
    eng_exp = build_all_engines(df, mode="expanding"); esi_exp = build_esi(eng_exp)
    print("\n  [Rolling 10Y/20Y — PRIMARY]")
    eng_roll = build_all_engines(df, mode="rolling"); esi_roll = build_esi(eng_roll)
    print("\n  [ML-weighted ESI]")
    try: esi_ml = build_esi_ml_weighted(eng_roll, recession)
    except Exception as e: print(f"  ML ESI failed: {e}"); esi_ml=esi_roll.copy(); esi_ml.name="ESI_ML"
    out=eng_roll.copy(); out["ESI"]=esi_roll; out["ESI_expanding"]=esi_exp.reindex(out.index)
    out["ESI_ML"]=esi_ml.reindex(out.index); out["RECESSION"]=recession
    out.to_csv(os.path.join(DATA_PROC,"engine_scores.csv"))
    eng_roll.to_csv(os.path.join(RESULTS_TAB,"T1_engine_scores.csv"))
    print(f"\n  ESI comparison at key events (rolling | expanding | ML):")
    print(f"  {'Date':<10}  {'Rolling':>8}  {'Expand':>8}  {'ML':>8}  Event")
    for dt,ev in {"1980-03":"Volcker","2008-10":"GFC","2020-04":"COVID",
                   "2022-06":"Inflation peak","2024-01":"Post-tight"}.items():
        try:
            t=pd.Timestamp(dt); i=esi_roll.index.get_indexer([t],method="nearest")[0]
            er=esi_roll.iloc[i]; ee=esi_exp.reindex(esi_roll.index).iloc[i]; em=esi_ml.reindex(esi_roll.index).iloc[i]
            print(f"  {dt:<10}  {er:>8.1f}  {ee:>8.1f}  {em:>8.1f}  {ev}")
        except: pass
    print()
    plot_engine_panel(eng_roll,esi_roll,recession,RESULTS_FIG)
    plot_esi_comparison(esi_exp,esi_roll,esi_ml,recession,RESULTS_FIG)
    plot_engine_radar(eng_roll,recession,RESULTS_FIG)
    plot_engine_heatmap_annual(eng_roll,esi_roll,RESULTS_FIG)
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
