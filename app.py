"""
app.py  —  ESI AI Chat Dashboard
=======================================
A clean, chat-first Streamlit dashboard for the Economic Stress Index (ESI).
Powered by Gemini 2.5 Flash via LangChain, with full project data as context.

Usage
-----
  streamlit run dashboard.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import threading
import numpy as np

from src.data.rss_feeds import fetch_rss_headlines, headlines_to_context_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from config import (
    DATA_PROC, RESULTS_TAB, STRESS_LABELS,
    ENGINE_FEATURES, FORECAST_HORIZON
)

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND MODEL RUNNER
#  Runs the regime-conditioned model in a daemon thread on startup so the
#  dashboard always has a fresh current_prediction.csv without blocking the UI.
# ═══════════════════════════════════════════════════════════════════════════════
CURRENT_PRED_PATH = os.path.join(RESULTS_TAB, "current_prediction.csv")

def _run_regime_model_background():
    """Re-runs regime-conditioned inference and saves current_prediction.csv."""
    try:
        import pandas as pd
        import numpy as np
        from src.models.regime_conditioned_ml import (
            classify_episode_regime, label_episodes_with_regime,
            train_regime_conditioned_models, predict_regime_conditioned,
            build_forward_target, PURE_ENGINES
        )
        feat_path   = os.path.join(DATA_PROC, "fred_features.csv")
        engine_path = os.path.join(DATA_PROC, "engine_scores.csv")
        panel_path  = os.path.join(DATA_PROC, "panel_engine_scores.csv")

        if not os.path.exists(feat_path) or not os.path.exists(engine_path):
            return  # pipeline hasn't run yet

        feat      = pd.read_csv(feat_path,   index_col=0, parse_dates=True)
        eng_raw   = pd.read_csv(engine_path, index_col=0, parse_dates=True)
        recession = feat["RECESSION"]
        engines   = [c for c in PURE_ENGINES if c in eng_raw.columns]
        eng_scores = eng_raw[engines]

        # Build episode regime labels
        if os.path.exists(panel_path):
            regime_df = label_episodes_with_regime(panel_path, engine_path)
        else:
            from src.data.download_oecd import RECESSION_DATES
            eps = []
            for cty, s, e in RECESSION_DATES:
                if cty != "USA":
                    continue
                start_ts = pd.Timestamp(s + "-01")
                i = eng_scores.index.get_indexer([start_ts], method="nearest")[0]
                row = eng_scores.iloc[max(0, i - 1)]
                regime = classify_episode_regime(row)
                eps.append({"country": cty, "recession_start": s, "recession_end": e,
                            "regime_type": regime,
                            **{e2: float(row.get(e2, np.nan)) for e2 in engines}})
            regime_df = pd.DataFrame(eps)

        models, month_regime = train_regime_conditioned_models(
            eng_scores, regime_df, recession, horizon=FORECAST_HORIZON
        )
        if not models:
            return

        # Current month inference
        latest   = eng_scores.dropna().index[-1]
        cur_regime = month_regime.get(latest, "Unknown")
        key      = cur_regime if cur_regime in models else list(models.keys())[0]
        x        = eng_scores.loc[latest, engines].fillna(50.0).values.reshape(1, -1)
        x_s      = models[key]["scaler"].transform(x)
        prob     = float(models[key]["model"].predict_proba(x_s)[0, 1])

        os.makedirs(RESULTS_TAB, exist_ok=True)
        pd.DataFrame([{
            "date":                latest.strftime("%Y-%m-%d"),
            "prob_recession_6m":   round(prob, 4),
            "regime":              cur_regime,
            "model_used":          key,
            "esi":                 round(float(eng_raw["ESI"].dropna().iloc[-1]), 1) if "ESI" in eng_raw.columns else None,
        }]).to_csv(CURRENT_PRED_PATH, index=False)
    except Exception as e:
        # Background thread — swallow errors silently so dashboard still loads
        try:
            os.makedirs(RESULTS_TAB, exist_ok=True)
            with open(os.path.join(RESULTS_TAB, "current_prediction_error.log"), "w") as f:
                import traceback
                f.write(traceback.format_exc())
        except Exception:
            pass


def maybe_refresh_prediction():
    """
    Spawn the background inference thread if:
      - current_prediction.csv doesn't exist yet, OR
      - it's older than 6 hours (stale after a new data pull)
    Uses st.session_state to avoid re-spawning on every Streamlit rerun.
    """
    if st.session_state.get("_model_thread_started"):
        return

    needs_refresh = True
    if os.path.exists(CURRENT_PRED_PATH):
        age_hours = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(CURRENT_PRED_PATH), unit="s")).total_seconds() / 3600
        needs_refresh = age_hours > 6

    if needs_refresh:
        t = threading.Thread(target=_run_regime_model_background, daemon=True)
        t.start()

    st.session_state["_model_thread_started"] = True

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ESI — AI Chat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  .stApp {
    background: #0a0c10;
    color: #cdd9e5;
  }

  section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
  }

  /* ── Chat messages ── */
  .msg-wrap {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-bottom: 18px;
  }

  .msg-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
    opacity: 0.5;
  }

  .msg-user .msg-label { color: #58a6ff; text-align: right; }
  .msg-ai   .msg-label { color: #3fb950; }

  .msg-bubble {
    padding: 12px 18px;
    border-radius: 2px;
    font-size: 14px;
    line-height: 1.7;
    max-width: 84%;
    white-space: pre-wrap;
  }

  .msg-user {
    align-items: flex-end;
  }
  .msg-user .msg-bubble {
    background: #1a2332;
    border: 1px solid #1f4068;
    color: #cdd9e5;
    border-radius: 8px 8px 2px 8px;
  }

  .msg-ai {
    align-items: flex-start;
  }
  .msg-ai .msg-bubble {
    background: #111820;
    border: 1px solid #21262d;
    color: #cdd9e5;
    border-radius: 2px 8px 8px 8px;
  }

  /* ── Sidebar elements ── */
  .api-hint {
    background: #111820;
    border: 1px solid #1f3a5f;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 12px;
    color: #58a6ff;
    line-height: 1.6;
    margin-bottom: 10px;
  }

  .data-chip {
    display: inline-block;
    background: #1a2332;
    border: 1px solid #21262d;
    color: #3fb950;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px 2px 2px 0;
  }
  .data-chip.missing {
    color: #f78166;
    border-color: #3d1f1f;
    background: #1a1010;
  }

  .snap-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8b949e;
    padding: 2px 0;
    border-bottom: 1px solid #1a1f2a;
  }
  .snap-key { color: #58a6ff; }
  .snap-val { color: #e6edf3; }

  /* ── Suggestions ── */
  .stButton > button {
    background: #111820 !important;
    border: 1px solid #21262d !important;
    color: #8b949e !important;
    font-size: 12px !important;
    border-radius: 6px !important;
    text-align: left !important;
    padding: 8px 12px !important;
    transition: border-color 0.15s, color 0.15s !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
  }
  .stButton > button:hover {
    border-color: #58a6ff !important;
    color: #cdd9e5 !important;
  }

  /* Send button override */
  div[data-testid="stFormSubmitButton"] > button {
    background: #1a3a5c !important;
    border: 1px solid #58a6ff !important;
    color: #58a6ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    border-radius: 6px !important;
    padding: 8px 24px !important;
    transition: background 0.15s !important;
  }
  div[data-testid="stFormSubmitButton"] > button:hover {
    background: #1f4a7a !important;
  }

  /* Text input */
  .stTextInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #cdd9e5 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.12) !important;
  }

  /* Header bar */
  .header-bar {
    display: flex;
    align-items: baseline;
    gap: 14px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 14px;
    margin-bottom: 20px;
  }
  .header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.02em;
  }
  .header-sub {
    font-size: 13px;
    color: #8b949e;
  }

  /* Badge */
  .badge {
    font-family: 'IBM Plex Mono', monospace;
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-ok  { background: #1a2d1a; color: #3fb950; border: 1px solid #2ea043; }
  .badge-err { background: #2d1a1a; color: #f78166; border: 1px solid #da3633; }

  /* Divider */
  hr { border-color: #21262d !important; }

  /* Scrollable chat area */
  .chat-scroll { max-height: 62vh; overflow-y: auto; padding-right: 4px; }

  /* Suggestion area label */
  .suggest-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #8b949e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    out = {}

    def _try(name, path):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                out[name] = df
            except Exception:
                pass

    _try("features",         os.path.join(DATA_PROC,   "fred_features.csv"))
    _try("engines",          os.path.join(DATA_PROC,   "engine_scores.csv"))
    _try("dfm",              os.path.join(DATA_PROC,   "dfm_factors.csv"))
    _try("regimes",          os.path.join(DATA_PROC,   "regimes.csv"))
    _try("backtest",         os.path.join(DATA_PROC,   "backtest_results.csv"))
    _try("engine_momentum",  os.path.join(DATA_PROC,   "engine_momentum.csv"))
    # All results tables
    _try("stress_tbl",       os.path.join(RESULTS_TAB, "04_stress_index.csv"))
    _try("ml_metrics",       os.path.join(RESULTS_TAB, "05_model_performance.csv"))
    _try("feat_importance",  os.path.join(RESULTS_TAB, "06_feature_importance.csv"))
    _try("backtest_tbl",     os.path.join(RESULTS_TAB, "backtest_results.csv"))
    _try("backtest_summary", os.path.join(RESULTS_TAB, "T4_backtest_summary.csv"))
    _try("regime_summary",   os.path.join(RESULTS_TAB, "T3_regime_summary.csv"))
    _try("regime_cond_perf", os.path.join(RESULTS_TAB, "T6_regime_conditioned_performance.csv"))
    _try("panel_perf",       os.path.join(RESULTS_TAB, "T5_panel_model_performance.csv"))
    _try("engine_scores_tbl",  os.path.join(RESULTS_TAB, "T1_engine_scores.csv"))
    # Live regime-conditioned prediction — written by background thread
    _try("current_prediction", CURRENT_PRED_PATH)

    return out


def get_latest_summary(data: dict) -> dict:
    snap = {}
    engines_df = data.get("engines")
    if engines_df is not None:
        last = engines_df.dropna(how="all").iloc[-1]
        snap["date"] = str(last.name.date() if hasattr(last.name, "date") else last.name)
        for eng in ["Inflation", "Labour", "Financial", "Monetary", "Real"]:
            if eng in last.index:
                snap[f"engine_{eng.lower()}"] = round(float(last[eng]), 1)
        if "ESI" in last.index:
            esi_val = float(last["ESI"])
            snap["esi"] = round(esi_val, 1)
            for lbl, (lo, hi) in STRESS_LABELS.items():
                if lo <= esi_val < hi:
                    snap["stress_level"] = lbl
                    break
        if "RECESSION" in last.index:
            snap["in_recession"] = bool(last["RECESSION"] == 1)

    # ── Recession probability ────────────────────────────────────────────────
    # Priority 1: current_prediction.csv written by the dashboard's background runner
    #             (or by the pipeline itself after our patch to regime_conditioned_ml.py)
    rec_prob = None
    cur_pred = data.get("current_prediction")
    if cur_pred is not None and "prob_recession_6m" in cur_pred.columns:
        val = cur_pred["prob_recession_6m"].dropna()
        if len(val):
            candidate = float(val.iloc[-1])
            if candidate <= 1.0:
                candidate *= 100
            if candidate > 0:
                rec_prob = round(candidate, 1)
                snap["recession_prob_source"] = "current_prediction.csv (regime-conditioned live)"

    # Priority 2: backtest_results.csv — last NON-ZERO forward probability
    if rec_prob is None or rec_prob == 0.0:
        for key in ("backtest", "backtest_tbl"):
            bt = data.get(key)
            if bt is not None:
                prob_cols = [c for c in bt.columns if "prob" in c.lower() and "forward" in c.lower()]
                if not prob_cols:
                    prob_cols = [c for c in bt.columns if "prob" in c.lower()]
                for col in prob_cols:
                    nonzero = bt[col].dropna()
                    nonzero = nonzero[nonzero > 0]
                    if len(nonzero):
                        rec_prob = round(float(nonzero.iloc[-1]) * 100, 1)
                        snap["recession_prob_source"] = f"{key}/{col} (last non-zero)"
                        break
                if rec_prob is not None and rec_prob != 0.0:
                    break

    if rec_prob is not None:
        snap["recession_probability_6m"] = rec_prob

    # ── Current regime ───────────────────────────────────────────────────────
    if cur_pred is not None and "regime" in cur_pred.columns:
        regime_val = cur_pred["regime"].dropna()
        if len(regime_val):
            snap["current_regime"] = str(regime_val.iloc[-1])
            snap["regime_model_used"] = str(cur_pred.get("model_used", pd.Series(["?"])).iloc[-1]) if "model_used" in cur_pred.columns else "regime-conditioned"

    if "current_regime" not in snap:
        reg = data.get("regimes")
        if reg is not None:
            last_reg = reg.dropna(how="all").iloc[-1]
            for col in reg.columns:
                if "label" in col.lower():
                    snap["current_regime"] = str(last_reg[col])
                    break
                elif "regime" in col.lower():
                    snap["current_regime"] = str(last_reg[col])

    return snap


def build_ai_context(data: dict, snap: dict) -> str:
    lines = [
        "You are an expert macroeconomic analyst assistant embedded in the ESI (Economic Stress Index) dashboard.",
        "The ESI is a 0–100 composite index built from 5 economic engines: Inflation, Labour, Financial, Monetary, Real.",
        "Higher scores indicate more economic stress. Thresholds: Low 0–30, Moderate 30–60, High 60–80, Extreme 80–100.",
        "",
        "=== PROJECT DATA SCIENCE MODEL ===",
        "- The pipeline uses FRED data (18 macro series) to compute engine scores via expanding/rolling z-scores.",
        "- A Dynamic Factor Model (DFM, 2 factors) extracts latent common factors across engines.",
        "- A Gaussian Mixture Model (6 regimes) labels the current stress regime.",
        "- ML models: XGBoost + LightGBM + Logistic Regression trained to predict recession 6 months ahead.",
        "- International panel (7 countries) used to boost training sample to ~45 recession episodes.",
        "- Regime-conditioned ML trains separate models per regime type for better precision.",
        "",
        "=== CURRENT STATE SNAPSHOT ===",
    ]
    for k, v in snap.items():
        lines.append(f"  {k}: {v}")

    eng = data.get("engines")
    if eng is not None:
        cols = ["ESI"] + [c for c in ["Inflation", "Labour", "Financial", "Monetary", "Real"] if c in eng.columns]
        recent = eng[cols].tail(6)
        lines += ["", "=== RECENT 6-MONTH ENGINE HISTORY ==="]
        lines.append(recent.round(1).to_string())

    mm = data.get("ml_metrics")
    if mm is not None:
        lines += ["", "=== ML MODEL PERFORMANCE ==="]
        lines.append(mm.round(3).to_string())

    # Backtest — prefer data/processed version, fall back to results table
    bt = data.get("backtest") if data.get("backtest") is not None else data.get("backtest_tbl")
    if bt is not None:
        # Show last non-trivial rows (skip trailing zeros in prob columns)
        prob_cols = [c for c in bt.columns if "prob" in c.lower()]
        if prob_cols:
            bt_nonzero = bt[bt[prob_cols[0]].fillna(0) > 0]
            bt_show = bt_nonzero.tail(12) if len(bt_nonzero) else bt.tail(12)
        else:
            bt_show = bt.tail(12)
        lines += ["", "=== BACKTEST RESULTS (last 12 non-zero rows) ==="]
        lines.append(bt_show.round(3).to_string())

    # Regime-conditioned performance and current prediction
    rc = data.get("regime_cond_perf")
    if rc is not None:
        lines += ["", "=== REGIME-CONDITIONED MODEL PERFORMANCE ==="]
        lines.append(rc.round(3).to_string())

    # Backtest summary table
    bs = data.get("backtest_summary")
    if bs is not None:
        lines += ["", "=== BACKTEST SUMMARY ==="]
        lines.append(bs.round(3).to_string())

    # Feature importance
    fi = data.get("feat_importance")
    if fi is not None:
        lines += ["", "=== FEATURE IMPORTANCE (top 10) ==="]
        lines.append(fi.head(10).round(4).to_string())

    # Engine momentum / transmission lags
    em = data.get("engine_momentum")
    if em is not None:
        lines += ["", "=== ENGINE MOMENTUM ==="]
        lines.append(em.round(3).to_string())

    # Live current prediction (regime-conditioned, freshest source)
    cp = data.get("current_prediction")
    if cp is not None:
        lines += ["", "=== LIVE REGIME-CONDITIONED PREDICTION (PRIMARY SOURCE) ==="]
        lines.append(cp.to_string(index=False))
        lines += [
            "NOTE: This is the PRIMARY recession probability. It is produced by the",
            "regime-conditioned logistic regression model run live on the current engine scores.",
            "Always cite this figure when asked about recession probability.",
        ]
    else:
        lines += [
            "",
            "=== RECESSION PROBABILITY NOTE ===",
            "current_prediction.csv not yet available — background model is still computing.",
            "The regime-conditioned model (Step 14) estimated ~49.6% for March 2026.",
            "This is the best available figure until the live file is written.",
        ]

    # ── Live news headlines ──────────────────────────────────────────────────
    try:
        rss = fetch_rss_headlines(max_per_source=4)
        if rss:
            lines.append(headlines_to_context_text(rss))
    except Exception:
        pass

    lines += [
        "",
        "=== YOUR ROLE ===",
        "Answer user questions about:",
        "  1. Current economic stress levels and what's driving them.",
        "  2. Recession risk forecasts (use the ML model outputs as primary source).",
        "  3. Historical comparisons (e.g. how does today compare to 2008?).",
        "  4. What each engine and indicator means.",
        "  5. Regime identification and what to expect given the current regime.",
        "  6. Forward-looking analysis using the model's 6-month horizon predictions.",
        "Be concise, data-driven, and professional. Reference specific numbers from the snapshot.",
        "If data is unavailable, say so clearly. Do not hallucinate data not present above.",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM
# ═══════════════════════════════════════════════════════════════════════════════
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.3,
    )


def chat_with_context(llm, history: list, user_msg: str, system_ctx: str) -> str:
    messages = [SystemMessage(content=system_ctx)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=user_msg))
    response = llm.invoke(messages)
    return response.content


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
# Kick off background model refresh (non-blocking)
maybe_refresh_prediction()

with st.spinner("Loading project data…"):
    DATA = load_data()

SNAP = get_latest_summary(DATA)
AI_CONTEXT = build_ai_context(DATA, SNAP)

EXPECTED_FILES = {
    "engines":      os.path.join(DATA_PROC,   "engine_scores.csv"),
    "features":     os.path.join(DATA_PROC,   "fred_features.csv"),
    "regimes":      os.path.join(DATA_PROC,   "regimes.csv"),
    "backtest":     os.path.join(DATA_PROC,   "backtest_results.csv"),
    "ml_metrics":   os.path.join(RESULTS_TAB, "ml_metrics.csv"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 ESI Intelligence")
    st.markdown("---")

    st.markdown("#### 🔑 Gemini API Key")
    st.markdown(
        '<div class="api-hint">'
        'Get a free key at<br>'
        '<a href="https://aistudio.google.com/app/apikey" target="_blank" '
        'style="color:#79c0ff;">aistudio.google.com/app/apikey</a>'
        '</div>',
        unsafe_allow_html=True,
    )
    gemini_key = st.text_input(
        "API Key",
        type="password",
        placeholder="AIza...",
        key="gemini_api_key",
        label_visibility="collapsed",
    )
    if gemini_key:
        st.markdown('<span class="badge badge-ok">✓ Key Set</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-err">No Key</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Loaded data status ──
    st.markdown("#### 📂 Data Status")
    for name, path in EXPECTED_FILES.items():
        loaded = name in DATA or (name == "backtest" and "backtest_tbl" in DATA)
        label_cls = "data-chip" if loaded else "data-chip missing"
        icon = "✓" if loaded else "✗"
        st.markdown(f'<span class="{label_cls}">{icon} {name}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Snapshot peek ──
    if SNAP:
        st.markdown("#### 📍 Live Snapshot")
        for k, v in SNAP.items():
            st.markdown(
                f'<div class="snap-row">'
                f'<span class="snap-key">{k}</span>: <span class="snap-val">{v}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("<small style='color:#8b949e;'>ESI v5 · Gemini 2.5 Flash · LangChain</small>",
                unsafe_allow_html=True)

    # ── RSS News Notification Bell ───────────────────────────────────────────
    st.markdown("---")

    # Fetch headlines (cached via session state to avoid refetch on every rerun)
    if "rss_headlines" not in st.session_state:
        try:
            st.session_state["rss_headlines"] = fetch_rss_headlines(max_per_source=5)
        except Exception:
            st.session_state["rss_headlines"] = []

    rss_items = st.session_state.get("rss_headlines", [])
    n_items   = len(rss_items)

    # Bell button with count badge
    bell_label = f"🔔 Economic News  **({n_items})**" if n_items else "🔔 Economic News  **(0)**"
    if st.button(bell_label, key="rss_bell_toggle", use_container_width=True):
        st.session_state["rss_open"] = not st.session_state.get("rss_open", False)

    # Refresh button
    if st.button("↺ Refresh feeds", key="rss_refresh", use_container_width=True):
        try:
            st.session_state["rss_headlines"] = fetch_rss_headlines(max_per_source=5)
        except Exception:
            st.session_state["rss_headlines"] = []
        st.session_state["rss_open"] = True
        st.rerun()

    # Expandable news list — only shown when open
    if st.session_state.get("rss_open", False) and rss_items:
        st.markdown(
            "<div style='margin-top:6px; max-height:340px; overflow-y:auto;'>",
            unsafe_allow_html=True,
        )
        current_source = None
        for item in rss_items:
            if item["source"] != current_source:
                current_source = item["source"]
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace; font-size:9px; "
                    f"color:#58a6ff; letter-spacing:0.1em; text-transform:uppercase; "
                    f"margin-top:10px; margin-bottom:3px;'>{current_source}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='font-size:11px; padding:4px 0; "
                f"border-bottom:1px solid #1a1f2a; line-height:1.5;'>"
                f"<a href='{item['link']}' target='_blank' "
                f"style='color:#cdd9e5; text-decoration:none;'>{item['title']}</a>"
                f"{'<br><span style=\"color:#8b949e; font-size:10px;\">' + item['summary'] + '</span>' if item['summary'] else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.get("rss_open", False) and not rss_items:
        st.caption("No headlines available. Check your internet connection.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="header-bar">'
    '<span class="header-title">ESI // AI Chat</span>'
    '<span class="header-sub">Economic Stress Index · Gemini 2.5 Flash · LangChain</span>'
    '</div>',
    unsafe_allow_html=True,
)

if not gemini_key:
    st.info("🔑 Enter your Gemini API key in the sidebar to start chatting.", icon="💡")
    st.stop()

# ── Initialise LLM + session state ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if ("llm_instance" not in st.session_state
        or st.session_state.get("llm_key") != gemini_key):
    try:
        st.session_state.llm_instance = get_llm(gemini_key)
        st.session_state.llm_key = gemini_key
        st.session_state.llm_error = None
    except Exception as e:
        st.session_state.llm_error = str(e)
        st.session_state.llm_instance = None

if st.session_state.get("llm_error"):
    st.error(f"❌ Failed to initialise Gemini: {st.session_state.llm_error}")
    st.stop()

# ── Starter suggestions (always visible, acts as quick-fire buttons) ──
STARTERS = [
    "What is the current ESI and stress level?",
    "Which engine is driving stress most right now?",
    "What's the 6-month recession probability?",
    "Compare current stress to 2008.",
    "What regime are we in and what does it imply?",
    "What indicators should I watch most closely?",
]

st.markdown('<div class="suggest-label">Suggested questions</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, q in enumerate(STARTERS):
    with cols[i % 3]:
        if st.button(q, key=f"s_{i}", use_container_width=True):
            st.session_state.pending_question = q

# ── Render chat history using native Streamlit chat components ──
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# ── Resolve input: pending question from button OR typed input ──
# st.chat_input must always be called (Streamlit requirement), then we pick
# whichever source has content — button press takes priority.
typed_input = st.chat_input("Ask about ESI, recession risk, regimes, engines…")
user_input = st.session_state.pop("pending_question", None) or typed_input

if user_input and user_input.strip():
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input.strip())

    # Get and show AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = chat_with_context(
                    st.session_state.llm_instance,
                    st.session_state.chat_history,
                    user_input.strip(),
                    AI_CONTEXT,
                )
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"❌ Gemini API error: {e}")

# ── Clear button ──
if st.session_state.chat_history:
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()