"""
config.py
---------
Central configuration for the Economic Stress Index project.
Single source of truth for all FRED codes, engine definitions, and parameters.
"""

# ── FRED API Key ──────────────────────────────────────────────────────────────
import os as _os
FRED_API_KEY = _os.environ.get("FRED_API_KEY", "588e1d7035224fbfcd6ac5460aaed780")

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "1970-01-01"
END_DATE   = None   # None = today

# ── FRED Series ───────────────────────────────────────────────────────────────
FRED_SERIES = {
    # Inflation
    "CPI"          : "CPIAUCSL",   # Consumer Price Index
    "CORE_CPI"     : "CPILFESL",   # Core CPI (ex food & energy)
    "INFL_EXP"     : "T5YIE",      # 5Y Breakeven Inflation (daily→monthly)
    # Labour
    "UNRATE"       : "UNRATE",     # Unemployment Rate
    "PAYEMS"       : "PAYEMS",     # Nonfarm Payroll Employment
    "ICSA"         : "ICSA",       # Initial Jobless Claims (weekly)
    # Interest rates
    "FED_FUNDS"    : "FEDFUNDS",   # Federal Funds Rate
    "T10Y"         : "GS10",       # 10-Year Treasury
    "T2Y"          : "GS2",        # 2-Year Treasury
    "YIELD_SPREAD" : "T10Y2Y",     # 10Y-2Y Spread (daily→monthly)
    # Credit / Financial stress
    "CREDIT_SPREAD": "BAA10Y",     # Moody's BAA Corporate Spread (daily→monthly)
    "FSI"          : "KCFSI",      # Kansas City Financial Stress Index
    "VIX"          : "VIXCLS",     # CBOE Volatility Index (daily→monthly, from 1990)
    "TED"          : "TEDRATE",    # TED Spread 3m LIBOR-T-bill (daily→monthly, ends 2023)
    # Real economy
    "INDPRO"       : "INDPRO",     # Industrial Production
    "RETAIL"       : "RSAFS",      # Retail & Food Services Sales (from 1992)
    "M2"           : "M2SL",       # M2 Money Supply
    # Recession label
    "RECESSION"    : "USREC",      # NBER Recession Indicator
}

# Weekly series → monthly mean; Daily series → monthly mean
WEEKLY_SERIES = {"ICSA"}
DAILY_SERIES  = {"INFL_EXP", "YIELD_SPREAD", "CREDIT_SPREAD", "VIX", "TED"}

# ── Custom FSI construction ───────────────────────────────────────────────────
# Features fed into PCA to build our own KCFSI-style index
# Each is a financial market stress signal with known recession-leading properties
CUSTOM_FSI_FEATURES = [
    "credit_spread",   # Corporate risk premium (BAA-10Y)
    "vix_zscore",      # Equity market fear (expanding z-score of VIX)
    "ted_spread",      # Interbank funding stress (3m LIBOR - 3m T-bill)
    "yield_spread",    # Term structure stress (-(10Y-2Y)), negated
]

# ── Economic Engine definitions (NO variable appears in more than one engine) ─
# Variables listed are the engineered features (from fred_features.csv)
ENGINE_FEATURES = {
    "Inflation" : ["inflation_yoy", "core_inflation_yoy", "infl_exp"],
    "Labour"    : ["unemployment_change", "payroll_growth", "jobless_claims_ma"],
    "Financial" : ["fsi", "custom_fsi", "credit_spread_chg", "vix_chg"],
    "Monetary"  : ["real_rate", "yield_spread", "fed_funds_change", "yield_curve_chg"],
    "Real"      : ["indpro_growth", "retail_growth", "m2_growth"],
}

ENGINE_COLORS = {
    "Inflation" : "#e6550d",   # orange-red
    "Labour"    : "#756bb1",   # purple
    "Financial" : "#31a354",   # green
    "Monetary"  : "#3182bd",   # blue
    "Real"      : "#d62728",   # red
}

# ── Stress Index ──────────────────────────────────────────────────────────────
STRESS_LABELS = {
    "Low"      : (0,  30),
    "Moderate" : (30, 60),
    "High"     : (60, 80),
    "Extreme"  : (80, 100),
}

# ── Dynamic Factor Model ──────────────────────────────────────────────────────
DFM_N_FACTORS    = 2    # must be < n_engines (5); 2 follows Brave & Butters (2012) / Chicago Fed NFCI convention
DFM_FACTOR_ORDER = 1    # AR(1) dynamics for each factor
DFM_START        = "1996-01-01"   # after sufficient data for all engines

# ── Stress Regime Detection ───────────────────────────────────────────────────
N_REGIMES = 6
REGIME_NAMES = {
    # Assigned programmatically based on dominant engine in centroid
    # Fallback names if auto-assignment fails:
    0: "Normal Growth",
    1: "Mild Stress",
    2: "Inflation Shock",
    3: "Labour Recession",
    4: "Financial Crisis",
    5: "Stagflation / Combined",
}

# ── ML / Backtest ─────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
CV_FOLDS              = 5
FORECAST_HORIZON      = 6     # months ahead for early warning
BACKTEST_START        = "1996-01-01"
BACKTEST_MIN_TRAIN_M  = 72    # months of history before first OOS prediction
BACKTEST_THRESHOLD    = 0.40  # probability threshold to flag "recession warning"

# ── PCA (legacy, still used inside engines) ──────────────────────────────────
N_PCA_COMPONENTS = 4

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC   = os.path.join(BASE_DIR, "data", "processed")
RESULTS_FIG = os.path.join(BASE_DIR, "results", "figures")
RESULTS_TAB = os.path.join(BASE_DIR, "results", "tables")

# ── Legacy: stress features list (used by EDA module) ────────────────────────
STRESS_FEATURES = [
    "inflation_yoy", "core_inflation_yoy", "infl_exp",
    "unemployment_change", "payroll_growth", "jobless_claims_ma",
    "fsi", "custom_fsi",
    "real_rate", "yield_spread", "fed_funds_change",
    "indpro_growth", "retail_growth", "m2_growth",
    "credit_spread", "vix_zscore", "ted_spread",
    # Fast-stress signals (v6): detect sudden breaks
    "credit_spread_chg", "yield_curve_chg", "vix_chg",
]

# ── v5 normalization windows ──────────────────────────────────────────────────
ROLLING_ZSCORE_WINDOW   = 120   # 10 years: rolling z-score window
ROLLING_PCTILE_WINDOW   = 240   # 20 years: rolling percentile window
CUSTOM_FSI_PCA_WINDOW   = 180   # 15 years: rolling PCA window for custom FSI (legacy, unused)
ESI_DISPERSION_LAMBDA   = 0.5   # ESI = mean(engines) + λ*std(engines)

# ── Adaptive min_periods ───────────────────────────────────────────────────────
# Replaces all hardcoded min_periods throughout the pipeline.
# Design: use 40% of window as baseline, never below ABS_MIN_PERIODS (12m = 1yr).
# When actual available data n < baseline, fall back to 30% of n.
# This allows the system to produce scores from the very first year of data
# instead of silently dropping the first 3–10 years.
ABS_MIN_PERIODS = 12          # absolute floor: never require fewer than 1 year
ADAPTIVE_PCT    = 0.40        # baseline: 40% of window
ADAPTIVE_FALLBACK_PCT = 0.30  # fallback when data < baseline: 30% of available


def adaptive_min(window: int, n_available: int = None) -> int:
    """
    Compute min_periods dynamically.

    Parameters
    ----------
    window      : rolling / expanding window length
    n_available : actual non-NaN observations available (optional)

    Returns
    -------
    int  — min_periods to use

    Logic
    -----
    1. baseline = max(ABS_MIN_PERIODS, window * ADAPTIVE_PCT)
    2. if n_available < baseline → max(ABS_MIN_PERIODS, n_available * ADAPTIVE_FALLBACK_PCT)
    3. else → baseline
    """
    baseline = max(ABS_MIN_PERIODS, int(window * ADAPTIVE_PCT))
    if n_available is not None and n_available < baseline:
        return max(ABS_MIN_PERIODS, int(n_available * ADAPTIVE_FALLBACK_PCT))
    return baseline

# ── Leading / Coincident split within engines ─────────────────────────────────
# Each engine variable tagged as 'leading' (L) or 'coincident' (C)
# Engine score = 0.6 * mean(leading z-scores) + 0.4 * mean(coincident z-scores)
ENGINE_LEAD_WEIGHTS  = {"leading": 0.6, "coincident": 0.4}

ENGINE_VARIABLE_TYPES = {
    # Inflation engine
    "infl_exp"           : "leading",    # forward-looking market expectation
    "inflation_yoy"      : "coincident",
    "core_inflation_yoy" : "coincident",
    # Labour engine
    "jobless_claims_ma"  : "leading",    # claims lead unemployment by 1-3m
    "building_permits"   : "leading",    # housing leads labour by 6-12m
    "unemployment_change": "coincident",
    "payroll_growth"     : "coincident",
    # Financial engine
    "credit_spread"      : "leading",    # credit markets price risk early
    "vix_zscore"         : "leading",
    "yield_spread_3m"    : "leading",    # T10Y-T3M stronger recession predictor
    "fsi"                : "coincident",
    "custom_fsi"         : "coincident",
    "ted_spread"         : "leading",
    # Financial engine — fast-stress (leading: sudden breaks precede slow indices)
    "credit_spread_chg"  : "leading",    # 3m credit spread acceleration
    "yield_curve_chg"    : "leading",    # 3m yield curve change (inversion speed)
    "vix_chg"            : "leading",    # 1m VIX spike (immediate fear signal)
    # Monetary engine
    "yield_spread"       : "leading",    # 10Y-2Y inverts before recessions
    "fed_funds_change"   : "leading",    # rate trajectory leads
    "real_rate"          : "coincident",
    # Real engine
    "new_orders_growth"  : "leading",    # ISM new orders leads IP by 2-4m
    "building_permits_growth": "leading",
    "indpro_growth"      : "coincident",
    "retail_growth"      : "coincident",
    "m2_growth"          : "coincident",
}

# ── New FRED series for v5 ────────────────────────────────────────────────────
FRED_SERIES_V5 = {
    "PERMIT"   : "PERMIT",     # Building permits — housing leading indicator
    "T10Y3M"   : "T10Y3M",     # 10Y-3M Treasury spread (daily→monthly)
    "NAPMNOI"  : "NAPMOI",     # ISM Manufacturing New Orders Index (monthly)
}
# Note: NAPMNOI ends 2023; use ISMMAN as alternative. Both will be tried.
