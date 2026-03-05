"""
src/features/engineer_features.py  —  v5
-----------------------------------------
All features signed: HIGHER = MORE STRESS.
New in v5:
  - building_permits      : housing leading indicator (negated: decline=stress)
  - yield_spread_3m       : 10Y-3M spread (negated: inversion=stress)
  - new_orders_growth     : ISM new orders proxy (negated: decline=stress)
  - building_permits_growth: same as permits but YoY% for Real engine
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_RAW, DATA_PROC, adaptive_min, ROLLING_ZSCORE_WINDOW


def yoy(series):
    return series.pct_change(12, fill_method=None) * 100


def expanding_zscore_single(series, min_periods=None):
    n = series.notna().sum()
    mp = adaptive_min(36, n) if min_periods is None else min_periods
    mu  = series.expanding(min_periods=mp).mean()
    sig = series.expanding(min_periods=mp).std().replace(0, np.nan)
    return (series - mu) / sig


def main():
    raw_path = os.path.join(DATA_RAW, "fred_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Run download_fred.py first: {raw_path}")

    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df  = pd.DataFrame(index=raw.index)

    # ── Inflation ─────────────────────────────────────────────────────────────
    df["inflation_yoy"]      = yoy(raw["CPI"])
    df["core_inflation_yoy"] = yoy(raw["CORE_CPI"])
    df["infl_exp"]           = raw["INFL_EXP"]

    # ── Labour ────────────────────────────────────────────────────────────────
    df["unemployment_change"] = raw["UNRATE"].diff(12)
    df["payroll_growth"]      = -yoy(raw["PAYEMS"])
    log_icsa = np.log(raw["ICSA"].replace(0, np.nan))
    ma_icsa  = log_icsa.rolling(3, min_periods=2).mean()
    # Adaptive long-run trend window: use full available length or 120m max
    n_icsa = int(ma_icsa.notna().sum())
    trend_window = min(120, max(24, n_icsa))
    trend_min = adaptive_min(trend_window, n_icsa)
    df["jobless_claims_ma"]   = ma_icsa - ma_icsa.rolling(trend_window, min_periods=trend_min).mean()
    # Building permits: housing starts lead labour market by 6-12 months
    if "PERMIT" in raw.columns:
        df["building_permits"]  = -yoy(raw["PERMIT"])   # negate: permits falling = stress

    # ── Monetary ──────────────────────────────────────────────────────────────
    df["real_rate"]           = raw["FED_FUNDS"] - df["inflation_yoy"]
    if "YIELD_SPREAD" in raw.columns:
        df["yield_spread"]    = -raw["YIELD_SPREAD"]       # 10Y-2Y negated
    else:
        df["yield_spread"]    = -(raw["T10Y"] - raw["T2Y"])
    df["fed_funds_change"]    = raw["FED_FUNDS"].diff(12)
    # 10Y-3M spread: empirically stronger recession predictor than 10Y-2Y
    if "T10Y3M" in raw.columns:
        df["yield_spread_3m"] = -raw["T10Y3M"]             # negated: inversion=stress
    elif "T10Y" in raw.columns and "FED_FUNDS" in raw.columns:
        # Approximate: 3-month rate ≈ Fed Funds (close enough before T10Y3M available)
        df["yield_spread_3m"] = -(raw["T10Y"] - raw["FED_FUNDS"])

    # ── Financial ─────────────────────────────────────────────────────────────
    df["credit_spread"]       = raw["CREDIT_SPREAD"]
    df["fsi"]                 = raw["FSI"]
    if "VIX" in raw.columns:
        df["vix_zscore"]      = expanding_zscore_single(raw["VIX"])
    if "TED" in raw.columns:
        df["ted_spread"]      = raw["TED"]

    # ── Real economy ──────────────────────────────────────────────────────────
    df["indpro_growth"]       = -yoy(raw["INDPRO"])
    df["m2_growth"]           = yoy(raw["M2"])
    if "RETAIL" in raw.columns:
        df["retail_growth"]   = -yoy(raw["RETAIL"])
    # ISM new orders: leads industrial production by 2-4 months
    if "NAPMNOI" in raw.columns:
        df["new_orders_growth"] = -raw["NAPMNOI"].diff(12)  # change in ISM level; neg=stress
    # Building permits also enters Real engine as YoY%
    if "PERMIT" in raw.columns:
        df["building_permits_growth"] = -yoy(raw["PERMIT"])

    # ── Fast-stress signals (unsmoothed, for Financial engine) ────────────────
    # These capture sudden BREAKS that YoY-smoothed signals miss.
    # credit_spread_chg: 3-month acceleration in credit spreads (stress = widening fast)
    if "CREDIT_SPREAD" in raw.columns:
        df["credit_spread_chg"]  = raw["CREDIT_SPREAD"].diff(3)      # 3m change
    # yield_curve_chg: 3-month change in yield curve (sudden inversion = stress)
    if "YIELD_SPREAD" in raw.columns:
        df["yield_curve_chg"]    = -raw["YIELD_SPREAD"].diff(3)      # negated: inversion=stress
    elif "T10Y" in raw.columns and "T2Y" in raw.columns:
        df["yield_curve_chg"]    = -(raw["T10Y"] - raw["T2Y"]).diff(3)
    # vix_chg: 1-month VIX spike (sudden fear = stress)
    if "VIX" in raw.columns:
        df["vix_chg"]            = raw["VIX"].diff(1).clip(lower=0)  # only spikes (up moves)

    # ── Recession label ───────────────────────────────────────────────────────
    df["RECESSION"]           = raw["RECESSION"]

    df = df.iloc[13:]
    os.makedirs(DATA_PROC, exist_ok=True)
    out_path = os.path.join(DATA_PROC, "fred_features.csv")
    df.to_csv(out_path)

    print(f"\n{'='*64}")
    print(f"  Features saved  →  {out_path}")
    print(f"  Shape           :  {df.shape}")
    print(f"  Date range      :  {df.index[0].date()}  to  {df.index[-1].date()}")
    print(f"\n  Feature coverage:")
    for col in [c for c in df.columns if c != "RECESSION"]:
        s = df[col]; first = s.first_valid_index(); n = s.notna().sum()
        tail = s.loc[first:].isnull().sum() if first else 0
        print(f"    {col:<30}  {n:>4}/{len(s)}  "
              f"from {str(first.date()) if first else 'N/A':>12}  "
              f"{tail} gaps")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
