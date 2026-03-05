"""
src/features/engineer_panel_features.py
-----------------------------------------
Builds country-level engine scores from the OECD panel data.

For each country, the same 5-engine structure is applied:
  Inflation : CPI YoY
  Labour    : unemployment change, employment growth (from IP proxy)
  Financial : CLI deviation from trend, yield spread
  Monetary  : real rate (IR3M - CPI YoY), yield spread (IRLT - IR3M)
  Real      : industrial production YoY

Variables available from OECD vs US-specific FRED:
──────────────────────────────────────────────────────────────────
Engine      US variables          International equivalent
─────────────────────────────────────────────────────────────────
Inflation   inflation_yoy         {C}_cpi_yoy
            core_inflation_yoy    (not harmonised → use CPI only)
            infl_exp              (not available internationally)
Labour      unemployment_change   {C}_unrate_change
            payroll_growth        (use IP as proxy for employment)
            jobless_claims_ma     (not available — omit)
Financial   fsi (KCFSI)           {C}_cli_stress (CLI deviation)
            custom_fsi            (not available — use CLI only)
            credit_spread         (BIS data, advanced feature)
Monetary    real_rate             {C}_real_rate
            yield_spread          {C}_yield_spread (IRLT - IR3M)
            fed_funds_change      {C}_rate_change
Real        indpro_growth         {C}_ip_yoy
            retail_growth         (available for some countries)
            m2_growth             (not harmonised — omit)
─────────────────────────────────────────────────────────────────

CLI stress note:
  The OECD CLI is amplitude-adjusted and centred at 100.
  CLI < 100: below trend (expansion) → stress = -(CLI - 100)
  So cli_stress = 100 - CLI: positive when economy below trend.
  This is the OECD's own recession leading indicator, repurposed
  as a financial stress proxy in the international panel.

Usage:
    python src/features/engineer_panel_features.py

Output:
    data/processed/panel_features.csv   (all countries, all engines, long format)
    data/processed/panel_engine_scores.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_RAW, DATA_PROC, adaptive_min

COUNTRIES = ["USA", "GBR", "DEU", "CAN", "JPN", "FRA", "AUS"]


def yoy(s):
    return s.pct_change(12, fill_method=None) * 100


def rolling_zscore(s, window=120, min_periods=None):
    n  = int(s.notna().sum())
    mp = adaptive_min(window, n) if min_periods is None else min_periods
    mu  = s.rolling(window=window, min_periods=mp).mean()
    sig = s.rolling(window=window, min_periods=mp).std().replace(0, np.nan)
    return (s - mu) / sig


def expanding_impute_median(s):
    vals = s.values.copy().astype(float)
    out  = vals.copy()
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            hist = vals[:i][~np.isnan(vals[:i])]
            if len(hist) > 0:
                out[i] = float(np.median(hist))
    return pd.Series(out, index=s.index, name=s.name)


def rolling_percentile_rank(s, window=240, min_periods=None):
    n     = int(s.notna().sum())
    mp    = adaptive_min(window, n) if min_periods is None else min_periods
    vals  = s.values.copy().astype(float)
    ranks = np.full(len(vals), np.nan)
    for i in range(mp, len(vals)):
        if np.isnan(vals[i]):
            continue
        start = max(0, i - window)
        hist  = vals[start:i][~np.isnan(vals[start:i])]
        if len(hist) < 10:
            continue
        ranks[i] = float((hist < vals[i]).sum()) / len(hist) * 100.0
    return pd.Series(ranks, index=s.index, name=s.name)


def build_country_features(panel: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Build stress-oriented features for a single country from the OECD panel.
    Returns a DataFrame with features named without the country prefix.
    """
    def get(series_key, default=None):
        col = f"{country}_{series_key}"
        if col in panel.columns:
            return panel[col].copy()
        return default

    df = pd.DataFrame(index=panel.index)

    # ── Inflation ─────────────────────────────────────────────────────────────
    cpi = get("cpi")
    if cpi is not None:
        df["inflation_yoy"] = yoy(cpi)
    # No core CPI or breakeven internationally — inflation engine uses CPI only

    # ── Labour ────────────────────────────────────────────────────────────────
    unrate = get("unrate")
    if unrate is not None:
        df["unemployment_change"] = unrate.diff(12)
    ip = get("ip")
    if ip is not None:
        df["ip_employment_proxy"] = -yoy(ip)   # IP as employment proxy; negate

    # ── Financial ─────────────────────────────────────────────────────────────
    cli = get("cli")
    if cli is not None:
        # CLI stress: above 100 = expanding above trend, below 100 = below trend
        # Negate so higher value = more stress (below trend = stress)
        df["cli_stress"]   = 100.0 - cli                   # positive when below trend
        df["cli_momentum"] = -(cli.diff(3))                  # falling CLI = stress

    # ── Monetary ──────────────────────────────────────────────────────────────
    ir3m = get("ir3m")
    irlt = get("irlt")
    if ir3m is not None and cpi is not None:
        df["real_rate"] = ir3m - df.get("inflation_yoy", pd.Series(np.nan, index=panel.index))
    if ir3m is not None and irlt is not None:
        df["yield_spread"]   = -(irlt - ir3m)   # negate: inversion = stress
    if ir3m is not None:
        df["rate_change"] = ir3m.diff(12)

    # ── Real ──────────────────────────────────────────────────────────────────
    if ip is not None:
        df["indpro_growth"] = -yoy(ip)   # negate: decline = stress

    # Drop the first 13 rows (need 12 months for YoY)
    df = df.iloc[13:]
    return df


def score_country_engines(features: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Build 5 engine scores for a single country using the international feature set.
    Uses rolling normalisation (same as US v5 pipeline).
    """
    # Engine definitions for international data
    ENGINE_FEATURES_INTL = {
        "Inflation" : ["inflation_yoy"],
        "Labour"    : ["unemployment_change", "ip_employment_proxy"],
        "Financial" : ["cli_stress", "cli_momentum"],
        "Monetary"  : ["real_rate", "yield_spread", "rate_change"],
        "Real"      : ["indpro_growth"],
    }

    engine_scores = {}
    for engine_name, feat_list in ENGINE_FEATURES_INTL.items():
        available = [f for f in feat_list if f in features.columns]
        if not available:
            continue

        # Rolling z-score each variable
        z_scores = {}
        for col in available:
            z = rolling_zscore(features[col])
            z_scores[col] = expanding_impute_median(z)
        z_df = pd.DataFrame(z_scores)

        # Equal-weight average within engine
        composite = z_df[available].mean(axis=1)

        # Rolling percentile rank → 0-100
        pct = rolling_percentile_rank(composite)
        engine_scores[engine_name] = pct

    return pd.DataFrame(engine_scores)


def build_panel_features(panel_path: str, recession_path: str):
    """
    Build features and engine scores for all countries.
    Returns:
        panel_features : long-format DataFrame with country column
        panel_engines  : long-format DataFrame with engine scores + recession
    """
    panel    = pd.read_csv(panel_path,    index_col=0, parse_dates=True)
    rec_df   = pd.read_csv(recession_path, index_col=0, parse_dates=True)

    all_features = []
    all_engines  = []

    for country in COUNTRIES:
        # Check which series are available for this country
        country_cols = [c for c in panel.columns if c.startswith(f"{country}_")]
        if not country_cols:
            print(f"    [{country}] No data found — skipping")
            continue

        print(f"    [{country}] {len(country_cols)} series available: "
              f"{[c.replace(country+'_','') for c in country_cols]}")

        # Build features
        features = build_country_features(panel, country)
        features["country"]   = country
        features["RECESSION"] = rec_df[country].reindex(features.index).fillna(0)

        # Build engine scores
        engines = score_country_engines(features, country)
        engines["country"]   = country
        engines["RECESSION"] = rec_df[country].reindex(engines.index).fillna(0)

        all_features.append(features)
        all_engines.append(engines)

        n_rec = engines["RECESSION"].sum()
        n_months = len(engines.dropna(how="all", subset=[c for c in engines.columns
                                                          if c not in ("country","RECESSION")]))
        print(f"           {n_months} months, {n_rec:.0f} recession months")

    panel_features = pd.concat(all_features, axis=0).sort_index()
    panel_engines  = pd.concat(all_engines,  axis=0).sort_index()

    return panel_features, panel_engines


def main():
    panel_path    = os.path.join(DATA_RAW, "oecd_panel_raw.csv")
    recession_path = os.path.join(DATA_RAW, "oecd_recession_dates.csv")

    if not os.path.exists(panel_path):
        print(f"\n  OECD panel not found: {panel_path}")
        print("  Run: python src/data/download_oecd.py  first")
        print("\n  Building recession dates only...")
        from src.data.download_oecd import build_recession_indicator, COUNTRIES as CTY_DICT
        rec_df = build_recession_indicator(CTY_DICT)
        rec_df.to_csv(recession_path)
        print(f"  Saved: {recession_path}")
        return

    os.makedirs(DATA_PROC, exist_ok=True)

    print(f"\n{'='*64}")
    print("  Building International Panel Features")
    print(f"{'='*64}\n")

    panel_features, panel_engines = build_panel_features(panel_path, recession_path)

    feat_path = os.path.join(DATA_PROC, "panel_features.csv")
    eng_path  = os.path.join(DATA_PROC, "panel_engine_scores.csv")

    panel_features.to_csv(feat_path)
    panel_engines.to_csv(eng_path)

    print(f"\n  Panel features → {feat_path}  {panel_features.shape}")
    print(f"  Panel engines  → {eng_path}  {panel_engines.shape}")

    # Recession episode summary
    print(f"\n  Recession episodes by country:")
    total_episodes = 0
    for cty in COUNTRIES:
        subset = panel_engines[panel_engines["country"] == cty]["RECESSION"]
        if len(subset) == 0:
            continue
        episodes = ((subset.diff() == 1) | (
            (subset == 1) & (subset.index == subset.index[0])
        )).sum()
        total_episodes += episodes
        print(f"    {cty:<6}  {episodes:>3} episodes  {subset.sum():.0f} months")

    print(f"\n  TOTAL: {total_episodes} recession episodes")
    print(f"  cf. US-only: ~6 episodes in backtest window")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
