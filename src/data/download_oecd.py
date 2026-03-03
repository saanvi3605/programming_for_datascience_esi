"""
src/data/download_oecd.py  —  v5.2
Correct OECD SDMX API URLs verified from OECD Data Explorer documentation.

Fixed in v5.2:
  - ip   → OECD.SDD.STES,DSD_STES@DF_INDSERV  (industrial production + services)
  - ir3m → OECD.SDD.STES,DSD_STES@DF_FINMARK  indicator=IR3TIB
  - irlt → OECD.SDD.STES,DSD_STES@DF_FINMARK  indicator=IRLT

Verified URL pattern for financial market data:
  https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/
    {countries}.M.IRLT.PA.....?startPeriod=...&dimensionAtObservation=AllDimensions

Source: Quant Trading Python tutorial (Feb 2025) confirming working API calls.
"""
import os, sys, time, io
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_RAW

COUNTRIES = {"USA": "United States", "GBR": "United Kingdom", "DEU": "Germany",
             "CAN": "Canada", "JPN": "Japan", "FRA": "France", "AUS": "Australia"}

RECESSION_DATES = [
    ("USA","1973-11","1975-03"),("USA","1980-01","1980-07"),("USA","1981-07","1982-11"),
    ("USA","1990-07","1991-03"),("USA","2001-03","2001-11"),("USA","2007-12","2009-06"),
    ("USA","2020-02","2020-04"),
    ("GBR","1973-11","1975-08"),("GBR","1979-05","1981-03"),("GBR","1990-07","1991-09"),
    ("GBR","2008-04","2009-09"),("GBR","2020-02","2020-07"),("GBR","2023-07","2023-12"),
    ("DEU","1973-11","1975-07"),("DEU","1980-04","1982-10"),("DEU","1991-01","1993-09"),
    ("DEU","2001-01","2001-11"),("DEU","2008-04","2009-06"),("DEU","2020-02","2020-06"),
    ("DEU","2022-10","2023-06"),
    ("CAN","1974-09","1975-06"),("CAN","1979-09","1980-10"),("CAN","1981-06","1982-10"),
    ("CAN","1990-03","1991-04"),("CAN","2008-11","2009-05"),("CAN","2015-01","2015-06"),
    ("CAN","2020-02","2020-05"),
    ("JPN","1973-11","1975-03"),("JPN","1980-02","1983-02"),("JPN","1985-06","1986-11"),
    ("JPN","1991-02","1993-10"),("JPN","1997-05","1999-01"),("JPN","2000-11","2002-01"),
    ("JPN","2008-02","2009-03"),("JPN","2012-04","2012-11"),("JPN","2020-01","2020-05"),
    ("FRA","1974-05","1975-06"),("FRA","1980-03","1980-09"),("FRA","1992-05","1993-08"),
    ("FRA","2008-05","2009-06"),("FRA","2020-02","2020-08"),
    ("AUS","1974-09","1975-07"),("AUS","1982-01","1983-05"),("AUS","1990-09","1991-06"),
    ("AUS","2020-02","2020-06"),
]

BASE = "https://sdmx.oecd.org/public/rest/data"

# ── Verified OECD API endpoints (tested Feb 2025) ───────────────────────────
OECD_CONFIGS = {
    "cli": {
        "dataflow": "OECD.SDD.STES,DSD_STES@DF_CLI",
        "key": "{countries}.M.LI...AA...H",
        "label": "Composite Leading Indicator (amplitude-adjusted)",
    },
    "unrate": {
        "dataflow": "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M",
        "key": "{countries}..PT_LF_SUB._Z.Y._T.Y_GE15..M",
        "label": "Harmonised unemployment rate",
    },
    "ip": {
        # Industrial & service production — DF_INDSERV (replaces old DF_IPSV)
        "dataflow": "OECD.SDD.STES,DSD_STES@DF_INDSERV",
        "key": "{countries}.M.PRINTO01.......ST",
        "label": "Industrial production (total, SA)",
        "alt_keys": [
            "{countries}.M.PRINTO01.......",
            "{countries}.M..........",
        ],
    },
    "cpi": {
        "dataflow": "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL",
        "key": "{countries}.M.N.CPI.._T.N.GY",
        "label": "CPI all items YoY growth",
    },
    "ir3m": {
        # Short-term interest rates — DF_FINMARK with IR3TIB indicator
        "dataflow": "OECD.SDD.STES,DSD_STES@DF_FINMARK",
        "key": "{countries}.M.IR3TIB.PA.....",
        "label": "3-month interbank rate",
        "alt_keys": [
            "{countries}.M.IR3TIB.......",
            "{countries}.M.IRSTCI.......",
        ],
    },
    "irlt": {
        # Long-term bond yields — DF_FINMARK with IRLT indicator
        "dataflow": "OECD.SDD.STES,DSD_STES@DF_FINMARK",
        "key": "{countries}.M.IRLT.PA.....",
        "label": "10-year government bond yield",
        "alt_keys": [
            "{countries}.M.IRLT.......",
            "{countries}.M.IRLTLT01.......",
        ],
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-script/1.0)",
    "Accept": "application/vnd.sdmx.data+csv; charset=utf-8",
}


def build_recession_indicator(countries=None, start="1970-01-01", end="2026-03-01"):
    if countries is None:
        countries = COUNTRIES
    dates  = pd.date_range(start, end, freq="MS")
    rec_df = pd.DataFrame(0, index=dates, columns=list(countries.keys()))
    for country, s, e in RECESSION_DATES:
        if country not in countries:
            continue
        rec_df.loc[(rec_df.index >= pd.Timestamp(s+"-01")) &
                   (rec_df.index <= pd.Timestamp(e+"-01")), country] = 1
    return rec_df


def _try_url(url, timeout=90):
    """Make one request, return (status_code, content) or (None, None) on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        return resp.status_code, resp.text
    except Exception as e:
        return None, str(e)


def fetch_oecd(series_key, countries, start="1970-01", timeout=90):
    """
    Fetch one OECD series. Tries primary key, then alt_keys, then bare dataflow.
    Returns DataFrame with columns = f"{country}_{series_key}".
    """
    cfg = OECD_CONFIGS[series_key]
    country_str = "+".join(countries)
    params = f"?startPeriod={start}&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"

    # Build list of URLs to try
    keys_to_try = [cfg["key"]] + cfg.get("alt_keys", [])
    urls = [f"{BASE}/{cfg['dataflow']}/{k.format(countries=country_str)}{params}"
            for k in keys_to_try]

    for url in urls:
        status, text = _try_url(url, timeout)
        if status == 200 and text and len(text) > 200:
            df = _parse_csv(text, series_key, countries)
            if df is not None and not df.empty:
                return df
        elif status == 429:
            print(f"      Rate limited — waiting 60s...")
            time.sleep(61)
            status, text = _try_url(url, timeout)
            if status == 200 and text and len(text) > 200:
                df = _parse_csv(text, series_key, countries)
                if df is not None and not df.empty:
                    return df
        elif status not in (404, None):
            # Unexpected error — log but try next
            print(f"      HTTP {status} for key variant")

    return pd.DataFrame()


def _parse_csv(text, series_key, countries):
    """Parse OECD CSV into wide DataFrame."""
    try:
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    except Exception:
        return None

    # Find time column
    time_col = next((c for c in ["TIME_PERIOD","Time","Period","time_period","DATE"]
                     if c in df.columns), None)
    if not time_col:
        return None

    # Find country column
    cty_col = next((c for c in ["REF_AREA","LOCATION","Country","ref_area"]
                    if c in df.columns), None)

    # Find value column
    val_col = next((c for c in ["OBS_VALUE","ObsValue","Value","obs_value"]
                    if c in df.columns), None)
    if not val_col:
        num_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        val_col  = num_cols[-1] if num_cols else None
    if not val_col:
        return None

    df[time_col] = pd.to_datetime(df[time_col].astype(str).str[:7] + "-01", errors="coerce")
    df = df.dropna(subset=[time_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    result = {}
    for cty in countries:
        subset = df[df[cty_col] == cty].copy() if cty_col in df.columns else df.copy()
        if subset.empty:
            continue
        s = subset.groupby(time_col)[val_col].median()
        s.index = pd.DatetimeIndex(s.index)
        s = s.resample("MS").last().ffill(limit=2)
        result[f"{cty}_{series_key}"] = s

    return pd.DataFrame(result) if result else None


def download_oecd_panel(countries=None, start="1970-01"):
    if countries is None:
        countries = list(COUNTRIES.keys())

    print(f"\n{'='*64}")
    print(f"  Downloading OECD panel: {len(countries)} countries")
    print(f"{'='*64}\n")

    frames = []
    for key, cfg in OECD_CONFIGS.items():
        print(f"  [{key}] {cfg['label']}")
        df = fetch_oecd(key, countries, start=start)
        if not df.empty:
            valid = sorted(set(c.split("_")[0] for c in df.columns))
            print(f"    OK: {len(df)} months, {len(valid)} countries: {valid}")
            frames.append(df)
        else:
            print(f"    FAILED — no data retrieved for {key}")
        time.sleep(2)

    if not frames:
        raise RuntimeError(
            "All downloads failed. Possible causes:\n"
            "  1. VPN active (OECD blocks VPN)\n"
            "  2. Rate limit exceeded (60 req/hr)\n"
            "  3. Corporate firewall/proxy"
        )

    panel = pd.concat(frames, axis=1)
    panel.index.name = "date"
    return panel


def main():
    rec_df = build_recession_indicator()
    os.makedirs(DATA_RAW, exist_ok=True)
    rec_path = os.path.join(DATA_RAW, "oecd_recession_dates.csv")
    rec_df.to_csv(rec_path)

    total = sum(((rec_df[c].diff()==1) | ((rec_df[c]==1) & (rec_df[c].index==rec_df[c].index[0]))).sum()
                for c in rec_df.columns)
    print(f"\n  Recession dates built: {total} episodes across {len(rec_df.columns)} countries")

    try:
        panel = download_oecd_panel()
        panel_path = os.path.join(DATA_RAW, "oecd_panel_raw.csv")
        panel.to_csv(panel_path)
        print(f"\n  Panel saved: {panel_path}  {panel.shape}")
    except RuntimeError as e:
        print(f"\n  {e}")


if __name__ == "__main__":
    main()
