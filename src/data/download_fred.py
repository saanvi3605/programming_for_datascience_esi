"""
src/data/download_fred.py  —  v5
Downloads all FRED series including new v5 leading indicators.
"""
import os, sys
import pandas as pd
from fredapi import Fred
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import FRED_SERIES, START_DATE, END_DATE, DATA_RAW, WEEKLY_SERIES, DAILY_SERIES

try:
    from config import FRED_API_KEY
except ImportError:
    FRED_API_KEY = None
FRED_API_KEY = FRED_API_KEY or os.environ.get("FRED_API_KEY", "")

# v5 additions to download alongside the main series
FRED_SERIES_V5_EXTRA = {
    "PERMIT" : "PERMIT",    # Building permits (monthly, from 1960)
    "T10Y3M" : "T10Y3M",    # 10Y-3M spread (daily→monthly, from 1982)
    "NAPMNOI": "NAPMOI",    # ISM Mfg New Orders (monthly, ends ~2023)
}
DAILY_SERIES_V5 = {"T10Y3M"}


def get_fred_client():
    if not FRED_API_KEY:
        raise ValueError(
            "\n  FRED API key not set.\n"
            "  Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "  Then add FRED_API_KEY = 'your_key' to config.py\n"
        )
    return Fred(api_key=FRED_API_KEY)


def to_monthly(s, label):
    all_daily  = DAILY_SERIES | DAILY_SERIES_V5
    all_weekly = WEEKLY_SERIES
    if label in all_weekly or label in all_daily:
        return s.resample("MS").mean()
    return s.resample("MS").last().ffill(limit=1)


def download_fred_series(series_dict, extra_dict, start, end=None):
    end_date = end or datetime.today().strftime("%Y-%m-%d")
    fred     = get_fred_client()
    all_series = {**series_dict, **extra_dict}
    frames   = {}

    print(f"\n{'='*64}")
    print(f"  Downloading {len(all_series)} FRED series")
    print(f"  Period: {start}  to  {end_date}")
    print(f"{'='*64}\n")

    for label, code in all_series.items():
        try:
            s = fred.get_series(code, observation_start=start, observation_end=end_date)
            s.name = label
            frames[label] = to_monthly(s, label)
            freq = ("weekly→monthly" if label in WEEKLY_SERIES
                    else "daily→monthly" if label in (DAILY_SERIES | DAILY_SERIES_V5)
                    else "monthly")
            print(f"  OK  {label:<16}  ({code:<12})  {len(frames[label]):>4} months  [{freq}]")
        except Exception as exc:
            print(f"  !!  {label:<16}  ({code:<12})  FAILED: {exc}")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df


def save_raw(df):
    os.makedirs(DATA_RAW, exist_ok=True)
    path = os.path.join(DATA_RAW, "fred_raw.csv")
    df.to_csv(path)
    return path


def main():
    df   = download_fred_series(FRED_SERIES, FRED_SERIES_V5_EXTRA, START_DATE, END_DATE)
    path = save_raw(df)
    print(f"\n{'='*64}")
    print(f"  Saved  →  {path}")
    print(f"  Shape  :  {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Dates  :  {df.index[0].date()}  to  {df.index[-1].date()}")
    print(f"\n  Coverage after first valid observation:")
    for col in df.columns:
        s = df[col]; first = s.first_valid_index()
        if first is None: continue
        total_miss = s.isnull().sum(); tail_miss = s.loc[first:].isnull().sum()
        if total_miss > 0:
            print(f"    {col:<16}  {total_miss:>4} total NaN  "
                  f"(from {str(first.date())})  {tail_miss} gaps after intro")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
