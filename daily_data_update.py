"""
daily_data_update.py
--------------------
Incremental daily update: fetches only new trading days since the last
stored date and appends them to the existing HuggingFace Dataset.

Run by GitHub Actions on weekdays after US market close, or locally:
    HF_TOKEN=your_token python daily_data_update.py
"""

import os
import sys
import yfinance as yf
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# ── Universe ──────────────────────────────────────────────────────────────────

EQUITY_ETFS = [
    "IWD", "IWF", "IWN", "IWO", "EFA", "EEM", "EWZ",
    "QQQ", "XLV", "XLF", "XLE", "XLI", "XLK", "XLY",
    "XLP", "XLB", "XLRE", "XLU", "XLC", "XBI", "XME",
    "XHB", "XSD", "XRT", "XAR", "XNTK",
]

FIXED_INCOME_ETFS = [
    "TIP", "SHY", "TLT", "LQD", "HYG",
    "PFF", "MBB", "SLV", "GLD", "VNQ",
]

BENCHMARKS    = ["SPY", "AGG"]
ALL_TICKERS   = EQUITY_ETFS + FIXED_INCOME_ETFS + BENCHMARKS

HF_REPO_ID    = "P2SAMAPA/p2-etf-trendfolios-replication-data"
HF_TOKEN      = os.environ.get("HF_TOKEN")

# Minimum new rows before we bother pushing (avoids spurious commits on holidays)
MIN_NEW_ROWS  = 1


# ── Load existing data ────────────────────────────────────────────────────────

def load_existing_prices() -> pd.DataFrame:
    """Load the current wide-format prices table from HuggingFace."""
    print("Loading existing prices from HuggingFace …")
    ds = load_dataset(HF_REPO_ID, "prices", split="train", token=HF_TOKEN)
    df = ds.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.columns.name = None
    last = df.index.max().date()
    print(f"  → {len(df)} rows | last stored date: {last}")
    return df


def load_existing_ohlcv() -> pd.DataFrame:
    """Load the current long-format OHLCV table from HuggingFace."""
    print("Loading existing OHLCV from HuggingFace …")
    ds = load_dataset(HF_REPO_ID, "ohlcv", split="train", token=HF_TOKEN)
    df = ds.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    print(f"  → {len(df):,} rows | last stored date: {df['date'].max().date()}")
    return df


# ── Fetch new data ────────────────────────────────────────────────────────────

def fetch_new_prices(last_stored_date: pd.Timestamp) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers from the day after
    last_stored_date up to today. Returns empty DataFrame if nothing new.
    """
    # yfinance start is inclusive, so add 1 day
    fetch_start = (last_stored_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Fetching prices: {fetch_start} → today …")
    raw = yf.download(
        ALL_TICKERS,
        start=fetch_start,
        end=fetch_end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        print("  → No new data returned.")
        return pd.DataFrame()

    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    close = close.sort_index()

    # Keep only rows strictly after last stored date
    close = close[close.index > last_stored_date]

    print(f"  → {len(close)} new trading day(s) fetched")
    return close


def fetch_new_ohlcv(last_stored_date: pd.Timestamp) -> pd.DataFrame:
    """
    Download OHLCV for all tickers from the day after last_stored_date.
    Returns long-format DataFrame.
    """
    fetch_start = (last_stored_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Fetching OHLCV: {fetch_start} → today …")
    raw = yf.download(
        ALL_TICKERS,
        start=fetch_start,
        end=fetch_end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    frames = []
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in raw.columns.get_level_values(0):
            df = raw[col].copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df.reset_index().melt(
                id_vars="date", var_name="ticker", value_name=col.lower()
            )
            frames.append(df.set_index(["date", "ticker"]))

    if not frames:
        return pd.DataFrame()

    ohlcv = pd.concat(frames, axis=1).reset_index()
    ohlcv = ohlcv[ohlcv["date"] > last_stored_date]
    ohlcv["date"] = ohlcv["date"].dt.strftime("%Y-%m-%d")
    ohlcv = ohlcv.sort_values(["ticker", "date"]).dropna(subset=["close"])
    return ohlcv


# ── Merge & push ──────────────────────────────────────────────────────────────

def merge_and_push_prices(existing: pd.DataFrame, new_rows: pd.DataFrame):
    """Append new rows to existing prices and push to HuggingFace."""
    combined = pd.concat([existing, new_rows]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    out = combined.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out = out.reset_index().rename(columns={"date": "date"})
    out.columns = [str(c) for c in out.columns]

    ds = Dataset.from_pandas(out, preserve_index=False)
    ds.push_to_hub(
        HF_REPO_ID,
        config_name="prices",
        split="train",
        token=HF_TOKEN,
        commit_message=f"daily update: prices +{len(new_rows)} row(s) "
                       f"through {new_rows.index.max().date()}",
    )
    print(f"  ✓ Prices pushed: {len(combined)} total rows "
          f"(+{len(new_rows)} new)")


def merge_and_push_ohlcv(existing: pd.DataFrame, new_rows: pd.DataFrame):
    """Append new OHLCV rows to existing table and push to HuggingFace."""
    combined = pd.concat([existing, new_rows], ignore_index=True)
    # Drop dupes on date+ticker
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.sort_values(["ticker", "date"])

    ds = Dataset.from_pandas(combined, preserve_index=False)
    ds.push_to_hub(
        HF_REPO_ID,
        config_name="ohlcv",
        split="train",
        token=HF_TOKEN,
        commit_message=f"daily update: OHLCV +{len(new_rows)} row(s)",
    )
    print(f"  ✓ OHLCV pushed: {len(combined):,} total rows "
          f"(+{len(new_rows)} new)")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_new_rows(new_prices: pd.DataFrame) -> bool:
    """
    Basic sanity checks on new data before pushing.
    Returns True if data looks valid.
    """
    if new_prices.empty:
        return False

    # Reject if all values are NaN (e.g. holiday / bad fetch)
    if new_prices.isna().all().all():
        print("  ⚠ All values NaN — skipping push.")
        return False

    # Reject if more than 80% of tickers are missing (likely a bad pull)
    coverage = new_prices.notna().mean(axis=1).min()
    if coverage < 0.20:
        print(f"  ⚠ Ticker coverage too low ({coverage:.0%}) — skipping push.")
        return False

    return True


def log_summary(new_prices: pd.DataFrame):
    """Print a concise summary of what was fetched."""
    if new_prices.empty:
        return
    for date, row in new_prices.iterrows():
        valid = row.notna().sum()
        total = len(row)
        print(f"  {date.date()} — {valid}/{total} tickers with data")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable not set.")

    print("=" * 60)
    print("TrendFolios ETF — Daily Incremental Update")
    print("=" * 60)

    # ── Prices ────────────────────────────────────────────────────────────────
    existing_prices = load_existing_prices()
    last_date       = existing_prices.index.max()

    new_prices = fetch_new_prices(last_date)

    if not validate_new_rows(new_prices):
        print("\nNothing new to push for prices. Exiting.")
        sys.exit(0)

    log_summary(new_prices)
    merge_and_push_prices(existing_prices, new_prices)

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    existing_ohlcv = load_existing_ohlcv()
    last_ohlcv_date = pd.to_datetime(existing_ohlcv["date"]).max()

    new_ohlcv = fetch_new_ohlcv(last_ohlcv_date)
    if not new_ohlcv.empty:
        merge_and_push_ohlcv(existing_ohlcv, new_ohlcv)
    else:
        print("  No new OHLCV rows to push.")

    print()
    print("=" * 60)
    print("Daily update complete.")
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
