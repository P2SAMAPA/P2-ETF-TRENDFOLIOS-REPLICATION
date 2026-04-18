"""
seed_data.py
------------
One-time script to fetch full historical OHLCV data for all ETFs
and push raw price data to HuggingFace Dataset.

Run locally once:
    pip install yfinance pandas datasets huggingface_hub
    HF_TOKEN=your_token python seed_data.py
"""

import os
import yfinance as yf
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

# ── Universe ──────────────────────────────────────────────────────────────────

EQUITY_ETFS = [
    "IWD", "IWF", "IWM", "IWO", "EFA", "EEM", "EWZ",
    "QQQ", "XLV", "XLF", "XLE", "XLI", "XLK", "XLY",
    "XLP", "XLB", "XLRE", "XLU", "XLC", "XBI", "XME",
    "XHB", "XSD", "XRT", "XAR", "XNTK",
]

FIXED_INCOME_ETFS = [
    "TIP", "SHY", "TLT", "LQD", "HYG",
    "PFF", "MBB", "SLV", "GLD", "VNQ",
]

BENCHMARKS = ["SPY", "AGG"]

ALL_TICKERS = EQUITY_ETFS + FIXED_INCOME_ETFS + BENCHMARKS

HF_REPO_ID = "P2SAMAPA/p2-etf-trendfolios-replication-data"
HF_TOKEN   = os.environ.get("HF_TOKEN")

START_DATE = "1997-01-01"


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    print(f"Fetching {len(tickers)} tickers from {start} …")
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=True,
        threads=True,
    )
    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    close = close.sort_index()
    print(f"  → {len(close)} trading days, {close.shape[1]} tickers")
    print(f"  → Date range: {close.index[0].date()} → {close.index[-1].date()}")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        print(f"  ⚠ Missing tickers: {missing}")
    return close


def fetch_ohlcv(tickers: list[str], start: str) -> pd.DataFrame:
    """Download full OHLCV for all tickers, return long-format DataFrame."""
    print(f"Fetching OHLCV for {len(tickers)} tickers …")
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=True,
        threads=True,
    )
    frames = []
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in raw.columns.get_level_values(0):
            df = raw[col].copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df.reset_index().melt(id_vars="date", var_name="ticker", value_name=col.lower())
            frames.append(df.set_index(["date", "ticker"]))

    ohlcv = pd.concat(frames, axis=1).reset_index()
    ohlcv["date"] = ohlcv["date"].dt.strftime("%Y-%m-%d")
    ohlcv = ohlcv.sort_values(["ticker", "date"]).dropna(subset=["close"])
    print(f"  → {len(ohlcv):,} rows")
    return ohlcv


# ── Push to HuggingFace ───────────────────────────────────────────────────────

def push_prices(close: pd.DataFrame):
    """Push wide-format adjusted close prices."""
    df = close.copy()
    df.index = df.index.strftime("%Y-%m-%d")
    df = df.reset_index()
    df.columns.name = None

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(
        HF_REPO_ID,
        config_name="prices",
        split="train",
        token=HF_TOKEN,
        commit_message="seed: raw adjusted close prices",
    )
    print(f"  ✓ Pushed prices ({len(df)} rows) → {HF_REPO_ID}:prices")


def push_ohlcv(ohlcv: pd.DataFrame):
    """Push long-format OHLCV data."""
    ds = Dataset.from_pandas(ohlcv, preserve_index=False)
    ds.push_to_hub(
        HF_REPO_ID,
        config_name="ohlcv",
        split="train",
        token=HF_TOKEN,
        commit_message="seed: raw OHLCV long format",
    )
    print(f"  ✓ Pushed OHLCV ({len(ohlcv):,} rows) → {HF_REPO_ID}:ohlcv")


def push_metadata():
    """Push universe metadata (ticker lists and benchmarks)."""
    rows = []
    for t in EQUITY_ETFS:
        rows.append({"ticker": t, "universe": "equity", "role": "strategy"})
    for t in FIXED_INCOME_ETFS:
        rows.append({"ticker": t, "universe": "fixed_income", "role": "strategy"})
    rows.append({"ticker": "SPY", "universe": "equity",       "role": "benchmark"})
    rows.append({"ticker": "AGG", "universe": "fixed_income", "role": "benchmark"})

    ds = Dataset.from_list(rows)
    ds.push_to_hub(
        HF_REPO_ID,
        config_name="metadata",
        split="train",
        token=HF_TOKEN,
        commit_message="seed: universe metadata",
    )
    print(f"  ✓ Pushed metadata ({len(rows)} tickers) → {HF_REPO_ID}:metadata")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable not set.")

    print("=" * 60)
    print("TrendFolios ETF — Data Seed")
    print("=" * 60)

    # 1. Fetch prices (wide format — fast, used by signal engine)
    close = fetch_prices(ALL_TICKERS, START_DATE)
    push_prices(close)

    # 2. Fetch OHLCV (long format — richer, used for volume/range checks)
    ohlcv = fetch_ohlcv(ALL_TICKERS, START_DATE)
    push_ohlcv(ohlcv)

    # 3. Push metadata
    push_metadata()

    print()
    print("=" * 60)
    print("Seed complete.")
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
