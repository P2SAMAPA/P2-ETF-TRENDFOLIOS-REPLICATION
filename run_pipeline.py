"""
run_pipeline.py
---------------
Main pipeline: loads raw prices from HuggingFace, runs signals,
builds portfolios, runs backtests, and pushes results back to HF.

Run by GitHub Actions weekly, or locally:
    HF_TOKEN=your_token python run_pipeline.py
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

import importlib
sig_module  = importlib.import_module("signal_engine")
port_module = importlib.import_module("portfolio")
bt_module   = importlib.import_module("backtest")

# ── Config ────────────────────────────────────────────────────────────────────

HF_REPO_ID = "P2SAMAPA/p2-etf-trendfolios-replication-data"
HF_TOKEN   = os.environ.get("HF_TOKEN")

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

BENCHMARKS = {"equity": "SPY", "fixed_income": "AGG"}

# Hard start dates — enforced to ensure meaningful full-universe coverage
UNIVERSE_START = {
    "equity":       "2005-01-01",
    "fixed_income": "2007-01-01",
}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Load wide-format adjusted close prices from HuggingFace."""
    print("Loading prices from HuggingFace …")
    ds = load_dataset(HF_REPO_ID, "prices", split="train", token=HF_TOKEN)
    df = ds.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.columns.name = None
    print(f"  → {len(df)} rows, {df.shape[1]} tickers")
    return df


def update_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch any new trading days since last stored date and append.
    Used in weekly GitHub Actions run.
    """
    import yfinance as yf

    last_date  = prices.index.max()
    today      = pd.Timestamp.today().normalize()

    if last_date >= today:
        print("Prices are up to date.")
        return prices

    print(f"Fetching new data from {last_date.date()} → {today.date()} …")
    all_tickers = EQUITY_ETFS + FIXED_INCOME_ETFS + list(BENCHMARKS.values())
    new_raw = yf.download(
        all_tickers,
        start=str(last_date.date()),
        auto_adjust=True,
        progress=False,
    )
    if new_raw.empty:
        print("  No new data.")
        return prices

    new_close = new_raw["Close"].copy()
    new_close.index = pd.to_datetime(new_close.index)
    new_close = new_close[new_close.index > last_date]

    updated = pd.concat([prices, new_close]).sort_index()
    updated = updated[~updated.index.duplicated(keep="last")]
    print(f"  → Added {len(new_close)} new rows")
    return updated


# ── Push helpers ──────────────────────────────────────────────────────────────

def df_to_dataset(df: pd.DataFrame, date_col: str = "date") -> Dataset:
    """Convert a DataFrame with DatetimeIndex to HF Dataset."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.strftime("%Y-%m-%d")
        out = out.reset_index().rename(columns={"index": date_col})
    out.columns = [str(c) for c in out.columns]
    return Dataset.from_pandas(out, preserve_index=False)


def push(ds: Dataset, config: str, msg: str):
    ds.push_to_hub(
        HF_REPO_ID,
        config_name=config,
        split="train",
        token=HF_TOKEN,
        commit_message=msg,
    )
    print(f"  ✓ Pushed → {config}")


# ── Per-universe pipeline ─────────────────────────────────────────────────────

def run_universe(
    prices_all: pd.DataFrame,
    tickers: list[str],
    benchmark_ticker: str,
    label: str,
    start_date: str,
):
    """Run signals → portfolio → backtest for one universe (equity or FI)."""
    print(f"\n{'='*60}")
    print(f"  Running: {label.upper()} ({len(tickers)} ETFs, benchmark={benchmark_ticker})")
    print(f"  Hard start date: {start_date}")
    print(f"{'='*60}")

    # Filter to tickers that actually exist in prices
    available = [t for t in tickers if t in prices_all.columns]
    missing   = [t for t in tickers if t not in prices_all.columns]
    if missing:
        print(f"  ⚠ Missing tickers skipped: {missing}")

    prices = prices_all[available].dropna(how="all")
    bench  = prices_all[benchmark_ticker].dropna()

    # Align to common dates AND enforce hard start date
    common = prices.index.intersection(bench.index)
    common = common[common >= pd.Timestamp(start_date)]
    prices = prices.loc[common]
    bench  = bench.loc[common]

    inception_year = pd.Timestamp(start_date).year
    print(f"  → Data from {prices.index[0].date()} to {prices.index[-1].date()} "
          f"({len(prices)} trading days)")

    # ── Signals ──────────────────────────────────────────────────────────────
    print("  Computing signals …")
    signals = sig_module.compute_signals(prices)

    # ── Portfolio ─────────────────────────────────────────────────────────────
    print("  Building portfolio …")
    port = port_module.build_portfolio(
        prices    = prices,
        inclusion = signals["inclusion"],
        benchmark_prices = bench,
    )

    # ── Backtest ──────────────────────────────────────────────────────────────
    print("  Running backtest …")
    bench_returns = bench.pct_change().dropna()
    bt = bt_module.run_backtest(
        port_returns  = port["portfolio_returns"],
        bench_returns = bench_returns,
        label         = label,
    )

    # ── Push results ──────────────────────────────────────────────────────────
    prefix = label  # e.g. "equity" or "fixed_income"

    # Weights
    push(df_to_dataset(port["weights"]),
         f"{prefix}_weights", f"update: {label} weights")

    # Inclusion signals
    push(df_to_dataset(signals["inclusion"]),
         f"{prefix}_inclusion", f"update: {label} inclusion signals")

    # Portfolio returns (gross + net)
    ret_df = pd.DataFrame({
        "portfolio_gross": port["portfolio_returns"],
        "portfolio_net":   bt["port_returns_net"],
        "benchmark":       bench_returns,
    })
    push(df_to_dataset(ret_df), f"{prefix}_returns",
         f"update: {label} daily returns")

    # Growth of $1
    growth_df = pd.DataFrame({
        "portfolio_gross": bt["growth_port"],
        "benchmark":       bt["growth_bench"],
    })
    push(df_to_dataset(growth_df), f"{prefix}_growth",
         f"update: {label} growth of $1")

    # Rolling excess returns
    rolling_df = pd.DataFrame({
        "rolling_1y": bt["rolling_1y"],
        "rolling_3y": bt["rolling_3y"],
        "rolling_5y": bt["rolling_5y"],
    })
    push(df_to_dataset(rolling_df), f"{prefix}_rolling",
         f"update: {label} rolling excess returns")

    # Summary performance table
    summary = bt["summary"].reset_index()
    push(Dataset.from_pandas(summary, preserve_index=False),
         f"{prefix}_summary", f"update: {label} summary table")

    # Calendar year table
    cal = bt["calendar"].reset_index()
    push(Dataset.from_pandas(cal, preserve_index=False),
         f"{prefix}_calendar", f"update: {label} calendar year returns")

    # ── Latest weights + optimal config — combined into one robust dataset ────
    latest_w = port["latest_weights"]
    if isinstance(latest_w, pd.Series):
        latest_w = latest_w.reset_index()
        latest_w.columns = ["ticker", "weight"]
    latest_w = latest_w[latest_w["weight"] > 1e-6].sort_values("weight", ascending=False)

    holdings_str   = ",".join(latest_w["ticker"].tolist()) if not latest_w.empty else ""
    optimal_period = int(port["latest_period"])
    optimal_n      = int(port["latest_n"])
    target_n_val   = int(port.get("latest_target_n", port["latest_n"]))
    optimal_method = str(port.get("latest_method", "inv_te"))
    best_ret_val   = round(float(port["latest_best_return"]), 6)
    as_of_val      = str(prices.index[-1].date())
    is_invested    = not latest_w.empty

    # Pad to at least 3 rows (HF parquet needs >1 row to avoid generation errors)
    if latest_w.empty:
        rows = [{"ticker": "CASH", "weight": 1.0}]
    else:
        rows = latest_w.to_dict("records")

    # Attach optimal config to EVERY row as extra columns — no single-row dataset needed
    for r in rows:
        r["optimal_period"]  = optimal_period
        r["optimal_n"]       = optimal_n
        r["target_n"]        = target_n_val
        r["optimal_method"]  = optimal_method
        r["best_ann_return"] = best_ret_val
        r["holdings"]        = holdings_str
        r["as_of"]           = as_of_val
        r["is_invested"]     = is_invested
        r["inception_year"]  = inception_year

    # Pad to minimum 3 rows to avoid HF parquet single-row generation errors
    while len(rows) < 3:
        rows.append({
            "ticker": "", "weight": 0.0,
            "optimal_period": optimal_period, "optimal_n": optimal_n,
            "target_n": target_n_val, "optimal_method": optimal_method,
            "best_ann_return": best_ret_val, "holdings": holdings_str,
            "as_of": as_of_val, "is_invested": is_invested,
            "inception_year": inception_year,
        })

    push(Dataset.from_list(rows),
         f"{prefix}_latest_weights", f"update: {label} latest weights + config")

    # Rolling optimal params history (keep as-is — large enough to not fail)
    opt_df = port["optimal_params"].reset_index()
    opt_df["date"] = opt_df["date"].dt.strftime("%Y-%m-%d")
    push(Dataset.from_pandas(opt_df, preserve_index=False),
         f"{prefix}_optimal_params", f"update: {label} optimal params history")

    print(f"  ✓ {label} pipeline complete.")
    return bt["summary"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable not set.")

    print("=" * 60)
    print("TrendFolios — Daily Pipeline Run")
    print("=" * 60)

    # Load and update prices
    prices = load_prices()
    prices = update_prices(prices)

    # Push updated prices back
    print("\nPushing updated prices …")
    push(df_to_dataset(prices), "prices", "update: prices refresh")

    # Run both universes
    eq_summary = run_universe(
        prices_all       = prices,
        tickers          = EQUITY_ETFS,
        benchmark_ticker = BENCHMARKS["equity"],
        label            = "equity",
        start_date       = UNIVERSE_START["equity"],
    )

    fi_summary = run_universe(
        prices_all       = prices,
        tickers          = FIXED_INCOME_ETFS,
        benchmark_ticker = BENCHMARKS["fixed_income"],
        label            = "fixed_income",
        start_date       = UNIVERSE_START["fixed_income"],
    )

    print("\n" + "=" * 60)
    print("EQUITY SUMMARY")
    print("=" * 60)
    print(eq_summary.to_string())

    print("\n" + "=" * 60)
    print("FIXED INCOME SUMMARY")
    print("=" * 60)
    print(fi_summary.to_string())

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
