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
):
    """Run signals → portfolio → backtest for one universe (equity or FI)."""
    print(f"\n{'='*60}")
    print(f"  Running: {label.upper()} ({len(tickers)} ETFs, benchmark={benchmark_ticker})")
    print(f"{'='*60}")

    # Filter to tickers that actually exist in prices
    available = [t for t in tickers if t in prices_all.columns]
    missing   = [t for t in tickers if t not in prices_all.columns]
    if missing:
        print(f"  ⚠ Missing tickers skipped: {missing}")

    prices    = prices_all[available].dropna(how="all")
    bench     = prices_all[benchmark_ticker].dropna()

    # Align to common dates
    common = prices.index.intersection(bench.index)
    prices = prices.loc[common]
    bench  = bench.loc[common]

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

    # Latest weights snapshot (for Streamlit "current allocation" card)
    latest_w = port["latest_weights"]
    if isinstance(latest_w, pd.Series):
        latest_w = latest_w.reset_index()
        latest_w.columns = ["ticker", "weight"]
    latest_w = latest_w[latest_w["weight"] > 0.001].sort_values("weight", ascending=False)
    push(Dataset.from_list(latest_w.to_dict("records")),
         f"{prefix}_latest_weights", f"update: {label} latest weights")

    # Rolling optimal params (period, n, best_ann_return over time)
    opt_df = port["optimal_params"].reset_index()
    opt_df["date"] = opt_df["date"].dt.strftime("%Y-%m-%d")
    push(Dataset.from_pandas(opt_df, preserve_index=False),
         f"{prefix}_optimal_params", f"update: {label} optimal params history")

    # Latest optimal snapshot — single row for hero box in Streamlit
    latest_opt = [{
        "optimal_period":      port["latest_period"],
        "optimal_n":           port["latest_n"],
        "best_ann_return":     round(port["latest_best_return"], 6),
        "holdings":            ",".join(latest_w["ticker"].tolist()),
        "as_of":               str(prices.index[-1].date()),
    }]
    push(Dataset.from_list(latest_opt),
         f"{prefix}_latest_optimal", f"update: {label} latest optimal snapshot")

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
    )

    fi_summary = run_universe(
        prices_all       = prices,
        tickers          = FIXED_INCOME_ETFS,
        benchmark_ticker = BENCHMARKS["fixed_income"],
        label            = "fixed_income",
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
