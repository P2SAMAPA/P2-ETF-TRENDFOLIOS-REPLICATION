"""
run_pipeline.py
-------------
Main pipeline: loads raw prices from HuggingFace, runs signals,
builds portfolios, runs backtests, and pushes results back to HF.

Run by GitHub Actions weekly, or locally:
 HF_TOKEN=your_token python run_pipeline.py
"""

import os
import json
import pandas as pd
import numpy as np
import traceback
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download

import importlib
sig_module = importlib.import_module("signal_engine")
sig_module_b = importlib.import_module("signal_engine_b")
port_module = importlib.import_module("portfolio")
port_module_b = importlib.import_module("portfolio_b")
bt_module = importlib.import_module("backtest")

# ── Config ────────────────────────────────────────────────────────────────────

HF_REPO_ID = "P2SAMAPA/p2-etf-trendfolios-replication-data"
HF_TOKEN = os.environ.get("HF_TOKEN")

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

BENCHMARKS = {"equity": "SPY", "fixed_income": "AGG"}

# Hard start dates — enforced to ensure meaningful full-universe coverage
UNIVERSE_START = {
    "equity": "2005-01-01",
    "fixed_income": "2007-01-01",
}

# ── Load data ─────────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Load wide-format adjusted close prices from HuggingFace."""
    print("Loading prices from HuggingFace …")
    
    # Bypass dataset metadata/schema issues by loading Parquet directly
    parquet_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="prices/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=HF_TOKEN,
        revision="main"
    )
    
    # Read parquet directly with pandas (no dataset schema casting)
    df = pd.read_parquet(parquet_path)
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.columns.name = None
    print(f" → {len(df)} rows, {df.shape[1]} tickers")
    print(f" → Columns: {list(df.columns)}")
    return df

def update_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch any new trading days since last stored date and append.
    Used in weekly GitHub Actions run.
    """
    import yfinance as yf

    last_date = prices.index.max()
    today = pd.Timestamp.today().normalize()

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
        print(" No new data.")
        return prices

    new_close = new_raw["Close"].copy()
    new_close.index = pd.to_datetime(new_close.index)
    new_close = new_close[new_close.index > last_date]

    updated = pd.concat([prices, new_close]).sort_index()
    updated = updated[~updated.index.duplicated(keep="last")]
    print(f" → Added {len(new_close)} new rows")
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
    print(f" ✓ Pushed → {config}")

# ── Per-universe pipeline ─────────────────────────────────────────────────────

def _push_option_results(
    prefix: str,
    option: str,  # "a" or "b"
    label: str,
    port: dict,
    bt: dict,
    signals: dict,
    bench_returns: pd.Series,
    prices: pd.DataFrame,
    inception_year: int,
):
    """Push all results for one option (A or B) to HuggingFace."""
    import json, tempfile, os as _os
    key = f"{prefix}_{option}"  # e.g. "equity_a" or "equity_b"

    # Config
    latest_w = port["latest_weights"]
    if isinstance(latest_w, pd.Series):
        latest_w = latest_w.reset_index()
        latest_w.columns = ["ticker", "weight"]
        latest_w = latest_w[latest_w["weight"] > 1e-6].sort_values("weight", ascending=False)

    config_data = {
        "option": option.upper(),
        "optimal_period": int(port["latest_period"]),
        "optimal_n": int(port["latest_n"]),
        "target_n": int(port.get("latest_target_n", port["latest_n"])),
        "optimal_method": str(port.get("latest_method", "inv_te")),
        "best_ann_return": round(float(port["latest_best_return"]), 6),
        "as_of": str(prices.index[-1].date()),
        "holdings": ",".join(latest_w["ticker"].tolist()),
        "inception_year": inception_year,
        "is_invested": not latest_w.empty,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                    delete=False, prefix=f"{key}_config_") as f:
        json.dump(config_data, f)
        tmp_path = f.name
    api = HfApi()
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=f"{key}_config.json",
        repo_id=HF_REPO_ID, repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"update: {label} option {option.upper()} config",
    )
    _os.unlink(tmp_path)
    print(f" ✓ Pushed → {key}_config.json")

    # Weights
    push(df_to_dataset(port["weights"]),
         f"{key}_weights", f"update: {label} option {option.upper()} weights")

    # Returns
    ret_df = pd.DataFrame({
        "portfolio_gross": port["portfolio_returns"],
        "portfolio_net": bt["port_returns_net"],
        "benchmark": bench_returns,
    })
    push(df_to_dataset(ret_df), f"{key}_returns",
         f"update: {label} option {option.upper()} returns")

    # Growth
    growth_df = pd.DataFrame({
        "portfolio_gross": bt["growth_port"],
        "benchmark": bt["growth_bench"],
    })
    push(df_to_dataset(growth_df), f"{key}_growth",
         f"update: {label} option {option.upper()} growth")

    # Rolling excess
    rolling_df = pd.DataFrame({
        "rolling_1y": bt["rolling_1y"],
        "rolling_3y": bt["rolling_3y"],
        "rolling_5y": bt["rolling_5y"],
    })
    push(df_to_dataset(rolling_df), f"{key}_rolling",
         f"update: {label} option {option.upper()} rolling")

    # Summary
    summary = bt["summary"].reset_index()
    push(Dataset.from_pandas(summary, preserve_index=False),
         f"{key}_summary", f"update: {label} option {option.upper()} summary")

    # Calendar
    cal = bt["calendar"].reset_index()
    push(Dataset.from_pandas(cal, preserve_index=False),
         f"{key}_calendar", f"update: {label} option {option.upper()} calendar")

    # Inclusion
    push(df_to_dataset(signals["inclusion"]),
         f"{key}_inclusion", f"update: {label} option {option.upper()} inclusion")

    print(f" ✓ {label} Option {option.upper()} push complete.")

def run_universe(
    prices_all: pd.DataFrame,
    tickers: list[str],
    benchmark_ticker: str,
    label: str,
    start_date: str,
):
    """Run Option A and Option B SEQUENTIALLY for one universe to avoid race conditions."""
    
    print(f"\n{'='*60}")
    print(f" Running: {label.upper()} ({len(tickers)} ETFs, benchmark={benchmark_ticker})")
    print(f" Hard start date: {start_date}")
    print(f"{'='*60}")

    available = [t for t in tickers if t in prices_all.columns]
    missing = [t for t in tickers if t not in prices_all.columns]
    if missing:
        print(f" ⚠ Missing tickers skipped: {missing}")

    prices = prices_all[available].dropna(how="all")
    bench = prices_all[benchmark_ticker].dropna()

    common = prices.index.intersection(bench.index)
    common = common[common >= pd.Timestamp(start_date)]
    prices = prices.loc[common]
    bench = bench.loc[common]

    inception_year = pd.Timestamp(start_date).year
    bench_returns = bench.pct_change().dropna()
    prefix_key = "equity" if label == "equity" else "fixed_income"

    print(f" → Data from {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f" → {len(prices.columns)} tickers available: {prices.columns.tolist()}")

    # ── Compute signals (sequential — shared base data needed) ────────────────
    print(" Computing signals (A + B) …")
    try:
        signals_a = sig_module.compute_signals(prices)
        print(f" ✓ Signals A computed: {signals_a['inclusion'].shape}")
    except Exception as e:
        print(f" ✗ Error computing signals A: {e}")
        traceback.print_exc()
        raise

    try:
        signals_b = sig_module_b.compute_signals_b(prices, bench)
        print(f" ✓ Signals B computed: {signals_b['inclusion'].shape}")
    except Exception as e:
        print(f" ✗ Error computing signals B: {e}")
        traceback.print_exc()
        raise

    # ── Run Option A ─────────────────────────────────────────────────────────
    summary_a = None
    try:
        print(f"\n [A] Building portfolio …")
        port_a = port_module.build_portfolio(
            prices=prices, inclusion=signals_a["inclusion"],
            benchmark_prices=bench,
        )
        bt_a = bt_module.run_backtest(
            port_returns=port_a["portfolio_returns"],
            bench_returns=bench_returns, label=label,
        )
        _push_option_results(prefix_key, "a", label, port_a, bt_a,
                            signals_a, bench_returns, prices, inception_year)
        summary_a = bt_a["summary"]
        print(f"\n ✓ {label} Option A complete")
    except Exception as e:
        print(f"\n ✗ Error in Option A: {e}")
        traceback.print_exc()
        raise

    # ── Run Option B ─────────────────────────────────────────────────────────
    summary_b = None
    try:
        print(f"\n [B] Building portfolio …")
        port_b = port_module_b.build_portfolio_b(
            prices=prices, inclusion=signals_b["inclusion"],
            benchmark_prices=bench,
        )
        bt_b = bt_module.run_backtest(
            port_returns=port_b["portfolio_returns"],
            bench_returns=bench_returns, label=label,
        )
        _push_option_results(prefix_key, "b", label, port_b, bt_b,
                            signals_b, bench_returns, prices, inception_year)
        summary_b = bt_b["summary"]
        print(f"\n ✓ {label} Option B complete")
    except Exception as e:
        print(f"\n ✗ Error in Option B: {e}")
        traceback.print_exc()
        raise

    print(f"\n ✓ {label} complete — both options pushed.")
    return summary_a, summary_b

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

    # Run both universes SEQUENTIALLY to avoid resource contention
    print("\nRunning Equity universe …")
    eq_summary_a, eq_summary_b = run_universe(
        prices_all=prices,
        tickers=EQUITY_ETFS,
        benchmark_ticker=BENCHMARKS["equity"],
        label="equity",
        start_date=UNIVERSE_START["equity"],
    )

    print("\nRunning Fixed Income universe …")
    fi_summary_a, fi_summary_b = run_universe(
        prices_all=prices,
        tickers=FIXED_INCOME_ETFS,
        benchmark_ticker=BENCHMARKS["fixed_income"],
        label="fixed_income",
        start_date=UNIVERSE_START["fixed_income"],
    )

    print("\n" + "=" * 60)
    print("EQUITY — Option A Summary")
    print("=" * 60)
    print(eq_summary_a.to_string())

    print("\n" + "=" * 60)
    print("EQUITY — Option B Summary")
    print("=" * 60)
    print(eq_summary_b.to_string())

    print("\n" + "=" * 60)
    print("FIXED INCOME — Option A Summary")
    print("=" * 60)
    print(fi_summary_a.to_string())

    print("\n" + "=" * 60)
    print("FIXED INCOME — Option B Summary")
    print("=" * 60)
    print(fi_summary_b.to_string())

    print("\n✓ All done.")

if __name__ == "__main__":
    main()
