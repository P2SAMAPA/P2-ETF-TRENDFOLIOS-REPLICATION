"""
portfolio.py
------------
Portfolio construction following TrendFolios® methodology, extended with:

1. Rolling optimal holding period  — evaluated over trailing 252 days,
   selects best from {3, 5, 10, 15} trading days by annualised return.

2. Rolling optimal portfolio size  — selects best N from {1, 2, 3} ETFs
   (ranked by inverse-TE) over the same trailing 252-day window.

Both decisions are made fresh at every rebalance date and applied immediately.
Weighting within the selected N ETFs follows inverse tracking-error (Eq. 7-8).
"""

import numpy as np
import pandas as pd
from itertools import product

# Candidate holding periods (trading days)
CANDIDATE_PERIODS = [3, 5, 10, 15]

# Candidate portfolio sizes
CANDIDATE_SIZES = [1, 2, 3]

# Optimisation lookback (trading days)
OPT_WINDOW = 252

# Tracking error lookback (trading days)
TE_WINDOW = 63

TRADING_DAYS = 252


# ── Core building blocks ──────────────────────────────────────────────────────

def _tracking_error_series(
    asset_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = TE_WINDOW,
) -> pd.Series:
    """Rolling tracking error of one asset vs benchmark."""
    return (asset_returns - bench_returns).rolling(window=window).std()


def _compute_all_te(
    prices: pd.DataFrame,
    bench_returns: pd.Series,
    window: int = TE_WINDOW,
) -> pd.DataFrame:
    """Tracking error DataFrame for all assets."""
    asset_rets = prices.pct_change()
    te = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        te[col] = _tracking_error_series(asset_rets[col], bench_returns, window)
    return te


def _inv_te_weights(
    te_row: pd.Series,
    inclusion_row: pd.Series,
    n_assets: int,
) -> pd.Series:
    """
    Pick top-n included assets by lowest TE, then inverse-TE weight them.
    Returns a full-length weight Series (zeros for excluded assets).
    """
    mask = (inclusion_row == 1) & te_row.notna() & (te_row > 0)
    eligible = te_row[mask].sort_values()

    selected = eligible.head(n_assets)

    w = pd.Series(0.0, index=te_row.index)
    if selected.empty:
        included = inclusion_row[inclusion_row == 1].index
        if len(included):
            w[included] = 1.0 / len(included)
        return w

    inv_te = 1.0 / selected
    w[selected.index] = inv_te / inv_te.sum()
    return w


def _simulate_strategy(
    asset_rets: pd.DataFrame,
    inclusion: pd.DataFrame,
    te: pd.DataFrame,
    period: int,
    n_assets: int,
    start_idx: int,
    end_idx: int,
) -> pd.Series:
    """
    Simulate portfolio returns for a (period, n_assets) combo over
    [start_idx, end_idx] rows. Rebalances every `period` days.
    """
    sim_index = asset_rets.index[start_idx:end_idx]
    port_vals = np.zeros(end_idx - start_idx)
    weights   = np.zeros(len(asset_rets.columns))

    rebalance_set = set(range(start_idx, end_idx, period))

    for i in range(start_idx, end_idx):
        if i in rebalance_set:
            w = _inv_te_weights(te.iloc[i], inclusion.iloc[i], n_assets)
            weights = w.values
        daily = asset_rets.iloc[i].fillna(0).values
        port_vals[i - start_idx] = float(weights @ daily)

    return pd.Series(port_vals, index=sim_index)


def _annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return, NaN-safe."""
    clean = returns.dropna()
    n = len(clean)
    if n < 5:
        return -np.inf
    cum = (1 + clean).prod()
    return float(cum ** (TRADING_DAYS / n) - 1)


# ── Rolling optimiser ─────────────────────────────────────────────────────────

def rolling_optimal_params(
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
    te: pd.DataFrame,
    candidate_periods: list = CANDIDATE_PERIODS,
    candidate_sizes: list = CANDIDATE_SIZES,
    opt_window: int = OPT_WINDOW,
) -> pd.DataFrame:
    """
    At each outer tick (every min(candidate_periods) days), evaluate all
    (period, n_assets) combinations over the trailing 252 days and select
    the combination with the highest annualised return.

    Returns DataFrame indexed by date:
        optimal_period | optimal_n | best_ann_return
    """
    asset_rets = prices.pct_change()
    outer_tick = min(candidate_periods)
    results    = []

    combos = list(product(candidate_periods, candidate_sizes))
    total  = len(range(opt_window, len(prices), outer_tick))
    done   = 0

    for i in range(opt_window, len(prices), outer_tick):
        date      = prices.index[i]
        win_start = i - opt_window
        win_end   = i

        best_ret    = -np.inf
        best_period = candidate_periods[0]
        best_n      = candidate_sizes[0]

        for period, n_assets in combos:
            rets = _simulate_strategy(
                asset_rets, inclusion, te,
                period, n_assets,
                win_start, win_end,
            )
            ann = _annualised_return(rets)
            if ann > best_ret:
                best_ret    = ann
                best_period = period
                best_n      = n_assets

        results.append({
            "date":            date,
            "optimal_period":  best_period,
            "optimal_n":       best_n,
            "best_ann_return": best_ret,
        })

        done += 1
        if done % 50 == 0:
            print(f"      optimiser: {done}/{total} steps done …")

    if not results:
        return pd.DataFrame(
            columns=["date", "optimal_period", "optimal_n", "best_ann_return"]
        )

    return pd.DataFrame(results).set_index("date")


# ── Main portfolio builder ────────────────────────────────────────────────────

def build_portfolio(
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> dict:
    """
    End-to-end portfolio construction with rolling optimisation.

    Returns
    -------
    {
        "weights"             : DataFrame — portfolio weights over time,
        "portfolio_returns"   : Series — daily portfolio returns,
        "latest_weights"      : Series — most recent non-zero weights,
        "optimal_params"      : DataFrame — rolling (period, n, return) history,
        "latest_period"       : int — most recent optimal holding period,
        "latest_n"            : int — most recent optimal N ETFs,
        "latest_best_return"  : float — annualised return of optimal combo,
    }
    """
    bench_returns = benchmark_prices.pct_change()
    asset_rets    = prices.pct_change()
    te            = _compute_all_te(prices, bench_returns, TE_WINDOW)

    print("    Rolling optimisation across periods × sizes …")
    opt_params = rolling_optimal_params(prices, inclusion, te)

    if opt_params.empty:
        print("    ⚠ Insufficient history — using defaults (10d, 2 ETFs)")
        opt_params = pd.DataFrame(
            [{"optimal_period": 10, "optimal_n": 2, "best_ann_return": np.nan}],
            index=[prices.index[-1]],
        )

    # ── Build weight series using rolling optimal params ──────────────────────
    opt_dates      = set(opt_params.index)
    current_period = CANDIDATE_PERIODS[1]   # sensible default until first opt
    current_n      = CANDIDATE_SIZES[1]
    last_rebalance = -999
    weights_list   = []
    current_w      = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if date in opt_dates:
            current_period = int(opt_params.loc[date, "optimal_period"])
            current_n      = int(opt_params.loc[date, "optimal_n"])

        if (i - last_rebalance) >= current_period:
            current_w      = _inv_te_weights(te.iloc[i], inclusion.iloc[i], current_n)
            last_rebalance = i

        weights_list.append(current_w.copy())

    weights = pd.DataFrame(weights_list, index=prices.index)
    weights.index.name = "date"

    # Daily returns — lag weights by 1 day (no look-ahead)
    port_ret = (weights.shift(1) * asset_rets).sum(axis=1)
    port_ret.name = "portfolio_return"

    # ── Latest state — force fresh weight calc using last available signals ──
    latest_row      = opt_params.iloc[-1]
    latest_period   = int(latest_row["optimal_period"])
    latest_n        = int(latest_row["optimal_n"])
    latest_best_ret = float(latest_row["best_ann_return"])

    # Recalculate weights at the very last row using today's inclusion + TE
    # This gives "what to hold NOW" regardless of where we are in the hold cycle
    latest_w = _inv_te_weights(
        te.iloc[-1],
        inclusion.iloc[-1],
        latest_n,
    )
    latest_w = latest_w[latest_w > 1e-6].sort_values(ascending=False)

    # Actual N is how many ETFs passed inclusion — may be less than optimal_n
    actual_n = len(latest_w)

    print(f"    ✓ Latest optimal: hold={latest_period}d | "
          f"target N={latest_n}, actual N={actual_n} (inclusion filtered) | "
          f"ann_return={latest_best_ret*100:.2f}%")
    print(f"    ✓ Current holdings: {list(latest_w.index)}")

    return {
        "weights":            weights,
        "portfolio_returns":  port_ret,
        "latest_weights":     latest_w,
        "optimal_params":     opt_params,
        "latest_period":      latest_period,
        "latest_n":           actual_n,        # actual holdings count, not target
        "latest_target_n":    latest_n,        # original optimiser target
        "latest_best_return": latest_best_ret,
    }
