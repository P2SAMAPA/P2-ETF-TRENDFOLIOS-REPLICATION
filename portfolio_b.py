"""
portfolio_b.py
--------------
Option B portfolio construction.
Uses the same rolling optimiser structure as portfolio.py but with:
  - Vol-targeted weighting (replaces inverse-TE)
  - Dual momentum inclusion (regime + absolute momentum gates)
  - Momentum-ranked selection within selected N assets
"""

import numpy as np
import pandas as pd
from itertools import product
from signal_engine_b import (
    vol_target_weights,
    VOL_TARGET,
    VOL_LOOKBACK,
    TRADING_DAYS,
)

CANDIDATE_PERIODS = [3, 5, 10, 15]
CANDIDATE_SIZES   = [1, 2, 3]
OPT_WINDOW        = 252


# ── Weight calculation ────────────────────────────────────────────────────────

def _momentum_score_row(
    prices_window: pd.DataFrame,
    inclusion_row: pd.Series,
) -> pd.Series:
    """Score included assets by 12-month return strength for ranking."""
    ret_12m = prices_window.pct_change(min(252, len(prices_window) - 1)).iloc[-1]
    score = ret_12m.where(inclusion_row == 1, other=-np.inf)
    return score


def _vol_weighted(
    asset_rets: pd.DataFrame,
    inclusion_row: pd.Series,
    n_assets: int,
    scores: pd.Series,
) -> pd.Series:
    """
    Select top-N included assets by 12m momentum score,
    then apply vol-targeting weights.
    """
    mask     = inclusion_row == 1
    eligible = scores[mask].sort_values(ascending=False)
    selected = list(eligible.head(n_assets).index)

    w = pd.Series(0.0, index=asset_rets.columns)
    if not selected:
        return w

    raw = {}
    for ticker in selected:
        rv = asset_rets[ticker].dropna()
        if len(rv) < VOL_LOOKBACK:
            raw[ticker] = 1.0
        else:
            ann_vol = rv.iloc[-VOL_LOOKBACK:].std() * np.sqrt(TRADING_DAYS)
            raw[ticker] = VOL_TARGET / ann_vol if ann_vol > 0 else 1.0

    total = sum(raw.values())
    if total > 0:
        for t, rw in raw.items():
            w[t] = rw / total

    return w


# ── Simulator ─────────────────────────────────────────────────────────────────

def _simulate(
    asset_rets: pd.DataFrame,
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
    period: int,
    n_assets: int,
    start_idx: int,
    end_idx: int,
) -> pd.Series:
    port_vals     = np.zeros(end_idx - start_idx)
    weights       = np.zeros(len(asset_rets.columns))
    rebalance_set = set(range(start_idx, end_idx, period))

    for i in range(start_idx, end_idx):
        if i in rebalance_set:
            inc_row = inclusion.iloc[i]
            if inc_row.sum() == 0:
                weights = np.zeros(len(asset_rets.columns))
            else:
                prices_window = prices.iloc[max(0, i - 252):i + 1]
                scores = _momentum_score_row(prices_window, inc_row)
                w = _vol_weighted(asset_rets.iloc[:i + 1], inc_row, n_assets, scores)
                weights = w.values
        daily = asset_rets.iloc[i].fillna(0).values
        port_vals[i - start_idx] = float(weights @ daily)

    return pd.Series(port_vals, index=asset_rets.index[start_idx:end_idx])


def _annualised_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    n = len(clean)
    if n < 5:
        return -np.inf
    return float((1 + clean).prod() ** (TRADING_DAYS / n) - 1)


# ── Rolling optimiser ─────────────────────────────────────────────────────────

def rolling_optimal_params_b(
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
    opt_window: int = OPT_WINDOW,
) -> pd.DataFrame:
    asset_rets = prices.pct_change()
    outer_tick = min(CANDIDATE_PERIODS)
    combos     = list(product(CANDIDATE_PERIODS, CANDIDATE_SIZES))
    total      = len(range(opt_window, len(prices), outer_tick))
    done       = 0
    results    = []

    for i in range(opt_window, len(prices), outer_tick):
        date      = prices.index[i]
        win_start = i - opt_window
        win_end   = i
        best_ret  = -np.inf
        best_p    = CANDIDATE_PERIODS[0]
        best_n    = CANDIDATE_SIZES[0]

        for period, n_assets in combos:
            rets = _simulate(asset_rets, prices, inclusion,
                             period, n_assets, win_start, win_end)
            ann  = _annualised_return(rets)
            if ann > best_ret:
                best_ret = ann
                best_p   = period
                best_n   = n_assets

        results.append({
            "date":            date,
            "optimal_period":  best_p,
            "optimal_n":       best_n,
            "best_ann_return": best_ret,
        })

        done += 1
        if done % 50 == 0:
            print(f"      [B] optimiser: {done}/{total} steps done …")

    if not results:
        return pd.DataFrame(columns=["date", "optimal_period",
                                     "optimal_n", "best_ann_return"])
    return pd.DataFrame(results).set_index("date")


# ── Main builder ──────────────────────────────────────────────────────────────

def build_portfolio_b(
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> dict:
    """
    Option B end-to-end portfolio construction.
    Returns same dict structure as portfolio.build_portfolio for compatibility.
    """
    asset_rets = prices.pct_change()

    print("    [B] Rolling optimisation …")
    opt_params = rolling_optimal_params_b(prices, inclusion)

    if opt_params.empty:
        print("    [B] ⚠ Insufficient history — using defaults (10d, 2 ETFs)")
        opt_params = pd.DataFrame(
            [{"optimal_period": 10, "optimal_n": 2, "best_ann_return": np.nan}],
            index=[prices.index[-1]],
        )

    # Build weight series
    opt_dates      = set(opt_params.index)
    current_period = CANDIDATE_PERIODS[1]
    current_n      = CANDIDATE_SIZES[1]
    last_rebalance = -999
    weights_list   = []
    current_w      = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if date in opt_dates:
            current_period = int(opt_params.loc[date, "optimal_period"])
            current_n      = int(opt_params.loc[date, "optimal_n"])

        if (i - last_rebalance) >= current_period:
            inc_row = inclusion.iloc[i]
            if inc_row.sum() == 0:
                current_w = pd.Series(0.0, index=prices.columns)
            else:
                prices_window = prices.iloc[max(0, i - 252):i + 1]
                scores = _momentum_score_row(prices_window, inc_row)
                current_w = _vol_weighted(
                    asset_rets.iloc[:i + 1], inc_row, current_n, scores
                )
            last_rebalance = i

        weights_list.append(current_w.copy())

    weights = pd.DataFrame(weights_list, index=prices.index)
    weights.index.name = "date"

    port_ret = (weights.shift(1) * asset_rets).sum(axis=1)
    port_ret.name = "portfolio_return"

    # Latest state
    latest_row      = opt_params.iloc[-1]
    latest_period   = int(latest_row["optimal_period"])
    latest_target_n = int(latest_row["optimal_n"])
    latest_best_ret = float(latest_row["best_ann_return"])

    # Fresh weight at last date
    last_inc = inclusion.iloc[-1]
    if last_inc.sum() == 0:
        latest_w = pd.Series(0.0, index=prices.columns)
    else:
        scores_last = _momentum_score_row(prices.iloc[-252:], last_inc)
        latest_w = _vol_weighted(
            asset_rets, last_inc, latest_target_n, scores_last
        )

    latest_w = latest_w[latest_w > 1e-6].sort_values(ascending=False)
    actual_n = len(latest_w)

    print(f"    [B] ✓ Latest optimal: hold={latest_period}d | "
          f"target N={latest_target_n}, actual N={actual_n} | "
          f"vol-targeted | ann_return={latest_best_ret*100:.2f}%")
    print(f"    [B] ✓ Current holdings: {list(latest_w.index)}")

    return {
        "weights":            weights,
        "portfolio_returns":  port_ret,
        "latest_weights":     latest_w,
        "optimal_params":     opt_params,
        "latest_period":      latest_period,
        "latest_n":           actual_n,
        "latest_target_n":    latest_target_n,
        "latest_method":      "vol_target",
        "latest_best_return": latest_best_ret,
    }
