"""
portfolio.py
------------
Portfolio construction with rolling optimisation across:

1. Holding period     — {3, 5, 10, 15} trading days
2. Portfolio size     — {1, 2, 3} ETFs
3. Weighting method   — TWO methods compete head-to-head:

   A) Inverse-TE (paper method)
      Select top-N included assets by LOWEST tracking error vs benchmark.
      Weight = 1/TE, normalised. Prioritises consistency/risk control.

   B) Momentum-ranked (return-chasing method)
      Score each included asset by its composite momentum signal strength
      (sum of binary votes across all 6 frequencies).
      Select top-N by HIGHEST score. Weight equally within selected set.
      Prioritises raw return potential.

At every rebalance date, ALL combinations of (period × N × method) are
simulated over the trailing 252 days. The combination with the highest
annualised return wins and is applied immediately.
"""

import numpy as np
import pandas as pd
from itertools import product

# Candidate holding periods (trading days)
CANDIDATE_PERIODS = [3, 5, 10, 15]

# Candidate portfolio sizes
CANDIDATE_SIZES = [1, 2, 3]

# Weighting methods
METHODS = ["inv_te", "momentum_rank"]

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


def _compute_momentum_scores(
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score each asset at each date by how many frequency timeframes show
    positive momentum (out of 6). Higher = stronger momentum signal.
    Uses normalised returns vs rolling mean across 6 frequencies.
    Returns DataFrame of scores (0–6) aligned to prices.
    """
    from signal_engine import FREQUENCIES, normalised_returns, VOL_WINDOW
    scores = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for label, freq in FREQUENCIES.items():
        rv      = normalised_returns(prices, freq)
        rv_mean = rv.rolling(window=VOL_WINDOW).mean()
        scores  = scores.add(rv.gt(rv_mean).astype(float), fill_value=0)
    # Zero out excluded assets
    scores = scores * inclusion.reindex(scores.index, fill_value=0)
    return scores


def _inv_te_weights(
    te_row: pd.Series,
    inclusion_row: pd.Series,
    n_assets: int,
) -> pd.Series:
    """
    Method A: Pick top-N included assets by LOWEST TE, inverse-TE weight them.
    """
    mask     = (inclusion_row == 1) & te_row.notna() & (te_row > 0)
    eligible = te_row[mask].sort_values()          # ascending = lowest TE first
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


def _momentum_rank_weights(
    score_row: pd.Series,
    inclusion_row: pd.Series,
    n_assets: int,
) -> pd.Series:
    """
    Method B: Pick top-N included assets by HIGHEST momentum score,
    weight equally within the selected set.
    Tiebreak: if scores are equal, take all tied assets up to N.
    """
    mask     = (inclusion_row == 1) & score_row.notna() & (score_row > 0)
    eligible = score_row[mask].sort_values(ascending=False)  # descending = strongest first
    selected = eligible.head(n_assets)
    w = pd.Series(0.0, index=score_row.index)
    if selected.empty:
        included = inclusion_row[inclusion_row == 1].index
        if len(included):
            w[included] = 1.0 / len(included)
        return w
    w[selected.index] = 1.0 / len(selected)
    return w


def _weights_for_method(
    method: str,
    te_row: pd.Series,
    score_row: pd.Series,
    inclusion_row: pd.Series,
    n_assets: int,
) -> pd.Series:
    """Dispatch to the correct weighting method."""
    if method == "inv_te":
        return _inv_te_weights(te_row, inclusion_row, n_assets)
    else:
        return _momentum_rank_weights(score_row, inclusion_row, n_assets)


def _simulate_strategy(
    asset_rets: pd.DataFrame,
    inclusion: pd.DataFrame,
    te: pd.DataFrame,
    scores: pd.DataFrame,
    period: int,
    n_assets: int,
    method: str,
    start_idx: int,
    end_idx: int,
) -> pd.Series:
    """
    Simulate returns for a (period, n_assets, method) combo over window.
    Rebalances every `period` days.
    """
    port_vals     = np.zeros(end_idx - start_idx)
    weights       = np.zeros(len(asset_rets.columns))
    rebalance_set = set(range(start_idx, end_idx, period))

    for i in range(start_idx, end_idx):
        if i in rebalance_set:
            w = _weights_for_method(
                method,
                te.iloc[i],
                scores.iloc[i],
                inclusion.iloc[i],
                n_assets,
            )
            weights = w.values
        daily = asset_rets.iloc[i].fillna(0).values
        port_vals[i - start_idx] = float(weights @ daily)

    return pd.Series(port_vals, index=asset_rets.index[start_idx:end_idx])


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
    scores: pd.DataFrame,
    candidate_periods: list = CANDIDATE_PERIODS,
    candidate_sizes: list = CANDIDATE_SIZES,
    opt_window: int = OPT_WINDOW,
) -> pd.DataFrame:
    """
    At each outer tick, simulate ALL combinations of:
        period × n_assets × method (inv_te vs momentum_rank)
    over the trailing 252 days. Pick winner by highest annualised return.

    Returns DataFrame indexed by date:
        optimal_period | optimal_n | optimal_method | best_ann_return
    """
    asset_rets = prices.pct_change()
    outer_tick = min(candidate_periods)
    results    = []

    combos = list(product(candidate_periods, candidate_sizes, METHODS))
    total  = len(range(opt_window, len(prices), outer_tick))
    done   = 0

    for i in range(opt_window, len(prices), outer_tick):
        date      = prices.index[i]
        win_start = i - opt_window
        win_end   = i

        best_ret    = -np.inf
        best_period = candidate_periods[0]
        best_n      = candidate_sizes[0]
        best_method = METHODS[0]

        for period, n_assets, method in combos:
            rets = _simulate_strategy(
                asset_rets, inclusion, te, scores,
                period, n_assets, method,
                win_start, win_end,
            )
            ann = _annualised_return(rets)
            if ann > best_ret:
                best_ret    = ann
                best_period = period
                best_n      = n_assets
                best_method = method

        results.append({
            "date":            date,
            "optimal_period":  best_period,
            "optimal_n":       best_n,
            "optimal_method":  best_method,
            "best_ann_return": best_ret,
        })

        done += 1
        if done % 50 == 0:
            print(f"      optimiser: {done}/{total} steps done …")

    if not results:
        return pd.DataFrame(
            columns=["date", "optimal_period", "optimal_n",
                     "optimal_method", "best_ann_return"]
        )

    return pd.DataFrame(results).set_index("date")


# ── Main portfolio builder ────────────────────────────────────────────────────

def build_portfolio(
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> dict:
    """
    End-to-end portfolio construction with rolling dual-method optimisation.

    Returns
    -------
    {
        "weights"             : DataFrame — portfolio weights over time,
        "portfolio_returns"   : Series — daily portfolio returns,
        "latest_weights"      : Series — current holdings (post-inclusion filter),
        "optimal_params"      : DataFrame — rolling optimal config history,
        "latest_period"       : int,
        "latest_n"            : int — actual ETFs held,
        "latest_target_n"     : int — optimiser's target N,
        "latest_method"       : str — "inv_te" or "momentum_rank",
        "latest_best_return"  : float,
    }
    """
    bench_returns = benchmark_prices.pct_change()
    asset_rets    = prices.pct_change()
    te            = _compute_all_te(prices, bench_returns, TE_WINDOW)

    print("    Computing momentum scores …")
    scores = _compute_momentum_scores(prices, inclusion)

    print("    Rolling optimisation: periods × sizes × methods …")
    opt_params = rolling_optimal_params(prices, inclusion, te, scores)

    if opt_params.empty:
        print("    ⚠ Insufficient history — using defaults (10d, 2 ETFs, inv_te)")
        opt_params = pd.DataFrame(
            [{"optimal_period": 10, "optimal_n": 2,
              "optimal_method": "inv_te", "best_ann_return": np.nan}],
            index=[prices.index[-1]],
        )

    # ── Build weight series using rolling optimal params ──────────────────────
    opt_dates      = set(opt_params.index)
    current_period = CANDIDATE_PERIODS[1]
    current_n      = CANDIDATE_SIZES[1]
    current_method = METHODS[0]
    last_rebalance = -999
    weights_list   = []
    current_w      = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if date in opt_dates:
            current_period = int(opt_params.loc[date, "optimal_period"])
            current_n      = int(opt_params.loc[date, "optimal_n"])
            current_method = str(opt_params.loc[date, "optimal_method"])

        if (i - last_rebalance) >= current_period:
            current_w = _weights_for_method(
                current_method,
                te.iloc[i],
                scores.iloc[i],
                inclusion.iloc[i],
                current_n,
            )
            last_rebalance = i

        weights_list.append(current_w.copy())

    weights = pd.DataFrame(weights_list, index=prices.index)
    weights.index.name = "date"

    # Daily returns — lag by 1 day to avoid look-ahead
    port_ret = (weights.shift(1) * asset_rets).sum(axis=1)
    port_ret.name = "portfolio_return"

    # ── Latest state — fresh calc at last row ─────────────────────────────────
    latest_row      = opt_params.iloc[-1]
    latest_period   = int(latest_row["optimal_period"])
    latest_target_n = int(latest_row["optimal_n"])
    latest_method   = str(latest_row["optimal_method"])
    latest_best_ret = float(latest_row["best_ann_return"])

    latest_w = _weights_for_method(
        latest_method,
        te.iloc[-1],
        scores.iloc[-1],
        inclusion.iloc[-1],
        latest_target_n,
    )
    latest_w = latest_w[latest_w > 1e-6].sort_values(ascending=False)
    actual_n = len(latest_w)

    method_label = "Inverse-TE" if latest_method == "inv_te" else "Momentum-Rank"
    print(f"    ✓ Latest optimal: hold={latest_period}d | "
          f"method={method_label} | target N={latest_target_n}, actual N={actual_n} | "
          f"ann_return={latest_best_ret*100:.2f}%")
    print(f"    ✓ Current holdings: {list(latest_w.index)}")

    return {
        "weights":            weights,
        "portfolio_returns":  port_ret,
        "latest_weights":     latest_w,
        "optimal_params":     opt_params,
        "latest_period":      latest_period,
        "latest_n":           actual_n,
        "latest_target_n":    latest_target_n,
        "latest_method":      latest_method,
        "latest_best_return": latest_best_ret,
    }
