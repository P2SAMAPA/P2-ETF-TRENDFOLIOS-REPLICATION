"""
portfolio.py
------------
Portfolio construction following TrendFolios® methodology.

Steps
-----
1. Filter universe to included assets (inclusion == 1)
2. Compute tracking error (TE) for each asset vs its benchmark
3. Weight = 1/TE, normalised to sum to 1  (Equation 7-8)
4. Rebalance every 2 weeks (10 trading days)
"""

import numpy as np
import pandas as pd

# Rebalance frequency (trading days)
REBALANCE_FREQ = 10

# Lookback for tracking error calculation (trading days)
TE_WINDOW = 63


def tracking_error(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = TE_WINDOW,
) -> pd.Series:
    """
    Equation (7): TE = std(Rp - Rb) over a rolling window.
    """
    diff = asset_returns - benchmark_returns
    return diff.rolling(window=window).std()


def compute_weights(
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
    benchmark_prices: pd.Series,
    te_window: int = TE_WINDOW,
) -> pd.DataFrame:
    """
    Equation (8): ωᵢ = (1/TEᵢ) / Σⱼ(1/TEⱼ)

    Parameters
    ----------
    prices           : adjusted close prices for strategy ETFs
    inclusion        : binary DataFrame from signal.majority_vote()
    benchmark_prices : single benchmark price series (SPY or AGG)
    te_window        : rolling window for TE calculation

    Returns
    -------
    weights : DataFrame aligned to prices index, columns = tickers
              weights are 0 for excluded assets, else inverse-TE normalised.
              Weights are forward-filled every 2 weeks (rebalance schedule).
    """
    # Daily returns
    asset_rets = prices.pct_change()
    bench_rets = benchmark_prices.pct_change()

    # Tracking error per asset
    te = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        te[col] = tracking_error(asset_rets[col], bench_rets, window=te_window)

    # Rebalance dates: every REBALANCE_FREQ trading days
    rebalance_dates = prices.index[::REBALANCE_FREQ]

    weights_list = []
    for date in prices.index:
        if date not in rebalance_dates:
            weights_list.append(None)
            continue

        included = inclusion.loc[date]
        te_today = te.loc[date]

        # Only included assets with valid TE
        mask = (included == 1) & te_today.notna() & (te_today > 0)
        if mask.sum() == 0:
            # Fallback: equal weight across all assets
            w = pd.Series(1.0 / len(prices.columns), index=prices.columns)
        else:
            inv_te = (1.0 / te_today).where(mask, 0.0)
            w = inv_te / inv_te.sum()

        weights_list.append(w)

    # Build weights DataFrame, forward fill between rebalance dates
    rebalance_weights = pd.DataFrame(
        [w for w in weights_list if w is not None],
        index=[d for d, w in zip(prices.index, weights_list) if w is not None],
    )
    weights = rebalance_weights.reindex(prices.index).ffill().fillna(0.0)
    return weights


def portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily portfolio returns from weights and asset prices.
    Weights are applied to next-day returns (no look-ahead).
    """
    asset_rets = prices.pct_change()
    # Shift weights by 1 day to avoid look-ahead bias
    w_lagged = weights.shift(1)
    port_ret = (w_lagged * asset_rets).sum(axis=1)
    port_ret.name = "portfolio_return"
    return port_ret


def build_portfolio(
    prices: pd.DataFrame,
    inclusion: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> dict:
    """
    End-to-end portfolio construction.

    Returns
    -------
    {
        "weights"           : DataFrame of portfolio weights over time,
        "portfolio_returns"  : Series of daily portfolio returns,
        "latest_weights"    : Series of most recent weights,
        "rebalance_dates"   : list of rebalance dates,
    }
    """
    weights = compute_weights(prices, inclusion, benchmark_prices)
    port_ret = portfolio_returns(prices, weights)
    rebalance_dates = list(prices.index[::REBALANCE_FREQ])

    return {
        "weights":           weights,
        "portfolio_returns":  port_ret,
        "latest_weights":    weights.iloc[-1],
        "rebalance_dates":   rebalance_dates,
    }
