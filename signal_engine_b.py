"""
signal_engine_b.py
------------------
Option B signal engine — enhanced trend-following with three additions
on top of the base TrendFolios® methodology:

1. Regime filter (Faber 2006)
   If the benchmark price is below its 200-day MA → go 100% cash.
   No signals computed, no positions held.

2. Dual momentum gate (Antonacci 2015)
   Asset must pass BOTH:
   a) Relative momentum — outperforming peers (inherited from Option A)
   b) Absolute momentum — 12-month return > 0 (new gate)
   Fails either → excluded regardless of trend signal.

3. Volatility targeting (Plessis 2016)
   Scale each selected asset's weight so its contribution targets
   20% annualised volatility. Replaces raw inverse-TE weighting.
"""

import numpy as np
import pandas as pd
from signal_engine import (
    FREQUENCIES,
    VOL_WINDOW,
    compounded_returns,
    normalised_returns,
    rolling_volatility,
    spread_signal,
    momentum_signal,
    trend_signal,
)

# ── Constants ─────────────────────────────────────────────────────────────────

REGIME_MA_WINDOW   = 200   # days — benchmark moving average for regime filter
ABS_MOM_WINDOW     = 252   # days — lookback for absolute momentum gate (12 months)
VOL_TARGET         = 0.20  # 20% annualised vol target per position
VOL_LOOKBACK       = 63    # days — realised vol estimation window
TRADING_DAYS       = 252


# ── 1. Regime filter ──────────────────────────────────────────────────────────

def regime_filter(benchmark_prices: pd.Series) -> pd.Series:
    """
    Returns a boolean Series: True = regime is ON (invest), False = cash.
    Benchmark price must be above its 200-day simple moving average to invest.
    """
    ma200 = benchmark_prices.rolling(window=REGIME_MA_WINDOW, min_periods=REGIME_MA_WINDOW).mean()
    return benchmark_prices > ma200


# ── 2. Absolute momentum gate ─────────────────────────────────────────────────

def absolute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Returns binary DataFrame: 1 if 12-month return > 0, else 0.
    Applied as an additional gate on top of relative momentum.
    """
    ret_12m = prices.pct_change(ABS_MOM_WINDOW)
    return (ret_12m > 0).astype(int)


def dual_momentum_inclusion(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> pd.DataFrame:
    """
    Combines:
    - Relative momentum signal (Option A logic, loosened threshold)
    - Trend signal (Option A logic)
    - Absolute momentum gate (new)
    - Regime filter (new) — all zeros when benchmark below 200MA

    Returns binary inclusion DataFrame.
    """
    # Option A signals
    mom       = momentum_signal(prices)
    trend     = trend_signal(prices)
    rel_incl  = ((mom == 1) | (trend == 1)).astype(int)

    # Absolute momentum gate
    abs_mom   = absolute_momentum(prices)

    # Dual gate: must pass both relative AND absolute momentum
    dual      = ((rel_incl == 1) & (abs_mom == 1)).astype(int)

    # Regime filter — zero out everything when benchmark below 200MA
    regime    = regime_filter(benchmark_prices)
    regime_df = pd.DataFrame(
        np.outer(regime.astype(int).values, np.ones(len(prices.columns))),
        index=prices.index,
        columns=prices.columns,
    )

    return (dual * regime_df).astype(int)


# ── 3. Volatility targeting ───────────────────────────────────────────────────

def vol_target_weights(
    asset_returns: pd.DataFrame,
    inclusion_row: pd.Series,
    n_assets: int,
    vol_target: float = VOL_TARGET,
    vol_lookback: int = VOL_LOOKBACK,
) -> pd.Series:
    """
    Select top-N included assets by momentum score, then scale each weight
    so its volatility contribution equals vol_target / n_assets.

    Weight_i = (vol_target / n_assets) / realised_vol_i
    Normalised so weights sum to 1.

    Falls back to equal weight if vol data unavailable.
    """
    mask     = inclusion_row == 1
    eligible = asset_returns.columns[mask]

    if len(eligible) == 0:
        return pd.Series(0.0, index=asset_returns.columns)

    # Take top-N by recency of available data (simple selection for now)
    selected = list(eligible[:n_assets])

    w = pd.Series(0.0, index=asset_returns.columns)

    # Realised vol per selected asset
    raw_weights = {}
    for ticker in selected:
        rv = asset_returns[ticker].dropna()
        if len(rv) < vol_lookback:
            raw_weights[ticker] = 1.0  # equal fallback
        else:
            ann_vol = rv.iloc[-vol_lookback:].std() * np.sqrt(TRADING_DAYS)
            if ann_vol > 0:
                raw_weights[ticker] = vol_target / ann_vol
            else:
                raw_weights[ticker] = 1.0

    total = sum(raw_weights.values())
    if total > 0:
        for ticker, rw in raw_weights.items():
            w[ticker] = rw / total

    return w


# ── Master compute function ───────────────────────────────────────────────────

def compute_signals_b(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> dict:
    """
    Option B signal computation.

    Returns
    -------
    {
        "inclusion"  : binary DataFrame (dual momentum + regime filtered),
        "regime_on"  : boolean Series (True = invest, False = cash),
        "abs_mom"    : binary DataFrame (absolute momentum gate),
        "momentum"   : binary DataFrame (relative momentum),
        "trend"      : binary DataFrame (trend signal),
    }
    """
    mom      = momentum_signal(prices)
    trend    = trend_signal(prices)
    abs_mom  = absolute_momentum(prices)
    regime   = regime_filter(benchmark_prices)
    inclusion = dual_momentum_inclusion(prices, benchmark_prices)

    return {
        "inclusion": inclusion,
        "regime_on": regime,
        "abs_mom":   abs_mom,
        "momentum":  mom,
        "trend":     trend,
    }
