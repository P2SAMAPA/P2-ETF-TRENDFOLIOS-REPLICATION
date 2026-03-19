"""
signal.py
---------
Computes momentum and trend-following signals for each ETF
following the TrendFolios® methodology (Lu et al., 2025).

Signal components
-----------------
1. Relative returns  Rᵥ at multiple calendar frequencies
2. Compounded returns CRₜ normalised back to daily
3. Volatility σᵥ at each frequency
4. Spread signal Sᵥ = (R¹ - R̄ᵥ) / (R̄ᵥ + σᵥ/100)
5. Binary inclusion matrix across timeframes
6. Majority-vote fusion of momentum + trend signals → include/exclude
"""

import numpy as np
import pandas as pd

# Calendar frequencies (trading days approx.)
FREQUENCIES = {
    "1d":   1,
    "5d":   5,
    "21d":  21,    # ~1 month
    "63d":  63,    # ~3 months
    "126d": 126,   # ~6 months
    "252d": 252,   # ~1 year
}

# Lookback window for volatility calculation
VOL_WINDOW = 63


def relative_returns(prices: pd.DataFrame, freq: int) -> pd.DataFrame:
    """
    Equation (1): Rᵥₜ = (Pₜ - Pₜ₋₁) / Pₜ₋₁  for given frequency ν.
    For ν=1 this is the standard daily return.
    For ν>1 we use price ratio ν days apart.
    """
    return prices.pct_change(freq) * 100


def compounded_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Equation (2)-(3): Compute CRₜ from daily returns, then normalise back
    to daily returns from the compounded series.
    Returns daily normalised returns R¹ₜ.
    """
    daily = prices.pct_change(1) * 100
    cr = (1 + daily / 100).cumprod()
    r1 = cr.pct_change(1) * 100
    return r1


def normalised_returns(prices: pd.DataFrame, freq: int) -> pd.DataFrame:
    """
    Equation (4): Rᵥ≥⁵ₜ = product of (1 + R¹ₜ₋ᵢ/100) for i in [0, ν) − 1.
    For ν=1 returns the simple daily return.
    """
    r1 = compounded_returns(prices)
    if freq == 1:
        return r1
    factor = (1 + r1 / 100)
    rolled = factor.rolling(window=freq).apply(np.prod, raw=True) - 1
    return rolled * 100


def rolling_volatility(ret: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Equation (5): σᵥₜ = rolling std of Rᵥ over `window` periods.
    """
    return ret.rolling(window=window).std()


def spread_signal(prices: pd.DataFrame, freq: int) -> pd.DataFrame:
    """
    Equation (6): Sᵥₜ = (R¹ₜ - R̄ᵥ) / (R̄ᵥ + σᵥₜ/100)
    where R̄ᵥ is the rolling mean of Rᵥ.
    """
    r1   = compounded_returns(prices)
    rv   = normalised_returns(prices, freq)
    rv_mean = rv.rolling(window=VOL_WINDOW).mean()
    sigma_v  = rolling_volatility(rv, window=VOL_WINDOW)
    denom = rv_mean + sigma_v / 100
    spread = (r1 - rv_mean) / denom.replace(0, np.nan)
    return spread


def momentum_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Line-based momentum signal (inspired by Moskowitz et al. 2011).
    Uses relative returns across frequencies; assigns 1 if return > 0
    for the majority of timeframes (cross-sectional rank + absolute sign).
    Returns binary Series per ticker (1 = include, 0 = exclude).
    """
    scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for label, freq in FREQUENCIES.items():
        rv = normalised_returns(prices, freq)
        rv_mean = rv.rolling(window=VOL_WINDOW).mean()
        # Positive return relative to its own mean → bullish
        scores = scores.add(rv.gt(rv_mean).astype(float), fill_value=0)

    n = len(FREQUENCIES)
    # Include if positive majority across frequencies
    return (scores >= (n / 2)).astype(int)


def trend_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Curve-based trend-following signal (inspired by Faber 2006).
    Uses spread signals across frequencies; assigns 1 if spread > 0
    for the majority of timeframes.
    Returns binary DataFrame (1 = in trend, 0 = not).
    """
    scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for label, freq in FREQUENCIES.items():
        s = spread_signal(prices, freq)
        scores = scores.add(s.gt(0).astype(float), fill_value=0)

    n = len(FREQUENCIES)
    return (scores >= (n / 2)).astype(int)


def majority_vote(mom: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    """
    Majority-of-vote fusion: include asset if BOTH momentum AND trend
    signals agree (1,1). This is the strict inclusion criterion from the paper.
    Returns binary inclusion DataFrame.
    """
    return ((mom == 1) & (trend == 1)).astype(int)


def compute_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Master function: compute all signal components and return a dict.

    Returns
    -------
    {
        "momentum"  : binary DataFrame,
        "trend"     : binary DataFrame,
        "inclusion" : binary DataFrame (majority vote),
        "spreads"   : dict of spread DataFrames per frequency,
        "vol"       : dict of volatility DataFrames per frequency,
    }
    """
    mom       = momentum_signal(prices)
    trend     = trend_signal(prices)
    inclusion = majority_vote(mom, trend)

    spreads = {
        label: spread_signal(prices, freq)
        for label, freq in FREQUENCIES.items()
    }
    vols = {
        label: rolling_volatility(normalised_returns(prices, freq), VOL_WINDOW)
        for label, freq in FREQUENCIES.items()
    }

    return {
        "momentum":  mom,
        "trend":     trend,
        "inclusion": inclusion,
        "spreads":   spreads,
        "vol":       vols,
    }
