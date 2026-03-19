"""
backtest.py
-----------
Performance metrics and backtest analytics for TrendFolios® strategy.

Metrics computed
----------------
- Annualised return
- Annualised standard deviation
- Sharpe ratio (no risk-free rate, pure return/risk)
- Maximum drawdown
- Tracking error vs benchmark
- Information ratio
- Excess return vs benchmark
- Calendar year returns
- Rolling excess returns (1Y, 3Y, 5Y)
- Growth of $1
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
ANNUAL_FEE   = 0.0055   # 0.55% AUM fee (deducted monthly)
MONTHLY_FEE  = ANNUAL_FEE / 12


def apply_fee(daily_returns: pd.Series, monthly_fee: float = MONTHLY_FEE) -> pd.Series:
    """Deduct monthly fee from daily returns."""
    net = daily_returns.copy()
    month_ends = daily_returns.resample("ME").last().index
    for me in month_ends:
        if me in net.index:
            net.loc[me] -= monthly_fee
    return net


def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return."""
    n = len(returns.dropna())
    if n == 0:
        return np.nan
    cumulative = (1 + returns.dropna()).prod()
    return float(cumulative ** (TRADING_DAYS / n) - 1)


def annualised_std(returns: pd.Series) -> float:
    """Annualised standard deviation."""
    return float(returns.dropna().std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    """Modified Sharpe: annualised return / annualised std (no risk-free rate)."""
    ann_ret = annualised_return(returns)
    ann_std = annualised_std(returns)
    if ann_std == 0 or np.isnan(ann_std):
        return np.nan
    return ann_ret / ann_std


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = (1 + returns.dropna()).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def tracking_error_ann(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Annualised tracking error."""
    diff = (port_returns - bench_returns).dropna()
    return float(diff.std() * np.sqrt(TRADING_DAYS))


def information_ratio(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Information ratio = excess return / tracking error."""
    exc = annualised_return(port_returns) - annualised_return(bench_returns)
    te  = tracking_error_ann(port_returns, bench_returns)
    if te == 0 or np.isnan(te):
        return np.nan
    return exc / te


def growth_of_dollar(returns: pd.Series) -> pd.Series:
    """Cumulative growth of $1 invested."""
    return (1 + returns.fillna(0)).cumprod()


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    """Annual return by calendar year."""
    return returns.resample("YE").apply(lambda r: (1 + r).prod() - 1)


def rolling_excess_return(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    window_years: int,
) -> pd.Series:
    """Rolling annualised excess return over a given window (in years)."""
    window = window_years * TRADING_DAYS
    excess = port_returns - bench_returns

    def ann_excess(x):
        n = len(x.dropna())
        if n < window * 0.5:
            return np.nan
        return float((1 + x.dropna()).prod() ** (TRADING_DAYS / max(n, 1)) - 1)

    return excess.rolling(window=window, min_periods=int(window * 0.5)).apply(
        ann_excess, raw=False
    )


def performance_table(
    port_returns_gross: pd.Series,
    bench_returns: pd.Series,
    label: str = "Strategy",
) -> pd.DataFrame:
    """
    Produce a summary performance table matching the paper's Table format.
    Computes gross and net (after 0.55% fee) metrics.
    """
    port_returns_net = apply_fee(port_returns_gross)

    periods = {
        "1-Year":        port_returns_gross.last("252D"),
        "3-Year":        port_returns_gross.last("756D"),
        "5-Year":        port_returns_gross.last("1260D"),
        "10-Year":       port_returns_gross.last("2520D"),
        "Since Inception": port_returns_gross,
    }
    periods_net = {
        k: apply_fee(v) for k, v in periods.items()
    }
    bench_periods = {
        "1-Year":        bench_returns.last("252D"),
        "3-Year":        bench_returns.last("756D"),
        "5-Year":        bench_returns.last("1260D"),
        "10-Year":       bench_returns.last("2520D"),
        "Since Inception": bench_returns,
    }

    rows = []
    for period in periods:
        pg = periods[period]
        pn = periods_net[period]
        b  = bench_periods[period]

        if len(pg.dropna()) < 21:
            continue

        row = {
            "Period":                    period,
            "Composite Gross Return":    annualised_return(pg),
            "Composite Net Return":      annualised_return(pn),
            "Index Return":              annualised_return(b),
            "Excess Return (gross)":     annualised_return(pg) - annualised_return(b),
            "Excess Return (net)":       annualised_return(pn) - annualised_return(b),
            "Composite Std Dev":         annualised_std(pg),
            "Index Std Dev":             annualised_std(b),
            "Composite Sharpe":          sharpe_ratio(pg),
            "Index Sharpe":              sharpe_ratio(b),
            "Tracking Error":            tracking_error_ann(pg, b),
            "Information Ratio":         information_ratio(pg, b),
            "Max Drawdown":              max_drawdown(pg),
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("Period")


def calendar_year_table(
    port_returns_gross: pd.Series,
    bench_returns: pd.Series,
) -> pd.DataFrame:
    """Calendar year performance table (gross, net, benchmark, excess)."""
    port_returns_net = apply_fee(port_returns_gross)

    gross  = calendar_year_returns(port_returns_gross)
    net    = calendar_year_returns(port_returns_net)
    bench  = calendar_year_returns(bench_returns)

    df = pd.DataFrame({
        "Strategy (Gross)":    gross,
        "Strategy (Net)":      net,
        "Benchmark":           bench,
        "Excess Return (Gross)": gross - bench,
        "Excess Return (Net)":   net   - bench,
    })
    df.index = df.index.year
    df.index.name = "Year"
    return df.sort_index(ascending=False)


def run_backtest(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    label: str = "Strategy",
) -> dict:
    """
    Full backtest output.

    Returns
    -------
    {
        "summary"          : performance table DataFrame,
        "calendar"         : calendar year table DataFrame,
        "growth_port"      : growth of $1 (gross),
        "growth_bench"     : growth of $1 (benchmark),
        "rolling_1y"       : rolling 1-year excess return,
        "rolling_3y"       : rolling 3-year excess return,
        "rolling_5y"       : rolling 5-year excess return,
        "port_returns_net" : net-of-fee daily returns,
    }
    """
    net = apply_fee(port_returns)

    return {
        "summary":           performance_table(port_returns, bench_returns, label),
        "calendar":          calendar_year_table(port_returns, bench_returns),
        "growth_port":       growth_of_dollar(port_returns),
        "growth_bench":      growth_of_dollar(bench_returns),
        "rolling_1y":        rolling_excess_return(port_returns, bench_returns, 1),
        "rolling_3y":        rolling_excess_return(port_returns, bench_returns, 3),
        "rolling_5y":        rolling_excess_return(port_returns, bench_returns, 5),
        "port_returns_net":  net,
    }
