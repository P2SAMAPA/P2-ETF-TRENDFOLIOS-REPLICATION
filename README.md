# TrendFolios® ETF Replication

An open-source replication of the **TrendFolios®** portfolio construction framework introduced in:

> **TrendFolios®: A Portfolio Construction Framework for Utilizing Momentum and Trend-Following In a Multi-Asset Portfolio**  
> Joseph Lu, Randall R. Rojas, Fiona C. Yeung, Patrick D. Convery  
> Conscious Capital Advisors / UCLA Department of Economics  
> arXiv: [2506.09330](https://arxiv.org/abs/2506.09330) · June 2025

---

## What is TrendFolios®?

TrendFolios® is a fully quantitative, rules-based portfolio construction framework that combines:

- **Momentum signals** (line-based, inspired by Moskowitz et al. 2011) — identifies assets with sustained upward price trends across multiple timeframes
- **Trend-following signals** (curve-based, inspired by Faber 2006) — identifies assets where short-term returns deviate positively from their rolling mean relative to volatility
- **Majority-vote fusion** — an asset is included only when both signals agree
- **Inverse tracking error weighting** — replaces standard inverse volatility with tracking error relative to a benchmark, so each asset contributes approximately equal relative risk

The framework is evaluated across equity, fixed income, and alternatives universes over 22+ years, demonstrating consistent excess returns and superior downside protection versus traditional benchmarks.

---

## This Replication

This repo implements the TrendFolios® signal and portfolio engine on a custom ETF universe, runs it weekly via GitHub Actions, stores all results on HuggingFace, and visualises them via a Streamlit dashboard.

### ETF Universe

**Equity** (benchmark: SPY)

| Category | ETFs |
|---|---|
| Size & Style | IWD, IWF, IWN, IWO |
| Geographic | EFA, EEM, EWZ |
| Broad Market | QQQ |
| SPDR Sectors | XLV, XLF, XLE, XLI, XLK, XLY, XLP, XLB, XLRE, XLU, XLC |
| Thematic | XBI, XME, XHB, XSD, XRT, XAR, XNTK |

**Fixed Income** (benchmark: AGG)

| Category | ETFs |
|---|---|
| Core Bonds | TIP, SHY, TLT, LQD, HYG |
| Credit & Preferred | PFF, MBB |
| Real Assets | SLV, GLD, VNQ |

---

## Architecture

```
Yahoo Finance / yfinance
        │
        ▼
  seed_data.py  (one-time historical fetch)
        │
        ▼
HuggingFace Dataset  ←──────────────────────────┐
        │                                         │
        ▼                                         │
  signal.py     → momentum + trend signals        │
  portfolio.py  → inverse TE weights              │
  backtest.py   → Sharpe, IR, rolling returns     │
        │                                         │
        └──── run_pipeline.py ───────────────────►┘
                     │
              GitHub Actions
              (weekly, Monday 02:00 UTC)
                     │
                     ▼
            Streamlit Dashboard
           (streamlit.io / app.py)
```

---

## Files

| File | Purpose |
|---|---|
| `seed_data.py` | One-time script: fetch full history, push to HuggingFace |
| `seed_data.yml` | GitHub Actions workflow: manual trigger for seeding |
| `signal.py` | Momentum + trend-following signal computation |
| `portfolio.py` | Inverse tracking error weighting, biweekly rebalance |
| `backtest.py` | Performance metrics: Sharpe, IR, drawdown, rolling returns |
| `run_pipeline.py` | Master runner: loads HF data → signals → backtest → push results |
| `pipeline.yml` | GitHub Actions: weekly scheduled run |
| `app.py` | Streamlit dashboard |
| `requirements.txt` | Python dependencies |

---

## Setup

### 1. Fork / clone this repo

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-TRENDFOLIOS-REPLICATION
cd P2-ETF-TRENDFOLIOS-REPLICATION
```

### 2. Add HuggingFace token to GitHub Secrets

In your repo: **Settings → Secrets → Actions → New secret**

- Name: `HF_TOKEN`
- Value: your HuggingFace write token (from hf.co/settings/tokens)

### 3. Seed the historical data (run once)

Trigger manually from **Actions → Seed Raw Data → Run workflow**.

This fetches all historical OHLCV data from 1997 and pushes it to:  
`P2SAMAPA/p2-etf-trendfolios-replication-data`

### 4. Run the pipeline

Trigger manually from **Actions → Weekly Pipeline Run → Run workflow**, or wait for the Monday 02:00 UTC schedule.

### 5. Deploy the dashboard

Connect your repo to [streamlit.io](https://streamlit.io) and set `app.py` as the entry point. Add `HF_TOKEN` as a Streamlit secret.

---

## Methodology Detail

### Signal Generation

Following equations (1)–(6) of Lu et al. (2025):

**Relative returns** at frequencies ν ∈ {1d, 5d, 21d, 63d, 126d, 252d}:

```
Rᵥₜ = (Pₜ - Pₜ₋₁) / Pₜ₋₁
```

**Compounded daily returns** (normalised):

```
CRₜ = CRₜ₋₁ × (1 + R¹ₜ/100)
R¹ₜ = (CRₜ - CRₜ₋₁) / CRₜ₋₁
```

**Spread signal**:

```
Sᵥₜ = (R¹ₜ - R̄ᵥ) / (R̄ᵥ + σᵥₜ/100)
```

**Majority-vote fusion**: asset included if both momentum AND trend signal agree across the majority of frequency timeframes.

### Portfolio Construction

**Tracking error** (Equation 7):

```
TE = std(Rₚ - R_b)   over 63-day rolling window
```

**Inverse TE weights** (Equation 8):

```
ωᵢ = (1/TEᵢ) / Σⱼ(1/TEⱼ)
```

Rebalanced every **10 trading days (~2 weeks)**.

### Performance Metrics

- Annualised return (geometric mean)
- Sharpe ratio (no risk-free rate — pure return/risk)
- Maximum drawdown
- Tracking error (annualised)
- Information ratio = excess return / tracking error
- Net-of-fee returns: 0.55% annual fee deducted monthly

---

## Key References

- Lu, J., Rojas, R.R., Yeung, F.C., Convery, P.D. (2025). *TrendFolios®: A Portfolio Construction Framework for Utilizing Momentum and Trend-Following In a Multi-Asset Portfolio*. arXiv:2506.09330.
- Jegadeesh, N. & Titman, S. (1993). *Returns to buying winners and selling losers*. Journal of Finance, 48, 65–91.
- Faber, M. (2006). *A Quantitative Approach to Tactical Asset Allocation*. Cambria Capital.
- Moskowitz, T.J., Ooi, Y.H., Pedersen, L.H. (2011). *Time Series Momentum*. Journal of Financial Economics.
- Antonacci, G. (2015). *Absolute Momentum: a Simple Rule-based Strategy and Universal Trend Following Overlay*. SSRN.
- Hurst, B., Ooi, Y.H., Pedersen, L.H. (2017). *A Century of Evidence on Trend-following Investing*. Journal of Portfolio Management.

---

## Disclaimer

This project is for **educational and research purposes only**. It is a replication study and not investment advice. Past performance of backtested strategies does not guarantee future results. All data sourced from public market data providers.
