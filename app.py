"""
app.py
------
TrendFolios® ETF Replication — Streamlit Dashboard
Reads all data from HuggingFace Dataset.

Fixes applied (v3):
  1. ROOT CAUSE FIX — Holdings now read from {key}_config.json ("holdings"
     field) rather than from the last row of the weights parquet.
     The weights parquet stores the FULL historical weight matrix; its last
     row reflects the last rebalance date (not today) and will be all-zeros
     on any day that falls between two rebalance events.  The config JSON is
     written by run_pipeline.py using latest_weights — the fresh calculation
     at the last data date — and is the authoritative source of truth for
     the current action card.

  2. Removed @st.cache_data from load_holdings_and_config() to prevent the
     Streamlit nested-cache serialisation bug (inner call to cached load()
     cannot be serialised into the outer cache entry).

  3. Added retry / back-off in load() for HuggingFace transient
     "generating the dataset" errors that appear immediately after a push.

  4. Improved warning messages to distinguish transient HF processing delays
     from genuine data-missing situations.
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datasets import load_dataset, get_dataset_split_names

# ── Config ────────────────────────────────────────────────────────────────────

HF_REPO_ID = "P2SAMAPA/p2-etf-trendfolios-replication-data"
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

st.set_page_config(
    page_title="TrendFolios® ETF Replication",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global font sizing ────────────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 17px !important; }
[data-testid="stMetricLabel"] { font-size: 16px !important; }
[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 700 !important; }
h2, h3 { font-size: 22px !important; }
[data-testid="stSidebar"] { font-size: 15px !important; }
table td, table th { font-size: 15px !important; }
.stDataFrame { font-size: 15px !important; }
[data-testid="stCaptionContainer"] { font-size: 15px !important; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────

BLUE  = "#2563EB"
GRAY  = "#94A3B8"
GREEN = "#16A34A"
RED   = "#DC2626"
AMBER = "#D97706"

CANDIDATE_PERIODS = [3, 5, 10, 15]
CANDIDATE_SIZES   = [1, 2, 3]

# ── Data loading ──────────────────────────────────────────────────────────────

def _get_split(config: str) -> str:
    try:
        splits = get_dataset_split_names(HF_REPO_ID, config, token=HF_TOKEN)
        return splits[0] if splits else "train"
    except Exception:
        return "train"


@st.cache_data(ttl=900)
def load(config: str) -> pd.DataFrame:
    """Load a HuggingFace dataset config into a DataFrame, with retries."""
    last_exc = None
    for attempt in range(4):
        try:
            split = _get_split(config)
            ds = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
            df = ds.to_pandas()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            return df
        except Exception as e:
            last_exc = e
            is_transient = any(k in str(e).lower() for k in
                               ("generating the dataset", "temporarily unavailable", "500", "503"))
            if is_transient and attempt < 3:
                time.sleep(4 * (attempt + 1))   # 4 s, 8 s, 12 s
                continue
            break
    raise last_exc


def _read_weights_for_tickers(key: str, tickers: list, as_of: str) -> dict:
    """
    Read per-ticker weights from the weights parquet for the given as_of date.
    Returns {ticker: weight} dict.  Falls back gracefully to equal weights on
    any error — the action card will still show the correct tickers from the
    config JSON.

    KEY FIX: We look for the row matching as_of date, or the LAST NON-ZERO
    row for those tickers — NOT blindly the last row of the DataFrame, which
    may be all-zeros if today is not a rebalance day.
    """
    try:
        weights_df = load(f"{key}_weights")
        if weights_df.empty:
            return {}

        target_date = pd.Timestamp(as_of) if as_of else None

        # First choice: exact date match
        if target_date is not None and target_date in weights_df.index:
            row = weights_df.loc[target_date]
        else:
            # Second choice: last row where any of our tickers has weight > 0
            relevant_cols = [t for t in tickers if t in weights_df.columns]
            if relevant_cols:
                nonzero_mask = (weights_df[relevant_cols] > 1e-6).any(axis=1)
                nonzero_rows = weights_df[nonzero_mask]
                row = nonzero_rows.iloc[-1] if not nonzero_rows.empty else weights_df.iloc[-1]
            else:
                row = weights_df.iloc[-1]

        result = {}
        for ticker in tickers:
            if ticker in row.index:
                v = pd.to_numeric(row[ticker], errors="coerce")
                result[ticker] = float(v) if pd.notna(v) and v > 1e-6 else 0.0
        return result

    except Exception:
        return {}


# NOTE: Not decorated with @st.cache_data — this function calls the cached
# load() internally, and Streamlit cannot serialise a nested cache call into
# an outer cache entry.  load() is already cached so performance is fine.
def load_holdings_and_config(key: str) -> tuple:
    """
    key = e.g. "equity_a", "equity_b", "fixed_income_a", "fixed_income_b"

    Returns (holdings_df, opt_dict) where:
      holdings_df  columns: ["ticker", "weight"]
      opt_dict     keys:    optimal_period, optimal_n, target_n,
                            optimal_method, best_ann_return, as_of,
                            inception_year, is_invested
    """
    inception_year = 2005 if key.startswith("equity") else 2007
    holdings = pd.DataFrame(columns=["ticker", "weight"])
    opt: dict = {}

    # ── Primary source: config JSON ───────────────────────────────────────────
    # This is written by run_pipeline.py from latest_weights — a fresh
    # re-calculation at the end of the pipeline run.  It is always current
    # and is the single source of truth for "what to hold today".
    try:
        from huggingface_hub import hf_hub_download
        import json

        json_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{key}_config.json",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(json_path) as f:
            cfg = json.load(f)

        opt = {
            "optimal_period":  int(cfg.get("optimal_period", 0)),
            "optimal_n":       int(cfg.get("optimal_n", 0)),
            "target_n":        int(cfg.get("target_n", 0)),
            "optimal_method":  str(cfg.get("optimal_method", "inv_te")),
            "best_ann_return": float(cfg.get("best_ann_return", float("nan"))),
            "as_of":           str(cfg.get("as_of", "")),
            "inception_year":  int(cfg.get("inception_year", inception_year)),
            "is_invested":     bool(cfg.get("is_invested", False)),
        }

        # Build holdings from the "holdings" ticker string in the config JSON.
        # This is the CORRECT list of what to hold — it comes from latest_weights
        # not from the weights parquet last row.
        tickers_str = cfg.get("holdings", "")
        ticker_list = [t.strip() for t in tickers_str.split(",") if t.strip()]

        if ticker_list:
            # Fetch per-ticker weights from the parquet using the smart lookup
            weights_map = _read_weights_for_tickers(key, ticker_list, cfg.get("as_of", ""))

            # Build holdings DataFrame
            rows = []
            for ticker in ticker_list:
                w = weights_map.get(ticker, 0.0)
                rows.append({"ticker": ticker, "weight": w})
            holdings = pd.DataFrame(rows)

            # If all weights are zero (weights parquet couldn't be read or
            # was all-zero), fall back to equal weighting — we still know
            # the tickers from the config JSON so the action card is correct
            if holdings["weight"].sum() < 1e-6:
                holdings["weight"] = 1.0 / len(holdings)
            else:
                # Re-normalise
                holdings["weight"] = holdings["weight"] / holdings["weight"].sum()

            holdings = holdings.sort_values("weight", ascending=False).reset_index(drop=True)

    except Exception as e:
        if "generating the dataset" in str(e).lower() or "404" in str(e):
            st.warning(
                f"⏳ `{key}_config.json` is still being processed by HuggingFace "
                f"— please refresh in ~30 seconds."
            )
        else:
            st.warning(f"Could not load `{key}_config.json`: {e}")

    return holdings, opt


def _coerce_int(v, default=0) -> int:
    try:
        f = float(v)
        return int(f) if not np.isnan(f) and f != 0 else default
    except Exception:
        return default


def _coerce_float(v) -> float:
    try:
        f = float(v)
        return f if not np.isnan(f) else float("nan")
    except Exception:
        return float("nan")

# ── Chart helpers ─────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def rolling_chart(rolling_df: pd.DataFrame, title: str):
    fig = go.Figure()
    colours = {"rolling_1y": BLUE, "rolling_3y": AMBER, "rolling_5y": GREEN}
    labels  = {"rolling_1y": "1-Year", "rolling_3y": "3-Year", "rolling_5y": "5-Year"}

    for col in ["rolling_1y", "rolling_3y", "rolling_5y"]:
        if col not in rolling_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=rolling_df.index, y=rolling_df[col] * 100,
            name=labels[col],
            line=dict(color=colours[col], width=1.5),
            fill="tozeroy",
            fillcolor=hex_to_rgba(colours[col]),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=GRAY, line_width=1)
    fig.update_layout(
        title=title,
        yaxis_title="Annualised Excess Return (%)",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    return fig


def fmt_pct(v):
    if pd.isna(v):
        return "—"
    color = GREEN if v >= 0 else RED
    return f"<span style='color:{color}'>{v*100:+.2f}%</span>"


def fmt_ratio(v):
    if pd.isna(v):
        return "—"
    color = GREEN if v >= 0 else RED
    return f"<span style='color:{color}'>{v:.2f}</span>"


def summary_html(df: pd.DataFrame) -> str:
    pct_cols = [
        "Composite Gross Return", "Composite Net Return", "Index Return",
        "Excess Return (gross)", "Excess Return (net)",
        "Composite Std Dev", "Index Std Dev", "Tracking Error", "Max Drawdown",
    ]
    ratio_cols = ["Composite Sharpe", "Index Sharpe", "Information Ratio"]

    rows_html = ""
    for period, row in df.iterrows():
        cells = f"<td><b>{period}</b></td>"
        for col in df.columns:
            v = row[col]
            if col in pct_cols:
                cells += f"<td>{fmt_pct(v)}</td>"
            elif col in ratio_cols:
                cells += f"<td>{fmt_ratio(v)}</td>"
            else:
                cells += f"<td>{v}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in ["Period"] + list(df.columns))
    return f"""
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead><tr style="background:#1e293b;color:#f1f5f9">{headers}</tr></thead>
<tbody>{rows_html}</tbody>
</table></div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 TrendFolios®")
    st.caption("ETF Momentum & Trend-Following Replication")
    universe = st.radio("Universe", ["Equity", "Fixed Income"])
    st.markdown("---")
    st.markdown("**Methodology**")
    st.markdown(
        "Replication of *TrendFolios®* (Lu et al., 2025). "
        "Momentum + trend-following signals fused via majority vote. "
        "Inverse tracking-error weighting. Rolling-optimised holding period & portfolio size."
    )
    st.markdown("---")
    st.markdown(
        "[📄 Research Paper](https://arxiv.org/abs/2506.09330) \n"
        "[💾 HuggingFace Dataset](https://huggingface.co/datasets/P2SAMAPA/p2-etf-trendfolios-replication-data) \n"
        "[🐙 GitHub](https://github.com/P2SAMAPA/P2-ETF-TRENDFOLIOS-REPLICATION)"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

prefix      = "equity" if universe == "Equity" else "fixed_income"
bench_label = "SPY"    if universe == "Equity" else "AGG"
today_label = pd.Timestamp.today().strftime("%d %b %Y")

st.title(f"TrendFolios® — {universe} Strategy")
st.caption(f"Benchmark: {bench_label} | Fee: 0.55% p.a.")

# ── Option tabs ───────────────────────────────────────────────────────────────

tab_a, tab_b = st.tabs([
    "📊 Option A — TrendFolios Base",
    "🚀 Option B — Enhanced (Regime + Dual Momentum + Vol Target)",
])

# ─────────────────────────────────────────────────────────────────────────────

def render_option(key: str, option_label: str, bench_label: str,
                  today_label: str, inception_year: int):
    """Render the full dashboard for one option."""

    with st.spinner(f"Loading {option_label} …"):
        try:
            rolling_df  = load(f"{key}_rolling")
            summary_df  = load(f"{key}_summary")
            calendar_df = load(f"{key}_calendar")
            latest_wts, latest_opt = load_holdings_and_config(key)
        except Exception as e:
            st.error(f"Could not load {option_label} data: {e}")
            st.info("Run the pipeline first, or refresh in ~30 seconds if it just ran.")
            return

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fmt_int(v, fallback="—"):
        try:
            i = int(float(v))
            return str(i) if i > 0 else fallback
        except Exception:
            return fallback

    def _fmt_pct(v, fallback="—"):
        try:
            f = float(v)
            return f"{f*100:.2f}%" if not np.isnan(f) else fallback
        except Exception:
            return fallback

    opt_period  = latest_opt.get("optimal_period")
    opt_n       = latest_opt.get("optimal_n")
    target_n    = latest_opt.get("target_n", opt_n)
    best_ret    = latest_opt.get("best_ann_return")
    as_of       = latest_opt.get("as_of", "")
    raw_method  = latest_opt.get("optimal_method", "inv_te")
    inc_year    = latest_opt.get("inception_year", inception_year)

    method_map = {
        "inv_te":        ("⚖️",  "Inverse-TE weighting"),
        "momentum_rank": ("🚀", "Momentum-rank weighting"),
        "vol_target":    ("🎯", "Vol-targeted weighting"),
    }
    m_emoji, m_label = method_map.get(raw_method, ("⚖️", raw_method))

    opt_period_str = _fmt_int(opt_period)
    opt_n_str      = _fmt_int(opt_n)
    target_n_str   = _fmt_int(target_n, opt_n_str)
    ret_str        = _fmt_pct(best_ret)
    as_of_str      = str(as_of) if as_of else "—"

    # ── Hero box ──────────────────────────────────────────────────────────────
    # is_invested driven by config JSON flag + holdings from ticker list —
    # NOT by the weights parquet last row (which is stale between rebalances).
    config_says_invested = latest_opt.get("is_invested", False)
    holdings = latest_wts[
        (latest_wts["weight"] > 1e-6) &
        (latest_wts["ticker"].str.strip() != "")
    ].sort_values("weight", ascending=False) if not latest_wts.empty and "ticker" in latest_wts.columns else pd.DataFrame(columns=["ticker", "weight"])

    is_invested = config_says_invested or len(holdings) > 0

    if is_invested and len(holdings) > 0:
        card_html = ""
        for _, row in holdings.iterrows():
            card_html += f"""
<div style="background:#f0f4ff;border:2.5px solid #2563EB;border-radius:14px;
            padding:22px 36px;display:inline-block;margin:8px 10px;text-align:center;min-width:130px">
  <div style="font-size:32px;font-weight:800;color:#1e3a8a;letter-spacing:2px">{row['ticker']}</div>
  <div style="font-size:18px;color:#2563EB;font-weight:700;margin-top:6px">{row['weight']*100:.1f}%</div>
</div>"""
    else:
        card_html = """
<div style="background:#fef9c3;border:2px solid #ca8a04;border-radius:12px;
            padding:20px 28px;display:inline-block;margin:6px 0;color:#713f12">
  <span style="font-size:22px;margin-right:10px">⚠️</span>
  <span style="font-size:18px;font-weight:600">No signal — stay in cash.</span>
</div>"""

    pill = ("background:#e8f0fe;border-radius:8px;padding:8px 16px;"
            "font-size:15px;color:#1e40af;font-weight:600;"
            "margin-right:10px;display:inline-block;margin-bottom:6px")

    config_html = (
        f'<span style="{pill}">⏱ Hold {opt_period_str}d</span>'
        f'<span style="{pill}">📦 Target {target_n_str} · Actual {opt_n_str}</span>'
        f'<span style="{pill}">{m_emoji} {m_label}</span>'
        f'<span style="{pill}">📈 {ret_str} best ann. return (trailing 252d)</span>'
        f'<span style="{pill}">📅 Signals as of {as_of_str}</span>'
    )

    st.markdown(f"""
<div style="background:#ffffff;border:2.5px solid #2563EB;border-radius:16px;padding:28px 32px;margin:8px 0 28px 0;">
  <div style="font-size:13px;letter-spacing:3px;color:#6b7280;text-transform:uppercase;margin-bottom:20px;font-weight:700">
    🎯 &nbsp;Action for {today_label} — Hold at Market Open
  </div>
  <div style="margin-bottom:24px">{card_html}</div>
  <div style="border-top:1.5px solid #e5e7eb;padding-top:14px">{config_html}</div>
</div>""", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────

    si_label = f"Since {inc_year}"
    try:
        sdf = summary_df.copy()
        if "Period" in sdf.columns:
            sdf = sdf.set_index("Period")
        si = sdf.loc["Since Inception"] if "Since Inception" in sdf.index else sdf.iloc[-1]
        st.caption(f"📅 Performance metrics: 1 Jan {inc_year} — {as_of_str}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"Gross Return ({si_label})",  f"{float(si['Composite Gross Return'])*100:.2f}%")
        c2.metric(f"Net Return ({si_label})",     f"{float(si['Composite Net Return'])*100:.2f}%")
        c3.metric(f"Excess Return ({si_label})",  f"{float(si['Excess Return (gross)'])*100:+.2f}%")
        c4.metric("Sharpe Ratio",                 f"{float(si['Composite Sharpe']):.2f}")
        c5.metric("Info Ratio",                   f"{float(si['Information Ratio']):.2f}")
    except Exception:
        pass

    st.markdown("---")

    # ── Rolling excess returns ────────────────────────────────────────────────

    if not rolling_df.empty:
        st.subheader("Rolling Annualised Excess Returns vs Benchmark")
        st.plotly_chart(
            rolling_chart(rolling_df, f"{option_label} — Excess Return over {bench_label}"),
            width="stretch",
        )
        st.markdown("---")

    # ── Performance summary ───────────────────────────────────────────────────

    st.subheader("Annualised Performance Summary")
    if not summary_df.empty:
        try:
            sdf = summary_df.copy()
            if "Period" in sdf.columns:
                sdf = sdf.set_index("Period")
            st.markdown(summary_html(sdf), unsafe_allow_html=True)
        except Exception:
            st.dataframe(summary_df, width="stretch")

    st.markdown("---")

    # ── Calendar year ─────────────────────────────────────────────────────────

    st.subheader("Calendar Year Returns")
    if not calendar_df.empty:
        try:
            cdf = calendar_df.copy()
            if "Year" in cdf.columns:
                cdf = cdf.set_index("Year")
            for col in cdf.columns:
                cdf[col] = cdf[col].map(lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—")
            st.dataframe(cdf.sort_index(ascending=False), width="stretch")
        except Exception:
            st.dataframe(calendar_df, width="stretch")


# ── Render both tabs ──────────────────────────────────────────────────────────

inc_year = 2005 if prefix == "equity" else 2007

with tab_a:
    render_option(
        key            = f"{prefix}_a",
        option_label   = "Option A — TrendFolios Base",
        bench_label    = bench_label,
        today_label    = today_label,
        inception_year = inc_year,
    )

with tab_b:
    render_option(
        key            = f"{prefix}_b",
        option_label   = "Option B — Enhanced",
        bench_label    = bench_label,
        today_label    = today_label,
        inception_year = inc_year,
    )
