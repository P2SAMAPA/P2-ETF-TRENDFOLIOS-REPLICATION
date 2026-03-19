"""
app.py
------
TrendFolios® ETF Replication — Streamlit Dashboard
Reads all data from HuggingFace Dataset.
"""

import os
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

# ── Colour palette ────────────────────────────────────────────────────────────

BLUE       = "#2563EB"
GRAY       = "#94A3B8"
GREEN      = "#16A34A"
RED        = "#DC2626"
AMBER      = "#D97706"

CANDIDATE_PERIODS = [3, 5, 10, 15]
CANDIDATE_SIZES   = [1, 2, 3]

# ── Data loading ──────────────────────────────────────────────────────────────

def _get_split(config: str) -> str:
    try:
        splits = get_dataset_split_names(HF_REPO_ID, config, token=HF_TOKEN)
        return splits[0] if splits else "train"
    except Exception:
        return "train"


@st.cache_data(ttl=3600)
def load(config: str) -> pd.DataFrame:
    split = _get_split(config)
    ds    = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
    df    = ds.to_pandas()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


@st.cache_data(ttl=3600)
def load_latest_optimal(prefix: str) -> dict:
    try:
        config = f"{prefix}_latest_optimal"
        split  = _get_split(config)
        ds     = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
        df     = ds.to_pandas()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_latest_weights(prefix: str) -> pd.DataFrame:
    try:
        config = f"{prefix}_latest_weights"
        split  = _get_split(config)
        ds     = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
        df     = ds.to_pandas()
        return df if not df.empty else pd.DataFrame(columns=["ticker", "weight"])
    except Exception:
        return pd.DataFrame(columns=["ticker", "weight"])


# ── Chart helpers ─────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def rolling_chart(rolling_df: pd.DataFrame, title: str):
    fig     = go.Figure()
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
    pct_cols   = [
        "Composite Gross Return", "Composite Net Return", "Index Return",
        "Excess Return (gross)", "Excess Return (net)",
        "Composite Std Dev", "Index Std Dev", "Tracking Error", "Max Drawdown",
    ]
    ratio_cols = ["Composite Sharpe", "Index Sharpe", "Information Ratio"]
    rows_html  = ""
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
        "[📄 Research Paper](https://arxiv.org/abs/2506.09330)  \n"
        "[💾 HuggingFace Dataset](https://huggingface.co/datasets/P2SAMAPA/p2-etf-trendfolios-replication-data)  \n"
        "[🐙 GitHub](https://github.com/P2SAMAPA/P2-ETF-TRENDFOLIOS-REPLICATION)"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

prefix      = "equity"       if universe == "Equity" else "fixed_income"
bench_label = "SPY"          if universe == "Equity" else "AGG"
today_label = pd.Timestamp.today().strftime("%d %b %Y")

st.title(f"TrendFolios® — {universe} Strategy")
st.caption(
    f"Benchmark: {bench_label} | "
    "Rolling optimised holding period & portfolio size | Fee: 0.55% p.a."
)

# Load all data
with st.spinner("Loading …"):
    try:
        rolling_df   = load(f"{prefix}_rolling")
        summary_df   = load(f"{prefix}_summary")
        calendar_df  = load(f"{prefix}_calendar")
        latest_opt   = load_latest_optimal(prefix)
        latest_wts   = load_latest_weights(prefix)
        data_loaded  = True
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Run the seed script and daily pipeline first.")
        data_loaded = False

if not data_loaded:
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — WHAT TO HOLD TODAY  (most prominent)
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")

# Gather data
opt_period   = latest_opt.get("optimal_period", "—")
opt_n        = latest_opt.get("optimal_n", "—")
best_ret     = latest_opt.get("best_ann_return", None)
as_of        = latest_opt.get("as_of", "")
ret_str      = f"{float(best_ret)*100:.2f}%" if best_ret and not pd.isna(float(best_ret)) else "—"

# Holdings: use latest_weights (top N by weight, already filtered by pipeline)
if not latest_wts.empty and "ticker" in latest_wts.columns:
    holdings = (
        latest_wts[latest_wts["weight"] > 0]
        .sort_values("weight", ascending=False)
        .head(int(opt_n) if str(opt_n).isdigit() else 3)
    )
else:
    holdings = pd.DataFrame(columns=["ticker", "weight"])

# Build holding cards HTML
pill_bg   = "rgba(255,255,255,0.18)"
card_html = ""
for _, row in holdings.iterrows():
    ticker = row["ticker"]
    weight = row["weight"]
    card_html += f"""
    <div style="background:{pill_bg};border-radius:10px;padding:14px 22px;
                display:inline-block;margin:6px 8px;text-align:center;min-width:90px">
        <div style="font-size:22px;font-weight:700;letter-spacing:1px">{ticker}</div>
        <div style="font-size:13px;opacity:0.8;margin-top:2px">{weight*100:.1f}%</div>
    </div>"""

if not card_html:
    card_html = '<div style="font-size:16px;opacity:0.7;padding:10px">No holdings signal — all assets excluded by momentum/trend filter</div>'

# Config pills
config_style = "background:rgba(255,255,255,0.12);border-radius:6px;padding:3px 12px;font-size:13px;margin-right:10px"
config_html  = (
    f'<span style="{config_style}">⏱ Hold {opt_period}d</span>'
    f'<span style="{config_style}">📦 {opt_n} ETF(s)</span>'
    f'<span style="{config_style}">📈 {ret_str} ann. return (trailing 252d)</span>'
    f'<span style="{config_style}">📅 Signals as of {as_of}</span>'
)

action_html = f"""
<div style="background:linear-gradient(135deg,#0f2444,#1a3a8f);
            border-radius:14px;padding:24px 28px;margin:8px 0 20px 0;color:white;">
    <div style="font-size:11px;letter-spacing:3px;opacity:0.6;
                text-transform:uppercase;margin-bottom:14px;">
        🎯 &nbsp;Action for {today_label} — Hold at Market Open
    </div>
    <div style="margin-bottom:18px">{card_html}</div>
    <div style="border-top:1px solid rgba(255,255,255,0.12);padding-top:12px">
        {config_html}
    </div>
</div>
"""
st.markdown(action_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KPI ROW
# ════════════════════════════════════════════════════════════════════════════════

try:
    sdf = summary_df.copy()
    if "Period" in sdf.columns:
        sdf = sdf.set_index("Period")
    si  = sdf.loc["Since Inception"] if "Since Inception" in sdf.index else sdf.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gross Return (SI)", f"{float(si['Composite Gross Return'])*100:.2f}%")
    c2.metric("Net Return (SI)",   f"{float(si['Composite Net Return'])*100:.2f}%")
    c3.metric("Excess Return",     f"{float(si['Excess Return (gross)'])*100:+.2f}%")
    c4.metric("Sharpe Ratio",      f"{float(si['Composite Sharpe']):.2f}")
    c5.metric("Info Ratio",        f"{float(si['Information Ratio']):.2f}")
except Exception:
    pass

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ROLLING EXCESS RETURNS
# ════════════════════════════════════════════════════════════════════════════════

if not rolling_df.empty:
    st.subheader("Rolling Annualised Excess Returns vs Benchmark")
    st.plotly_chart(
        rolling_chart(rolling_df, f"{universe} Strategy — Excess Return over {bench_label}"),
        width="stretch",
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ANNUALISED PERFORMANCE SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("Annualised Performance Summary")
if not summary_df.empty:
    try:
        sdf = summary_df.copy()
        if "Period" in sdf.columns:
            sdf = sdf.set_index("Period")
        st.markdown(summary_html(sdf), unsafe_allow_html=True)
    except Exception as e:
        st.dataframe(summary_df, width="stretch")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CALENDAR YEAR RETURNS
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("Calendar Year Returns")
if not calendar_df.empty:
    try:
        cdf = calendar_df.copy()
        if "Year" in cdf.columns:
            cdf = cdf.set_index("Year")
        for col in cdf.columns:
            cdf[col] = cdf[col].map(
                lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—"
            )
        st.dataframe(cdf.sort_index(ascending=False), width="stretch")
    except Exception:
        st.dataframe(calendar_df, width="stretch")
