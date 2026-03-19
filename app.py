"""
app.py
------
Streamlit dashboard for TrendFolios® ETF replication.
Reads all data from HuggingFace Dataset.

Run locally:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

HF_REPO_ID = "P2SAMAPA/p2-etf-trendfolios-replication-data"
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

st.set_page_config(
    page_title="TrendFolios® ETF Replication",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load(config: str) -> pd.DataFrame:
    ds = load_dataset(HF_REPO_ID, config, split="train", token=HF_TOKEN)
    df = ds.to_pandas()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


@st.cache_data(ttl=3600)
def load_latest_weights(prefix: str) -> pd.DataFrame:
    ds = load_dataset(HF_REPO_ID, f"{prefix}_latest_weights", split="train", token=HF_TOKEN)
    return ds.to_pandas()


# ── Colour palette ────────────────────────────────────────────────────────────

BLUE   = "#2563EB"
GRAY   = "#94A3B8"
GREEN  = "#16A34A"
RED    = "#DC2626"
AMBER  = "#D97706"


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def growth_chart(growth_df: pd.DataFrame, title: str, port_label: str, bench_label: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=growth_df.index, y=growth_df["portfolio_gross"],
        name=port_label, line=dict(color=BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=growth_df.index, y=growth_df["benchmark"],
        name=bench_label, line=dict(color=GRAY, width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title=title, yaxis_title="Growth of $1",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    return fig


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
            fillcolor=colours[col] + "22",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=GRAY, line_width=1)
    fig.update_layout(
        title=title, yaxis_title="Annualised Excess Return (%)",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    return fig


def weights_chart(latest: pd.DataFrame, title: str):
    latest = latest.sort_values("weight", ascending=True)
    fig = go.Figure(go.Bar(
        x=latest["weight"] * 100,
        y=latest["ticker"],
        orientation="h",
        marker_color=BLUE,
        text=[f"{w*100:.1f}%" for w in latest["weight"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Weight (%)",
        margin=dict(l=0, r=60, t=40, b=0),
    )
    return fig


def summary_html(summary_df: pd.DataFrame) -> str:
    pct_cols = [
        "Composite Gross Return", "Composite Net Return", "Index Return",
        "Excess Return (gross)", "Excess Return (net)",
        "Composite Std Dev", "Index Std Dev", "Tracking Error",
        "Max Drawdown",
    ]
    ratio_cols = ["Composite Sharpe", "Index Sharpe", "Information Ratio"]

    rows_html = ""
    for period, row in summary_df.iterrows():
        cells = f"<td><b>{period}</b></td>"
        for col in summary_df.columns:
            v = row[col]
            if col in pct_cols:
                cells += f"<td>{fmt_pct(v)}</td>"
            elif col in ratio_cols:
                cells += f"<td>{fmt_ratio(v)}</td>"
            else:
                cells += f"<td>{v}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in ["Period"] + list(summary_df.columns))
    return f"""
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="background:#1e293b;color:#f1f5f9">{headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """


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
        "Inverse tracking-error weighting. Biweekly rebalance."
    )
    st.markdown("---")
    st.markdown(
        "[📄 Research Paper](https://arxiv.org/abs/2506.09330)  \n"
        "[💾 HuggingFace Dataset](https://huggingface.co/datasets/P2SAMAPA/p2-etf-trendfolios-replication-data)  \n"
        "[🐙 GitHub](https://github.com/P2SAMAPA/P2-ETF-TRENDFOLIOS-REPLICATION)"
    )


# ── Main content ──────────────────────────────────────────────────────────────

prefix      = "equity" if universe == "Equity" else "fixed_income"
bench_label = "SPY"    if universe == "Equity" else "AGG"

st.title(f"TrendFolios® — {universe} Strategy")
st.caption(f"Benchmark: {bench_label} | Rebalance: Biweekly | Fee: 0.55% p.a.")

# Load data
with st.spinner("Loading data from HuggingFace …"):
    try:
        growth_df  = load(f"{prefix}_growth")
        rolling_df = load(f"{prefix}_rolling")
        summary_df = load(f"{prefix}_summary")
        calendar_df = load(f"{prefix}_calendar")
        latest_wts  = load_latest_weights(prefix)
        inclusion_df = load(f"{prefix}_inclusion")
        data_loaded = True
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Run the seed script and weekly pipeline first.")
        data_loaded = False

if data_loaded:
    # ── KPI row ───────────────────────────────────────────────────────────────
    try:
        since_inception = summary_df[summary_df.index == "Since Inception"].iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Gross Return (SI)", f"{since_inception['Composite Gross Return']*100:.2f}%")
        col2.metric("Net Return (SI)",   f"{since_inception['Composite Net Return']*100:.2f}%")
        col3.metric("Excess Return",     f"{since_inception['Excess Return (gross)']*100:+.2f}%")
        col4.metric("Sharpe Ratio",      f"{since_inception['Composite Sharpe']:.2f}")
        col5.metric("Info Ratio",        f"{since_inception['Information Ratio']:.2f}")
    except Exception:
        pass

    st.markdown("---")

    # ── Charts row ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            growth_chart(growth_df, f"Growth of $1 — {universe}", universe, bench_label),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            rolling_chart(rolling_df, f"Rolling Annualised Excess Return — {universe}"),
            use_container_width=True,
        )

    # ── Current allocation ────────────────────────────────────────────────────
    st.subheader("Current Portfolio Allocation")
    col_w, col_inc = st.columns([1, 2])
    with col_w:
        if not latest_wts.empty:
            st.plotly_chart(
                weights_chart(latest_wts, "Latest Weights"),
                use_container_width=True,
            )
    with col_inc:
        st.markdown("**Inclusion Signal (recent 60 days)**")
        try:
            inc_recent = inclusion_df.tail(60).T
            inc_recent.columns = [str(d.date()) for d in inc_recent.columns]
            # Style: green=1, red=0
            styled = inc_recent.style.applymap(
                lambda v: "background-color:#bbf7d0" if v == 1 else "background-color:#fecaca"
            ).format("{:.0f}")
            st.dataframe(styled, use_container_width=True, height=400)
        except Exception:
            st.dataframe(inclusion_df.tail(60).T, use_container_width=True)

    # ── Performance table ─────────────────────────────────────────────────────
    st.subheader("Annualised Performance Summary")
    if not summary_df.empty:
        st.markdown(summary_html(summary_df), unsafe_allow_html=True)

    # ── Calendar year ─────────────────────────────────────────────────────────
    st.subheader("Calendar Year Returns")
    if not calendar_df.empty:
        try:
            cal_display = calendar_df.copy()
            for col in cal_display.columns:
                if col != "Year":
                    cal_display[col] = cal_display[col].map(
                        lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—"
                    )
            st.dataframe(cal_display.set_index("Year"), use_container_width=True)
        except Exception:
            st.dataframe(calendar_df, use_container_width=True)
