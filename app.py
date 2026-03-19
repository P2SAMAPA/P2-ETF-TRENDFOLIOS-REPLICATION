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

def _get_split(config: str) -> str:
    """Return the first available split name for a given config."""
    from datasets import get_dataset_split_names
    try:
        splits = get_dataset_split_names(HF_REPO_ID, config, token=HF_TOKEN)
        return splits[0] if splits else "train"
    except Exception:
        return "train"


@st.cache_data(ttl=3600)
def load(config: str) -> pd.DataFrame:
    split = _get_split(config)
    ds = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
    df = ds.to_pandas()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


@st.cache_data(ttl=3600)
def load_latest_weights(prefix: str) -> pd.DataFrame:
    config = f"{prefix}_latest_weights"
    split  = _get_split(config)
    ds = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
    return ds.to_pandas()


@st.cache_data(ttl=3600)
def load_latest_optimal(prefix: str) -> dict:
    try:
        config = f"{prefix}_latest_optimal"
        split  = _get_split(config)
        ds = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
        return ds.to_pandas().iloc[0].to_dict()
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_optimal_params(prefix: str) -> pd.DataFrame:
    try:
        config = f"{prefix}_optimal_params"
        split  = _get_split(config)
        ds = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
        df = ds.to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
    except Exception:
        return pd.DataFrame()


# ── Colour palette ────────────────────────────────────────────────────────────

BLUE   = "#2563EB"
GRAY   = "#94A3B8"
GREEN  = "#16A34A"
RED    = "#DC2626"
AMBER  = "#D97706"

CANDIDATE_PERIODS = [3, 5, 10, 15]
CANDIDATE_SIZES   = [1, 2, 3]


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


def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) for Plotly fillcolor."""
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
st.caption(f"Benchmark: {bench_label} | Rolling optimised holding period & portfolio size | Fee: 0.55% p.a.")

# Load data
with st.spinner("Loading data from HuggingFace …"):
    try:
        growth_df    = load(f"{prefix}_growth")
        rolling_df   = load(f"{prefix}_rolling")
        summary_df   = load(f"{prefix}_summary")
        calendar_df  = load(f"{prefix}_calendar")
        latest_wts   = load_latest_weights(prefix)
        inclusion_df = load(f"{prefix}_inclusion")
        latest_opt   = load_latest_optimal(prefix)
        opt_params   = load_optimal_params(prefix)
        data_loaded  = True
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Run the seed script and daily pipeline first.")
        data_loaded = False

if data_loaded:
    # ── Hero box — optimal strategy snapshot ─────────────────────────────────
    if latest_opt:
        period    = latest_opt.get("optimal_period", "—")
        n_etfs    = latest_opt.get("optimal_n", "—")
        best_ret  = latest_opt.get("best_ann_return", None)
        holdings  = latest_opt.get("holdings", "")
        as_of     = latest_opt.get("as_of", "")
        ret_str   = f"{best_ret*100:.2f}%" if best_ret and not pd.isna(best_ret) else "—"
        hold_list = holdings.split(",") if holdings else []

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#1e3a5f,#1e40af);
                        border-radius:12px;padding:20px 28px;margin-bottom:16px;color:white;">
                <div style="font-size:11px;letter-spacing:2px;opacity:0.7;
                            text-transform:uppercase;margin-bottom:6px;">
                    🤖 Model Optimal Configuration — as of {as_of}
                </div>
                <div style="display:flex;gap:48px;flex-wrap:wrap;align-items:center">
                    <div>
                        <div style="font-size:32px;font-weight:700;line-height:1">{period}d</div>
                        <div style="font-size:12px;opacity:0.75;margin-top:2px">Optimal hold period</div>
                    </div>
                    <div>
                        <div style="font-size:32px;font-weight:700;line-height:1">{n_etfs}</div>
                        <div style="font-size:12px;opacity:0.75;margin-top:2px">ETFs held</div>
                    </div>
                    <div>
                        <div style="font-size:32px;font-weight:700;line-height:1;color:#4ade80">{ret_str}</div>
                        <div style="font-size:12px;opacity:0.75;margin-top:2px">Best ann. return (trailing 252d)</div>
                    </div>
                    <div>
                        <div style="font-size:15px;font-weight:600;line-height:1.4">
                            {" · ".join(f'<span style="background:rgba(255,255,255,0.15);'
                                        f'border-radius:6px;padding:2px 10px">{t}</span>'
                                        for t in hold_list)}
                        </div>
                        <div style="font-size:12px;opacity:0.75;margin-top:4px">Current holdings</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    try:
        # summary may have Period as index or as a column depending on HF round-trip
        sdf = summary_df.copy()
        if "Period" in sdf.columns:
            sdf = sdf.set_index("Period")
        si_row = sdf.loc["Since Inception"] if "Since Inception" in sdf.index else sdf.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Gross Return (SI)", f"{float(si_row['Composite Gross Return'])*100:.2f}%")
        col2.metric("Net Return (SI)",   f"{float(si_row['Composite Net Return'])*100:.2f}%")
        col3.metric("Excess Return",     f"{float(si_row['Excess Return (gross)'])*100:+.2f}%")
        col4.metric("Sharpe Ratio",      f"{float(si_row['Composite Sharpe']):.2f}")
        col5.metric("Info Ratio",        f"{float(si_row['Information Ratio']):.2f}")
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

    # ── Optimal params history ────────────────────────────────────────────────
    if not opt_params.empty:
        st.subheader("Rolling Optimal Configuration History")
        col_p, col_n = st.columns(2)
        with col_p:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=opt_params.index, y=opt_params["optimal_period"],
                mode="lines", name="Optimal Period",
                line=dict(color=BLUE, width=2),
                fill="tozeroy", fillcolor=hex_to_rgba(BLUE, 0.08),
            ))
            fig_p.update_layout(
                title="Optimal Holding Period (days)",
                yaxis=dict(tickvals=CANDIDATE_PERIODS, title="Days"),
                margin=dict(l=0, r=0, t=40, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_p, use_container_width=True)
        with col_n:
            fig_n = go.Figure()
            fig_n.add_trace(go.Scatter(
                x=opt_params.index, y=opt_params["optimal_n"],
                mode="lines", name="Optimal N",
                line=dict(color=AMBER, width=2),
                fill="tozeroy", fillcolor=hex_to_rgba(AMBER, 0.08),
            ))
            fig_n.update_layout(
                title="Optimal Number of ETFs Held",
                yaxis=dict(tickvals=CANDIDATE_SIZES, title="N ETFs"),
                margin=dict(l=0, r=0, t=40, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_n, use_container_width=True)

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
            styled = inc_recent.style.applymap(
                lambda v: "background-color:#bbf7d0" if v == 1 else "background-color:#fecaca"
            ).format("{:.0f}")
            st.dataframe(styled, use_container_width=True, height=400)
        except Exception:
            st.dataframe(inclusion_df.tail(60).T, use_container_width=True)

    # ── Performance table ─────────────────────────────────────────────────────
    st.subheader("Annualised Performance Summary")
    if not summary_df.empty:
        sdf = summary_df.copy()
        if "Period" in sdf.columns:
            sdf = sdf.set_index("Period")
        st.markdown(summary_html(sdf), unsafe_allow_html=True)

    # ── Calendar year ─────────────────────────────────────────────────────────
    st.subheader("Calendar Year Returns")
    if not calendar_df.empty:
        try:
            cal_display = calendar_df.copy()
            if "Year" in cal_display.columns:
                cal_display = cal_display.set_index("Year")
            for col in cal_display.columns:
                cal_display[col] = cal_display[col].map(
                    lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—"
                )
            st.dataframe(cal_display.sort_index(ascending=False), use_container_width=True)
        except Exception:
            st.dataframe(calendar_df, use_container_width=True)
