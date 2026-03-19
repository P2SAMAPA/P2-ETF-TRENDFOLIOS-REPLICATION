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

# ── Global font sizing ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Body text */
    html, body, [class*="css"] { font-size: 17px !important; }
    /* Metric labels and values */
    [data-testid="stMetricLabel"] { font-size: 16px !important; }
    [data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 700 !important; }
    /* Subheaders */
    h2, h3 { font-size: 22px !important; }
    /* Sidebar text */
    [data-testid="stSidebar"] { font-size: 15px !important; }
    /* Table text */
    table td, table th { font-size: 15px !important; }
    /* Dataframe */
    .stDataFrame { font-size: 15px !important; }
    /* Caption */
    [data-testid="stCaptionContainer"] { font-size: 15px !important; }
</style>
""", unsafe_allow_html=True)

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


@st.cache_data(ttl=900)
def load(config: str) -> pd.DataFrame:
    split = _get_split(config)
    ds    = load_dataset(HF_REPO_ID, config, split=split, token=HF_TOKEN)
    df    = ds.to_pandas()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


@st.cache_data(ttl=900)
def load_holdings_and_config(prefix: str) -> tuple[pd.DataFrame, dict]:
    """
    Two separate reliable sources:
    - {prefix}_config.json  : plain JSON file in HF repo → config values
    - {prefix}_weights      : large parquet dataset → current holdings (last row)
    """
    holdings       = pd.DataFrame(columns=["ticker", "weight"])
    opt            = {}
    inception_year = 2005 if prefix == "equity" else 2007

    # ── Load config from JSON file ────────────────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download
        json_path = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = f"{prefix}_config.json",
            repo_type = "dataset",
            token     = HF_TOKEN,
        )
        import json
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
    except Exception as e:
        st.warning(f"Could not load {prefix}_config.json: {e}")

    # ── Load holdings from last row of weights dataset ────────────────────────
    try:
        weights_df = load(f"{prefix}_weights")
        if not weights_df.empty:
            last = weights_df.iloc[-1]
            as_of = str(weights_df.index[-1].date())
            w = last.apply(pd.to_numeric, errors="coerce")
            w = w[w > 1e-6].sort_values(ascending=False)
            if not w.empty:
                holdings = pd.DataFrame({
                    "ticker": w.index.tolist(),
                    "weight": w.values.tolist(),
                })
            if not opt:
                opt["as_of"] = as_of
            opt["optimal_n"] = len(holdings)
            opt["is_invested"] = len(holdings) > 0
    except Exception as e:
        st.warning(f"Could not load {prefix}_weights: {e}")

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
        rolling_df          = load(f"{prefix}_rolling")
        summary_df          = load(f"{prefix}_summary")
        calendar_df         = load(f"{prefix}_calendar")
        latest_wts, latest_opt = load_holdings_and_config(prefix)
        data_loaded         = True
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

# Gather data — coerce types robustly from parquet round-trip
def _safe(v, fmt=str, fallback="—"):
    try:
        return fmt(v) if v is not None and str(v) not in ("nan", "None", "", "0") else fallback
    except Exception:
        return fallback

opt_period   = latest_opt.get("optimal_period")
opt_n        = latest_opt.get("optimal_n")
target_n     = latest_opt.get("target_n", opt_n)
best_ret     = latest_opt.get("best_ann_return")
as_of        = latest_opt.get("as_of", "")
raw_method   = latest_opt.get("optimal_method", "inv_te")
method_label = "Inverse-TE weighting" if raw_method == "inv_te" else "Momentum-rank weighting"
method_emoji = "⚖️" if raw_method == "inv_te" else "🚀"

def _fmt_int(v, fallback="—") -> str:
    """Format an int value, showing fallback if None/0/nan."""
    try:
        i = int(float(v))
        return str(i) if i > 0 else fallback
    except Exception:
        return fallback

def _fmt_pct(v, fallback="—") -> str:
    """Format a float as percentage, showing fallback if None/nan."""
    try:
        f = float(v)
        return f"{f*100:.2f}%" if not np.isnan(f) else fallback
    except Exception:
        return fallback

opt_period_str = _fmt_int(opt_period)
opt_n_str      = _fmt_int(opt_n)
target_n_str   = _fmt_int(target_n, opt_n_str)
ret_str        = _fmt_pct(best_ret)
as_of_str      = str(as_of) if as_of else "—"

# Holdings already filtered and sorted by load_holdings_and_config
holdings    = latest_wts
is_invested = len(holdings) > 0

# ── ETF cards ─────────────────────────────────────────────────────────────────
if is_invested:
    card_html = ""
    for _, row in holdings.iterrows():
        card_html += f"""
        <div style="background:#f0f4ff;border:2.5px solid #2563EB;border-radius:14px;
                    padding:22px 36px;display:inline-block;margin:8px 10px;
                    text-align:center;min-width:130px">
            <div style="font-size:32px;font-weight:800;color:#1e3a8a;
                        letter-spacing:2px">{row['ticker']}</div>
            <div style="font-size:18px;color:#2563EB;font-weight:700;
                        margin-top:6px">{row['weight']*100:.1f}%</div>
        </div>"""
else:
    card_html = """
    <div style="background:#fef9c3;border:2px solid #ca8a04;border-radius:12px;
                padding:20px 28px;display:inline-block;margin:6px 0;color:#713f12">
        <span style="font-size:22px;margin-right:10px">⚠️</span>
        <span style="font-size:18px;font-weight:600">
            No signal — all ETFs excluded by momentum/trend filter. Stay in cash.
        </span>
    </div>"""

# ── Config footer ──────────────────────────────────────────────────────────────
pill = (
    "background:#e8f0fe;border-radius:8px;padding:8px 16px;"
    "font-size:15px;color:#1e40af;font-weight:600;"
    "margin-right:10px;display:inline-block;margin-bottom:6px"
)
config_html = (
    f'<span style="{pill}">⏱ Hold {opt_period_str}d</span>'
    f'<span style="{pill}">📦 Target {target_n_str} ETF(s) · Actual {opt_n_str}</span>'
    f'<span style="{pill}">{method_emoji} {method_label}</span>'
    f'<span style="{pill}">📈 {ret_str} best ann. return (trailing 252d)</span>'
    f'<span style="{pill}">📅 Signals as of {as_of_str}</span>'
)

action_html = f"""
<div style="background:#ffffff;border:2.5px solid #2563EB;border-radius:16px;
            padding:28px 32px;margin:8px 0 28px 0;">
    <div style="font-size:13px;letter-spacing:3px;color:#6b7280;
                text-transform:uppercase;margin-bottom:20px;font-weight:700">
        🎯 &nbsp;Action for {today_label} — Hold at Market Open
    </div>
    <div style="margin-bottom:24px">{card_html}</div>
    <div style="border-top:1.5px solid #e5e7eb;padding-top:14px">
        {config_html}
    </div>
</div>
"""
st.markdown(action_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KPI ROW
# ════════════════════════════════════════════════════════════════════════════════

inception_year = latest_opt.get("inception_year", 2005 if prefix == "equity" else 2007)
si_label = f"Since {inception_year}"

try:
    sdf = summary_df.copy()
    if "Period" in sdf.columns:
        sdf = sdf.set_index("Period")
    si  = sdf.loc["Since Inception"] if "Since Inception" in sdf.index else sdf.iloc[-1]
    st.caption(f"📅 Performance metrics shown from 1 Jan {inception_year} — 18 Mar 2026")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Gross Return ({si_label})",  f"{float(si['Composite Gross Return'])*100:.2f}%")
    c2.metric(f"Net Return ({si_label})",    f"{float(si['Composite Net Return'])*100:.2f}%")
    c3.metric(f"Excess Return ({si_label})", f"{float(si['Excess Return (gross)'])*100:+.2f}%")
    c4.metric("Sharpe Ratio",                f"{float(si['Composite Sharpe']):.2f}")
    c5.metric("Info Ratio",                  f"{float(si['Information Ratio']):.2f}")
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
