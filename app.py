"""
app.py — Stock Market Index Simulator
A real-time Streamlit dashboard simulating index construction & rebalancing.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import deque

from data_engine import (
    initialize_dataframe,
    simulate_price_tick,
    construct_index,
    compute_index_value,
    rebalance,
    sector_summary,
    filter_by_sector,
    top_movers,
)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Index Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

/* Root palette */
:root {
    --bg: #0d0f14;
    --surface: #141720;
    --surface2: #1c2030;
    --accent: #00e5a0;
    --accent2: #3d8ef8;
    --red: #ff4d6d;
    --green: #00e5a0;
    --muted: #5a6484;
    --text: #e4e8f5;
    --text-dim: #8892b0;
    --border: #252b3b;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Hide Streamlit default chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: var(--accent);
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; color: var(--text) !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00b377) !important;
    color: #0d0f14 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0, 229, 160, 0.3);
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

/* Badge chips */
.badge-in { background: rgba(0,229,160,0.15); color: var(--green); border: 1px solid rgba(0,229,160,0.3); border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }
.badge-added { background: rgba(61,142,248,0.15); color: var(--accent2); border: 1px solid rgba(61,142,248,0.3); border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }
.badge-dropped { background: rgba(255,77,109,0.15); color: var(--red); border: 1px solid rgba(255,77,109,0.3); border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }

/* Index value banner */
.index-banner {
    background: linear-gradient(135deg, #141720, #1c2030);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 16px;
}
.index-value { font-family: 'Space Mono', monospace; font-size: 2.8rem; font-weight: 700; color: var(--accent); }
.index-label { font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1.5px; }
.index-change-pos { color: var(--green); font-size: 1.1rem; font-weight: 600; }
.index-change-neg { color: var(--red); font-size: 1.1rem; font-weight: 600; }

/* Ticker tape style movers */
.mover-gain { color: var(--green); font-weight: 600; }
.mover-loss { color: var(--red); font-weight: 600; }

/* Alert boxes */
.rebalance-alert {
    background: rgba(61,142,248,0.08);
    border: 1px solid rgba(61,142,248,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────

def init_state():
    if "df" not in st.session_state:
        st.session_state.df = initialize_dataframe()

    if "index_df" not in st.session_state:
        st.session_state.index_df = construct_index(
            st.session_state.df, top_n=st.session_state.get("top_n", 10)
        )

    if "base_mcap" not in st.session_state:
        st.session_state.base_mcap = st.session_state.index_df["market_cap"].sum()

    if "index_history" not in st.session_state:
        st.session_state.index_history = deque(maxlen=120)  # last 2 min at 1s ticks
        now = datetime.now().strftime("%H:%M:%S")
        st.session_state.index_history.append(
            {"time": now, "index_value": 1000.0}
        )

    if "tick" not in st.session_state:
        st.session_state.tick = 0

    if "rebalance_log" not in st.session_state:
        st.session_state.rebalance_log = []

    if "last_rebalance_tick" not in st.session_state:
        st.session_state.last_rebalance_tick = 0

    if "added_tickers" not in st.session_state:
        st.session_state.added_tickers = set()

    if "removed_tickers" not in st.session_state:
        st.session_state.removed_tickers = set()

    if "prev_index_value" not in st.session_state:
        st.session_state.prev_index_value = 1000.0


init_state()


# ─── Sidebar Controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    top_n = st.slider("Index Size (Top N)", min_value=5, max_value=20, value=10, step=1)
    volatility = st.slider("Volatility", min_value=0.2, max_value=3.0, value=1.0, step=0.1,
                            help="Controls how much prices move each tick")
    auto_rebalance = st.toggle("Auto Rebalance", value=True)
    rebalance_every = st.slider("Rebalance Frequency (ticks)", min_value=5, max_value=30,
                                 value=10, step=5, disabled=not auto_rebalance)
    sector_filter = st.selectbox(
        "Filter Universe by Sector",
        ["All"] + sorted(st.session_state.df["sector"].unique().tolist())
    )

    st.divider()
    rebalance_btn = st.button("🔄 Rebalance Now", use_container_width=True)

    st.divider()
    live_mode = st.toggle("▶ Live Simulation", value=True)
    refresh_rate = st.slider("Refresh Rate (sec)", min_value=1, max_value=5, value=2)

    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem; color: #5a6484; font-family: Space Mono, monospace;'>
    <b>INDEX ENGINE</b><br>
    Market-cap weighted<br>
    Top-N selection<br>
    Auto rebalancing<br>
    Pandas-native logic
    </div>
    """, unsafe_allow_html=True)


# ─── Tick Logic ───────────────────────────────────────────────────────────────

def run_tick():
    """One simulation step: update prices, rebalance if needed, log index value."""
    st.session_state.tick += 1
    df = simulate_price_tick(st.session_state.df, volatility=volatility)
    st.session_state.df = df

    current_tickers = set(st.session_state.index_df["ticker"].tolist())

    # Decide whether to rebalance
    do_rebalance = rebalance_btn or (
        auto_rebalance and
        (st.session_state.tick - st.session_state.last_rebalance_tick) >= rebalance_every
    )

    if do_rebalance:
        new_idx, added, removed = rebalance(df, current_tickers, top_n=top_n)
        st.session_state.index_df = new_idx
        st.session_state.last_rebalance_tick = st.session_state.tick
        st.session_state.added_tickers = set(added)
        st.session_state.removed_tickers = set(removed)

        if added or removed:
            log_entry = {
                "tick": st.session_state.tick,
                "time": datetime.now().strftime("%H:%M:%S"),
                "added": ", ".join(added) if added else "—",
                "removed": ", ".join(removed) if removed else "—",
            }
            st.session_state.rebalance_log.insert(0, log_entry)
            st.session_state.rebalance_log = st.session_state.rebalance_log[:15]
    else:
        # Just refresh weights without changing constituents
        updated = construct_index(df, top_n=top_n)
        # Filter to current constituents only to avoid inadvertent swaps
        current_rows = df[df["ticker"].isin(current_tickers)].copy()
        total_mc = current_rows["market_cap"].sum()
        current_rows["weight"] = (current_rows["market_cap"] / total_mc * 100).round(3)
        st.session_state.index_df = current_rows.sort_values("market_cap", ascending=False)

    # Update index value history
    idx_value = compute_index_value(
        st.session_state.index_df,
        base_value=1000.0,
        base_mcap=st.session_state.base_mcap,
    )
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.index_history.append({"time": now, "index_value": idx_value})
    st.session_state.prev_index_value = list(st.session_state.index_history)[-2]["index_value"] \
        if len(st.session_state.index_history) > 1 else 1000.0


if live_mode:
    run_tick()


# ─── Main Layout ──────────────────────────────────────────────────────────────

# Title bar
col_title, col_tick = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div style='font-family: Space Mono, monospace;'>
        <span style='font-size:1.6rem; font-weight:700; color:#e4e8f5;'>📊 STOCK INDEX</span>
        <span style='font-size:1.6rem; font-weight:300; color:#5a6484;'> SIMULATOR</span>
    </div>
    <div style='font-size:0.72rem; color:#5a6484; letter-spacing:2px; margin-top:2px;'>
        REAL-TIME · MARKET-CAP WEIGHTED · PANDAS-POWERED
    </div>
    """, unsafe_allow_html=True)
with col_tick:
    st.metric("Tick", st.session_state.tick, delta=None)


st.divider()

# ─── Index Value Banner ───────────────────────────────────────────────────────

history_list = list(st.session_state.index_history)
current_idx_val = history_list[-1]["index_value"]
delta_val = current_idx_val - st.session_state.prev_index_value
delta_pct = (delta_val / st.session_state.prev_index_value * 100) if st.session_state.prev_index_value else 0
sign = "▲" if delta_val >= 0 else "▼"
chg_class = "index-change-pos" if delta_val >= 0 else "index-change-neg"

col_banner, col_kpi1, col_kpi2, col_kpi3 = st.columns([2, 1, 1, 1])

with col_banner:
    st.markdown(f"""
    <div class="index-banner">
        <div class="index-label">SIMULATED MARKET INDEX (SMI-{top_n})</div>
        <div style='display:flex; align-items:baseline; gap:16px;'>
            <div class="index-value">{current_idx_val:,.2f}</div>
            <div class="{chg_class}">{sign} {abs(delta_val):.2f} ({delta_pct:+.3f}%)</div>
        </div>
        <div style='font-size:0.72rem; color:#5a6484; margin-top:4px;'>
            Base: 1,000.00 · Constituents: {top_n} · Tick #{st.session_state.tick}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Total market cap of index
idx_mcap = st.session_state.index_df["market_cap"].sum() / 1e12
all_mcap = st.session_state.df["market_cap"].sum() / 1e12
avg_weight = st.session_state.index_df["weight"].mean() if "weight" in st.session_state.index_df.columns else 0
sectors_in_idx = st.session_state.index_df["sector"].nunique()

with col_kpi1:
    st.metric("Index Mkt Cap", f"${idx_mcap:.2f}T", delta=None)
with col_kpi2:
    st.metric("Avg Weight", f"{avg_weight:.2f}%", delta=None)
with col_kpi3:
    st.metric("Sectors Covered", sectors_in_idx, delta=None)


# ─── Top Movers ───────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">⚡ Top Movers</div>', unsafe_allow_html=True)
gainers, losers = top_movers(st.session_state.df, n=5)

col_g, col_l = st.columns(2)

def fmt_mover_row(row, color_class):
    arrow = "▲" if row["pct_change"] >= 0 else "▼"
    return f'<span style="color:#8892b0;">{row["ticker"]}</span> <b>{row["name"]}</b> — ' \
           f'<span class="{color_class}">{arrow} {abs(row["pct_change"]):.3f}%</span> ' \
           f'<span style="color:#5a6484;">(${row["price"]:.2f})</span>'

with col_g:
    st.markdown("**🟢 Gainers**")
    for _, row in gainers.iterrows():
        st.markdown(fmt_mover_row(row, "mover-gain"), unsafe_allow_html=True)

with col_l:
    st.markdown("**🔴 Losers**")
    for _, row in losers[::-1].iterrows():
        st.markdown(fmt_mover_row(row, "mover-loss"), unsafe_allow_html=True)


st.divider()

# ─── Main Tables ─────────────────────────────────────────────────────────────

left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown('<div class="section-header">🌐 Company Universe</div>', unsafe_allow_html=True)

    display_df = filter_by_sector(st.session_state.df, sector_filter).copy()
    idx_tickers = set(st.session_state.index_df["ticker"].tolist())

    # Tag index membership
    display_df["In Index"] = display_df["ticker"].apply(
        lambda t: "✅ INDEX" if t in idx_tickers else ""
    )
    display_df["Change"] = display_df["pct_change"].apply(
        lambda x: f"▲ {x:.3f}%" if x >= 0 else f"▼ {abs(x):.3f}%"
    )
    display_df["Mkt Cap ($B)"] = (display_df["market_cap"] / 1e9).round(2)
    display_df["Price ($)"] = display_df["price"].round(2)

    show_cols = ["ticker", "name", "sector", "Price ($)", "Change", "Mkt Cap ($B)", "In Index"]
    st.dataframe(
        display_df[show_cols].rename(columns={"ticker": "Ticker", "name": "Company", "sector": "Sector"}),
        height=430,
        use_container_width=True,
        hide_index=True,
    )

with right_col:
    st.markdown(f'<div class="section-header">🏆 Index Constituents (Top {top_n})</div>',
                unsafe_allow_html=True)

    idx_display = st.session_state.index_df.copy()
    idx_display["Weight (%)"] = idx_display["weight"].apply(lambda x: f"{x:.3f}%")
    idx_display["Mkt Cap ($B)"] = (idx_display["market_cap"] / 1e9).round(2)
    idx_display["Price ($)"] = idx_display["price"].round(2)
    idx_display["Δ (%)"] = idx_display["pct_change"].apply(
        lambda x: f"▲ {x:.3f}%" if x >= 0 else f"▼ {abs(x):.3f}%"
    )

    # Highlight newly added / removed tickers
    def tag_status(ticker):
        if ticker in st.session_state.added_tickers:
            return "🆕 ADDED"
        return ""

    idx_display["Status"] = idx_display["ticker"].apply(tag_status)

    show_idx_cols = ["ticker", "name", "sector", "Price ($)", "Δ (%)", "Mkt Cap ($B)", "Weight (%)", "Status"]
    st.dataframe(
        idx_display[show_idx_cols].rename(columns={"ticker": "Ticker", "name": "Company", "sector": "Sector"}),
        height=430,
        use_container_width=True,
        hide_index=True,
    )


st.divider()

# ─── Charts ───────────────────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.markdown('<div class="section-header">📈 Index Value Over Time</div>', unsafe_allow_html=True)
    hist_df = pd.DataFrame(list(st.session_state.index_history))
    hist_df = hist_df.set_index("time")
    st.line_chart(hist_df["index_value"], height=220, use_container_width=True)

with chart_col2:
    st.markdown('<div class="section-header">🏭 Sector Market Cap ($B)</div>', unsafe_allow_html=True)
    sec_df = sector_summary(st.session_state.df)[["sector", "total_market_cap_B"]].set_index("sector")
    st.bar_chart(sec_df["total_market_cap_B"], height=220, use_container_width=True)


st.divider()

# ─── Sector Summary Table ─────────────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Sector Breakdown</div>', unsafe_allow_html=True)
sec_summary = sector_summary(st.session_state.df)
sec_summary["avg_price"] = sec_summary["avg_price"].round(2)
sec_summary = sec_summary.rename(columns={
    "sector": "Sector",
    "total_market_cap_B": "Total Mkt Cap ($B)",
    "avg_price": "Avg Price ($)",
    "num_companies": "Companies",
})
st.dataframe(
    sec_summary[["Sector", "Companies", "Total Mkt Cap ($B)", "Avg Price ($)"]],
    use_container_width=True,
    hide_index=True,
)


st.divider()

# ─── Rebalance Log ────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🔄 Rebalance History</div>', unsafe_allow_html=True)

if st.session_state.rebalance_log:
    log_df = pd.DataFrame(st.session_state.rebalance_log)
    log_df = log_df.rename(columns={
        "tick": "Tick", "time": "Time", "added": "Added", "removed": "Removed"
    })
    st.dataframe(log_df, use_container_width=True, hide_index=True, height=200)
else:
    st.markdown(
        '<div class="rebalance-alert">No rebalancing events yet — '
        'constituents have been stable since inception.</div>',
        unsafe_allow_html=True,
    )


# ─── Auto-refresh ────────────────────────────────────────────────────────────

if live_mode:
    time.sleep(refresh_rate)
    st.rerun()
