import streamlit as st
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timezone
from collections import Counter

# ====================================
# CONFIGURATION
# ====================================

DB_FILE = "nexus_signals.db"
PERFORMANCE_FILE = "performance.json"

st.set_page_config(
    page_title="Nexus Intelligence Dashboard",
    layout="wide"
)

# ====================================
# DATABASE HELPERS
# ====================================

def fetch_latest_signals():
    conn = sqlite3.connect(DB_FILE)

    query = """
        SELECT *
        FROM signals
        WHERE ts >= datetime('now', '-24 hours')
        ORDER BY ts DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def fetch_active_signals():
    conn = sqlite3.connect(DB_FILE)

    query = """
        SELECT *
        FROM signals
        WHERE status = 'ACTIVE'
        ORDER BY ts DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_performance():
    if not os.path.exists(PERFORMANCE_FILE):
        return {}
    with open(PERFORMANCE_FILE, "r") as f:
        return json.load(f)

# ====================================
# CONSENSUS CLASSIFIER (4 Engines)
# ====================================

def classify_consensus(directions):
    counts = Counter(directions)

    if len(counts) == 1 and len(directions) == 4:
        return "DIAMOND 💎"
    elif any(v >= 3 for v in counts.values()):
        return "PLATINUM 🏆"
    elif any(v == 2 for v in counts.values()):
        return "GOLD 🥇"
    else:
        return "SPLIT ⚪"

# ====================================
# UI HEADER
# ====================================

st.title("NEXUS Intelligence System")
st.markdown("### Autonomous Multi-Engine Trading Network")
st.markdown("---")

# ====================================
# SYSTEM ARCHITECTURE VISUAL
# ====================================

st.subheader("System Architecture")

st.graphviz_chart("""
digraph {
    rankdir=LR;

    Core -> DB;
    Range -> DB;
    Hybrid -> DB;
    AI -> DB;

    DB -> Consensus;
    Consensus -> Dispatcher;
    Dispatcher -> User;

    DB -> Monitor;
    Monitor -> Audit;
}
""")

st.markdown("---")

# ====================================
# LOAD DATA
# ====================================

try:
    df_all = fetch_latest_signals()
    df_active = fetch_active_signals()
    performance = load_performance()
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

# ====================================
# NETWORK STATUS PANEL
# ====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Signals", len(df_active))

with col2:
    total_last_24h = len(df_all)
    st.metric("Signals (24H)", total_last_24h)

with col3:
    engines_online = df_all["engine"].nunique()
    st.metric("Active Engines", engines_online)

st.markdown("---")

# ====================================
# SIGNAL VIEW BY ASSET
# ====================================

st.subheader("Live Signal Board")

if df_active.empty:
    st.info("No active signals.")
else:
    assets = df_active["asset"].unique()

    for asset in assets:
        st.markdown(f"## {asset}")

        asset_df = df_active[df_active["asset"] == asset]

        cols = st.columns(4)

        directions = []

        for idx, engine in enumerate(["NEXUS_CORE", "RANGE", "HYBRID", "AI"]):
            with cols[idx]:
                st.markdown(f"**{engine}**")

                engine_df = asset_df[asset_df["engine"] == engine]

                if not engine_df.empty:
                    row = engine_df.iloc[0]
                    directions.append(row["signal"])

                    st.write(f"Direction: {row['signal']}")
                    st.write(f"Entry: {round(row['entry'], 4)}")
                    st.write(f"SL: {round(row['sl'], 4)}")
                    st.write(f"TP: {round(row['tp'], 4)}")
                    st.write(f"Confidence: {round(row.get('confidence', 0), 2)}")
                else:
                    st.write("No Signal")

        if directions:
            consensus = classify_consensus(directions)
            st.markdown(f"### Consensus: {consensus}")

        st.markdown("---")

# ====================================
# ENGINE PERFORMANCE PANEL
# ====================================

st.subheader("Engine Governance Status")

if not performance:
    st.info("No performance data available.")
else:
    perf_df = pd.DataFrame(performance).T
    st.dataframe(perf_df)

# ====================================
# RESOLUTION SUMMARY
# ====================================

st.subheader("Resolved Trades (Last 24H)")

resolved = df_all[df_all["status"].isin(["TP", "SL"])]

if resolved.empty:
    st.write("No resolved trades.")
else:
    summary = resolved.groupby("status").size()
    st.bar_chart(summary)

# ====================================
# FOOTER
# ====================================

st.markdown("---")
st.caption(f"Last Updated: {datetime.now(timezone.utc).isoformat()} UTC")
