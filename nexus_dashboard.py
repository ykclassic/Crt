import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from collections import Counter

# ===============================
# CONFIGURATION
# ===============================

TREND_DB = "trending_signals.db"
RANGE_DB = "ranging_signals.db"
HYBRID_DB = "hybrid_signals.db"

st.set_page_config(
    page_title="Nexus Trading System",
    layout="wide"
)

# ===============================
# DATABASE UTIL
# ===============================

def fetch_latest_signals(db_path):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT pair, direction, entry, stop_loss, take_profit, timestamp
        FROM signals
        WHERE timestamp = (
            SELECT MAX(timestamp) FROM signals
        )
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ===============================
# CONSENSUS LOGIC
# ===============================

def classify_consensus(signals):
    directions = [s["direction"] for s in signals if s is not None]
    direction_count = Counter(directions)

    if len(direction_count) == 1 and len(directions) == 3:
        return "DIAMOND 💎"
    elif any(count == 2 for count in direction_count.values()):
        return "GOLD 🥇"
    else:
        return "STANDARD ⚪"


# ===============================
# LOAD DATA
# ===============================

try:
    trending_df = fetch_latest_signals(TREND_DB)
    ranging_df = fetch_latest_signals(RANGE_DB)
    hybrid_df = fetch_latest_signals(HYBRID_DB)
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

# ===============================
# HEADER
# ===============================

st.title("NEXUS Trading System Dashboard")
st.markdown("---")

# ===============================
# ARCHITECTURE VISUAL
# ===============================

st.subheader("System Workflow")

st.graphviz_chart("""
digraph {
    rankdir=LR;

    config -> trending;
    config -> ranging;
    config -> hybrid;

    trending -> trending_db;
    ranging -> ranging_db;
    hybrid -> hybrid_db;

    trending_db -> consensus;
    ranging_db -> consensus;
    hybrid_db -> consensus;

    consensus -> dispatcher;
}
""")

st.markdown("---")

# ===============================
# SIGNAL DISPLAY
# ===============================

st.subheader("Engine Signals")

pairs = sorted(
    set(trending_df["pair"])
    | set(ranging_df["pair"])
    | set(hybrid_df["pair"])
)

for pair in pairs:
    st.markdown(f"### {pair}")

    t_signal = trending_df[trending_df["pair"] == pair]
    r_signal = ranging_df[ranging_df["pair"] == pair]
    h_signal = hybrid_df[hybrid_df["pair"] == pair]

    col1, col2, col3 = st.columns(3)

    def display_signal(column, title, df):
        with column:
            st.markdown(f"**{title}**")
            if not df.empty:
                row = df.iloc[0]
                st.write(f"Direction: {row['direction']}")
                st.write(f"Entry: {row['entry']}")
                st.write(f"Stop Loss: {row['stop_loss']}")
                st.write(f"Take Profit: {row['take_profit']}")
            else:
                st.write("No Signal")

    display_signal(col1, "Trending Engine", t_signal)
    display_signal(col2, "Ranging Engine", r_signal)
    display_signal(col3, "Hybrid Engine", h_signal)

    signals = []
    if not t_signal.empty:
        signals.append(t_signal.iloc[0])
    if not r_signal.empty:
        signals.append(r_signal.iloc[0])
    if not h_signal.empty:
        signals.append(h_signal.iloc[0])

    consensus = classify_consensus(signals)

    st.markdown(f"### Consensus Result: {consensus}")
    st.markdown("---")

# ===============================
# FOOTER
# ===============================

st.caption(f"Last updated: {datetime.utcnow()} UTC")
