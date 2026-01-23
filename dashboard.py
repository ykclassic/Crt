import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Nexus Intelligence Suite", layout="wide", page_icon="📈")

DB_FILES = {
    "Core Engine": "nexus_core.db",
    "Hybrid V1": "hybrid_v1.db",
    "Rangemaster": "rangemaster.db",
    "AI Predict": "nexus_ai.db"
}
PERFORMANCE_FILE = "performance.json"

# --- DATA LOADING ---
def load_performance():
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, "r") as f:
            return json.load(f)
    return {}

def load_signals(db_path):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading {db_path}: {e}")
        return pd.DataFrame()

# --- HEADER ---
st.title("🛡️ Nexus Intelligence Command Center")
st.markdown(f"**Last Sync:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# --- TOP ROW: PERFORMANCE METRICS ---
perf_data = load_performance()
cols = st.columns(len(DB_FILES))

for i, (name, db_file) in enumerate(DB_FILES.items()):
    strategy_id = db_file.replace(".db", "")
    data = perf_data.get(strategy_id, {"win_rate": 0, "status": "OFFLINE"})
    
    with cols[i]:
        color = "normal" if data['status'] == "LIVE" else "inverse"
        st.metric(label=name, value=f"{data['win_rate']}%", delta=data['status'], delta_color=color)
        if data['status'] == "RECOVERY":
            st.warning("⚠️ In Recovery Mode")

# --- MIDDLE ROW: VISUAL ANALYSIS ---
st.subheader("📊 Strategy Insights")
col_left, col_right = st.columns([2, 1])

all_signals = []
for name, db in DB_FILES.items():
    df = load_signals(db)
    if not df.empty:
        df['engine'] = name
        all_signals.append(df)

if all_signals:
    master_df = pd.concat(all_signals)
    
    with col_left:
        # Confidence vs Time Scatter
        fig = px.scatter(master_df, x="ts", y="conf" if "conf" in master_df.columns else "confidence", 
                         color="engine", title="Signal Conviction Over Time",
                         hover_data=["asset", "signal"])
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Distribution of Signals
        fig2 = px.pie(master_df, names='signal', title="Directional Bias", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

# --- BOTTOM ROW: LIVE SIGNAL FEED ---
st.subheader("📡 Live Signal Feed (Consolidated)")
if all_signals:
    # Clean up column names for display
    display_df = master_df.copy()
    if "conf" in display_df.columns and "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].fillna(display_df["conf"])
    
    # Sort by timestamp
    display_df = display_df.sort_values(by="ts", ascending=False)
    
    st.dataframe(
        display_df[["ts", "engine", "asset", "signal", "entry", "sl", "tp"]].head(20),
        use_container_width=True,
        hide_index=True
    )

# --- DEEP ANALYSIS: BOLLINGER BANDS VIEW ---
st.divider()
st.subheader("🔍 Engine Deep Dive: Rangemaster")
range_df = load_signals("rangemaster.db")
if not range_df.empty:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=range_df['ts'], y=range_df['entry'], mode='markers', name='Entry Price',
                             marker=dict(color=range_df['signal'].map({'LONG': 'green', 'SHORT': 'red'}), size=10)))
    fig3.update_layout(title="Rangemaster Executions", xaxis_title="Timestamp", yaxis_title="Price")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No Rangemaster data available yet.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
if st.sidebar.button("🔄 Force Refresh Data"):
    st.rerun()

st.sidebar.markdown("""
---
### System Status
- **GitHub Actions:** Active
- **Database:** SQLite (Persistent)
- **Neural Network:** MLP Fallback Active
""")
