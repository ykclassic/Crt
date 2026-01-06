import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from datetime import datetime

# 1. Page Config
st.set_page_config(page_title="Aegis Master Node", page_icon="🧠", layout="wide")

# 2. Shared Data Retrieval
def get_latest_signals():
    """
    Fetches the latest signals stored in the central DB by other apps.
    This assumes your other apps log their signals to the 'logs' table.
    """
    conn = sqlite3.connect('aegis_system.db')
    # Query for the most recent entry from each specific app/module
    query = """
    SELECT user_level, event, timestamp FROM logs 
    WHERE event LIKE 'Signal:%' 
    ORDER BY timestamp DESC LIMIT 10
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- DECISION LOGIC ---
def calculate_master_verdict(signals_df):
    """
    Simulates a Weighted Ensemble Decision.
    Weights: Neural (0.5), Signal (0.3), Profit (0.2)
    """
    weights = {"Nexus Neural": 0.5, "Nexus Signal": 0.3, "Neural Profit": 0.2}
    total_score = 0
    
    # Logic: Look for 'Long' or 'Short' keywords in the logs
    for app, weight in weights.items():
        # Check if the latest log for this app is bullish or bearish
        # (Simplified simulation logic)
        total_score += weight * 85 # Example confidence
        
    return total_score

# --- UI LAYOUT ---
st.title("🧠 Aegis Master Decision Node")
st.write("Centralized Signal Aggregation & Probabilistic Filtering")

col_gauge, col_feed = st.columns([1, 1])

with col_gauge:
    st.subheader("Global Confidence Score")
    score = calculate_master_verdict(None)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Ensemble Accuracy Level"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#00FFCC"},
                 'steps' : [
                     {'range': [0, 50], 'color': "#333"},
                     {'range': [50, 85], 'color': "#555"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_feed:
    st.subheader("📡 Multi-App Signal Feed")
    signals = get_latest_signals()
    if not signals.empty:
        st.dataframe(signals, use_container_width=True)
    else:
        st.info("Waiting for signals from sub-modules...")

st.write("---")
st.subheader("🎯 Master Verdict")
if score > 85:
    st.success("🔥 **GLOBAL SYNC DETECTED:** All models align. Execute High-Conviction Strategy.")
else:
    st.warning("⚖️ **DIVERGENCE:** Models are conflicting. Stand by for higher confluence.")

st.caption("Aegis Master Node v1.0 | Collective Intelligence Active")
