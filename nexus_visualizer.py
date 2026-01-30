import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
import json
from config import DB_FILE, PERFORMANCE_FILE

# 1. Page Configuration
st.set_page_config(page_title="Nexus Command Center", page_icon="🤖", layout="wide")

# Professional Dark Theme Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    div[data-testid="stExpander"] { border: 1px solid #30363d; }
    </style>
    """, unsafe_content_allow_html=True)

# 2. Resilient Data Loading
def get_data():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    try:
        # Use uri=True to allow read-only access if the file is busy
        conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
        conn.close()
        df['ts'] = pd.to_datetime(df['ts'])
        return df
    except Exception as e:
        st.sidebar.error(f"DB Access Error: {e}")
        return pd.DataFrame()

# 3. Sidebar - Performance Stats
st.sidebar.title("🛡️ System Health")
if os.path.exists(PERFORMANCE_FILE):
    try:
        with open(PERFORMANCE_FILE, 'r') as f:
            perf_data = json.load(f)
            for engine, stats in perf_data.items():
                st.sidebar.metric(f"{engine.upper()} Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    except:
        st.sidebar.warning("Performance logs initializing...")

# 4. Main Dashboard UI
st.title("📈 Nexus AI Intelligence Dashboard")
data = get_data()

if data.empty:
    st.warning("📡 System Online: Waiting for incoming signals...")
    st.info("Ensure the AI Engine has run at least once to populate the database.")
else:
    # --- ROW 1: METRICS ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Signals Captured", len(data))
    with m2:
        high_conf = len(data[data['confidence'] >= 80])
        st.metric("Elite Signals (Gold/Diamond)", high_conf)
    with m3:
        last_asset = data['asset'].iloc[0]
        st.metric("Latest Target", last_asset)

    # --- ROW 2: VISUALS ---
    st.subheader("📊 Intelligence Visualization")
    fig = px.line(data, x='ts', y='confidence', color='asset', 
                 title="Signal Confidence History", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- ROW 3: THE DOWNLOAD CENTER ---
    st.divider()
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        st.subheader("📑 Intelligence Logs")
    
    with col_b:
        # CONVERT DATA FOR DOWNLOAD
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Trade Logs (CSV)",
            data=csv,
            file_name=f"nexus_signals_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            help="Export full trade history for Excel/external analysis"
        )

    st.dataframe(data, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Nexus AI v5.2 | CTO Monitored")
