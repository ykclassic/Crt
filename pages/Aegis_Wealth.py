import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
import sys
import json
from datetime import datetime

# --- CTO PATH BRIDGE ---
# This allows the script to see config.py in the folder above
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config import DB_FILE, PERFORMANCE_FILE, TOTAL_CAPITAL, RISK_PER_TRADE
except ImportError:
    st.error("❌ Link Broken: Could not find config.py in the root directory.")
    st.stop()

# 1. Page Config
st.set_page_config(page_title="Aegis Wealth Dashboard", page_icon="💰", layout="wide")

# Custom Dark Mode Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_content_allow_html=True)

# 2. Resilient Data Loading
def load_wealth_data():
    # Adjust path to find DB in the root folder
    db_path = os.path.join(os.path.dirname(__file__), '..', DB_FILE)
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        # mode=ro ensures we don't interfere with the AI Engine's writing
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
        conn.close()
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"⚠️ Database Error: {e}")
        return pd.DataFrame()

# 3. Main Interface
st.title("💰 Aegis Wealth Intelligence")
df = load_wealth_data()

if df.empty:
    st.warning("📡 Connecting to Nexus Node... No data found in root/nexus.db")
else:
    # --- TOP ROW: KPI METRICS ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Total Opportunities", len(df))
    
    with c2:
        # Calculate Risk in tens of dollars ($10 to $100)
        base_risk = TOTAL_CAPITAL * RISK_PER_TRADE
        clamped_risk = max(10, min(100, round(base_risk / 10) * 10))
        st.metric("Risk Per Trade", f"${clamped_risk}")
    
    with c3:
        win_rate = "TBD"
        if os.path.exists(os.path.join(os.path.dirname(__file__), '..', PERFORMANCE_FILE)):
            with open(os.path.join(os.path.dirname(__file__), '..', PERFORMANCE_FILE), 'r') as f:
                perf = json.load(f)
                win_rate = f"{perf.get('ai', {}).get('win_rate', 0):.1f}%"
        st.metric("AI Win Rate", win_rate)

    # --- VISUALIZATION ---
    st.subheader("📊 Signal Distribution")
    fig = px.scatter(df, x="ts", y="confidence", color="asset", size="confidence",
                     template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA & DOWNLOAD ---
    st.divider()
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        st.subheader("📑 Detailed Trade Logs")
    
    with col_right:
        # DOWNLOAD BUTTON
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name=f"Aegis_Wealth_Export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

    st.dataframe(df, use_container_width=True)

st.sidebar.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
