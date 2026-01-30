import streamlit as st
import os
import sys
import json
import sqlite3
from datetime import datetime

# --- CTO PATH BRIDGE ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- SAFETY IMPORT CHECK ---
try:
    import pandas as pd
    import plotly.express as px
except ImportError:
    st.error("🔬 **Technical Requirement Missing:** Plotly or Pandas not found.")
    st.info("Please ensure `plotly` and `pandas` are in your `requirements.txt` file in the root folder.")
    st.stop()

try:
    from config import DB_FILE, PERFORMANCE_FILE, TOTAL_CAPITAL, RISK_PER_TRADE
except ImportError:
    st.error("❌ Link Broken: Could not find `config.py` in the root directory.")
    st.stop()

# 1. Page Config
st.set_page_config(page_title="Aegis Wealth Dashboard", page_icon="💰", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_content_allow_html=True)

# 2. Data Loader
def load_wealth_data():
    db_path = os.path.join(os.path.dirname(__file__), '..', DB_FILE)
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
        conn.close()
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        return df
    except Exception as e:
        return pd.DataFrame()

# 3. Main UI
st.title("💰 Aegis Wealth Intelligence")
df = load_wealth_data()

if df.empty:
    st.warning("📡 Connecting to Nexus... Database currently empty or not found.")
else:
    # --- METRICS ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Signals", len(df))
    with c2:
        # Clamped Risk $10 - $100
        risk_val = max(10, min(100, round((TOTAL_CAPITAL * RISK_PER_TRADE) / 10) * 10))
        st.metric("Risk Per Trade", f"${risk_val}")
    with c3:
        st.metric("Status", "ENCRYPTED")

    # --- CHART ---
    st.subheader("📊 Opportunity Timeline")
    fig = px.scatter(df, x="ts", y="confidence", color="asset", size="confidence",
                     template="plotly_dark", hover_data=['entry', 'tp', 'sl'])
    st.plotly_chart(fig, use_container_width=True)

    # --- DOWNLOAD & TABLE ---
    st.divider()
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Trade Logs (CSV)", data=csv, file_name="aegis_wealth_export.csv", mime='text/csv')
    st.dataframe(df, use_container_width=True)
