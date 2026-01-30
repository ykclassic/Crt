import sys
import subprocess
import os

# --- 1. EMERGENCY AUTO-INSTALLER (MUST BE FIRST) ---
# This ensures Plotly and Pandas exist before the rest of the script runs.
def check_dependencies():
    try:
        import pandas
        import plotly
    except ImportError:
        # If missing, install them manually on the Streamlit server
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly==5.18.0", "pandas==2.2.0"])
        # Standard Python restart is not needed; Streamlit will rerun on the next line
        return False
    return True

dependencies_ready = check_dependencies()

# --- 2. STANDARD IMPORTS ---
import streamlit as st

if not dependencies_ready:
    st.warning("⚙️ Initializing Aegis Wealth Environment... Please wait 30 seconds.")
    st.stop()

import json
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px

# --- 3. PATH BRIDGE TO ROOT ---
# This allows the script to find config.py in the main folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config import DB_FILE, PERFORMANCE_FILE, TOTAL_CAPITAL, RISK_PER_TRADE
except ImportError:
    st.error("❌ Configuration Link Failed: Ensure config.py is in the root folder.")
    st.stop()

# --- 4. PAGE CONFIGURATION ---
st.set_page_config(page_title="Aegis Wealth Dashboard", page_icon="💰", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_content_allow_html=True)

# --- 5. DATA LOADING LOGIC ---
def load_wealth_data():
    # Look for the database in the root folder (one level up)
    db_path = os.path.join(os.path.dirname(__file__), '..', DB_FILE)
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    try:
        # mode=ro (Read Only) to prevent database locks during AI runs
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
        conn.close()
        
        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        return df
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")
        return pd.DataFrame()

# --- 6. USER INTERFACE ---
st.title("💰 Aegis Wealth Intelligence")
df = load_wealth_data()

if df.empty:
    st.info("📡 Nexus Node Connected. Waiting for the next signal cycle to populate logs...")
    st.image("https://via.placeholder.com/800x200.png?text=Waiting+for+Market+Intelligence...", use_column_width=True)
else:
    # --- ROW 1: KEY PERFORMANCE INDICATORS ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Signals", len(df))
        
    with col2:
        # Risk Sizing Logic: $10 to $100 in tens
        raw_risk = TOTAL_CAPITAL * RISK_PER_TRADE
        clamped_risk = max(10, min(100, round(raw_risk / 10) * 10))
        st.metric("Recommended Risk", f"${clamped_risk}", help="Calculated based on 2% of equity, capped at $100.")
        
    with col3:
        st.metric("System Status", "PROTECTED", delta="Active")

    # --- ROW 2: VISUALIZATION ---
    st.subheader("📊 Signal Confidence & Trend")
    fig = px.scatter(
        df, x="ts", y="confidence", color="asset", 
        size="confidence", template="plotly_dark",
        labels={"ts": "Time", "confidence": "AI Confidence %"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- ROW 3: DATA EXPORT & TABLE ---
    st.divider()
    left, right = st.columns([3, 1])
    
    with left:
        st.subheader("📑 Intelligence Audit Trail")
        
    with right:
        # CSV Export Feature
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export to CSV",
            data=csv,
            file_name=f"Aegis_Wealth_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

    st.dataframe(df, use_container_width=True)

st.sidebar.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
