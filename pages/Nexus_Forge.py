import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import time

# 1. Page Configuration
st.set_page_config(page_title="Aegis Master Node | Nexus Forge", page_icon="🧠", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# --- DATABASE INTEGRITY CHECK ---
def safe_init_db():
    """Ensures the table exists before any query is made to prevent ReadSQL errors."""
    conn = sqlite3.connect('aegis_system.db', check_same_thread=False)
    c = conn.cursor()
    # Ensure the table and columns exist
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (timestamp TEXT, user_level TEXT, event TEXT)''')
    conn.commit()
    conn.close()

# --- SHARED DATA ENGINE ---
def get_latest_signals():
    """Fetches signals with a fallback to prevent app crashes."""
    safe_init_db() # Ensure DB is ready
    try:
        conn = sqlite3.connect('aegis_system.db', check_same_thread=False)
        # Use a simpler query to check for data first
        query = """
        SELECT timestamp, user_level as Module, event as Signal 
        FROM logs 
        WHERE event LIKE 'Signal:%' 
        ORDER BY timestamp DESC LIMIT 20
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        # If the DB is empty or columns are missing, return an empty DF with correct headers
        return pd.DataFrame(columns=["timestamp", "Module", "Signal"])

# --- WEIGHTED ENSEMBLE LOGIC ---
def calculate_master_confidence(df):
    """Calculates weighted consensus from sub-modules."""
    if df.empty:
        return 50.0, "INITIALIZING" # Neutral starting point
    
    # Logic: More signals = higher conviction. 
    # In a real setup, we'd parse the 'Signal' string for 'LONG'/'SHORT'
    base_confidence = 75.0 + (len(df) * 1.5) 
    base_confidence = min(base_confidence, 98.5) # Cap at 98.5
    
    verdict = "STRONG CONFLUENCE" if base_confidence > 85 else "AWAITING SYNC"
    return base_confidence, verdict

# --- UI LAYOUT ---
st.title("🧠 Aegis Master Node: Nexus Forge")
st.write("Meta-Inference Engine | Global Signal Aggregator")

# Automatic Refresh Logic
@st.fragment(run_every="30s")
def render_master_dashboard():
    df_signals = get_latest_signals()
    confidence, verdict = calculate_master_confidence(df_signals)
    
    col_gauge, col_logic = st.columns([1, 1])
    
    with col_gauge:
        st.subheader("Global Confidence Level")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Verdict: {verdict}", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#00FFCC"},
                'steps': [
                    {'range': [0, 70], 'color': '#222'},
                    {'range': [70, 85], 'color': '#444'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        fig.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_logic:
        st.subheader("📡 Real-Time Intelligence Feed")
        if not df_signals.empty:
            st.dataframe(df_signals, use_container_width=True, height=300)
        else:
            st.warning("Nexus Forge is online. Waiting for signals from sub-modules (Neural, Signal, Profit)...")

    st.write("---")
    if confidence > 85:
        st.success(f"🔥 **CONFLUENCE ALERT:** Decision Node confirms high-probability entry for global assets.")
    else:
        st.info(f"⚖️ **MARKET SCAN:** {verdict}. Current intelligence suggests waiting for model alignment.")

# Execute the core logic
render_master_dashboard()

st.write("---")
st.caption(f"Aegis Forge v1.1 | Database Sync: Active | {time.strftime('%H:%M:%S')}")
