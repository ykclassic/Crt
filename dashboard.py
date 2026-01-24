import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math

# --- CONFIGURATION ---
st.set_page_config(page_title="Nexus Intelligence Suite", layout="wide", page_icon="🧠")

DB_FILES = {
    "Core Engine": "nexus_core.db",
    "Hybrid V1": "hybrid_v1.db",
    "Rangemaster": "rangemaster.db",
    "AI Predict": "nexus_ai.db"
}
PERFORMANCE_FILE = "performance.json"
MODEL_FILE = "nexus_brain.pkl"

# --- DATA UTILITIES ---
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
        if "conf" in df.columns and "confidence" not in df.columns:
            df = df.rename(columns={"conf": "confidence"})
        if "reason" not in df.columns:
            df["reason"] = "LEGACY"
        return df
    except:
        return pd.DataFrame()

# --- HEADER ---
st.title("🛡️ Nexus Intelligence Suite: Visual Command")
st.markdown(f"**System Status:** Statistical Auditing Active | **Last Sync:** {datetime.now().strftime('%H:%M:%S')}")

perf_data = load_performance()
m_cols = st.columns(len(DB_FILES))

for i, (name, db_file) in enumerate(DB_FILES.items()):
    strat_id = db_file.replace(".db", "")
    stats = perf_data.get(strat_id, {"win_rate": 0.0, "status": "LIVE", "sample_size": 0})
    
    # Calculate Trust Score
    wr = stats.get("win_rate", 0)
    ss = stats.get("sample_size", 0)
    # Formula: (WR * sqrt(SS)) / 10
    trust_score = round((wr * math.sqrt(ss)) / 10, 1) if ss > 0 else 0.0

    with m_cols[i]:
        st.subheader(name)
        status_color = "green" if stats.get('status') == "LIVE" else "orange"
        st.markdown(f"**Status:** :{status_color}[{stats.get('status', 'OFFLINE')}]")
        
        # Display Metrics
        c1, c2 = st.columns(2)
        c1.metric("Win Rate", f"{wr}%")
        c2.metric("Sample Size", ss)
        
        # Trust Score Highlight
        st.metric("Trust Score", f"{trust_score}/100", help="Combined score of accuracy and statistical significance.")
        st.progress(min(trust_score / 100, 1.0))

st.divider()

# --- NEURAL NETWORK TESTER ---
st.subheader("🧠 Neural Network Simulator (AI Gatekeeper)")
col1, col2, col3 = st.columns(3)
with col1:
    test_rsi = st.slider("Current RSI", 0.0, 100.0, 50.0)
with col2:
    test_vol = st.number_input("Volume % Change (1h)", value=0.0, step=0.1)
with col3:
    test_dist = st.slider("Distance from EMA20 (%)", -10.0, 10.0, 0.0)

if st.button("🔮 Run AI Prediction"):
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                model, scaler = pickle.load(f)
            features = np.array([[test_rsi, test_vol, test_dist]])
            features_scaled = scaler.transform(features)
            prob = model.predict_proba(features_scaled)[0][1]
            prediction = "BULLISH" if prob > 0.5 else "BEARISH"
            color = "green" if prediction == "BULLISH" else "red"
            st.markdown(f"### Result: :{color}[{prediction}] ({round(prob * 100, 2)}% Confidence)")
            st.progress(prob)
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model file not found. Run training script first.")

st.divider()

# --- ANALYTICS & LIVE FEED ---
tab1, tab2 = st.tabs(["📊 Analytics", "📡 Live Signal Feed"])

all_data = []
for name, db in DB_FILES.items():
    df = load_signals(db)
    if not df.empty:
        df['Engine'] = name
        all_data.append(df)

if all_data:
    master_df = pd.concat(all_data, sort=False).fillna("N/A")
    master_df = master_df.sort_values("ts", ascending=False)
    
    with tab1:
        fig = px.scatter(master_df, x="ts", y="confidence", color="reason", 
                         title="Signal Conviction by Reason",
                         hover_data=["asset", "Engine"])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(
            master_df[["ts", "Engine", "asset", "signal", "confidence", "reason", "entry", "sl", "tp"]].head(20),
            use_container_width=True, hide_index=True
        )

st.sidebar.title("🛠️ System Control")
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.rerun()
