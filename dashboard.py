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
        # Attempting to load with 'reason' column if it exists
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# --- HEADER & KEY METRICS ---
st.title("🛡️ Nexus Intelligence Suite: Visual Command")
st.markdown(f"**System Status:** Deep Network Monitoring | **Last Sync:** {datetime.now().strftime('%H:%M:%S')}")

perf_data = load_performance()
m_cols = st.columns(len(DB_FILES))

for i, (name, db_file) in enumerate(DB_FILES.items()):
    strat_id = db_file.replace(".db", "")
    stats = perf_data.get(strat_id, {"win_rate": 50.0, "status": "LIVE"})
    
    with m_cols[i]:
        delta_val = "✅ LIVE" if stats['status'] == "LIVE" else "⚠️ RECOVERY"
        st.metric(label=name, value=f"{stats['win_rate']}%", delta=delta_val, 
                  delta_color="normal" if stats['status'] == "LIVE" else "inverse")

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
            st.write(f"### Result: **{prediction}** ({round(prob * 100, 2)}% Confidence)")
            st.progress(prob)
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model file not found. Run training script first.")

st.divider()

# --- ANALYTICS & LIVE FEED WITH REASONING ---
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
        # Confidence vs Time
        fig = px.scatter(master_df, x="ts", y="confidence", color="reason", 
                         title="Conviction by Technical Reason",
                         hover_data=["asset", "Engine"])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Show specific reasoning column in the main feed
        st.dataframe(
            master_df[["ts", "Engine", "asset", "signal", "confidence", "reason", "entry", "sl", "tp"]].head(20),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("No signal data found. Run your engines on GitHub to generate data.")

st.sidebar.title("🛠️ System Control")
for strat_id, stats in perf_data.items():
    if stats['status'] == "RECOVERY":
        st.sidebar.error(f"{strat_id}: IN RECOVERY")
    else:
        st.sidebar.success(f"{strat_id}: HEALTHY")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()
