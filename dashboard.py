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
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        
        # --- DATA CLEANING FIX ---
        # 1. Standardize 'conf' vs 'confidence'
        if "conf" in df.columns and "confidence" not in df.columns:
            df = df.rename(columns={"conf": "confidence"})
        
        # 2. Ensure 'reason' exists (for older database rows)
        if "reason" not in df.columns:
            df["reason"] = "LEGACY"
        
        return df
    except Exception as e:
        return pd.DataFrame()

# --- HEADER & KEY METRICS ---
st.title("🛡️ Nexus Intelligence Suite: Visual Command")
st.markdown(f"**System Status:** Monitoring Active | **Last Sync:** {datetime.now().strftime('%H:%M:%S')}")

perf_data = load_performance()
m_cols = st.columns(len(DB_FILES))

for i, (name, db_file) in enumerate(DB_FILES.items()):
    strat_id = db_file.replace(".db", "")
    stats = perf_data.get(strat_id, {"win_rate": 50.0, "status": "LIVE"})
    
    with m_cols[i]:
        status_label = "✅ LIVE" if stats.get('status') == "LIVE" else "⚠️ RECOVERY"
        st.metric(label=name, value=f"{stats.get('win_rate', 50.0)}%", delta=status_label, 
                  delta_color="normal" if stats.get('status') == "LIVE" else "inverse")

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
        st.error("Model file (nexus_brain.pkl) not found. Run training script first.")

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
    # Concatenate and fill missing columns gracefully
    master_df = pd.concat(all_data, sort=False)
    master_df["reason"] = master_df["reason"].fillna("TECHNICAL")
    master_df["confidence"] = pd.to_numeric(master_df["confidence"], errors='coerce').fillna(50.0)
    master_df = master_df.sort_values("ts", ascending=False)
    
    with tab1:
        # Visualizing the logic behind trades
        try:
            fig = px.scatter(master_df, x="ts", y="confidence", color="reason", 
                             title="Signal Conviction by Reason",
                             hover_data=["asset", "Engine", "signal"],
                             color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)
            
            # Asset distribution
            fig2 = px.histogram(master_df, x="asset", color="Engine", barmode="group", title="Asset Activity Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Charts are refreshing... (Reason: Data Syncing)")

    with tab2:
        # Clean display for the table
        cols_to_show = ["ts", "Engine", "asset", "signal", "confidence", "reason", "entry", "sl", "tp"]
        # Ensure only existing columns are used
        available_cols = [c for c in cols_to_show if c in master_df.columns]
        st.dataframe(master_df[available_cols].head(30), use_container_width=True, hide_index=True)
else:
    st.info("Waiting for signals... Ensure your engines are running on GitHub Actions.")

st.sidebar.title("🛠️ System Control")
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("### Engine Health")
for name, db in DB_FILES.items():
    if os.path.exists(db):
        st.sidebar.success(f"{name}: Connected")
    else:
        st.sidebar.error(f"{name}: DB Missing")
