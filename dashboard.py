import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import pickle
import numpy as np
import plotly.express as px
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
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        # Standardization
        if "conf" in df.columns and "confidence" not in df.columns:
            df = df.rename(columns={"conf": "confidence"})
        if "reason" not in df.columns:
            df["reason"] = "LEGACY"
        return df
    except:
        return pd.DataFrame()

# --- CONSENSUS LOGIC ---
def calculate_consensus(all_dfs):
    if not all_dfs:
        return pd.DataFrame()
    
    # Combine only the latest signal per asset per engine
    latest_signals = []
    for engine_name, df in all_dfs.items():
        if not df.empty:
            temp = df.sort_values('ts').groupby('asset').tail(1).copy()
            temp['Engine'] = engine_name
            latest_signals.append(temp)
    
    if not latest_signals:
        return pd.DataFrame()

    master = pd.concat(latest_signals)
    
    # Group by asset and signal to find agreements
    consensus = master.groupby(['asset', 'signal']).agg({
        'Engine': 'count',
        'confidence': 'mean',
        'reason': lambda x: ', '.join(x.unique())
    }).reset_index()
    
    consensus = consensus.rename(columns={'Engine': 'Engine_Count', 'confidence': 'Avg_Confidence'})
    # Return only where 2 or more engines agree (User asked for all 4, but we show top-tier)
    return consensus.sort_values('Engine_Count', ascending=False)

# --- HEADER ---
st.title("🛡️ Nexus Intelligence Suite: Visual Command")
st.markdown(f"**System Status:** Confluence Monitoring | **Last Sync:** {datetime.now().strftime('%H:%M:%S')}")

perf_data = load_performance()
m_cols = st.columns(len(DB_FILES))

for i, (name, db_file) in enumerate(DB_FILES.items()):
    strat_id = db_file.replace(".db", "")
    stats = perf_data.get(strat_id, {"win_rate": 0.0, "status": "LIVE", "sample_size": 0})
    wr, ss = stats.get("win_rate", 0), stats.get("sample_size", 0)
    trust_score = round((wr * math.sqrt(ss)) / 10, 1) if ss > 0 else 0.0

    with m_cols[i]:
        st.subheader(name)
        st.metric("Trust Score", f"{trust_score}/100", delta=f"{wr}% WR")
        st.progress(min(trust_score / 100, 1.0))

st.divider()

# --- THE CONSENSUS TABLE ---
st.subheader("💎 Diamond Confluence (Engine Agreement)")
all_dfs = {name: load_signals(db) for name, db in DB_FILES.items()}
consensus_df = calculate_consensus(all_dfs)

if not consensus_df.empty:
    # Filter for the "Holy Grail": All 4 engines agreeing
    perfect_match = consensus_df[consensus_df['Engine_Count'] >= 4]
    strong_match = consensus_df[consensus_df['Engine_Count'] == 3]
    
    if not perfect_match.empty:
        st.success("🎯 **DIAMOND CONSENSUS DETECTED: All 4 engines are in agreement!**")
        st.table(perfect_match)
    elif not strong_match.empty:
        st.warning("⚡ **STRONG CONSENSUS: 3 engines are in agreement.**")
        st.dataframe(strong_match, use_container_width=True, hide_index=True)
    else:
        st.info("Scanning for engine alignment... Currently showing partial agreements.")
        st.dataframe(consensus_df[consensus_df['Engine_Count'] >= 2], use_container_width=True, hide_index=True)
else:
    st.info("No active signal confluence detected yet.")

st.divider()

# --- NEURAL NETWORK SIMULATOR ---
st.subheader("🧠 Neural Network Simulator (AI Gatekeeper)")
c1, c2, c3 = st.columns(3)
with c1: test_rsi = st.slider("Current RSI", 0.0, 100.0, 50.0)
with c2: test_vol = st.number_input("Volume % Change (1h)", value=0.0, step=0.1)
with c3: test_dist = st.slider("Distance from EMA20 (%)", -10.0, 10.0, 0.0)

if st.button("🔮 Run AI Prediction"):
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        prob = model.predict_proba(scaler.transform(np.array([[test_rsi, test_vol, test_dist]])))[0][1]
        res = "BULLISH" if prob > 0.5 else "BEARISH"
        st.markdown(f"### Result: :{'green' if res=='BULLISH' else 'red'}[{res}] ({round(prob*100,2)}% Conf)")
        st.progress(prob)
    else: st.error("Model file not found.")

st.divider()

# --- LIVE FEED ---
tab1, tab2 = st.tabs(["📊 Analytics", "📡 Live Signal Feed"])
master_list = [df.assign(Engine=name) for name, df in all_dfs.items() if not df.empty]

if master_list:
    master_df = pd.concat(master_list, sort=False).fillna("N/A").sort_values("ts", ascending=False)
    with tab1:
        st.plotly_chart(px.scatter(master_df, x="ts", y="confidence", color="reason", title="Signal Conviction"), use_container_width=True)
    with tab2:
        st.dataframe(master_df[["ts", "Engine", "asset", "signal", "confidence", "reason"]].head(20), use_container_width=True, hide_index=True)

if st.sidebar.button("🔄 Refresh"): st.rerun()
