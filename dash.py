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
st.set_page_config(page_title="Nexus Intelligence Suite", layout="wide", page_icon="🛡️")

DB_FILES = {
    "Core Engine": "nexus_core.db",
    "Hybrid V1": "hybrid_v1.db",
    "Rangemaster": "rangemaster.db",
    "AI Predict": "nexus_ai.db"
}
JOURNAL_DB = "nexus_journal.db"
PERFORMANCE_FILE = "performance.json"
MODEL_FILE = "nexus_brain.pkl"

# --- DATABASE UTILITIES ---
def init_journal():
    conn = sqlite3.connect(JOURNAL_DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS journal 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     ts TEXT, 
                     category TEXT, 
                     entry TEXT)''')
    conn.close()

def save_journal_entry(category, text):
    conn = sqlite3.connect(JOURNAL_DB)
    conn.execute("INSERT INTO journal (ts, category, entry) VALUES (?, ?, ?)",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), category, text))
    conn.commit()
    conn.close()

def load_journal(search_term=""):
    if not os.path.exists(JOURNAL_DB): return pd.DataFrame()
    conn = sqlite3.connect(JOURNAL_DB)
    query = "SELECT ts as 'Time', category as 'Type', entry as 'Observation' FROM journal"
    if search_term:
        query += f" WHERE entry LIKE '%{search_term}%' OR category LIKE '%{search_term}%'"
    query += " ORDER BY id DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_signals(db_path):
    if not os.path.exists(db_path): return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        if "conf" in df.columns and "confidence" not in df.columns:
            df = df.rename(columns={"conf": "confidence"})
        if "reason" not in df.columns:
            df["reason"] = "LEGACY"
        return df
    except: return pd.DataFrame()

# --- INITIALIZE ---
init_journal()

# --- HEADER & STATS ---
st.title("🛡️ Nexus Intelligence Suite: Visual Command")
st.markdown(f"**System Status:** Statistical Auditing Active | **Last Sync:** {datetime.now().strftime('%H:%M:%S')}")

perf_data = {}
if os.path.exists(PERFORMANCE_FILE):
    with open(PERFORMANCE_FILE, "r") as f: perf_data = json.load(f)

m_cols = st.columns(len(DB_FILES))
for i, (name, db_file) in enumerate(DB_FILES.items()):
    strat_id = db_file.replace(".db", "")
    stats = perf_data.get(strat_id, {"win_rate": 0.0, "status": "LIVE", "sample_size": 0})
    wr, ss = stats.get("win_rate", 0), stats.get("sample_size", 0)
    trust = round((wr * math.sqrt(ss)) / 10, 1) if ss > 0 else 0.0
    with m_cols[i]:
        st.subheader(name)
        st.metric("Trust Score", f"{trust}/100", f"{wr}% Win Rate")
        st.progress(min(trust/100, 1.0))

st.divider()

# --- SECTION 1: CONFLUENCE & JOURNAL ---
col_j1, col_j2 = st.columns([1, 1])

with col_j1:
    st.subheader("💎 Confluence (Engine Agreement)")
    all_dfs = {name: load_signals(db) for name, db in DB_FILES.items()}
    master_list = []
    for name, df in all_dfs.items():
        if not df.empty:
            t = df.sort_values('ts').groupby('asset').tail(1).copy()
            t['Engine'] = name
            master_list.append(t)
    
    if master_list:
        m_df = pd.concat(master_list)
        consensus = m_df.groupby(['asset', 'signal']).agg({
            'Engine': 'count', 
            'confidence': 'mean', 
            'reason': lambda x: ' | '.join(x.unique())
        }).reset_index().rename(columns={'Engine': 'Agreements', 'confidence': 'Avg_Conf'})
        
        # Highlight Diamond (4) and Gold (3)
        st.dataframe(consensus.sort_values('Agreements', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Scanning for engine alignment...")

with col_j2:
    st.subheader("📓 Observation Journal")
    with st.expander("📝 Log New Entry", expanded=False):
        cat = st.selectbox("Type", ["Market Observation", "Diamond Audit", "AI Note", "Error"])
        note = st.text_area("Observations")
        if st.button("Save Entry"):
            if note:
                save_journal_entry(cat, note)
                st.success("Saved!")
                st.rerun()
    
    search = st.text_input("🔍 Search Logs", placeholder="BTC, Diamond, etc.")
    history = load_journal(search)
    st.dataframe(history, use_container_width=True, hide_index=True, height=200)

st.divider()

# --- SECTION 2: NEURAL NETWORK SIMULATOR ---
st.subheader("🧠 Neural Network Simulator (AI Gatekeeper)")
s1, s2, s3 = st.columns(3)
with s1: t_rsi = st.slider("Current RSI", 0.0, 100.0, 50.0)
with s2: t_vol = st.number_input("Volume % Change (1h)", value=0.0, step=0.1)
with s3: t_dist = st.slider("Distance from EMA20 (%)", -10.0, 10.0, 0.0)

if st.button("🔮 Run AI Prediction"):
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        feat = scaler.transform(np.array([[t_rsi, t_vol, t_dist]]))
        prob = model.predict_proba(feat)[0][1]
        res = "BULLISH" if prob > 0.5 else "BEARISH"
        st.markdown(f"### Result: :{'green' if res=='BULLISH' else 'red'}[{res}] ({round(prob*100,2)}% Confidence)")
        st.progress(prob)
    else: st.error("Model file not found. Ensure train_brain.py has run.")

st.divider()

# --- SECTION 3: ANALYTICS & RAW FEED ---
t_visuals, t_raw = st.tabs(["📊 Performance Visuals", "📡 Live Signal Feed"])

if master_list:
    full_df = pd.concat([df.assign(Engine=n) for n, df in all_dfs.items() if not df.empty])
    full_df = full_df.sort_values('ts', ascending=False)
    
    with t_visuals:
        fig = px.scatter(full_df, x="ts", y="confidence", color="reason", 
                         title="Signal Conviction by Technical Reason",
                         hover_data=["asset", "Engine", "signal"])
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.histogram(full_df, x="asset", color="Engine", barmode="group", title="Engine Activity by Asset")
        st.plotly_chart(fig2, use_container_width=True)

    with t_raw:
        st.dataframe(full_df[["ts", "Engine", "asset", "signal", "confidence", "reason", "entry", "sl", "tp"]].head(50), 
                     use_container_width=True, hide_index=True)
else:
    st.warning("No data found in engine databases.")

if st.sidebar.button("🔄 Refresh Data"): st.rerun()
