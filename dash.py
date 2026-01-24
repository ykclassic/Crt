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

# --- UTILITIES & DATABASE ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def init_journal():
    conn = sqlite3.connect(JOURNAL_DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS journal 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     ts TEXT, category TEXT, entry TEXT)''')
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
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        # Handle column naming inconsistencies
        if "conf" in df.columns and "confidence" not in df.columns:
            df = df.rename(columns={"conf": "confidence"})
        if "reason" not in df.columns:
            df["reason"] = "N/A"
        return df
    except: return pd.DataFrame()

# --- INITIALIZE ---
init_journal()

# --- HEADER & GLOBAL METRICS ---
st.title("🛡️ Nexus Intelligence Suite: Visual Command")
st.markdown(f"**System Status:** Full Auditing Enabled | **Local Time:** {datetime.now().strftime('%H:%M:%S')}")

perf_data = {}
if os.path.exists(PERFORMANCE_FILE):
    try:
        with open(PERFORMANCE_FILE, "r") as f: perf_data = json.load(f)
    except: pass

m_cols = st.columns(len(DB_FILES))
for i, (name, db_file) in enumerate(DB_FILES.items()):
    strat_id = db_file.replace(".db", "")
    stats = perf_data.get(strat_id, {"win_rate": 0.0, "sample_size": 0})
    wr, ss = stats.get("win_rate", 0), stats.get("sample_size", 0)
    trust = round((wr * math.sqrt(ss)) / 10, 1) if ss > 0 else 0.0
    with m_cols[i]:
        st.subheader(name)
        st.metric("Trust Score", f"{trust}/100", f"{wr}% WR ({ss} Trades)")
        st.progress(min(trust/100, 1.0))

st.divider()

# --- SECTION 1: CONFLUENCE & JOURNALING ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("💎 Market Confluence (Diamond Checker)")
    all_dfs = {name: load_signals(path) for name, path in DB_FILES.items()}
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
        }).reset_index().rename(columns={'Engine': 'Matches', 'confidence': 'Avg_Conf'})
        st.dataframe(consensus.sort_values('Matches', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Scanning for engine alignment...")

with col_right:
    st.subheader("📓 Digital Observation Journal")
    with st.expander("📝 Log New Entry", expanded=False):
        cat = st.selectbox("Category", ["Market Observation", "Diamond Audit", "AI Accuracy Note", "System Error"])
        note = st.text_area("Your Observation")
        if st.button("Save to Database"):
            if note:
                save_journal_entry(cat, note)
                st.success("Saved!")
                st.rerun()

    search = st.text_input("🔍 Search Past Notes", placeholder="e.g. BTC, Short, Diamond")
    history = load_journal(search)
    
    if not history.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Export CSV", convert_df_to_csv(history), "nexus_journal.csv", "text/csv")
        with c2:
            if st.button("📤 Forward to Discord"):
                history.to_csv("temp_journal.csv", index=False)
                try:
                    from nexus_reporter import forward_to_discord
                    forward_to_discord("temp_journal.csv", f"Journal Export - {search if search else 'All'}")
                    st.success("Sent to Discord!")
                except: st.error("Reporter script not found.")
        st.dataframe(history, use_container_width=True, hide_index=True, height=180)

st.divider()

# --- SECTION 2: AI NEURAL SIMULATOR ---
st.subheader("🧠 Neural Network Simulator (AI Gatekeeper)")
s1, s2, s3 = st.columns(3)
with s1: t_rsi = st.slider("Current RSI", 0, 100, 50)
with s2: t_vol = st.number_input("Volume % Change (1h)", value=0.0, step=0.1)
with s3: t_dist = st.slider("Distance from EMA20 (%)", -10.0, 10.0, 0.0)

if st.button("🔮 Run Live Prediction"):
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        feat = scaler.transform(np.array([[t_rsi, t_vol, t_dist]]))
        prob = model.predict_proba(feat)[0][1]
        res = "BULLISH" if prob > 0.5 else "BEARISH"
        st.markdown(f"### Result: :{'green' if res=='BULLISH' else 'red'}[{res}] ({round(prob*100,2)}% Confidence)")
        st.progress(prob)
    else: st.error("Brain file (nexus_brain.pkl) not found. Run train_brain.py.")

st.divider()

# --- SECTION 3: ANALYTICS & RAW DATA ---
tab_v, tab_f = st.tabs(["📊 Performance Analytics", "📡 Live Signal Feed"])

all_raw = pd.concat([df.assign(Engine=n) for n, df in all_dfs.items() if not df.empty]) if all_dfs else pd.DataFrame()

with tab_v:
    if not all_raw.empty:
        fig = px.scatter(all_raw, x="ts", y="confidence", color="Engine", 
                         title="Signal Distribution & Conviction", hover_data=["asset", "reason"])
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.histogram(all_raw, x="asset", color="Engine", barmode="group", title="Asset Activity by Engine")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Awaiting market data for visualization.")

with tab_f:
    if not all_raw.empty:
        cf1, cf2 = st.columns([1, 4])
        with cf1:
            st.download_button("📥 Export Feed", convert_df_to_csv(all_raw), "nexus_signals.csv", "text/csv")
        with cf2:
            if st.button("📤 Forward Signal Feed to Discord"):
                all_raw.to_csv("temp_signals.csv", index=False)
                try:
                    from nexus_reporter import forward_to_discord
                    forward_to_discord("temp_signals.csv", "Full Master Signal Feed")
                    st.success("Feed sent to Discord!")
                except: st.error("Reporter script missing.")
        
        st.dataframe(all_raw.sort_values('ts', ascending=False), use_container_width=True, hide_index=True)

if st.sidebar.button("🔄 Global System Refresh"): st.rerun()
