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

def load_journal():
    if not os.path.exists(JOURNAL_DB): return pd.DataFrame()
    conn = sqlite3.connect(JOURNAL_DB)
    df = pd.read_sql_query("SELECT ts as 'Time', category as 'Type', entry as 'Observation' FROM journal ORDER BY id DESC", conn)
    conn.close()
    return df

def load_signals(db_path):
    if not os.path.exists(db_path): return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        if "conf" in df.columns: df = df.rename(columns={"conf": "confidence"})
        if "reason" not in df.columns: df["reason"] = "LEGACY"
        return df
    except: return pd.DataFrame()

# --- INITIALIZE ---
init_journal()

# --- HEADER & STATS ---
st.title("🛡️ Nexus Intelligence Suite: Visual Command")
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
        st.metric(name, f"{trust}/100", f"{wr}% WR")
        st.progress(min(trust/100, 1.0))

st.divider()

# --- THE DIGITAL JOURNAL (NEW SECTION) ---
st.subheader("📓 Observation Journal")
j_col1, j_col2 = st.columns([1, 2])

with j_col1:
    st.write("📝 **Log New Entry**")
    category = st.selectbox("Category", ["Market Observation", "Diamond Consensus Audit", "AI Accuracy Note", "System Error"])
    note_text = st.text_area("What did you notice?", placeholder="Example: Diamond Consensus on BTC was a clean win...")
    if st.button("Save Entry"):
        if note_text:
            save_journal_entry(category, note_text)
            st.success("Entry Saved!")
            st.rerun()

with j_col2:
    st.write("📖 **Historical Logs**")
    history_df = load_journal()
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No journal entries yet. Start logging your observations!")

st.divider()

# --- CONSENSUS & ANALYTICS ---
all_dfs = {name: load_signals(db) for name, db in DB_FILES.items()}
tab1, tab2 = st.tabs(["📊 Market Confluence", "📡 Raw Live Feed"])

with tab1:
    # Logic for Confluence
    master_list = []
    for name, df in all_dfs.items():
        if not df.empty:
            t = df.sort_values('ts').groupby('asset').tail(1).copy()
            t['Engine'] = name
            master_list.append(t)
    
    if master_list:
        master_df = pd.concat(master_list)
        consensus = master_df.groupby(['asset', 'signal']).agg({'Engine': 'count', 'confidence': 'mean', 'reason': lambda x: ' | '.join(x.unique())}).reset_index()
        st.dataframe(consensus.sort_values('Engine', ascending=False), use_container_width=True)
    else:
        st.info("Awaiting new signals for confluence analysis.")

with tab2:
    if master_list:
        raw_df = pd.concat(master_list).sort_values('ts', ascending=False)
        st.dataframe(raw_df[["ts", "Engine", "asset", "signal", "confidence", "reason"]], use_container_width=True)

if st.sidebar.button("🔄 Force Refresh"): st.rerun()
