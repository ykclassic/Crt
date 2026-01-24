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

# --- UTILITIES ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def init_journal():
    try:
        conn = sqlite3.connect(JOURNAL_DB)
        conn.execute('''CREATE TABLE IF NOT EXISTS journal 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         ts TEXT, category TEXT, entry TEXT)''')
        conn.close()
    except Exception as e:
        st.error(f"Failed to initialize Journal DB: {e}")

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
        if "conf" in df.columns and "confidence" not in df.columns:
            df = df.rename(columns={"conf": "confidence"})
        return df
    except Exception as e:
        st.warning(f"Note: {db_path} could not be read (might be empty).")
        return pd.DataFrame()

def discord_forward_helper(df, title):
    temp_name = "dispatch_temp.csv"
    df.to_csv(temp_name, index=False)
    try:
        # We import inside the function to prevent the app from crashing if the file is missing
        if os.path.exists("nexus_reporter.py"):
            from nexus_reporter import forward_to_discord
            forward_to_discord(temp_name, title)
            st.success(f"Forwarded to Discord!")
            os.remove(temp_name)
        else:
            st.error("Error: nexus_reporter.py not found in directory.")
    except Exception as e:
        st.error(f"Forwarding error: {e}")

# --- INITIALIZE ---
init_journal()

# --- HEADER & STATS ---
st.title("🛡️ Nexus Intelligence Suite: Visual Command")

# Load performance with safety
perf_data = {}
if os.path.exists(PERFORMANCE_FILE):
    try:
        with open(PERFORMANCE_FILE, "r") as f:
            perf_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading performance.json: {e}")

m_cols = st.columns(len(DB_FILES))
for i, (name, db_file) in enumerate(DB_FILES.items()):
    strat_id = db_file.replace(".db", "")
    stats = perf_data.get(strat_id, {"win_rate": 0.0, "sample_size": 0})
    wr, ss = stats.get("win_rate", 0), stats.get("sample_size", 0)
    trust = round((wr * math.sqrt(ss)) / 10, 1) if ss > 0 else 0.0
    with m_cols[i]:
        st.metric(name, f"{trust}/100", f"{wr}% WR")
        st.progress(min(trust/100, 1.0))

st.divider()

# --- TOP: CONFLUENCE & JOURNAL ---
c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("💎 Market Confluence")
    all_dfs = {name: load_signals(path) for name, path in DB_FILES.items()}
    master_list = [df.assign(Engine=name) for name, df in all_dfs.items() if not df.empty]
    
    if master_list:
        m_df = pd.concat(master_list)
        # Ensure 'reason' exists
        if 'reason' not in m_df.columns: m_df['reason'] = "N/A"
        
        consensus = m_df.groupby(['asset', 'signal']).agg({
            'Engine': 'count', 'confidence': 'mean', 'reason': lambda x: ' | '.join(x.astype(str).unique())
        }).reset_index().rename(columns={'Engine': 'Matches', 'confidence': 'Avg_Conf'})
        st.dataframe(consensus.sort_values('Matches', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No active signals found. Run your engines to populate data.")

with c_right:
    st.subheader("📓 Digital Journal")
    with st.expander("📝 Log Entry"):
        cat = st.selectbox("Category", ["Market Observation", "Diamond Audit", "AI Note"])
        note = st.text_area("Details")
        if st.button("Save Note"):
            save_journal_entry(cat, note); st.rerun()
    
    j_search = st.text_input("🔍 Search Logs")
    history = load_journal(j_search)
    if not history.empty:
        jc1, jc2 = st.columns(2)
        jc1.download_button("📥 Download CSV", convert_df_to_csv(history), "journal.csv")
        if jc2.button("📤 Discord Forward", key="j_discord"):
            discord_forward_helper(history, "Journal Observation Report")
        st.dataframe(history, use_container_width=True, hide_index=True, height=150)

st.divider()

# --- MIDDLE: AI SIMULATOR ---
st.subheader("🧠 AI Neural Simulator")
s1, s2, s3 = st.columns(3)
t_rsi = s1.slider("RSI", 0, 100, 50)
t_vol = s2.number_input("Volume % Change", value=0.0)
t_dist = s3.slider("Distance from Mean %", -10.0, 10.0, 0.0)

if st.button("🔮 Run Prediction"):
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f: model, scaler = pickle.load(f)
            feat = scaler.transform(np.array([[t_rsi, t_vol, t_dist]]))
            prob = model.predict_proba(feat)[0][1]
            st.markdown(f"### Result: {'🟢 BULLISH' if prob > 0.5 else '🔴 BEARISH'} ({round(prob*100,2)}%)")
        except Exception as e:
            st.error(f"AI Model Error: {e}")
    else:
        st.error(f"Model file ({MODEL_FILE}) not found. Run train_brain.py first.")

st.divider()

# --- BOTTOM: ANALYTICS & SIGNAL FEED ---
st.subheader("📡 Intelligence Feed & Signal Exports")

if master_list:
    full_signals = pd.concat(master_list).sort_values('ts', ascending=False)
    
    ec1, ec2, _ = st.columns([2, 2, 5])
    with ec1:
        st.download_button("📥 Download All Signals", convert_df_to_csv(full_signals), "nexus_signals.csv")
    with ec2:
        if st.button("📤 Forward to Discord", key="s_discord"):
            discord_forward_helper(full_signals, "Master Signal Feed Audit")
    
    tab_vis, tab_raw = st.tabs(["📊 Performance Visuals", "📜 Raw Signal Data"])
    
    with tab_vis:
        try:
            st.plotly_chart(px.scatter(full_signals, x="ts", y="confidence", color="Engine", hover_data=["asset"]), use_container_width=True)
        except Exception as e:
            st.error(f"Visuals Error: {e}")
    
    with tab_raw:
        st.dataframe(full_signals, use_container_width=True, hide_index=True)
else:
    st.warning("Awaiting database population...")

if st.sidebar.button("🔄 Refresh System"): st.rerun()
