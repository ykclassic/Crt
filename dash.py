import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import numpy as np
import plotly.express as px
import math
import logging
from datetime import datetime
from config import DB_FILE, JOURNAL_DB, PERFORMANCE_FILE, MODEL_FILE, ENGINES, WEBHOOK_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

st.set_page_config(page_title="Nexus Intelligence Suite", layout="wide", page_icon="🛡️")

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
    base_query = "SELECT ts as 'Time', category as 'Type', entry as 'Observation' FROM journal"
    params = []
    if search_term:
        base_query += " WHERE entry LIKE ? OR category LIKE ?"
        params = [f"%{search_term}%", f"%{search_term}%"]
    base_query += " ORDER BY id DESC"
    df = pd.read_sql_query(base_query, conn, params=params)
    conn.close()
    return df

def load_signals(engine_filter=None):
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT *, engine as Engine FROM signals"
    if engine_filter:
        query += " WHERE engine = ?"
        df = pd.read_sql_query(query, conn, params=(engine_filter,))
    else:
        df = pd.read_sql_query(query + " ORDER BY id DESC LIMIT 500", conn)
    conn.close()
    if not df.empty:
        df["confidence"] = pd.to_numeric(df["confidence"], errors='coerce').fillna(0.0).round(2)
        df["reason"] = df["reason"].fillna("Technical Evaluation")
    return df

def discord_forward_helper(df, title):
    temp_name = "dispatch_temp.csv"
    df.to_csv(temp_name, index=False)
    try:
        from nexus_reporter import forward_to_discord
        forward_to_discord(temp_name, title)
        st.success(f"Forwarded {title} to Discord!")
        os.remove(temp_name)
    except Exception as e:
        st.error(f"Forwarding error: {e}")

init_journal()

st.title("🛡️ Nexus Intelligence Suite: Visual Command")

perf_data = {}
if os.path.exists(PERFORMANCE_FILE):
    try:
        with open(PERFORMANCE_FILE, "r") as f: 
            perf_data = json.load(f)
    except: pass

m_cols = st.columns(len(ENGINES))
for i, (strat_id, name) in enumerate(ENGINES.items()):
    stats = perf_data.get(strat_id, {"win_rate": 0.0, "closed_sample": 0})
    wr, ss = stats.get("win_rate", 0), stats.get("closed_sample", 0)
    trust = round((wr * math.sqrt(ss)) / 10, 1) if ss > 0 else 0.0
    with m_cols[i]:
        st.metric(name, f"{trust}/100", f"{wr}% WR ({ss} Closed)")
        st.progress(min(trust/100, 1.0))

st.divider()

c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("💎 Market Confluence")
    master_df = load_signals()
    
    if not master_df.empty:
        consensus = master_df.groupby(['asset', 'signal', 'timeframe']).agg({
            'Engine': 'nunique', 
            'confidence': 'mean', 
            'reason': lambda x: ' + '.join(x.unique())
        }).reset_index().rename(columns={'Engine': 'Matches', 'confidence': 'Avg_Conf'})
        
        consensus['Avg_Conf'] = consensus['Avg_Conf'].round(2)
        st.dataframe(consensus.sort_values('Matches', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Awaiting market data confluence...")

with c_right:
    st.subheader("📓 Digital Journal")
    with st.expander("📝 Log Entry"):
        cat = st.selectbox("Category", ["Market Observation", "Diamond Audit", "AI Note"])
        note = st.text_area("Details")
        if st.button("Save Note"):
            save_journal_entry(cat, note)
            st.rerun()
    
    history = load_journal(st.text_input("🔍 Search Logs"))
    if not history.empty:
        jc1, jc2 = st.columns(2)
        jc1.download_button("📥 Download CSV", convert_df_to_csv(history), "journal.csv")
        if jc2.button("📤 Discord Forward", key="j_discord"):
            discord_forward_helper(history, "Journal Observation Report")
        st.dataframe(history, use_container_width=True, hide_index=True, height=150)

st.divider()

st.subheader("🧠 AI Neural Simulator")
s1, s2, s3 = st.columns(3)
t_rsi = s1.slider("RSI", 0, 100, 50)
t_vol = s2.number_input("Volume % Change", value=0.0)
t_dist = s3.slider("Distance from Mean %", -10.0, 10.0, 0.0)

if st.button("🔮 Run Prediction"):
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f: 
            model, scaler = pickle.load(f)
        prob = model.predict_proba(scaler.transform(np.array([[t_rsi, t_vol, t_dist]])))[0][1]
        st.markdown(f"### Result: {'🟢 BULLISH' if prob > 0.5 else '🔴 BEARISH'} ({round(prob*100,2)}%)")
    else: 
        st.error("Model file missing.")

st.divider()

st.subheader("📡 Intelligence Feed & Signal Exports")

full_signals = load_signals()
if not full_signals.empty:
    ec1, ec2 = st.columns([2, 2])
    with ec1:
        st.download_button("📥 Download All Signals", convert_df_to_csv(full_signals), "nexus_signals.csv")
    with ec2:
        if st.button("📤 Forward to Discord", key="s_discord"):
            discord_forward_helper(full_signals, "Master Signal Feed Audit")
    
    tab_vis, tab_raw = st.tabs(["📊 Performance Visuals", "📜 Raw Signal Data"])
    with tab_vis:
        st.plotly_chart(px.scatter(full_signals, x="ts", y="confidence", color="Engine", hover_data=["asset", "timeframe", "reason"]), use_container_width=True)
    with tab_raw:
        def apply_row_style(row):
            conf = row.get('confidence', 0)
            color = '#00ff00' if conf > 80 else '#ff4b4b' if conf < 60 else 'white'
            return [f"color: {color}"] * len(row)
        styled = full_signals.style.apply(apply_row_style, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.warning("Awaiting signal data...")

if st.sidebar.button("🔄 Refresh System"): 
    st.rerun()
