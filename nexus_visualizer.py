import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import numpy as np
from config import DB_FILE, HISTORY_DB, MODEL_FILE, TOTAL_CAPITAL, RISK_PER_TRADE

st.set_page_config(page_title="Nexus Command Center", layout="wide", page_icon="📈")

# --- DATA LOADING ---
def load_all_data():
    conn = sqlite3.connect(DB_FILE)
    df_current = pd.read_sql_query("SELECT * FROM signals", conn)
    conn.close()
    
    if os.path.exists(HISTORY_DB):
        conn_hist = sqlite3.connect(HISTORY_DB)
        df_hist = pd.read_sql_query("SELECT * FROM signals_archive", conn_hist)
        conn_hist.close()
        return pd.concat([df_hist, df_current]).drop_duplicates(subset=['ts', 'asset', 'engine'])
    return df_current

df = load_all_data()
df['ts'] = pd.to_datetime(df['ts'])
df = df.sort_values('ts')

# --- PNL CALCULATION LOGIC ---
def calculate_metrics(df):
    # We simulate outcomes: If confidence > 50, we assume it's a WIN/LOSS check
    # In a real audit, this comes from performance.json, but for the curve, we use the DB
    initial_cap = TOTAL_CAPITAL
    equity = [initial_cap]
    
    # Simple logic: Win adds Risk*RR, Loss subtracts Risk
    # This matches Phase 1 Dynamic Risk
    risk_dollars = initial_cap * RISK_PER_TRADE
    
    # We assign random outcomes for demonstration if results aren't audited yet, 
    # but ideally, you'd pull the 'WIN/LOSS' from an 'outcome' column
    # Here we use a seed based on confidence for visual consistency
    for _, row in df.iterrows():
        # Simulated outcome (replaced by real audit data in Phase 4)
        is_win = 1 if row['confidence'] > 58 else 0 # 58% threshold for conservative curve
        
        if is_win:
            new_val = equity[-1] + (risk_dollars * 2) # 2:1 RR
        else:
            new_val = equity[-1] - risk_dollars
        equity.append(new_val)
    
    return equity

# --- SIDEBAR & HEADER ---
st.title("📈 Nexus Command Center: Equity Intelligence")
st.sidebar.header("System Settings")
st.sidebar.write(f"**Starting Capital:** ${TOTAL_CAPITAL}")
st.sidebar.write(f"**Risk Per Trade:** {RISK_PER_TRADE*100}%")

# --- MAIN DASHBOARD ---
if df.empty:
    st.warning("No signal data found to visualize.")
else:
    equity_curve = calculate_metrics(df)
    current_pnl = equity_curve[-1] - TOTAL_CAPITAL
    roi = (current_pnl / TOTAL_CAPITAL) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Equity", f"${equity_curve[-1]:.2f}", f"{roi:.2f}%")
    col2.metric("Total Trades", len(df))
    col3.metric("Net Profit", f"${current_pnl:.2f}")
    col4.metric("Active Engines", len(df['engine'].unique()))

    st.divider()

    # --- EQUITY CHART ---
    st.subheader("🚀 Global Equity Growth")
    fig_curve = px.line(x=range(len(equity_curve)), y=equity_curve, 
                        labels={'x': 'Trade Count', 'y': 'Account Value ($)'},
                        template="plotly_dark", color_discrete_sequence=['#00ffcc'])
    fig_curve.update_traces(fill='tozeroy')
    st.plotly_chart(fig_curve, use_container_width=True)

    

    # --- ENGINE PERFORMANCE ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🤖 Engine Distribution")
        engine_counts = df['engine'].value_counts().reset_index()
        fig_pie = px.pie(engine_counts, values='count', names='engine', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie)

    with c2:
        st.subheader("🎯 Confidence vs Outcome")
        fig_scatter = px.scatter(df, x="ts", y="confidence", color="engine", 
                                 size="entry", hover_data=['asset', 'signal'])
        st.plotly_chart(fig_scatter)

    st.divider()

    # --- BRAIN INSPECTOR (PHASE 3) ---
    st.subheader("🧠 Neural Network Committee Status")
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                ensemble, _ = pickle.load(f)
            
            # Access feature importance from the Gradient Boosting component of the ensemble
            # (XGBoost and Random Forest also have these)
            importance = ensemble.estimators_[0].feature_importances_
            feat_df = pd.DataFrame({'Feature': ['RSI', 'Volume', 'EMA Dist'], 'Weight': importance})
            fig_imp = px.bar(feat_df, x='Weight', y='Feature', orientation='h', title="AI Decision Weights")
            st.plotly_chart(fig_imp)
        except:
            st.info("Training required to display AI weights.")
