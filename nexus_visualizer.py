import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import pickle
import os
import json
from datetime import datetime
from config import DB_FILE, MODEL_FILE, PERFORMANCE_FILE, TOTAL_CAPITAL, RISK_PER_TRADE

# 1. Page Configuration
st.set_page_config(
    page_title="Nexus Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Robust Data Loaders
def safe_query(query):
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.sidebar.error(f"DB Query Error: {e}")
        return pd.DataFrame()

# 3. Sidebar: System Status
st.sidebar.title("🛡️ System Health")

if os.path.exists(PERFORMANCE_FILE):
    with open(PERFORMANCE_FILE, "r") as f:
        perf_data = json.load(f)
        for engine, stats in perf_data.items():
            status = stats.get("status", "UNKNOWN")
            color = "🟢" if status == "LIVE" else "🔴"
            st.sidebar.markdown(f"{color} **{engine.upper()}**: {status}")
            st.sidebar.caption(f"Win Rate: {stats.get('win_rate', 0)}% | Trades: {stats.get('total_trades', 0)}")
else:
    st.sidebar.warning("No performance data found.")

# 4. Main Dashboard Logic
st.title("📈 Nexus Full-Suite Intelligence")

# KPIs and Data Processing
df = safe_query("SELECT * FROM signals")

if df.empty:
    st.info("👋 Welcome to Nexus. No trade data found yet. Run your engines to populate the dashboard.")
else:
    # Calculate Equity Curve (Only on Audited Results)
    df_audited = df[df['result'].notnull()].copy()
    
    if not df_audited.empty:
        equity = [TOTAL_CAPITAL]
        for _, row in df_audited.sort_values('ts').iterrows():
            change = (TOTAL_CAPITAL * RISK_PER_TRADE * 2) if row['result'] == 1 else -(TOTAL_CAPITAL * RISK_PER_TRADE)
            equity.append(equity[-1] + change)
        
        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Account Equity", f"${equity[-1]:.2f}", f"{(equity[-1]-TOTAL_CAPITAL):+.2f}")
        m2.metric("Active Signals", len(df[df['result'].isnull()]))
        m3.metric("Total Audited", len(df_audited))
        
        win_rate = (df_audited['result'].sum() / len(df_audited)) * 100
        m4.metric("Global Win Rate", f"{win_rate:.1f}%")

        # Plot Equity
        st.subheader("🚀 Equity Growth")
        fig_equity = px.line(x=range(len(equity)), y=equity, template="plotly_dark", 
                             labels={'x': 'Trade Count', 'y': 'Balance ($)'})
        fig_equity.update_traces(line_color='#00ffcc', fill='tozeroy')
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.warning("Signals detected, but none have been audited yet. Waiting for price to hit TP/SL.")

    # AI Feature Importance
    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🧠 AI Brain Analysis")
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    pipeline = pickle.load(f)
                
                # Extracting feature importance from the model inside the pipeline
                model = pipeline.named_steps['model']
                # Gradient Boosting / Random Forest support
                importances = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'estimators_'): # Voting Classifier
                    importances = model.estimators_[0].feature_importances_
                
                if importances is not None:
                    feat_df = pd.DataFrame({'Feature': ['RSI', 'Volume', 'EMA Dist'], 'Importance': importances})
                    fig_brain = px.bar(feat_df, x='Importance', y='Feature', orientation='h', template="plotly_dark")
                    st.plotly_chart(fig_brain, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render Brain logic: {e}")
        else:
            st.info("AI Model not trained yet.")

    with col_right:
        st.subheader("📡 Recent Signals")
        st.dataframe(df.sort_values('ts', ascending=False).head(10), use_container_width=True)
