import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from config import DB_FILE, PERFORMANCE_FILE, TOTAL_CAPITAL

# 1. Page Config
st.set_page_config(page_title="Nexus Command Center", page_icon="🤖", layout="wide")

# Custom CSS for Dark Mode Professionalism
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_content-allow-html=True)

# 2. Data Access Layer
def load_data():
    if not os.path.exists(DB_FILE):
        st.error(f"Database {DB_FILE} not found. Run the engine first!")
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
        conn.close()
        # Ensure timestamp is datetime
        df['ts'] = pd.to_datetime(df['ts'])
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

# 3. Sidebar Status
st.sidebar.title("🛡️ Nexus System Status")
if os.path.exists(PERFORMANCE_FILE):
    with open(PERFORMANCE_FILE, 'r') as f:
        perf = json.load(f)
        for engine, stats in perf.items():
            st.sidebar.subheader(f"Engine: {engine.upper()}")
            st.sidebar.write(f"📈 Win Rate: {stats.get('win_rate', 0):.1f}%")
            st.sidebar.write(f"📝 Total Trades: {stats.get('total_trades', 0)}")
            st.sidebar.progress(min(stats.get('win_rate', 0) / 100, 1.0))

# 4. Dashboard Main Logic
st.title("📈 Nexus AI Intelligence Dashboard")
df = load_data()

if df.empty:
    st.info("Waiting for first signals to be generated...")
else:
    # --- TOP ROW: KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_signals = len(df)
    diamond_signals = len(df[(df['confidence'] >= 90) | (df['confidence'] <= 10)])
    latest_price = df['entry'].iloc[0]
    
    col1.metric("Total Signals", total_signals)
    col2.metric("Diamond Consensus", diamond_signals)
    col3.metric("Last Entry", f"${latest_price:,.2f}")
    col4.metric("Market Status", "LIVE", delta="Active")

    # --- ROW 2: EQUITY & SIGNAL DISTRIBUTION ---
    st.divider()
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📊 Signal Confidence Timeline")
        # Visualizing signal strength over time
        fig = px.scatter(df, x="ts", y="confidence", color="signal", 
                         size="confidence", hover_data=['asset', 'entry'],
                         color_discrete_map={"LONG": "#00ffcc", "SHORT": "#ff4b4b"},
                         template="plotly_dark")
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("🎯 Asset Distribution")
        asset_counts = df['asset'].value_counts()
        fig_pie = px.pie(values=asset_counts.values, names=asset_counts.index, 
                         hole=0.4, template="plotly_dark",
                         color_discrete_sequence=px.colors.sequential.Cyan_r)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- ROW 3: RECENT SIGNALS TABLE ---
    st.divider()
    st.subheader("📑 Recent Intelligence Logs")
    
    # Clean up dataframe for display
    display_df = df[['ts', 'asset', 'signal', 'entry', 'confidence', 'rsi', 'dist_ema']].copy()
    display_df['rsi'] = display_df['rsi'].round(2)
    display_df['dist_ema'] = (display_df['dist_ema'] * 100).round(2).astype(str) + '%'
    
    st.dataframe(display_df.head(20), use_container_width=True)

    # --- ROW 4: FEATURE ANALYSIS ---
    st.divider()
    st.subheader("🧠 Technical Confluence Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("RSI vs Confidence")
        fig_rsi = px.density_heatmap(df, x="rsi", y="confidence", text_auto=True, template="plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    with c2:
        st.write("EMA Distance Distribution")
        fig_ema = px.histogram(df, x="dist_ema", nbins=20, template="plotly_dark", color_discrete_sequence=['#00ffcc'])
        st.plotly_chart(fig_ema, use_container_width=True)

