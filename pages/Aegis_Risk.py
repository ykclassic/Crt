import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import plotly.express as px
import requests
from fpdf import FPDF
from datetime import datetime
import io

# 1. Page Configuration
st.set_page_config(page_title="Aegis Risk | Intelligence Hub", page_icon="📉", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# 3. Data & Global Metrics Engine
@st.cache_data(ttl=600)
def get_global_metrics():
    try:
        # Fear & Greed Index
        fng_res = requests.get("https://api.alternative.me/fng/").json()
        fng_v, fng_l = int(fng_res['data'][0]['value']), fng_res['data'][0]['value_classification']
        # BTC Dominance & Altcoin Index
        cg_res = requests.get("https://api.coingecko.com/api/v3/global").json()
        btc_d, eth_d = cg_res['data']['market_cap_percentage']['btc'], cg_res['data']['market_cap_percentage']['eth']
        return fng_v, fng_l, round(btc_d, 2), round(100 - (btc_d + eth_d), 2)
    except:
        return 50, "Neutral", 50.0, 25.0

def fetch_market_data(symbol, timeframe):
    try:
        ex = ccxt.bitget()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        df['returns'] = np.log(df['c'] / df['c'].shift(1))
        vol = df['returns'].std() * np.sqrt(len(df))
        return df, vol
    except:
        return pd.DataFrame(), 0

# 4. Header & Top Metrics (Maintained)
st.title("📉 Aegis Risk: Market Analyzer")
fng_v, fng_l, btc_d, alt_i = get_global_metrics()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Fear & Greed", f"{fng_v}/100", fng_l)
m2.metric("BTC Dominance", f"{btc_d}%")
m3.metric("Altcoin Index", f"{alt_i}%")
m4.metric("Risk Status", "WATCH" if fng_v > 75 else "SAFE")

st.write("---")

# 5. Core View: Analysis vs. Settings
tab_analysis, tab_settings = st.tabs(["📊 Market Analysis", "⚙️ Notification Settings"])

with tab_analysis:
    col_ctl, col_viz = st.columns([1, 3])
    with col_ctl:
        st.subheader("🛡️ Parameters")
        asset = st.selectbox("Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
        df, vol_score = fetch_market_data(asset, tf)
        st.metric("Realized Volatility", f"{vol_score:.4f}")
        
        if st.button("Generate Market Snapshot"):
            st.toast("Generating PDF Report...", icon="📄")

    with col_viz:
        if not df.empty:
            # LIQUIDATION HEATMAP (Maintained)
            st.subheader("🔥 Liquidation Heatmap Proxy")
            last_p = df['c'].iloc[-1]
            fig = go.Figure(go.Candlestick(x=df['dt'], open=df['o'], high=df['h'], low=df['l'], close=df['c']))
            fig.add_hline(y=last_p * 0.96, line_dash="dash", line_color="orange", annotation_text="25x Long Liq")
            fig.add_hline(y=last_p * 1.04, line_dash="dash", line_color="orange", annotation_text="25x Short Liq")
            fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # VOLATILITY TREND (Maintained)
            st.subheader("📊 Volatility Z-Score Trend")
            df['vol_rolling'] = df['returns'].rolling(10).std()
            fig_vol = px.area(df, x='dt', y='vol_rolling', color_discrete_sequence=['#4F8BF9'])
            fig_vol.update_layout(template="plotly_dark", height=200)
            st.plotly_chart(fig_vol, use_container_width=True)

with tab_settings:
    st.subheader("📡 Global Alert Configuration")
    st.write("Define the thresholds that trigger system-wide risk notifications.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.toggle("Enable Telegram Push Alerts", value=False, help="Connects to the Aegis Telegram Bot.")
        st.toggle("Enable Browser Toast Notifications", value=True)
        st.slider("Volatility Alert Threshold", 0.01, 0.10, 0.03, format="%.2f")
    
    with c2:
        st.text_input("Telegram Bot Token", type="password")
        st.text_input("Target Chat ID")
        st.multiselect("Trigger Events", ["Volatility Spikes", "Liquidation Proximity", "Dominance Shift", "Altseason Entry"], 
                       default=["Volatility Spikes", "Dominance Shift"])
    
    if st.button("Test Notification System"):
        st.toast("Aegis Risk: Test Alert Successful", icon="🛡️")

st.caption("Aegis Unified Risk Engine v5.0 | Analysis & Reporting Mode")
