import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.express as px
from datetime import datetime

# 1. Page Config
st.set_page_config(page_title="Aegis Risk | Scanner", page_icon="📉", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# 3. Risk Engine Logic
def fetch_volatility_data(symbol="BTC/USDT"):
    try:
        ex = ccxt.binance()
        # Fetch 100 hourly candles
        ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        
        # Calculate Log Returns
        df['returns'] = np.log(df['c'] / df['c'].shift(1))
        
        # Calculate Rolling Volatility (Standard Deviation)
        df['volatility'] = df['returns'].rolling(window=24).std() * np.sqrt(24)
        return df
    except Exception as e:
        st.error(f"Risk Engine Error: {e}")
        return pd.DataFrame()

# 4. Header
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title("📉 Aegis Risk: Volatility Scanner")
    st.write("Current Focus: **Global Market Turbulence & Drawdown Protection**")
with col_h2:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

st.write("---")

# 5. Main Scanner
with st.spinner("Scanning Global Markets..."):
    risk_df = fetch_volatility_data()

if not risk_df.empty:
    current_vol = risk_df['volatility'].iloc[-1]
    avg_vol = risk_df['volatility'].mean()
    
    # Determine Risk Level
    risk_score = (current_vol / avg_vol) * 50 # Base 50
    status = "STABLE" if risk_score < 70 else "HIGH TURBULENCE"
    color = "inverse" if risk_score > 70 else "normal"

    # Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.metric("Volatility Index", f"{current_vol:.4f}", delta=f"{((current_vol/avg_vol)-1)*100:.1f}%", delta_color=color)
    m2.metric("Market Status", status)
    m3.metric("System Drawdown", "0.00%", help="Calculated from Aegis Wealth database.")

    # Volatility Chart
    st.subheader("📊 Volatility Trend (24h Rolling)")
    fig = px.area(risk_df, x=pd.to_datetime(risk_df['ts'], unit='ms'), y='volatility', 
                  title="Hourly Market Agitation", color_discrete_sequence=['#ff4b4b' if status != "STABLE" else '#4F8BF9'])
    fig.update_layout(template="plotly_dark", xaxis_title="Time", yaxis_title="Std Dev")
    st.plotly_chart(fig, use_container_width=True)

    # 6. Actionable Alerts
    st.write("---")
    st.subheader("🚨 Risk Mitigation Protocols")
    if risk_score > 75:
        st.error(f"CRITICAL: Market agitation is {risk_score:.1f}% above baseline.")
        st.button("FORCE LIQUIDATION (Emergency Move to USDT)")
    elif risk_score > 60:
        st.warning("CAUTION: Reducing recommended leverage for Nexus Signal.")
    else:
        st.success("OPTIMAL: Market conditions suitable for automated strategies.")

else:
    st.info("Awaiting data from Exchange API...")
