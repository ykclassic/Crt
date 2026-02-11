# app.py - ProfitForge AI Multi-Timeframe Edition
# Requirements: pip install streamlit ccxt pandas numpy plotly scikit-learn pandas-ta

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timezone
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Error handling for missing pandas_ta library
try:
    import pandas_ta as ta
except ImportError:
    st.error("Missing library: **pandas-ta**. Please run `pip install pandas-ta` or add it to your requirements.txt file.")
    st.stop()

# ---------------------------  
# Data Fetching Engine (Cached)
# ---------------------------  
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol, timeframe, limit=500):
    exchange = ccxt.xt({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        return df
    except:
        return pd.DataFrame()

def get_trend_bias(df):
    if df.empty or len(df) < 50: return "NEUTRAL"
    ema = df.ta.ema(length=50)
    current_price = df['close'].iloc[-1]
    current_ema = ema.iloc[-1]
    return "BULLISH" if current_price > current_ema else "BEARISH"

# ---------------------------  
# AI Feature Engineering
# ---------------------------  
def add_ml_features(df):
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    return df.dropna()

@st.cache_resource(show_spinner="Training 1H AI Model...")
def train_ai_model(df):
    # Target: 1.5% move in next 6 hours
    df['target'] = (df['close'].pct_change(6).shift(-6) > 0.015).astype(int)
    features = ['RSI_14', 'MACDh_12_26_9', 'ATRr_14'] # Key core features
    
    # Ensure all columns exist
    feat_cols = [c for c in df.columns if any(f in c for f in features)]
    data = df.dropna(subset=['target'] + feat_cols)
    
    if len(data) < 100: return None, None, feat_cols
    
    X = data[feat_cols]
    y = data['target']
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X_s, y)
    return model, scaler, feat_cols

# ---------------------------  
# Main Application UI
# ---------------------------  
st.set_page_config(layout="wide", page_title="ProfitForge AI Pro")
st.title("🤖 ProfitForge AI: Multi-Timeframe Strategy")

with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Market", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
    risk_reward = st.slider("Risk:Reward Ratio", 1.5, 4.0, 2.0)
    refresh = st.button("Manual Refresh")

# 1. FETCH DATA FOR ALL TIMEFRAMES
df_1d = fetch_data(symbol, "1d", 100)
df_4h = fetch_data(symbol, "4h", 200)
df_1h = fetch_data(symbol, "1h", 600)

if not df_1d.empty and not df_4h.empty and not df_1h.empty:
    # 2. CALCULATE BIAS
    bias_1d = get_trend_bias(df_1d)
    bias_4h = get_trend_bias(df_4h)
    
    # UI Header for Trend Confirmation
    c1, c2, c3 = st.columns(3)
    c1.metric("1D Macro Bias", bias_1d, delta=None, delta_color="normal")
    c2.metric("4H Confirmation", bias_4h, delta=None, delta_color="normal")
    
    # 3. AI ANALYSIS ON 1H
    df_1h_feat = add_ml_features(df_1h)
    model, scaler, feat_cols = train_ai_model(df_1h_feat)
    
    signal = "NEUTRAL"
    reason = "Waiting for Bias Alignment"
    
    # Logic: Only Signal if 1D and 4H match
    if bias_1d == bias_4h and bias_1d != "NEUTRAL":
        if model:
            latest_data = df_1h_feat[feat_cols].iloc[-1:]
            prob = model.predict_proba(scaler.transform(latest_data))[0][1]
            
            if bias_1d == "BULLISH" and prob > 0.65:
                signal = "LONG"
                reason = "Trend & AI Convergence"
            elif bias_1d == "BEARISH" and prob < 0.35:
                signal = "SHORT"
                reason = "Trend & AI Convergence"
            else:
                reason = f"AI Confidence ({round(prob*100)}%) too low for {bias_1d} entry"
    
    c3.metric("Trade Signal", signal)

    st.divider()

    # 4. ENTRY, SL, TP RECOMMENDATION
    if signal != "NEUTRAL":
        entry = df_1h['close'].iloc[-1]
        atr = df_1h_feat[[c for c in df_1h_feat.columns if 'ATRr' in c][0]].iloc[-1]
        
        if signal == "LONG":
            sl = entry - (1.5 * atr)
            tp = entry + (risk_reward * 1.5 * atr)
        else:
            sl = entry + (1.5 * atr)
            tp = entry - (risk_reward * 1.5 * atr)
            
        st.subheader(f"🚀 Recommended {signal} Setup")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.write(f"**Entry Price:**\n# ${entry:,.2f}")
        res_col2.write(f"**Stop Loss:**\n# :red[${sl:,.2f}]")
        res_col3.write(f"**Take Profit:**\n# :green[${tp:,.2f}]")
        
        # Risk Summary
        st.info(f"**Reasoning:** 1D & 4H are {bias_1d}. AI confirms {signal} setup on 1H with volatility-adjusted SL based on ATR.")
    else:
        st.warning(f"**Market Status:** {reason}. No high-probability setup detected.")

    # 5. CHARTING
    st.subheader(f"{symbol} 1H Execution Chart")
    fig = go.Figure(data=[go.Candlestick(x=df_1h.index, open=df_1h['open'], high=df_1h['high'], low=df_1h['low'], close=df_1h['close'])])
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Could not fetch data from exchange. Check your internet connection.")
