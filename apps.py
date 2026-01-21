# app.py - ProfitForge — AI-Powered Crypto Trading App (Optimized)
# Requirements: pip install streamlit ccxt pandas numpy plotly scikit-learn pandas_ta

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timezone
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta

# ---------------------------  
# Trading Session Detector  
# ---------------------------  
def get_trading_session():
    utc_now = datetime.now(timezone.utc)
    hour = utc_now.hour
    minute = utc_now.minute
    current_time = f"{hour:02d}:{minute:02d} UTC"

    if 0 <= hour < 8:
        session = "Asian Session"
        color = "orange"
        note = "Range-bound • Lower volatility"
    elif 8 <= hour < 12:
        session = "London Open"
        color = "blue"
        note = "Breakouts • Increasing volume"
    elif 12 <= hour < 16:
        session = "NY + London Overlap"
        color = "purple"
        note = "Highest volatility • Major moves"
    elif 16 <= hour < 21:
        session = "New York Session"
        color = "green"
        note = "Trend continuation • US news impact"
    else:
        session = "Quiet Hours"
        color = "gray"
        note = "Low volume • Consolidation"

    return session, color, note, current_time

# ---------------------------  
# Configuration  
# ---------------------------  
SYMBOLS = ["BTC/USDT", "SOL/USDT", "BNB/USDT", "ETH/USDT", "XRP/USDT"]

# ---------------------------  
# Data Fetching - Real XT.com via ccxt  
# ---------------------------  
@st.cache_data(ttl=60, show_spinner="Fetching live data from XT.com...")
def fetch_klines(symbol, timeframe, limit=500):
    exchange = ccxt.xt({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    try:
        # XT.com sometimes has strict limits, lowered default to 500 for stability
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) == 0:
            raise Exception("Empty response from XT.com")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return pd.DataFrame()

# ---------------------------  
# Feature Engineering with pandas_ta
# ---------------------------  
def add_features(df):
    if df.empty:
        return df
    df = df.copy()
    
    # We explicitly set lengths to ensure column names match the ML model expectations
    # Expected by model: RSI_14, MACD_12_26_9, BBL_5_2.0, EMA_50, ATRr_14, OBV, STOCHk_14_3_3
    
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    # Explicitly using length=5 to match the hardcoded feature list in the ML model
    df.ta.bbands(length=5, std=2.0, append=True) 
    df.ta.ema(length=50, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    
    return df.dropna()

# ---------------------------  
# AI Model (Random Forest Classifier)
# ---------------------------  
@st.cache_resource(show_spinner="Training AI Model...")
def train_model(df_train):
    # Features must match exactly what pandas_ta generates
    features = ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
                'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'EMA_50', 'ATRr_14', 'OBV', 'STOCHk_14_3_3']
    
    # Verify columns exist
    missing_cols = [col for col in features if col not in df_train.columns]
    if missing_cols:
        st.warning(f"Missing features for AI: {missing_cols}")
        return None, None, features

    # Target: price up >1.5% in next 6 candles
    df_train['future_return'] = df_train['close'].pct_change(6).shift(-6)
    df_train['target'] = (df_train['future_return'] > 0.015).astype(int)
    
    data_for_training = df_train.dropna(subset=['target'] + features)
    
    X = data_for_training[features]
    y = data_for_training['target']
    
    if len(X) < 50:
        return None, None, features

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduced estimators for speed in a live app context
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
    model.fit(X_scaled, y)
    
    return model, scaler, features

# ---------------------------  
# Signal Engine with AI Integration
# ---------------------------  
def generate_ai_signal(df):
    if df is None or len(df) < 60:
        return {"signal": "NEUTRAL", "ai_confidence": 0, "reason": "Insufficient data"}

    df_feat = add_features(df)
    
    # Train on historical, predict on latest
    model, scaler, features = train_model(df_feat)
    
    if model is None:
        return {"signal": "NEUTRAL", "ai_confidence": 0, "reason": "Model training failed (low data)"}

    try:
        latest = df_feat.iloc[-1:]
        X_latest = latest[features]
        X_scaled = scaler.transform(X_latest)
        
        prob = model.predict_proba(X_scaled)[0]
        confidence_long = round(prob[1] * 100, 1)
        confidence_short = round(prob[0] * 100, 1)
        
        # Threshold logic
        signal = "NEUTRAL"
        if confidence_long > 65: # Slightly lowered threshold for responsiveness
            signal = "LONG"
        elif confidence_short > 65:
            signal = "SHORT"
        
        ai_confidence = max(confidence_long, confidence_short)
        
        entry = round(latest['close'].iloc[-1], 2)
        atr = round(latest['ATRr_14'].iloc[-1], 2)
        
        # Fallback if ATR is NaN or 0
        if np.isnan(atr) or atr == 0:
            atr = entry * 0.01 

        sl = tp = None
        if signal == "LONG":
            sl = round(entry - 1.5 * atr, 2)
            tp = round(entry + 3.0 * atr, 2)
        elif signal == "SHORT":
            sl = round(entry + 1.5 * atr, 2)
            tp = round(entry - 3.0 * atr, 2)

        return {
            "signal": signal,
            "ai_confidence": ai_confidence,
            "confidence_long": confidence_long,
            "confidence_short": confidence_short,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "reason": f"AI Confidence: {ai_confidence}%"
        }
    except Exception as e:
        return {"signal": "NEUTRAL", "ai_confidence": 0, "reason": f"Prediction Error: {str(e)}"}

# ---------------------------  
# Paper Trader  
# ---------------------------  
class PaperTrader:
    def __init__(self, balance=100000):
        self.balance = balance
        self.positions = []

    def open(self, symbol, signal):
        if signal['signal'] == "NEUTRAL":
            return
        atr = signal.get('atr', 100)
        risk = self.balance * 0.02
        # Safely calculate quantity
        dist = 1.5 * atr
        if dist == 0: dist = signal['entry'] * 0.01
        
        qty = round(risk / dist, 6)
        
        pos = {
            "symbol": symbol,
            "side": signal['signal'],
            "entry": signal['entry'],
            "qty": qty,
            "sl": signal['sl'],
            "tp": signal['tp'],
            "confidence": signal['ai_confidence'],
            "time": datetime.now().strftime("%H:%M:%S")
        }
        self.positions.append(pos)
        # Simple balance deduction (margin logic simplified)
        self.balance -= (qty * signal['entry'] * 0.1) # Assuming 10x leverage for paper logic or just cost
        st.success(f"AI Paper {signal['signal']} opened: {qty} @ ${signal['entry']} (Conf: {signal['ai_confidence']}%)")

    def df(self):
        return pd.DataFrame(self.positions) if self.positions else pd.DataFrame(columns=["symbol","side","entry","qty","sl","tp","confidence","time"])

if 'trader' not in st.session_state:
    st.session_state.trader = PaperTrader()

# ---------------------------  
# UI Dashboard  
# ---------------------------  
st.set_page_config(layout="wide", page_title="ProfitForge AI")
st.title("🤖 ProfitForge AI — Intelligent Crypto Trading")

# Session Banner
session, color, note, current_time = get_trading_session()
gradient_map = {
    "orange": "#FF8E53, #FE6B8B",
    "blue": "#4FACFE, #00F2FE",
    "purple": "#667eea, #764ba2",
    "green": "#43E97B, #38F9D7",
    "gray": "#888888, #AAAAAA"
}
gradient = gradient_map.get(color, "#888888, #AAAAAA")

st.markdown(f"""
<div style='background: linear-gradient(90deg, {gradient}); color: white; padding: 15px; border-radius: 12px; text-align: center; font-size: 1.4rem; font-weight: bold; margin-bottom: 20px;'>
    {session} • {note} <br>
    <span style='font-size: 1rem;'>UTC: {current_time}</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("AI Controls")
    # Use key to prevent state reset issues
    symbol = st.selectbox("Symbol", SYMBOLS, key="sb_symbol")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2, key="sb_timeframe")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False) # Default off to prevent blocking UI by default
    
    st.divider()
    st.metric("Paper Balance", f"${st.session_state.trader.balance:,.2f}")
    if st.button("Reset Account"):
        st.session_state.trader = PaperTrader()
        st.rerun()

# Fetch data
df = fetch_klines(symbol, timeframe)

if df.empty:
    st.warning("No data received. Exchange might be busy or symbol invalid.")
else:
    # Generate AI signal
    ai_signal = generate_ai_signal(df)

    col1, col2 = st.columns([1.3, 2.7])

    with col1:
        st.subheader("AI-Powered Signal")
        sig = ai_signal.get('signal', 'NEUTRAL')
        sig_color = "green" if sig == "LONG" else "red" if sig == "SHORT" else "gray"
        
        st.markdown(f"""
        <div style='border: 2px solid {sig_color}; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px;'>
            <h1 style='color:{sig_color}; margin:0; font-size: 3em;'>{sig}</h1>
            <p style='margin:0; opacity: 0.8;'>{ai_signal.get('reason', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if sig != "NEUTRAL":
            c1, c2 = st.columns(2)
            c1.metric("Entry", f"${ai_signal['entry']}")
            c2.metric("Confidence", f"{ai_signal['ai_confidence']}%")
            
            c3, c4 = st.columns(2)
            c3.metric("Stop Loss", f"${ai_signal['sl']}", delta="-Risk", delta_color="inverse")
            c4.metric("Take Profit", f"${ai_signal['tp']}", delta="+Target")
            
            if st.button(f"Execute Paper {sig}", type="primary", use_container_width=True):
                st.session_state.trader.open(symbol, ai_signal)
                st.rerun()

    with col2:
        st.subheader(f"{symbol} {timeframe} - AI Analysis")
        
        # Safe Ichimoku Calculation for Plotting
        try:
            ichi = df.ta.ichimoku()
            # ichi returns (df_main, df_span) tuple
            if isinstance(ichi, tuple):
                ichi_df = ichi[0] 
                # Span A/B names vary by version, usually ISA_9 / ISB_26
                span_a = ichi_df[ichi_df.columns[0]] 
                span_b = ichi_df[ichi_df.columns[1]]
            else:
                span_a = span_b = None
        except:
            span_a = span_b = None

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
        
        # Bollinger Bands (using explicit Length 5 to match ML logic, or standard 20 for visual? Keeping 20 for visual usually better, but let's match ML)
        df_bb = df.ta.bbands(length=20) # Visualizing standard bands for context
        if df_bb is not None:
            # Columns usually BBU_20_2.0 etc
            bbu = df_bb[df_bb.columns[2]] # Upper
            bbl = df_bb[df_bb.columns[0]] # Lower
            fig.add_trace(go.Scatter(x=df_bb.index, y=bbu, line=dict(color="rgba(128,0,128,0.5)", dash="dot"), name="BB Upper"))
            fig.add_trace(go.Scatter(x=df_bb.index, y=bbl, line=dict(color="rgba(128,0,128,0.5)", dash="dot"), name="BB Lower"))
        
        # Ichimoku
        if span_a is not None:
            fig.add_trace(go.Scatter(x=span_a.index, y=span_a, line=dict(color="rgba(0, 255, 0, 0.5)", width=1), name="Senkou A"))
            fig.add_trace(go.Scatter(x=span_b.index, y=span_b, fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(color="rgba(255, 0, 0, 0.5)", width=1), name="Cloud"))

        # Signal Lines
        if ai_signal['signal'] != "NEUTRAL":
            fig.add_hline(y=ai_signal['entry'], line_dash="dash", line_color="cyan", annotation_text="ENTRY")
            fig.add_hline(y=ai_signal['sl'], line_dash="dot", line_color="red", annotation_text="SL")
            fig.add_hline(y=ai_signal['tp'], line_dash="dot", line_color="lime", annotation_text="TP")

        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📋 AI Paper Trading Log")
    positions_df = st.session_state.trader.df()
    if not positions_df.empty:
        st.dataframe(positions_df.sort_values("time", ascending=False), use_container_width=True)
    else:
        st.info("No trades executed yet.")

# Non-blocking Auto-refresh Logic
if auto_refresh:
    time.sleep(30)
    st.rerun()
