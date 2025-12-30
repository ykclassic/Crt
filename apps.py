# app.py - ProfitForge — AI-Powered Crypto Trading App (Integrated ML Signals)
# Requirements: pip install streamlit ccxt pandas numpy plotly scikit-learn xgboost pandas_ta

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timezone
import time
import joblib
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
def fetch_klines(symbol, timeframe, limit=1000):
    exchange = ccxt.xt({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) == 0:
            raise Exception("Empty response from XT.com")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    except Exception as e:
        st.error("Unable to connect to XT.com exchange")
        st.info(f"Error details: {str(e)}")
        st.stop()

# ---------------------------  
# Feature Engineering with pandas_ta
# ---------------------------  
def add_features(df):
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.ema(length=50, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.stoch(append=True)
    return df.dropna()

# ---------------------------  
# AI Model (Random Forest Classifier)
# ---------------------------  
@st.cache_resource
def load_or_train_model(_df):
    features = ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
                'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'EMA_50', 'ATRr_14', 'OBV', 'STOCHk_14_3_3']
    
    # Target: price up >1.5% in next 6 candles (adjustable)
    _df['future_return'] = _df['close'].pct_change(6).shift(-6)
    _df['target'] = (_df['future_return'] > 0.015).astype(int)
    
    X = _df[features]
    y = _df['target'].dropna()
    X = X.loc[y.index]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_scaled, y)
    
    return model, scaler, features

# ---------------------------  
# Signal Engine with AI Integration
# ---------------------------  
def generate_ai_signal(df):
    if len(df) < 100:
        return {"signal": "NEUTRAL", "ai_confidence": 0, "reason": "Insufficient data"}

    df_feat = add_features(df)
    
    model, scaler, features = load_or_train_model(df_feat)
    
    latest = df_feat.iloc[-1:]
    X_latest = latest[features]
    X_scaled = scaler.transform(X_latest)
    
    prob = model.predict_proba(X_scaled)[0]
    confidence_long = round(prob[1] * 100, 1)
    confidence_short = round(prob[0] * 100, 1)
    
    # Combine with rule-based confluence
    rule_signal = "NEUTRAL"
    if confidence_long > 70:
        rule_signal = "LONG"
    elif confidence_short > 70:
        rule_signal = "SHORT"
    
    # Final AI-powered signal
    signal = rule_signal
    ai_confidence = max(confidence_long, confidence_short)
    
    entry = round(latest['close'].iloc[-1], 2)
    atr = round(latest['ATRr_14'].iloc[-1], 2)
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
        "reason": f"AI Confidence: {ai_confidence}% {'(Bullish)' if signal=='LONG' else '(Bearish)' if signal=='SHORT' else ''}"
    }

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
        qty = round(risk / (1.5 * atr), 6) if atr > 0 else 0.001
        pos = {
            "symbol": symbol,
            "side": signal['signal'],
            "entry": signal['entry'],
            "qty": qty,
            "sl": signal['sl'],
            "tp": signal['tp'],
            "confidence": signal['ai_confidence'],
            "time": datetime.utcnow().strftime("%H:%M")
        }
        self.positions.append(pos)
        self.balance -= qty * signal['entry']
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
gradient = gradient_map[color]
st.markdown(f"""
<div style='background: linear-gradient(90deg, {gradient}); color: white; padding: 15px; border-radius: 12px; text-align: center; font-size: 1.4rem; font-weight: bold; margin-bottom: 20px;'>
    {session} • {note} <br>
    <span style='font-size: 1rem;'>UTC: {current_time}</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("AI Controls")
    symbol = st.selectbox("Symbol", SYMBOLS)
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    st.metric("Paper Balance", f"${st.session_state.trader.balance:,.2f}")

# Fetch data
df = fetch_klines(symbol, timeframe)

# Generate AI signal
ai_signal = generate_ai_signal(df)

col1, col2 = st.columns([1.3, 2.7])

with col1:
    st.subheader("AI-Powered Signal")
    sig_color = "green" if ai_signal['signal'] == "LONG" else "red" if ai_signal['signal'] == "SHORT" else "gray"
    st.markdown(f"<h1 style='color:{sig_color}; text-align:center'>{ai_signal['signal']}</h1>", unsafe_allow_html=True)
    
    st.markdown("### Trade Setup")
    st.write(f"**Entry:** ${ai_signal['entry']}")
    st.write(f"**SL:** ${ai_signal['sl']} | **TP:** ${ai_signal['tp']}")
    st.write(f"**AI Confidence:** {ai_signal['ai_confidence']}%")
    st.caption(ai_signal['reason'])

    st.write(f"Long Prob: {ai_signal['confidence_long']}% | Short Prob: {ai_signal['confidence_short']}%")

    if ai_signal['signal'] != "NEUTRAL":
        if st.button(f"Execute AI Paper {ai_signal['signal']}", type="primary", use_container_width=True):
            st.session_state.trader.open(symbol, ai_signal)

with col2:
    st.subheader(f"{symbol} {timeframe} - AI Analysis")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
    
    # Bollinger Bands
    df_bb = df.ta.bbands()
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb['BBU_5_2.0'], line=dict(color="purple", dash="dot"), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb['BBL_5_2.0'], line=dict(color="purple", dash="dot"), name="BB Lower"))
    
    # Ichimoku (simplified)
    ichi = df.ta.ichimoku()
    if ichi is not None and len(ichi) > 1:
        isa = ichi[0]['ISA_9']
        isb = ichi[0]['ISB_26']
        fig.add_trace(go.Scatter(x=isa.index, y=isa, line=dict(color="green"), name="Senkou A"))
        fig.add_trace(go.Scatter(x=isb.index, y=isb, fill='tonexty', fillcolor='rgba(255,0,0,0.15)', line=dict(color="red"), name="Cloud"))

    fig.add_hline(y=ai_signal['entry'], line_dash="dash", line_color="cyan", annotation_text=f"Entry (${ai_signal['entry']})")
    if ai_signal['sl']:
        fig.add_hline(y=ai_signal['sl'], line_dash="dot", line_color="red", annotation_text="SL")
        fig.add_hline(y=ai_signal['tp'], line_dash="dot", line_color="lime", annotation_text="TP")

    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("AI Paper Trading Positions")
positions_df = st.session_state.trader.df()
if not positions_df.empty:
    st.dataframe(positions_df.sort_values("time", ascending=False))
else:
    st.info("No active AI paper trades")

if auto_refresh:
    time.sleep(30)
    st.rerun()

st.caption("ProfitForge AI • Machine Learning Signals • Live XT.com Data • December 28, 2025")