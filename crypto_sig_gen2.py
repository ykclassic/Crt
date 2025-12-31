import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ==============================
# PROFITFORGE - PRODUCTION READY
# ==============================

st.set_page_config(page_title="ProfitForge", page_icon="🔥", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 3.2rem; color: #00FF9F; text-align: center; font-weight: bold; text-shadow: 0 0 15px #00FF9F;}
    .live-price {font-size: 2.2rem; color: #00FF9F; text-align: center; margin: 20px 0; animation: glow 2s infinite;}
    @keyframes glow {0% {text-shadow: 0 0 10px #00FF9F;} 50% {text-shadow: 0 0 30px #00FF9F;} 100% {text-shadow: 0 0 10px #00FF9F;}}
    .signal-strong-buy {background: linear-gradient(135deg, rgba(0,255,159,0.2), rgba(0,100,50,0.3)); border-left: 8px solid #00FF00; padding: 20px; border-radius: 15px; box-shadow: 0 8px 25px rgba(0,255,0,0.3);}
    .signal-buy {background: rgba(0,255,159,0.1); border-left: 6px solid #00FF9F; padding: 20px; border-radius: 15px;}
    .signal-sell {background: rgba(255,90,90,0.1); border-left: 6px solid #FF5A5A; padding: 20px; border-radius: 15px;}
    .signal-strong-sell {background: linear-gradient(135deg, rgba(255,90,90,0.2), rgba(139,0,0,0.3)); border-left: 8px solid #FF0000; padding: 20px; border-radius: 15px; box-shadow: 0 8px 25px rgba(255,0,0,0.3);}
    .stButton>button {background: linear-gradient(45deg, #00FF9F, #0080FF); color: black; font-weight: bold; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔥 ProfitForge</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #A0FFC0; font-size: 1.4rem;'>Live Crypto Signals • Real-Time Analysis • Smart Trade Recommendations</div>", unsafe_allow_html=True)

# -----------------------------
# Technical Indicators
# -----------------------------
class TechnicalIndicators:
    @staticmethod
    def bollinger(df, window=20, std=2):
        rolling_mean = df['Close'].rolling(window).mean()
        rolling_std = df['Close'].rolling(window).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * std)
        df['BB_Lower'] = rolling_mean - (rolling_std * std)
        df['BB_Mid'] = rolling_mean
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        return df

    @staticmethod
    def macd(df, fast=12, slow=26, signal=9):
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df

    @staticmethod
    def rsi(df, window=14):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def ichimoku(df):
        high9 = df['High'].rolling(9).max()
        low9 = df['Low'].rolling(9).min()
        df['Conversion'] = (high9 + low9) / 2

        high26 = df['High'].rolling(26).max()
        low26 = df['Low'].rolling(26).min()
        df['Base'] = (high26 + low26) / 2

        df['SpanA'] = ((df['Conversion'] + df['Base']) / 2).shift(26)
        df['SpanB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
        return df

    @staticmethod
    def sma(df, periods=[20, 50, 200]):
        for p in periods:
            df[f'SMA_{p}'] = df['Close'].rolling(p).mean()
        return df

# -----------------------------
# Data Fetching
# -----------------------------
@st.cache_data(ttl=180)  # Cache OHLCV for 3 minutes
def fetch_ohlcv(exchange_id, symbol, timeframe='1h', limit=500):
    exchanges = {'xt': ccxt.xt, 'gateio': ccxt.gateio, 'bitget': ccxt.bitget, 'binance': ccxt.binance}
    try:
        exchange = exchanges[exchange_id]({'enableRateLimit': True, 'timeout': 30000})
        symbol_norm = symbol.replace('/USDT', '_USDT') if exchange_id == 'gateio' else symbol
        ohlcv = exchange.fetch_ohlcv(symbol_norm, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.warning(f"{exchange_id.upper()} failed: {e}")
        return pd.DataFrame()

def fetch_live_price(exchange_id, symbol):
    try:
        exchanges = {'xt': ccxt.xt, 'gateio': ccxt.gateio, 'bitget': ccxt.bitget, 'binance': ccxt.binance}
        exchange = exchanges[exchange_id]({'enableRateLimit': True})
        symbol_norm = symbol.replace('/USDT', '_USDT') if exchange_id == 'gateio' else symbol
        ticker = exchange.fetch_ticker(symbol_norm)
        return ticker['last']
    except:
        return None

# -----------------------------
# Signal Engine
# -----------------------------
def generate_signal(df, symbol, timeframe):
    if df.empty or len(df) < 100:
        return None

    ti = TechnicalIndicators()
    df = ti.bollinger(df)
    df = ti.macd(df)
    df = ti.rsi(df)
    df = ti.ichimoku(df)
    df = ti.sma(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last['Close']

    # Composite Score
    score = 50
    reasons = []

    # RSI
    if last['RSI'] < 30:
        score += 30
        reasons.append("RSI Oversold (+30)")
    elif last['RSI'] > 70:
        score -= 30
        reasons.append("RSI Overbought (-30)")

    # MACD
    if last['MACD'] > last['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        score += 25
        reasons.append("MACD Bullish Cross (+25)")
    elif last['MACD'] < last['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        score -= 25
        reasons.append("MACD Bearish Cross (-25)")

    # Ichimoku Cloud
    cloud_top = max(last.get('SpanA', price), last.get('SpanB', price))
    cloud_bottom = min(last.get('SpanA', price), last.get('SpanB', price))
    if price > cloud_top:
        score += 20
        reasons.append("Price Above Cloud (+20)")
    elif price < cloud_bottom:
        score -= 20
        reasons.append("Price Below Cloud (-20)")

    # SMA Trend
    if last['SMA_20'] > last['SMA_50'] > last['SMA_200']:
        score += 15
        reasons.append("Bullish SMA Alignment (+15)")
    elif last['SMA_20'] < last['SMA_50'] < last['SMA_200']:
        score -= 15
        reasons.append("Bearish SMA Alignment (-15)")

    score = max(0, min(100, score))

    # Determine Signal
    if score >= 80:
        signal = "STRONG BUY"
        card = "signal-strong-buy"
    elif score >= 65:
        signal = "BUY"
        card = "signal-buy"
    elif score <= 20:
        signal = "STRONG SELL"
        card = "signal-strong-sell"
    elif score <= 35:
        signal = "SELL"
        card = "signal-sell"
    else:
        signal = "HOLD"
        card = ""

    # Trading Style Recommendation
    style_map = {"1m": "Scalping", "5m": "Scalping", "15m": "Day Trading", "1h": "Day Trading", "4h": "Swing Trading", "1d": "Swing Trading"}
    base_style = style_map.get(timeframe, "Day Trading")
    if score >= 80 or score <= 20:
        base_style = "Swing Trading"  # Strong signals best held longer

    return {
        "signal": signal,
        "score": score,
        "reasons": reasons,
        "style": base_style,
        "price": price,
        "card": card,
        "entry": round(price * 1.001 if "BUY" in signal else price * 0.999, 6),
        "sl": round(price * 0.98 if "BUY" in signal else price * 1.02, 6),
        "tp1": round(price * 1.03 if "BUY" in signal else price * 0.97, 6),
        "tp2": round(price * 1.06 if "BUY" in signal else price * 0.94, 6)
    }

# -----------------------------
# Chart
# -----------------------------
def plot_chart(df, symbol):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=('Price & Indicators', 'Volume', 'RSI & MACD'))

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=symbol), row=1, col=1)

    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='#00FF9F', dash='dot'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], fill='tonexty', fillcolor='rgba(0,255,159,0.1)', name='BB Lower'), row=1, col=1)

    if 'SpanA' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(color='green'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanB'], fill='tonexty',
                                fillcolor='rgba(0,255,159,0.3)' if df['SpanA'].iloc[-1] > df['SpanB'].iloc[-1] else 'rgba(255,90,90,0.3)',
                                name='Ichimoku Cloud'), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(0,255,159,0.6)'), row=2, col=1)

    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#FF5A5A')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=900, title=f"{symbol} - Live Analysis", template='plotly_dark', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width='stretch')

# -----------------------------
# Sidebar & Controls
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/anvil.png", caption="ProfitForge")
    st.title("⚙️ Controls")

    exchange = st.selectbox("Exchange", ["xt", "gateio", "bitget", "binance"], index=0)
    symbol = st.selectbox("Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)

    if st.button("🔥 Forge Signal Now"):
        st.session_state.forced_refresh = True

# -----------------------------
# Main Logic
# -----------------------------
if st.button("Refresh") or 'forced_refresh' in st.session_state or auto_refresh:
    if 'forced_refresh' in st.session_state:
        del st.session_state.forced_refresh

    with st.spinner("Forging signal from live data..."):
        df = fetch_ohlcv(exchange, symbol, timeframe)
        if df.empty:
            for fallback in ['gateio', 'bitget', 'binance']:
                df = fetch_ohlcv(fallback, symbol, timeframe)
                if not df.empty:
                    st.info(f"Switched to {fallback.upper()}")
                    break

        if not df.empty:
            signal = generate_signal(df.copy(), symbol, timeframe)
            live_price = fetch_live_price(exchange, symbol)

            st.session_state.df = df
            st.session_state.signal = signal
            st.session_state.symbol = symbol
            st.session_state.live_price = live_price or df['Close'].iloc[-1]

    if auto_refresh:
        time.sleep(30)
        st.rerun()

# -----------------------------
# Display
# -----------------------------
if 'signal' in st.session_state:
    sig = st.session_state.signal
    live_price = st.session_state.get('live_price', sig['price'])

    st.markdown(f"<div class='live-price'>LIVE: ${live_price:,.6f}</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Signal Strength", f"{sig['score']}%")
    col2.metric("Recommendation", sig['signal'])
    col3.metric("Best Style", sig['style'])

    st.markdown(f"<div class='{sig['card']}'>"
                f"<h2 style='text-align: center;'>{sig['signal']} 🔥</h2>"
                f"<h3>Recommended Trading Style: <strong>{sig['style']}</strong></h3>"
                f"<p><strong>Entry:</strong> ${sig['entry']}<br>"
                f"<strong>Stop Loss:</strong> ${sig['sl']}<br>"
                f"<strong>Take Profit 1:</strong> ${sig['tp1']} | <strong>TP2:</strong> ${sig['tp2']}</p>"
                f"<p><strong>Reasons:</strong> {' • '.join(sig['reasons'])}</p>"
                f"</div>", unsafe_allow_html=True)

    plot_chart(st.session_state.df, st.session_state.symbol)
else:
    st.info("👈 Select settings and click **Forge Signal Now** or enable auto-refresh.")

st.caption("ProfitForge • Live Crypto Intelligence • December 2025")