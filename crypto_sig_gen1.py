import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==============================
# PROFITFORGE - CRYPTO SIGNAL GENERATOR
# ==============================

st.set_page_config(page_title="ProfitForge", page_icon="🔥", layout="wide")

# Enhanced UI with Forge Theme (Dark, Fiery Green Accents)
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #00FF9F; text-align: center; font-weight: bold; text-shadow: 0 0 10px #00FF9F;}
    .sub-header {font-size: 1.5rem; color: #A0FFC0; text-align: center; margin-bottom: 2rem;}
    .signal-buy {background: linear-gradient(90deg, rgba(0,255,159,0.1), rgba(0,0,0,0)); padding: 20px; border-radius: 15px; border-left: 6px solid #00FF9F; margin: 15px 0; box-shadow: 0 4px 15px rgba(0,255,159,0.2);}
    .signal-sell {background: linear-gradient(90deg, rgba(255,90,90,0.1), rgba(0,0,0,0)); padding: 20px; border-radius: 15px; border-left: 6px solid #FF5A5A; margin: 15px 0; box-shadow: 0 4px 15px rgba(255,90,90,0.2);}
    .stButton>button {background: linear-gradient(45deg, #00FF9F, #00B35B); color: black; font-weight: bold; border-radius: 10px; height: 3em;}
    .sidebar .sidebar-content {background-color: #111111;}
</style>
""", unsafe_allow_html=True)

# Logo & Header
st.markdown("<h1 class='main-header'>🔥 ProfitForge</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Forge Your Profits – Advanced Crypto Signal Generator</div>", unsafe_allow_html=True)

# Technical Indicators
class TechnicalIndicators:
    @staticmethod
    def bollinger(df, window=20, std=2):
        rolling_mean = df['Close'].rolling(window).mean()
        rolling_std = df['Close'].rolling(window).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * std)
        df['BB_Middle'] = rolling_mean
        df['BB_Lower'] = rolling_mean - (rolling_std * std)
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

# Data Fetcher with Fallback
def fetch_data(exchange_id, symbol, timeframe='1h', limit=500):
    exchanges = {'xt': ccxt.xt, 'gateio': ccxt.gateio, 'bitget': ccxt.bitget, 'binance': ccxt.binance}
    
    if exchange_id not in exchanges:
        st.error("Unsupported exchange")
        return pd.DataFrame()
    
    try:
        exchange_class = exchanges[exchange_id]
        exchange = exchange_class({'timeout': 30000, 'enableRateLimit': True})
        symbol_norm = symbol.replace('/USDT', '_USDT') if exchange_id == 'gateio' else symbol
        ohlcv = exchange.fetch_ohlcv(symbol_norm, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        st.success(f"🔗 Data from {exchange_id.upper()}")
        return df
    except Exception as e:
        st.warning(f"{exchange_id.upper()} failed: {e}")
        return pd.DataFrame()

# Signal Generation
def generate_signals(df, symbol, timeframe):
    if df.empty: return []
    ti = TechnicalIndicators()
    df = ti.bollinger(df)
    df = ti.macd(df)
    df = ti.rsi(df)
    df = ti.ichimoku(df)
    df = ti.sma(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last['Close']

    signals = []
    cloud_top = max(last.get('SpanA', price), last.get('SpanB', price))
    cloud_bottom = min(last.get('SpanA', price), last.get('SpanB', price))
    cloud = "BULLISH 🔥" if price > cloud_top else "BEARISH ❄️" if price < cloud_bottom else "NEUTRAL ⚡"

    if last['RSI'] < 30:
        signals.append({"type": "LONG", "name": "RSI Oversold", "confidence": 0.85, "desc": f"RSI {last['RSI']:.1f} – Strong Buy Zone"})
    if last['RSI'] > 70:
        signals.append({"type": "SHORT", "name": "RSI Overbought", "confidence": 0.85, "desc": f"RSI {last['RSI']:.1f} – Strong Sell Zone"})
    if prev['MACD'] <= prev['MACD_Signal'] and last['MACD'] > last['MACD_Signal']:
        signals.append({"type": "LONG", "name": "MACD Bullish Cross", "confidence": 0.80, "desc": "Momentum Shift Upward"})

    for s in signals:
        s['symbol'] = symbol
        s['timeframe'] = timeframe
        s['price'] = price
        s['cloud'] = cloud
        if s['type'] == 'LONG':
            s['entry'] = round(price * 1.001, 6)
            s['sl'] = round(price * 0.98, 6)
            s['tp'] = [round(price * 1.03, 6), round(price * 1.05, 6)]
            s['rr'] = round((s['tp'][-1] - s['entry']) / (s['entry'] - s['sl']), 2)
        else:
            s['entry'] = round(price * 0.999, 6)
            s['sl'] = round(price * 1.02, 6)
            s['tp'] = [round(price * 0.97, 6), round(price * 0.95, 6)]
            s['rr'] = round((s['entry'] - s['tp'][-1]) / (s['sl'] - s['entry']), 2)

    return signals

# Updated Chart (No Deprecation Warning)
def plot_chart(df, symbol):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=('Candles & Indicators', 'Volume', 'RSI'))
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=symbol), row=1, col=1)
    
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='#00FF9F', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', fill='tonexty', fillcolor='rgba(0,255,159,0.1)'), row=1, col=1)
    
    if 'SpanA' in df.columns and 'SpanB' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(color='green'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanB'], fill='tonexty', 
                                fillcolor='rgba(0,255,159,0.3)' if df['SpanA'].iloc[-1] > df['SpanB'].iloc[-1] else 'rgba(255,90,90,0.3)',
                                name='Ichimoku Cloud'), row=1, col=1)
    
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(0,255,159,0.6)'), row=2, col=1)
    
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#FF5A5A')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5A5A", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00FF9F", row=3, col=1)
    
    fig.update_layout(height=900, title=f"{symbol} - Forged Analysis", template='plotly_dark', xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, width='stretch')

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/hammer.png", caption="ProfitForge")  # Simple hammer icon as logo
    st.title("⚙️ Forge Settings")
    
    exchange = st.selectbox("Exchange", ["xt", "gateio", "bitget", "binance"], index=0)
    symbol = st.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    
    if st.button("🔥 Forge Signals Now"):
        df = fetch_data(exchange, symbol, timeframe)
        if df.empty:
            for fallback in ['gateio', 'bitget', 'binance'] if exchange != 'binance' else []:
                df = fetch_data(fallback, symbol, timeframe)
                if not df.empty: break
        
        if not df.empty:
            signals = generate_signals(df.copy(), symbol, timeframe)
            st.session_state.df = df
            st.session_state.signals = signals
            st.session_state.symbol = symbol
        else:
            st.error("All exchanges failed – Check internet/VPN")

# Main Display
if 'df' in st.session_state:
    df = st.session_state.df
    signals = st.session_state.signals or []

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🚀 Signals", "📈 Chart"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
        col2.metric("24h Change", f"{((df['Close'].iloc[-1]/df['Close'].iloc[-24])-1)*100:+.2f}%" if len(df)>24 else "N/A")
        col3.metric("Forged Signals", len(signals))
        st.markdown("### 🔥 Market Overview Ready")

    with tab2:
        st.markdown("### 🔥 Active Signals")
        for s in signals:
            card_class = "signal-buy" if s['type'] == "LONG" else "signal-sell"
            direction = "🟢 LONG" if s['type'] == "LONG" else "🔴 SHORT"
            st.markdown(f"<div class='{card_class}'>"
                        f"<h3>{direction} – {s['name']}</h3>"
                        f"<p><b>Confidence:</b> {s['confidence']:.0%} | <b>Cloud:</b> {s['cloud']} | <b>R/R Ratio:</b> {s.get('rr', 'N/A')}</p>"
                        f"<p><b>Entry:</b> ${s['entry']} | <b>Stop Loss:</b> ${s['sl']} | <b>TPs:</b> ${s['tp'][0]} → ${s['tp'][1]}</p>"
                        f"<p>{s['desc']}</p></div>", unsafe_allow_html=True)

    with tab3:
        plot_chart(df, st.session_state.symbol)
else:
    st.info("👈 Select settings and click **Forge Signals Now** to begin.")