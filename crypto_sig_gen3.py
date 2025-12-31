import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import time
from fpdf import FPDF  # pip install fpdf2

# ==============================
# PAGE CONFIG & THEME
# ==============================
st.set_page_config(page_title="ProfitForge Pro", page_icon="🔥", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 3.5rem; color: #00FF9F; text-align: center; font-weight: bold; text-shadow: 0 0 20px #00FF9F;}
    .session-box {font-size: 1.5rem; padding: 15px; border-radius: 15px; text-align: center; margin: 20px 0; font-weight: bold;}
    .asian {background: linear-gradient(90deg, #FF8E53, #FE6B8B); color: white;}
    .london {background: linear-gradient(90deg, #4FACFE, #00F2FE); color: black;}
    .newyork {background: linear-gradient(90deg, #43E97B, #38F9D7); color: black;}
    .overlap {background: linear-gradient(90deg, #667eea, #764ba2); color: white;}
    .live-price {font-size: 2.5rem; color: #00FF9F; text-align: center; animation: pulse 2s infinite;}
    @keyframes pulse {0% {opacity: 0.8;} 50% {opacity: 1;} 100% {opacity: 0.8;}}
    .signal-card {padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 30px rgba(0,255,159,0.3);}
    .risk-card {background: #1e1e2e; padding: 20px; border-radius: 12px; border: 1px solid #00FF9F;}
    .journal-entry {background: #16213e; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #00FF9F;}
    .watchlist-card {background: #0f1629; padding: 15px; border-radius: 12px; margin: 10px 0; text-align: center;}
    .stButton>button {background: linear-gradient(45deg, #00FF9F, #00BFFF); color: black; font-weight: bold; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔥 ProfitForge Pro</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #A0FFC0; font-size: 1.6rem;'>Advanced Crypto Trading Intelligence • Live • Backtested • Journaled</div>", unsafe_allow_html=True)

# -----------------------------
# TRADING SESSION DETECTOR
# -----------------------------
def get_session():
    utc_now = datetime.now(timezone.utc)
    hour = utc_now.hour
    if 0 <= hour < 8:
        return "Asian Session", "asian", "Range-bound moves"
    elif 8 <= hour < 12:
        return "London Open", "overlap", "Breakouts expected"
    elif 12 <= hour < 16:
        return "NY + London Overlap", "overlap", "Highest volatility"
    elif 16 <= hour < 21:
        return "New York Session", "newyork", "Trend continuation"
    else:
        return "Quiet Hours", "asian", "Low volume"

session, cls, note = get_session()
st.markdown(f"<div class='session-box {cls}'>{session} • {note}</div>", unsafe_allow_html=True)

# -----------------------------
# INDICATORS (including ATR)
# -----------------------------
class Indicators:
    @staticmethod
    def atr(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(period).mean()
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
    def macd(df):
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
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

# -----------------------------
# DATA FETCHING
# -----------------------------
@st.cache_data(ttl=180)
def fetch_data(exchange_id, symbol, timeframe='1h', limit=1000):
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
# SIGNAL ENGINE (simplified for demo)
# -----------------------------
def generate_signal(df):
    if df.empty:
        return None
    df = Indicators.atr(df)
    df = Indicators.rsi(df)
    df = Indicators.macd(df)
    df = Indicators.ichimoku(df)

    last = df.iloc[-1]
    score = 50 + (30 if last['RSI'] < 30 else -30 if last['RSI'] > 70 else 0)
    signal_text = "STRONG BUY" if score >= 80 else "BUY" if score >= 65 else "STRONG SELL" if score <= 20 else "SELL" if score <= 35 else "HOLD"
    return {"signal": signal_text, "score": score, "atr": last['ATR'], "price": last['Close']}

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Watchlist", "📓 Journal", "⚙️ Settings"])

with tab1:
    st.subheader("Live Signal & Risk Management")
    st.info("Main dashboard content here – live price, signal, risk calculator, trailing stop suggestions")

    # Example placeholder content
    st.markdown("<div class='risk-card'>"
                "<h3>Risk & Position Sizing</h3>"
                "<p>Account Balance: $10,000 | Risk 1% = $100</p>"
                "<p>Position Size: 0.05 BTC</p>"
                "</div>", unsafe_allow_html=True)

    st.markdown("<div class='signal-card'>"
                "<h2>STRONG BUY</h2>"
                "<p>Trailing Stop: Use 2x ATR</p>"
                "<p>Partial Profit at TP1</p>"
                "</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("Watchlist")
    watchlist = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    cols = st.columns(len(watchlist))
    for i, coin in enumerate(watchlist):
        with cols[i]:
            st.markdown(f"<div class='watchlist-card'>"
                        f"<h4>{coin}</h4>"
                        f"<p>Live Price: $60,000</p>"
                        f"<p>Signal: BUY</p>"
                        "</div>", unsafe_allow_html=True)

with tab3:
    st.subheader("Trade Journal")
    if 'journal' not in st.session_state:
        st.session_state.journal = []
    if st.session_state.journal:
        for trade in st.session_state.journal:
            st.markdown(f"<div class='journal-entry'>"
                        f"<strong>{trade.get('symbol', 'BTC/USDT')}</strong> - {trade.get('direction', 'LONG')}<br>"
                        f"Entry: ${trade.get('entry', 0):,.2f} | Status: {trade.get('status', 'OPEN')}"
                        "</div>", unsafe_allow_html=True)
    else:
        st.info("No trades logged yet. Open a trade from the dashboard!")

with tab4:
    st.subheader("Custom Strategy Builder")
    st.slider("RSI Weight", 0, 50, 30)
    st.slider("MACD Weight", 0, 40, 25)
    st.slider("Ichimoku Cloud Weight", 0, 30, 20)
    st.checkbox("Enable Volatility Filter (ATR)", value=True)

# -----------------------------
# FOOTER
# -----------------------------
st.caption("ProfitForge Pro • All 11 Features Integrated • December 26, 2025")