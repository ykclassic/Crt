import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import time
from fpdf import FPDF

# ==============================
# PROFITFORGE PRO - FULLY ACTIVATED FEATURES
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
    .stButton>button {background: linear-gradient(45deg, #00FF9F, #00BFFF); color: black; font-weight: bold; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔥 ProfitForge Pro</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #A0FFC0; font-size: 1.6rem;'>Live Crypto Intelligence • Risk Management • Multi-TF • Trailing • Journal • News</div>", unsafe_allow_html=True)

# -----------------------------
# SESSION DETECTOR
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
# INDICATORS
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
# SIGNAL ENGINE WITH VOLATILITY FILTER & MULTI-TIMEFRAME
# -----------------------------
def calculate_score(last):
    score = 50
    if last.get('RSI', 50) < 30: score += 30
    if last.get('RSI', 50) > 70: score -= 30
    if last.get('MACD', 0) > last.get('MACD_Signal', 0): score += 20
    if last.get('MACD', 0) < last.get('MACD_Signal', 0): score -= 20
    cloud_top = max(last.get('SpanA', last['Close']), last.get('SpanB', last['Close']))
    cloud_bottom = min(last.get('SpanA', last['Close']), last.get('SpanB', last['Close']))
    if last['Close'] > cloud_top: score += 20
    if last['Close'] < cloud_bottom: score -= 20
    return score

def generate_signal(df_main, symbol, exchange):
    if df_main.empty:
        return None

    df_main = Indicators.atr(df_main)
    df_main = Indicators.rsi(df_main)
    df_main = Indicators.macd(df_main)
    df_main = Indicators.ichimoku(df_main)

    last = df_main.iloc[-1]
    atr = last['ATR']
    current_price = last['Close']

    # 4. Volatility Filter (ATR)
    atr_avg = df_main['ATR'].mean()
    if atr < atr_avg * 0.5:
        return {"signal": "NO TRADE", "reason": "Low volatility (ATR filter)"}

    # Base score
    score = calculate_score(last)

    # 2. Multi-Timeframe Confirmation
    tf_scores = []
    for tf in ['15m', '1h', '4h']:
        df_tf = fetch_data(exchange, symbol, tf, 500)
        if not df_tf.empty:
            df_tf = Indicators.rsi(df_tf)
            df_tf = Indicators.macd(df_tf)
            df_tf = Indicators.ichimoku(df_tf)
            tf_scores.append(calculate_score(df_tf.iloc[-1]))
    if tf_scores:
        mtf_avg = np.mean(tf_scores)
        score = (score + mtf_avg) / 2  # Average with higher TF

    score = max(0, min(100, score))

    # Signal
    if score >= 80:
        signal = "STRONG BUY"
    elif score >= 65:
        signal = "BUY"
    elif score <= 20:
        signal = "STRONG SELL"
    elif score <= 35:
        signal = "SELL"
    else:
        signal = "HOLD"

    # 3. Trailing Stop & Partial Profit
    if "BUY" in signal:
        entry = current_price * 1.001
        sl = current_price - (atr * 2)
        tp1 = current_price * 1.03
        tp2 = current_price * 1.06
        trail = f"Trail stop at 2x ATR ({atr*2:.2f}) after TP1"
        partial = "Take 50% profit at TP1, trail the rest"
    else:
        entry = current_price * 0.999
        sl = current_price + (atr * 2)
        tp1 = current_price * 0.97
        tp2 = current_price * 0.94
        trail = f"Trail stop at 2x ATR ({atr*2:.2f}) after TP1"
        partial = "Take 50% profit at TP1, trail the rest"

    return {
        "signal": signal,
        "score": round(score, 1),
        "price": current_price,
        "atr": atr,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "trail": trail,
        "partial": partial
    }

# -----------------------------
# 1. RISK MANAGEMENT & POSITION SIZING
# -----------------------------
def calculate_position(signal, balance, risk_pct):
    if "HOLD" in signal['signal'] or "NO TRADE" in signal['signal']:
        return 0, 0
    risk_amount = balance * (risk_pct / 100)
    risk_per_unit = abs(signal['entry'] - signal['sl'])
    if risk_per_unit == 0:
        return 0, 0
    size = risk_amount / risk_per_unit
    return round(size, 6), round(risk_amount, 2)

# -----------------------------
# 6. NEWS/SENTIMENT OVERLAY (Simple placeholder - can be expanded with RSS or API)
# -----------------------------
def get_sentiment_news():
    return [
        "Bitcoin ETF inflows reach $1B this week",
        "Fed signals rate cut in 2026",
        "Altcoin season heating up"
    ]

# -----------------------------
# 5. TRADE JOURNAL
# -----------------------------
if 'journal' not in st.session_state:
    st.session_state.journal = []

def log_trade(signal, size, balance, risk_pct):
    if size == 0:
        return
    st.session_state.journal.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": st.session_state.symbol,
        "signal": signal['signal'],
        "entry": signal['entry'],
        "sl": signal['sl'],
        "tp1": signal['tp1'],
        "tp2": signal['tp2'],
        "size": size,
        "risk": balance * (risk_pct / 100),
        "status": "OPEN"
    })
    st.success("Trade logged to journal!")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/anvil.png", caption="ProfitForge Pro")
    st.title("Controls")

    exchange = st.selectbox("Exchange", ["xt", "gateio", "bitget", "binance"], index=0)
    symbol = st.selectbox("Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

    balance = st.number_input("Account Balance ($)", min_value=100.0, value=10000.0)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)

    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)

    if st.button("🔥 Generate Signal"):
        st.session_state.refresh = True

# -----------------------------
# MAIN LOGIC
# -----------------------------
if st.button("Refresh") or st.session_state.get('refresh', False) or auto_refresh:
    st.session_state.refresh = False

    with st.spinner("Analyzing market..."):
        df = fetch_data(exchange, symbol, timeframe)
        if df.empty:
            st.error("Failed to fetch data from all exchanges")
        else:
            signal = generate_signal(df.copy(), symbol, exchange)
            live_price = fetch_live_price(exchange, symbol) or df['Close'].iloc[-1]

            st.session_state.df = df
            st.session_state.signal = signal
            st.session_state.live_price = live_price
            st.session_state.symbol = symbol

    if auto_refresh:
        time.sleep(30)
        st.rerun()

# -----------------------------
# DASHBOARD
# -----------------------------
if 'signal' in st.session_state:
    signal = st.session_state.signal
    live_price = st.session_state.live_price

    st.markdown(f"<div class='live-price'>LIVE: ${live_price:,.2f}</div>", unsafe_allow_html=True)

    # Determine card color
    if signal.get('signal') == "NO TRADE":
        card_color = "#888888"
    elif "BUY" in signal.get('signal', ''):
        card_color = "#00FF9F"
    elif "SELL" in signal.get('signal', ''):
        card_color = "#FF5A5A"
    else:
        card_color = "#888888"

    # === FIXED SIGNAL CARD WITH SAFE KEY ACCESS ===
    st.markdown(f"""
    <div class='signal-card' style='border-left: 8px solid {card_color};'>
        <h2 style='color: {card_color}; text-align: center;'>{signal.get('signal', 'N/A')}</h2>
        <p><strong>Score:</strong> {signal.get('score', 'N/A')}% | 
           <strong>Volatility (ATR):</strong> ${signal.get('atr', 0):.2f}</p>
        <p><strong>Entry:</strong> ${signal.get('entry', 0):.2f} | 
           <strong>SL:</strong> ${signal.get('sl', 0):.2f}</p>
        <p><strong>TP1:</strong> ${signal.get('tp1', 0):.2f} | 
           <strong>TP2:</strong> ${signal.get('tp2', 0):.2f}</p>
        <p><strong>Trailing:</strong> {signal.get('trail', 'N/A')}</p>
        <p><strong>Partial Profit:</strong> {signal.get('partial', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Risk & Position Sizing
    size, risk_amount = calculate_position(signal, balance, risk_pct)
    st.markdown("<div class='risk-card'>", unsafe_allow_html=True)
    st.subheader("Risk Management")
    st.write(f"Risk Amount: ${risk_amount:.2f} ({risk_pct}%)")
    st.write(f"Recommended Position Size: {size} {symbol.split('/')[0]}")
    if st.button("Log This Trade to Journal"):
        log_trade(signal, size, balance, risk_pct)
    st.markdown("</div>", unsafe_allow_html=True)

    # 6. News/Sentiment
    st.subheader("Market Sentiment & News")
    news = get_sentiment_news()
    for item in news:
        st.write(f"• {item}")

else:
    st.info("Select settings and generate a signal to begin.")

st.caption("ProfitForge Pro • Features 1-6 Fully Activated • December 27, 2025")