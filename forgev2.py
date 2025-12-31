import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="ProfitForge Safe Live XT", layout="wide")
st.title("🔥 ProfitForge — Live Price & Session (Safe with XT)")

# =========================
# TRADING SESSION FUNCTION
# =========================
def get_session():
    utc_now = datetime.now(timezone.utc)
    hour = utc_now.hour
    if 0 <= hour < 8:
        return "Asian Session", "Range-bound moves", "#FF8E53"
    elif 8 <= hour < 12:
        return "London Open", "Breakouts expected", "#667eea"
    elif 12 <= hour < 16:
        return "NY + London Overlap", "Highest volatility", "#764ba2"
    elif 16 <= hour < 21:
        return "New York Session", "Trend continuation", "#43E97B"
    else:
        return "Quiet Hours", "Low volume", "#888888"

# =========================
# USER SELECTIONS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    exchange_id = st.selectbox("Exchange", ["binance", "bitget", "gateio", "xt"])

with col2:
    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

with col3:
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])

refresh_interval = st.slider("Refresh interval (seconds)", 2, 10, 3)

# =========================
# CCXT SAFE FETCH FUNCTIONS
# =========================
def get_exchange(exchange_id):
    try:
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class({"enableRateLimit": True, "timeout": 30000})
    except Exception as e:
        st.warning(f"Exchange init failed: {str(e)}")
        return None

def fetch_ohlcv_safe(exchange_id, symbol, timeframe, retries=3, delay=2):
    exchange = get_exchange(exchange_id)
    if not exchange:
        return pd.DataFrame()
    try:
        exchange.load_markets()
        if symbol not in exchange.symbols:
            st.error(f"Symbol {symbol} not available on {exchange_id.upper()}")
            return pd.DataFrame()
    except ccxt.BaseError as e:
        st.warning(f"Load markets failed: {str(e)}")
        return pd.DataFrame()
    
    for i in range(retries):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
            df = pd.DataFrame(data, columns=["timestamp","Open","High","Low","Close","Volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except ccxt.BaseError as e:
            st.warning(f"Attempt {i+1}/{retries} failed: {str(e)}")
    return pd.DataFrame()

def fetch_live_price_safe(exchange_id, symbol):
    exchange = get_exchange(exchange_id)
    if not exchange:
        return None
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except ccxt.BaseError:
        return None

# =========================
# AUTORELOAD (non-blocking)
# =========================
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

# =========================
# FETCH DATA WITH FALLBACK
# =========================
exchanges = [exchange_id] + [ex for ex in ["binance","bitget","gateio","xt"] if ex != exchange_id]
df = pd.DataFrame()
live_price = None

for ex in exchanges:
    df = fetch_ohlcv_safe(ex, symbol, timeframe)
    live_price = fetch_live_price_safe(ex, symbol)
    if not df.empty and live_price is not None:
        exchange_id = ex  # use working exchange
        break

if df.empty or live_price is None:
    st.warning("Failed to fetch data from all exchanges. Try again later.")
    st.stop()

# =========================
# DISPLAY SESSION
# =========================
session_name, note, color = get_session()
st.markdown(f"<h3 style='text-align:center; color:{color};'>💹 {session_name} — {note}</h3>", unsafe_allow_html=True)

# =========================
# DISPLAY LIVE PRICE
# =========================
prev_close = df['Close'].iloc[-2] if len(df) > 1 else live_price
st.metric(label=f"Live {symbol} Price on {exchange_id.upper()}", value=f"${live_price:,.2f}", delta=f"{live_price - prev_close:.2f}")

# =========================
# DISPLAY CANDLESTICK CHART
# =========================
fig = go.Figure()
fig.add_candlestick(
    x=df.index[-100:],
    open=df["Open"][-100:],
    high=df["High"][-100:],
    low=df["Low"][-100:],
    close=df["Close"][-100:],
    name="Price"
)
fig.update_layout(height=500, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
