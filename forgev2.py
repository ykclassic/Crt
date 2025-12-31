import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from datetime import datetime
import time

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="ProfitForge Live", layout="wide")

st.title("🔥 ProfitForge — Live Price Edition")

# =========================
# USER SELECTIONS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    exchange_id = st.selectbox("Exchange", ["binance", "bitget", "gateio"])

with col2:
    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

with col3:
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])

refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 1)

# =========================
# FUNCTIONS
# =========================
@st.cache_data(ttl=60)
def fetch_data(exchange_id, symbol, timeframe):
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
    df = pd.DataFrame(
        data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def fetch_live_price(exchange_id, symbol):
    try:
        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except:
        return None

# =========================
# LIVE FEED
# =========================
live_placeholder = st.empty()

while True:
    live_price = fetch_live_price(exchange_id, symbol)
    df = fetch_data(exchange_id, symbol, timeframe)

    # Display live price
    if live_price is not None:
        live_placeholder.metric(
            label=f"Live {symbol} Price on {exchange_id.upper()}",
            value=f"${live_price:,.2f}",
            delta=f"{live_price - df['Close'].iloc[-2]:.2f}" if len(df) > 1 else 0
        )
    else:
        live_placeholder.error("Failed to fetch live price.")

    # Simple candlestick chart with last 100 candles
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

    time.sleep(refresh_interval)
