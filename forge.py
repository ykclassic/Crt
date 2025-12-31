import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="ProfitForge Lite", layout="wide")

# =========================
# INDICATORS (SAFE)
# =========================

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(df):
    fast = df['Close'].ewm(span=12, adjust=False).mean()
    slow = df['Close'].ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# =========================
# DATA FETCH
# =========================
@st.cache_data(ttl=120)
def fetch_data(exchange_id, symbol, timeframe):
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)

    df = pd.DataFrame(
        data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


# =========================
# SIGNAL ENGINE
# =========================
def generate_signal(df):
    df = df.copy()
    df["ATR"] = atr(df)
    df["RSI"] = rsi(df)
    df["MACD"], df["MACD_SIGNAL"] = macd(df)

    last = df.iloc[-1]

    rsi_score = 1 if last.RSI < 30 else 0 if last.RSI > 70 else 0.5
    macd_score = 1 if last.MACD > last.MACD_SIGNAL else 0

    score = (rsi_score * 0.5 + macd_score * 0.5) * 100

    if score >= 70:
        signal = "BUY"
    elif score <= 30:
        signal = "SELL"
    else:
        signal = "HOLD"

    atr_val = last.ATR
    price = last.Close

    return {
        "signal": signal,
        "score": round(score, 1),
        "price": price,
        "atr": atr_val,
        "entry": price,
        "sl": price - atr_val * 2 if signal == "BUY" else price + atr_val * 2,
        "tp": price + atr_val * 3 if signal == "BUY" else price - atr_val * 3,
    }


# =========================
# UI
# =========================
st.title("🔥 ProfitForge — Simple Edition")

col1, col2, col3 = st.columns(3)

with col1:
    exchange = st.selectbox("Exchange", ["binance", "bitget", "gateio"])

with col2:
    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

with col3:
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])

if st.button("Generate Signal"):
    df = fetch_data(exchange, symbol, timeframe)

    if df.empty:
        st.error("No data received.")
    else:
        signal = generate_signal(df)

        st.metric("Signal", signal["signal"])
        st.metric("Score", f"{signal['score']}%")
        st.metric("Price", f"${signal['price']:.2f}")

        st.write(f"**Entry:** {signal['entry']:.2f}")
        st.write(f"**Stop Loss:** {signal['sl']:.2f}")
        st.write(f"**Take Profit:** {signal['tp']:.2f}")

        # Chart
        fig = go.Figure()
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        )

        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

st.caption("Educational use only — no financial advice.")
