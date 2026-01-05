# profitforge_v7_async.py
import streamlit as st
import ccxt.async_support as ccxt  # async version
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import asyncio

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="ProfitForge Pro v7 Async", layout="wide")
st.title("🔥 ProfitForge Pro v7 - Async Multi-TF Live Signals + Backtest + Alerts")

st.cache_data.clear()

TRADING_PAIRS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT",
    "ADA/USDT","LTC/USDT","DOGE/USDT","MATIC/USDT","AVAX/USDT"
]
TIMEFRAMES = ["5m","15m","30m","1h","4h","1d"]

# -----------------------------
# SESSION DETECTOR
# -----------------------------
def get_session():
    utc_now = datetime.now(timezone.utc) + timedelta(hours=1)
    hour = utc_now.hour
    if 0 <= hour < 8: return "Asian Session", "Range-bound moves"
    elif 8 <= hour < 12: return "London Open", "Breakouts expected"
    elif 12 <= hour < 16: return "NY + London Overlap", "Highest volatility"
    elif 16 <= hour < 21: return "New York Session", "Trend continuation"
    else: return "Quiet Hours", "Low volume"

session_name, session_note = get_session()

# -----------------------------
# INDICATORS
# -----------------------------
class Indicators:
    @staticmethod
    def atr(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(period).mean()
        return df

    @staticmethod
    def rsi(df, window=14):
        delta = df['Close'].diff()
        gain = delta.where(delta>0,0).rolling(window).mean()
        loss = -delta.where(delta<0,0).rolling(window).mean()
        rs = gain/loss
        df['RSI'] = 100-(100/(1+rs))
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
        df['Conversion'] = (high9+low9)/2
        high26 = df['High'].rolling(26).max()
        low26 = df['Low'].rolling(26).min()
        df['Base'] = (high26+low26)/2
        df['SpanA'] = ((df['Conversion']+df['Base'])/2).shift(26)
        df['SpanB'] = ((df['High'].rolling(52).max()+df['Low'].rolling(52).min())/2).shift(26)
        return df

# -----------------------------
# ASYNC DATA FETCHING
# -----------------------------
async def fetch_data_async(exchange_id, symbol, timeframe="1h", limit=500):
    exchanges = {
        "binance": ccxt.binance(),
        "bitget": ccxt.bitget(),
        "gateio": ccxt.gateio(),
        "xt": ccxt.xt()
    }
    exchange = exchanges.get(exchange_id)
    if not exchange:
        return pd.DataFrame()

    try:
        markets = await exchange.load_markets()
        if symbol not in markets:
            st.warning(f"{symbol} not available on {exchange_id.upper()}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"{exchange_id.upper()} load_markets error: {e}")
        return pd.DataFrame()

    for attempt in range(3):
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            st.warning(f"{exchange_id.upper()} fetch error ({attempt+1}/3): {e}")
            await asyncio.sleep(2)
        except Exception as e:
            st.warning(f"{exchange_id.upper()} unexpected fetch error: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# -----------------------------
# SIGNAL ENGINE
# -----------------------------
def calculate_score(last):
    score = 50
    if last.get("RSI",50)<30: score+=30
    if last.get("RSI",50)>70: score-=30
    if last.get("MACD",0) > last.get("MACD_Signal",0): score+=20
    if last.get("MACD",0) < last.get("MACD_Signal",0): score-=20
    cloud_top = max(last.get("SpanA", last["Close"]), last.get("SpanB", last["Close"]))
    cloud_bottom = min(last.get("SpanA", last["Close"]), last.get("SpanB", last["Close"]))
    if last["Close"]>cloud_top: score+=20
    if last["Close"]<cloud_bottom: score-=20
    return score

def generate_signal(df):
    if df.empty or len(df)<52: return "NO DATA",0,None
    df = Indicators.atr(df.copy())
    df = Indicators.rsi(df)
    df = Indicators.macd(df)
    df = Indicators.ichimoku(df)
    last = df.iloc[-1]
    score = calculate_score(last)
    if score>=80: signal="STRONG BUY"
    elif score>=65: signal="BUY"
    elif score<=20: signal="STRONG SELL"
    elif score<=35: signal="SELL"
    else: signal="HOLD"
    is_buy = "BUY" in signal
    entry = last["Close"]*(1.001 if is_buy else 0.999)
    atr = last["ATR"]
    sl = entry-(atr*2) if is_buy else entry+(atr*2)
    tp1 = entry*1.03 if is_buy else entry*0.97
    tp2 = entry*1.06 if is_buy else entry*0.94
    return signal, score, {"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2}

# -----------------------------
# MULTI-TF ASYNC SIGNAL
# -----------------------------
async def multi_tf_signal_async(exchange_id, symbol):
    tasks = [fetch_data_async(exchange_id, symbol, tf) for tf in ["5m","15m","1h","4h"]]
    dfs = await asyncio.gather(*tasks)
    scores = []
    for df in dfs:
        if not df.empty:
            _,score,_ = generate_signal(df)
            scores.append(score)
    return np.mean(scores) if scores else 50

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    exchange_id = st.selectbox("Exchange", ["binance","bitget","gateio","xt"])
    symbol = st.selectbox("Trading Pair", TRADING_PAIRS)
    timeframe = st.selectbox("Base Timeframe", TIMEFRAMES)
    auto_refresh = st.checkbox("Auto-refresh 60s", True)

# -----------------------------
# DASHBOARD ASYNC
# -----------------------------
async def main_dashboard():
    st.subheader(f"Trading Dashboard - {session_name} ({session_note})")
    st.markdown(f"**Current Time (UTC+1): {datetime.now(timezone.utc)+timedelta(hours=1):%Y-%m-%d %H:%M:%S}**")

    df = await fetch_data_async(exchange_id, symbol, timeframe)
    if df.empty:
        st.warning("No data available")
        return

    live_price = df["Close"].iloc[-1]
    score = await multi_tf_signal_async(exchange_id, symbol)
    signal, _, levels = generate_signal(df)

    st.markdown(f"**Live {symbol} Price: ${live_price:,.2f}**")
    st.markdown(f"**Signal: {signal} ({score:.1f}%)**")
    if levels:
        st.markdown(f"Entry: {levels['entry']:.2f} | SL: {levels['sl']:.2f} | TP1: {levels['tp1']:.2f} | TP2: {levels['tp2']:.2f}")

    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index[-100:], open=df["Open"][-100:], high=df["High"][-100:],
        low=df["Low"][-100:], close=df["Close"][-100:]
    )])
    fig.update_layout(height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Notification
    if signal in ["BUY","STRONG BUY","SELL","STRONG SELL"]:
        st.balloons() if "BUY" in signal else st.warning(f"{signal} signal triggered!")

# -----------------------------
# RUN ASYNC
# -----------------------------
if auto_refresh:
    asyncio.run(main_dashboard())
    time.sleep(60)
    st.experimental_rerun()
else:
    asyncio.run(main_dashboard())
