# profitforge_v4.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go

# -----------------------------
# CLEAR CACHE ON STARTUP
# -----------------------------
st.cache_data.clear()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ProfitForge Pro", layout="wide")
st.title("🔥 ProfitForge Pro - CCXT Polling Edition")

# -----------------------------
# PLACEHOLDERS
# -----------------------------
session_ph = st.empty()
time_ph = st.empty()
price_ph = st.empty()
signal_ph = st.empty()
levels_ph = st.empty()
chart_ph = st.empty()

# -----------------------------
# USER INPUTS
# -----------------------------
TRADING_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                 "ADA/USDT", "LTC/USDT", "DOGE/USDT", "MATIC/USDT", "AVAX/USDT"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

with st.sidebar:
    exchange_id = st.selectbox("Exchange", ["binance", "bitget", "gateio", "xt"], index=0)
    symbol = st.selectbox("Pair", TRADING_PAIRS, index=0)
    timeframe = st.selectbox("Chart Timeframe", TIMEFRAMES, index=4)
    refresh_btn = st.button("Refresh Now")

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
        df['Lagging'] = df['Close'].shift(-26)
        return df

# -----------------------------
# SESSION DETECTOR
# -----------------------------
def get_session():
    utc_now = datetime.now(timezone.utc)
    hour = utc_now.hour
    if 0 <= hour < 8:
        return "Asian Session", "#FFB86C"
    elif 8 <= hour < 12:
        return "London Open", "#8A2BE2"
    elif 12 <= hour < 16:
        return "NY + London Overlap", "#FF416C"
    elif 16 <= hour < 21:
        return "New York Session", "#43E97B"
    else:
        return "Quiet Hours", "#888888"

# -----------------------------
# SIGNAL CALCULATION
# -----------------------------
def calculate_score(last):
    score = 50
    if last.get('RSI',50)<30: score+=30
    if last.get('RSI',50)>70: score-=30
    if last.get('MACD',0)>last.get('MACD_Signal',0): score+=20
    if last.get('MACD',0)<last.get('MACD_Signal',0): score-=20
    cloud_top = max(last.get('SpanA',last['Close']), last.get('SpanB', last['Close']))
    cloud_bottom = min(last.get('SpanA',last['Close']), last.get('SpanB', last['Close']))
    if last['Close']>cloud_top: score+=20
    if last['Close']<cloud_bottom: score-=20
    return score

def generate_signal(df):
    if df.empty or len(df)<52: return "NO DATA",0,None
    df=Indicators.atr(df.copy())
    df=Indicators.rsi(df)
    df=Indicators.macd(df)
    df=Indicators.ichimoku(df)
    last=df.iloc[-1]
    atr=last['ATR']
    current_price=last['Close']
    atr_avg=df['ATR'].mean()
    if atr<atr_avg*0.5: return "NO TRADE",0,None
    score=calculate_score(last)
    if score>=80: signal="STRONG BUY"
    elif score>=65: signal="BUY"
    elif score<=20: signal="STRONG SELL"
    elif score<=35: signal="SELL"
    else: signal="HOLD"
    is_buy="BUY" in signal
    entry=current_price*(1.001 if is_buy else 0.999)
    sl=current_price-(atr*2) if is_buy else current_price+(atr*2)
    tp1=current_price*1.03 if is_buy else current_price*0.97
    tp2=current_price*1.06 if is_buy else current_price*0.94
    return signal, score, {"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2}

# -----------------------------
# FETCH DATA
# -----------------------------
def fetch_data(exchange_id,symbol,timeframe,limit=300):
    try:
        exchange_cls=getattr(ccxt,exchange_id)
        exchange=exchange_cls({"enableRateLimit":True})
        ohlcv=exchange.fetch_ohlcv(symbol,timeframe=timeframe,limit=limit)
        df=pd.DataFrame(ohlcv,columns=['timestamp','Open','High','Low','Close','Volume'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.warning(f"{exchange_id.upper()} error: {e}")
        return pd.DataFrame()

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    session_name,color=get_session()
    session_ph.markdown(f"<div style='text-align:center;color:white;background:{color};padding:8px;border-radius:12px;'>{session_name}</div>",unsafe_allow_html=True)
    utc_now=datetime.now(timezone.utc)+timedelta(hours=1)
    time_ph.markdown(f"<div style='text-align:center;color:#00FF9F;font-size:16px;'>Current Time (UTC+1): {utc_now.strftime('%Y-%m-%d %H:%M:%S')}</div>",unsafe_allow_html=True)

    df=fetch_data(exchange_id,symbol,timeframe)
    if df.empty: return st.warning("No data available yet.")

    live_price=df['Close'].iloc[-1]
    price_ph.markdown(f"<h2 style='text-align:center;color:#00FF9F;'>${live_price:,.2f}</h2>",unsafe_allow_html=True)

    signal,score,levels=generate_signal(df)
    signal_ph.markdown(f"<h3 style='text-align:center;color:#00FF9F;'>Signal: {signal} ({score}%)</h3>",unsafe_allow_html=True)
    if levels:
        levels_ph.markdown(f"<div style='text-align:center;color:#FFDD00;'>Entry: {levels['entry']:.2f} | SL: {levels['sl']:.2f} | TP1: {levels['tp1']:.2f} | TP2: {levels['tp2']:.2f}</div>",unsafe_allow_html=True)

    fig=go.Figure(data=[go.Candlestick(
        x=df.index[-50:],
        open=df["Open"][-50:], high=df["High"][-50:],
        low=df["Low"][-50:], close=df["Close"][-50:]
    )])
    fig.update_layout(height=400,xaxis_rangeslider_visible=False)
    chart_ph.plotly_chart(fig,use_container_width=True)

# -----------------------------
# RUN
# -----------------------------
main()
