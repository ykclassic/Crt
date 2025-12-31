# profitforge_v5.py
import streamlit as st
import ccxtpro
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="ProfitForge Pro v5", layout="wide")
st.title("🔥 ProfitForge Pro v5 - Live + Backtest + Alerts")

# Clear cache at startup
st.cache_data.clear()

# Trading Pairs and Timeframes
TRADING_PAIRS = ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT",
                 "ADA/USDT","LTC/USDT","DOGE/USDT","MATIC/USDT","AVAX/USDT"]
TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d"]

# Persistent journal file
JOURNAL_FILE = "trade_journal.csv"
if not os.path.exists(JOURNAL_FILE):
    pd.DataFrame(columns=["time","symbol","signal","entry","sl","tp1","tp2","status"]).to_csv(JOURNAL_FILE,index=False)

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
        rs = gain / loss
        df['RSI'] = 100 - (100/(1+rs))
        return df

    @staticmethod
    def macd(df):
        exp1 = df['Close'].ewm(span=12,adjust=False).mean()
        exp2 = df['Close'].ewm(span=26,adjust=False).mean()
        df['MACD'] = exp1-exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9,adjust=False).mean()
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
# SIGNAL CALCULATION
# -----------------------------
def calculate_score(last):
    score = 50
    if last.get('RSI',50)<30: score+=30
    if last.get('RSI',50)>70: score-=30
    if last.get('MACD',0)>last.get('MACD_Signal',0): score+=20
    if last.get('MACD',0)<last.get('MACD_Signal',0): score-=20
    cloud_top = max(last.get('SpanA',last['Close']),last.get('SpanB',last['Close']))
    cloud_bottom = min(last.get('SpanA',last['Close']),last.get('SpanB',last['Close']))
    if last['Close']>cloud_top: score+=20
    if last['Close']<cloud_bottom: score-=20
    return score

def generate_signal(df_main, symbol, exchange_id):
    if df_main.empty or len(df_main)<52:
        return "NO DATA",0,None
    df_main=Indicators.atr(df_main.copy())
    df_main=Indicators.rsi(df_main)
    df_main=Indicators.macd(df_main)
    df_main=Indicators.ichimoku(df_main)
    last=df_main.iloc[-1]
    score=calculate_score(last)
    if score>=80: signal="STRONG BUY"
    elif score>=65: signal="BUY"
    elif score<=20: signal="STRONG SELL"
    elif score<=35: signal="SELL"
    else: signal="HOLD"
    is_buy="BUY" in signal
    entry=last['Close']*(1.001 if is_buy else 0.999)
    atr=last['ATR']
    sl=entry-(atr*2) if is_buy else entry+(atr*2)
    tp1=entry*1.03 if is_buy else entry*0.97
    tp2=entry*1.06 if is_buy else entry*0.94
    return signal,score,{"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2}

# -----------------------------
# FETCH DATA ASYNC
# -----------------------------
async def fetch_ohlcv(exchange_id, symbol, timeframe='1h', limit=500):
    try:
        exchange_cls=getattr(ccxtpro,exchange_id)
        exchange=exchange_cls({"enableRateLimit":True})
        await exchange.load_markets()
        data=await exchange.fetch_ohlcv(symbol,timeframe=timeframe,limit=limit)
        df=pd.DataFrame(data,columns=['timestamp','Open','High','Low','Close','Volume'])
        df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
        df.set_index('timestamp', inplace=True)
        await exchange.close()
        return df
    except Exception as e:
        st.warning(f"{exchange_id.upper()} error: {e}")
        return pd.DataFrame()

# -----------------------------
# BACKTESTING
# -----------------------------
def backtest(df):
    df=Indicators.atr(df.copy())
    df=Indicators.rsi(df)
    df=Indicators.macd(df)
    df=Indicators.ichimoku(df)
    equity=10000
    trades=[]
    for i in range(52,len(df)):
        last=df.iloc[i]
        signal,_,levels=generate_signal(df.iloc[:i+1],'','')
        if levels and "BUY" in signal:
            entry=levels['entry']
            sl=levels['sl']
            tp=levels['tp1']
            pnl=(tp-entry)/entry*100
            equity+=equity*pnl/100
            trades.append(pnl)
    return equity,trades

# -----------------------------
# PERSISTENT TRADE JOURNAL
# -----------------------------
def log_trade(symbol,signal,levels):
    df=pd.read_csv(JOURNAL_FILE)
    df=df.append({"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "symbol":symbol,"signal":signal,"entry":levels['entry'],
                  "sl":levels['sl'],"tp1":levels['tp1'],"tp2":levels['tp2'],
                  "status":"OPEN"},ignore_index=True)
    df.to_csv(JOURNAL_FILE,index=False)

# -----------------------------
# STREAMLIT DASHBOARD
# -----------------------------
async def main():
    st.subheader("Trading Dashboard")
    utc_now=datetime.now(timezone.utc)+timedelta(hours=1)
    st.markdown(f"**Current Time (UTC+1): {utc_now.strftime('%Y-%m-%d %H:%M:%S')}**")
    df=await fetch_ohlcv(exchange_id,symbol,timeframe)
    if df.empty: return st.warning("No data yet")
    live_price=df['Close'].iloc[-1]
    st.markdown(f"**Live Price {symbol}: ${live_price:,.2f}**")
    signal,score,levels=generate_signal(df,symbol,exchange_id)
    st.markdown(f"**Signal: {signal} ({score}%)**")
    if levels:
        st.markdown(f"Entry: {levels['entry']:.2f} | SL: {levels['sl']:.2f} | TP1: {levels['tp1']:.2f} | TP2: {levels['tp2']:.2f}")
    # Plot chart
    fig=go.Figure(data=[go.Candlestick(x=df.index[-50:],open=df["Open"][-50:],high=df["High"][-50:],
                                       low=df["Low"][-50:],close=df["Close"][-50:])])
    fig.update_layout(height=400,xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
    # Streamlit notification
    if signal in ["BUY","STRONG BUY","SELL","STRONG SELL"]:
        st.balloons() if "BUY" in signal else st.warning(f"{signal} signal triggered!")

# -----------------------------
# RUN ASYNC
# -----------------------------
asyncio.run(main())
