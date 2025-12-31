import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import ccxt.pro as ccxtpro
import plotly.graph_objects as go
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="ProfitForge Pro WS", layout="wide")
st.title("🔥 ProfitForge Pro — Live Signals + Trade Management (WebSocket)")

# =========================
# TRADING SESSION
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

session_name, note, color = get_session()
st.markdown(f"<h3 style='text-align:center;color:{color};'>💹 {session_name} — {note}</h3>", unsafe_allow_html=True)

# =========================
# USER INPUTS
# =========================
col1, col2, col3 = st.columns(3)
with col1:
    exchange_id = st.selectbox("Exchange", ["binance","bitget","gateio","xt"])
with col2:
    symbol = st.selectbox("Symbol", ["BTC/USDT","ETH/USDT","SOL/USDT"])
with col3:
    timeframe = st.selectbox("Timeframe", ["1m","5m","15m","1h","4h","1d"])

balance = st.number_input("Account Balance ($)", min_value=100.0, value=10000.0)
risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)

# =========================
# INDICATORS
# =========================
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
        gain = delta.where(delta>0,0).rolling(window).mean()
        loss = -delta.where(delta<0,0).rolling(window).mean()
        rs = gain/loss
        df['RSI'] = 100-(100/(1+rs))
        return df

    @staticmethod
    def macd(df):
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1-exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df

    @staticmethod
    def ichimoku(df):
        high9 = df['High'].rolling(9).max()
        low9 = df['Low'].rolling(9).min()
        df['Conversion']=(high9+low9)/2
        high26 = df['High'].rolling(26).max()
        low26 = df['Low'].rolling(26).min()
        df['Base']=(high26+low26)/2
        df['SpanA']=((df['Conversion']+df['Base'])/2).shift(26)
        df['SpanB']=((df['High'].rolling(52).max()+df['Low'].rolling(52).min())/2).shift(26)
        df['Lagging']=df['Close'].shift(-26)
        return df

# =========================
# SIGNAL ENGINE
# =========================
def calculate_score(last):
    score=50
    if last.get('RSI',50)<30: score+=30
    if last.get('RSI',50)>70: score-=30
    if last.get('MACD',0)>last.get('MACD_Signal',0): score+=20
    if last.get('MACD',0)<last.get('MACD_Signal',0): score-=20
    cloud_top=max(last.get('SpanA',last['Close']),last.get('SpanB',last['Close']))
    cloud_bottom=min(last.get('SpanA',last['Close']),last.get('SpanB',last['Close']))
    if last['Close']>cloud_top: score+=20
    if last['Close']<cloud_bottom: score-=20
    return score

def generate_signal(df):
    if df.empty or len(df)<52: return "NO DATA",0,None
    df=Indicators.atr(df)
    df=Indicators.rsi(df)
    df=Indicators.macd(df)
    df=Indicators.ichimoku(df)
    last=df.iloc[-1]
    atr_avg=df['ATR'].mean()
    if last['ATR']<atr_avg*0.5: return "NO TRADE",0,None
    score=calculate_score(last)
    signal="HOLD"
    if score>=80: signal="STRONG BUY"
    elif score>=65: signal="BUY"
    elif score<=20: signal="STRONG SELL"
    elif score<=35: signal="SELL"
    is_buy="BUY" in signal
    entry=last['Close']*(1.001 if is_buy else 0.999)
    sl=last['Close']-(last['ATR']*2) if is_buy else last['Close']+(last['ATR']*2)
    tp1=last['Close']*1.03 if is_buy else last['Close']*0.97
    tp2=last['Close']*1.06 if is_buy else last['Close']*0.94
    return signal,score,{"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"atr":last['ATR']}

# =========================
# POSITION SIZE
# =========================
def calc_position(levels,balance,risk_pct):
    if not levels: return 0,0
    risk_amount=balance*(risk_pct/100)
    risk_per_unit=abs(levels['entry']-levels['sl'])
    if risk_per_unit<=0: return 0,0
    size=risk_amount/risk_per_unit
    return round(size,6),round(risk_amount,2)

# =========================
# MAIN ASYNC LOOP
# =========================
async def main():
    exchange_cls=getattr(ccxtpro, exchange_id)
    exchange=exchange_cls({"enableRateLimit":True})
    await exchange.load_markets()
    df=pd.DataFrame()

    # Fetch initial OHLCV
    try:
        ohlcv=await exchange.fetch_ohlcv(symbol,timeframe=timeframe,limit=300)
        df=pd.DataFrame(ohlcv,columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms")
        df.set_index("timestamp",inplace=True)
    except: pass

    chart_placeholder = st.empty()
    signal_placeholder = st.empty()
    info_placeholder = st.empty()

    while True:
        try:
            ticker=await exchange.watch_ticker(symbol)
            price=ticker['last']
        except:
            price=None

        if price:
            new_row={"Open":price,"High":price,"Low":price,"Close":price,"Volume":ticker.get("quoteVolume",0)}
            df=df.append(pd.DataFrame([new_row], index=[datetime.utcnow()]))
            if len(df)>300: df=df.iloc[-300:]

            signal,score,levels=generate_signal(df)

            # Signal display
            signal_placeholder.markdown(f"<h2 style='text-align:center;color:#00FF9F;'>Signal: {signal} ({score}%)</h2>", unsafe_allow_html=True)

            # Position info
            if levels:
                size,risk_amount=calc_position(levels,balance,risk_pct)
                info_placeholder.write(f"Entry: ${levels['entry']:.2f}, SL: ${levels['sl']:.2f}, TP1: ${levels['tp1']:.2f}, TP2: ${levels['tp2']:.2f}, ATR: {levels['atr']:.2f}")
                info_placeholder.write(f"Recommended Size: {size} {symbol.split('/')[0]}, Risk: ${risk_amount:.2f}")

            # Candlestick chart
            fig=go.Figure()
            fig.add_candlestick(
                x=df.index[-100:],
                open=df["Open"][-100:],
                high=df["High"][-100:],
                low=df["Low"][-100:],
                close=df["Close"][-100:],
                name="Price"
            )
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        await asyncio.sleep(1)  # 1-second interval

# =========================
# RUN ASYNC
# =========================
asyncio.run(main())
