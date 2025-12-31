# profitforge_single.py
import streamlit as st
import ccxtpro
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ProfitForge Pro", layout="wide")
st.title("🔥 ProfitForge Pro - Single File Edition")

# -----------------------------
# PLACEHOLDERS
# -----------------------------
session_ph = st.empty()
price_ph = st.empty()
signal_ph = st.empty()
chart_ph = st.empty()
info_ph = st.empty()
journal_ph = st.empty()

# -----------------------------
# USER INPUTS
# -----------------------------
with st.sidebar:
    exchange_id = st.selectbox("Exchange", ["binance", "bitget", "gateio", "xt"], index=0)
    symbol = st.selectbox("Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"], index=0)
    timeframe = st.selectbox("Chart Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    balance = st.number_input("Account Balance ($)", min_value=100.0, value=10000.0)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)

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
# SIGNAL CALCULATION
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

def generate_signal(df):
    if df.empty or len(df) < 52:
        return "NO DATA", 0, None

    df = Indicators.atr(df.copy())
    df = Indicators.rsi(df)
    df = Indicators.macd(df)
    df = Indicators.ichimoku(df)

    last = df.iloc[-1]
    atr = last['ATR']
    current_price = last['Close']

    atr_avg = df['ATR'].mean()
    if atr < atr_avg * 0.5:
        return "NO TRADE", 0, None

    score = calculate_score(last)

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

    is_buy = "BUY" in signal
    entry = current_price * (1.001 if is_buy else 0.999)
    sl = current_price - (atr * 2) if is_buy else current_price + (atr * 2)
    tp1 = current_price * 1.03 if is_buy else current_price * 0.97
    tp2 = current_price * 1.06 if is_buy else current_price * 0.94

    return signal, score, {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "atr": atr}

def calculate_position(levels, balance, risk_pct):
    if not levels: return 0, 0
    risk_amount = balance * (risk_pct / 100)
    risk_per_unit = abs(levels['entry'] - levels['sl'])
    if risk_per_unit <= 0: return 0, 0
    size = risk_amount / risk_per_unit
    return round(size,6), round(risk_amount,2)

# -----------------------------
# TRADE JOURNAL
# -----------------------------
if 'journal' not in st.session_state:
    st.session_state.journal = []

def log_trade(signal, size, levels):
    if size==0 or not levels: return
    st.session_state.journal.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "signal": signal,
        "entry": levels['entry'],
        "sl": levels['sl'],
        "size": size,
        "risk_$": size * abs(levels['entry']-levels['sl']),
        "status": "OPEN"
    })

# -----------------------------
# ASYNC MAIN LOOP
# -----------------------------
async def main_loop():
    exchange_cls = getattr(ccxtpro, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    await exchange.load_markets()
    df = pd.DataFrame()

    while True:
        # --- Time & Session ---
        now = datetime.now(timezone.utc)
        hour = now.hour
        if 0 <= hour < 8: session_name,color="#FFB86C"
        elif 8 <= hour < 12: session_name,color="#8A2BE2"
        elif 12 <= hour < 16: session_name,color="#FF416C"
        elif 16 <= hour < 21: session_name,color="#43E97B"
        else: session_name,color="#888888"

        session_ph.markdown(f"<div style='text-align:center;color:white;background:{color};padding:8px;border-radius:12px;'>UTC {now.strftime('%H:%M:%S')} — {session_name}</div>", unsafe_allow_html=True)

        # --- Fetch price ---
        try:
            ticker = await exchange.watch_ticker(symbol)
            price = ticker['last']
        except: price=None

        if price:
            price_ph.markdown(f"<h2 style='text-align:center;color:#00FF9F;'>${price:,.2f}</h2>", unsafe_allow_html=True)

            new_row={"Open":price,"High":price,"Low":price,"Close":price,"Volume":ticker.get("quoteVolume",0)}
            df=df.append(pd.DataFrame([new_row],index=[datetime.utcnow()]))
            if len(df)>300: df=df.iloc[-300:]

            signal,score,levels=generate_signal(df)
            signal_ph.markdown(f"<h2 style='text-align:center;color:#00FF9F;'>Signal: {signal} ({score}%)</h2>", unsafe_allow_html=True)

            # Position
            size,risk_amount=calculate_position(levels,balance,risk_pct)
            info_ph.write(f"Entry: {levels['entry']:.2f}, SL: {levels['sl']:.2f}, TP1: {levels['tp1']:.2f}, TP2: {levels['tp2']:.2f}, ATR: {levels['atr']:.2f}")
            info_ph.write(f"Size: {size}, Risk: ${risk_amount:.2f}")

            # Candlestick
            fig=go.Figure()
            fig.add_candlestick(
                x=df.index[-50:],
                open=df["Open"][-50:], high=df["High"][-50:],
                low=df["Low"][-50:], close=df["Close"][-50:]
            )
            fig.update_layout(height=400, xaxis_rangeslider_visible=False)
            chart_ph.plotly_chart(fig,use_container_width=True)

        await asyncio.sleep(1)

# -----------------------------
# RUN ASYNC LOOP
# -----------------------------
if __name__=="__main__":
    asyncio.run(main_loop())
