import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sqlite3

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="Aegis Sentinel Pro", layout="wide")
st.title("📡 Aegis Sentinel – Advanced Signal Engine + Full Backtesting")
st.caption("1H Entry • 4H/1D Confirmation • ML Confidence • TP/SL • Multi-Exchange Backtest & Analytics")

# ============================
# DATABASE SETUP
# ============================
conn = sqlite3.connect("signals.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS signals (
    timestamp TEXT,
    exchange TEXT,
    pair TEXT,
    direction TEXT,
    entry REAL,
    take_profit REAL,
    stop_loss REAL,
    confidence REAL,
    session TEXT
)
""")
conn.commit()

# ============================
# CONFIGURATION
# ============================
EXCHANGES = {
    "Bitget": ccxt.bitget(),
    "Gate.io": ccxt.gateio(),
    "XT": ccxt.xt()
}

ALL_SYMBOLS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","SUI/USDT","PEPE/USDT",
    "LINK/USDT","BNB/USDT","ADA/USDT","DOGE/USDT","MATIC/USDT"
]

st.sidebar.header("🔹 Trading Pair Selection")
selected_symbols = st.sidebar.multiselect(
    "Select assets to monitor", ALL_SYMBOLS, default=["BTC/USDT","ETH/USDT","SOL/USDT"]
)

st.sidebar.header("🔹 Strategy Parameters")
ema_fast_span = st.sidebar.slider("EMA Fast Span",5,50,20)
ema_slow_span = st.sidebar.slider("EMA Slow Span",10,200,50)
rsi_period = st.sidebar.slider("RSI Period",5,50,14)
atr_multiplier = st.sidebar.slider("ATR Multiplier (for TP/SL)",0.5,3.0,1.5)
confidence_threshold = st.sidebar.slider("ML Confidence Threshold",0.0,1.0,0.65)
bollinger_period = st.sidebar.slider("Bollinger Band Period",10,50,20)
bollinger_std = st.sidebar.slider("Bollinger Band Std Dev",1.0,3.0,2.0)
macd_fast = st.sidebar.slider("MACD Fast EMA",5,30,12)
macd_slow = st.sidebar.slider("MACD Slow EMA",10,60,26)
macd_signal = st.sidebar.slider("MACD Signal EMA",5,30,9)

TIMEFRAMES = {"entry":"1h","confirm_4h":"4h","confirm_1d":"1d"}
MAX_BARS = 200
REFRESH_SECONDS = 60

# ============================
# HARDEN CCXT
# ============================
for ex in EXCHANGES.values():
    ex.enableRateLimit = True
    ex.timeout = 20000

# ============================
# DATA FETCHING
# ============================
@st.cache_data(ttl=300)
def fetch_ohlcv(_exchange, symbol, timeframe):
    data = _exchange.fetch_ohlcv(symbol, timeframe, limit=MAX_BARS)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# ============================
# TECHNICAL INDICATORS
# ============================
def compute_structure(df):
    df["ema_fast"] = df["close"].ewm(span=ema_fast_span).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_span).mean()
    df["rsi"] = 100 - (100 / (1 + df["close"].pct_change().rolling(rsi_period).mean()))
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["macd"] = df["close"].ewm(span=macd_fast).mean() - df["close"].ewm(span=macd_slow).mean()
    df["macd_signal"] = df["macd"].ewm(span=macd_signal).mean()
    df["bb_mid"] = df["close"].rolling(bollinger_period).mean()
    df["bb_upper"] = df["bb_mid"] + bollinger_std*df["close"].rolling(bollinger_period).std()
    df["bb_lower"] = df["bb_mid"] - bollinger_std*df["close"].rolling(bollinger_period).std()
    return df

def trend_bias(df):
    if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]:
        return 1
    elif df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]:
        return -1
    return 0

def get_trading_session():
    utc_hour = datetime.utcnow().hour
    if 0 <= utc_hour < 8: return "Asian"
    elif 8 <= utc_hour < 16: return "London"
    else: return "NY"

# ============================
# ML CONFIDENCE
# ============================
def ml_confidence(entry_df, confirm_4h, confirm_1d):
    weights = {"trend_alignment":0.3,"momentum":0.2,"volatility":0.2,"macd":0.2,"bollinger":0.1}
    trends = [trend_bias(entry_df), trend_bias(confirm_4h), trend_bias(confirm_1d)]
    trend_alignment = 1 if len(set(trends))==1 else 0
    momentum = min(max((entry_df["rsi"].iloc[-1]-50)/50,-1),1)
    volatility = entry_df["close"].pct_change().std()
    volatility_score = 1 - min(volatility*10,1)
    macd_signal_val = 1 if entry_df["macd"].iloc[-1] > entry_df["macd_signal"].iloc[-1] else 0
    bb_signal_val = 1 if entry_df["close"].iloc[-1] > entry_df["bb_mid"].iloc[-1] else 0
    raw = (weights["trend_alignment"]*trend_alignment +
           weights["momentum"]*abs(momentum) +
           weights["volatility"]*volatility_score +
           weights["macd"]*macd_signal_val +
           weights["bollinger"]*bb_signal_val)
    return round(1 / (1 + np.exp(-5*(raw-0.5))),4)

# ============================
# SIGNAL ENGINE
# ============================
def generate_signals(symbols):
    signals=[]
    session=get_trading_session()
    for ex_name,ex in EXCHANGES.items():
        for symbol in symbols:
            try:
                entry_df=compute_structure(fetch_ohlcv(ex,symbol,TIMEFRAMES["entry"]))
                confirm_4h=compute_structure(fetch_ohlcv(ex,symbol,TIMEFRAMES["confirm_4h"]))
                confirm_1d=compute_structure(fetch_ohlcv(ex,symbol,TIMEFRAMES["confirm_1d"]))
                bias=trend_bias(entry_df)
                if bias==0: continue
                confidence=ml_confidence(entry_df,confirm_4h,confirm_1d)
                if confidence<confidence_threshold: continue
                entry_price=entry_df["close"].iloc[-1]
                atr=entry_df["atr"].iloc[-1]
                tp=entry_price+atr*atr_multiplier if bias==1 else entry_price-atr*atr_multiplier
                sl=entry_price-atr if bias==1 else entry_price+atr
                reward_risk=round(abs(tp-entry_price)/abs(entry_price-sl),2) if sl!=entry_price else np.nan
                signal={
                    "Exchange":ex_name,
                    "Pair":symbol,
                    "Direction":"LONG" if bias==1 else "SHORT",
                    "Entry":round(entry_price,4),
                    "Take Profit":round(tp,4),
                    "Stop Loss":round(sl,4),
                    "Reward/Risk":reward_risk,
                    "ML Confidence":confidence,
                    "Session":session,
                    "Time (UTC)":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                }
                signals.append(signal)
                # Audit logging
                cursor.execute("""
                INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?)
                """,(signal["Time (UTC)"],signal["Exchange"],signal["Pair"],signal["Direction"],
                    signal["Entry"],signal["Take Profit"],signal["Stop Loss"],signal["ML Confidence"],
                    signal["Session"],signal["Time (UTC)"]))
                conn.commit()
            except Exception as e:
                st.warning(f"{ex_name} {symbol}: {str(e)}")
    return signals

# ============================
# BACKTEST ENGINE
# ============================
def backtest(symbol):
    equity_dict={}
    metrics=[]
    for ex_name, ex in EXCHANGES.items():
        equity=10000
        equity_curve=[]
        peak=equity
        max_drawdown=0
        gross_profit=0
        gross_loss=0
        trades=0
        wins=0
        df=compute_structure(fetch_ohlcv(ex,symbol,"1h"))
        for i in range(50,len(df)):
            sub=df.iloc[:i]
            bias=trend_bias(sub)
            if bias==0: continue
            entry=sub["close"].iloc[-1]
            atr=sub["atr"].iloc[-1]
            tp=entry+atr*atr_multiplier if bias==1 else entry-atr*atr_multiplier
            sl=entry-atr if bias==1 else entry+atr
            trades+=1
            profit=tp-entry if bias==1 else entry-tp
            equity+=profit
            equity_curve.append(equity)
            if equity>peak: peak=equity
            drawdown=(peak-equity)/peak
            if drawdown>max_drawdown: max_drawdown=drawdown
            if profit>0: 
                wins+=1
                gross_profit+=profit
            else:
                gross_loss+=abs(profit)
        win_rate=wins/trades if trades>0 else np.nan
        profit_factor=gross_profit/gross_loss if gross_loss>0 else np.nan
        metrics.append({"Exchange":ex_name,"Trades":trades,"Win Rate":win_rate,"Max Drawdown":max_drawdown,"Profit Factor":profit_factor})
        equity_dict[ex_name]=equity_curve
    return equity_dict, metrics

# ============================
# STREAMLIT UI
# ============================
tabs=st.tabs(["📈 Live Signals","📊 Backtesting","📜 Audit Logs"])

with tabs[0]:
    st.subheader("Live Signals")
    if not selected_symbols:
        st.info("Please select trading pairs in the sidebar.")
    else:
        signals=generate_signals(selected_symbols)
        if signals:
            df=pd.DataFrame(signals)
            def color_rows(row):
                return ['background-color: #b6fcb6' if row['Direction']=='LONG' else 'background-color: #fbb6b6' for _ in row]
            st.dataframe(df.style.apply(color_rows,axis=1),use_container_width=True)
        else:
            st.info("No valid signals at this time.")

with tabs[1]:
    st.subheader("Backtesting")
    for symbol in selected_symbols:
        equity_dict, metrics=backtest(symbol)
        st.write(f"Symbol: {symbol}")
        metric_df=pd.DataFrame(metrics)
        st.dataframe(metric_df, use_container_width=True)
        # Equity curves per exchange
        for ex_name, eq_curve in equity_dict.items():
            st.line_chart(eq_curve, height=250, use_container_width=True, 
                          width=None, key=f"{symbol}_{ex_name}_eq")

with tabs[2]:
    st.subheader("Audit Logs")
    df_logs=pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 100",conn)
    st.dataframe(df_logs,use_container_width=True)
