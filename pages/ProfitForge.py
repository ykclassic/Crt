# profitforge_v10.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import itertools

# === INSERTED DUMMY CREDENTIALS ===
TELEGRAM_TOKEN = "8367963721:AAH6B819_DevFNpZracbJ5EmHrDR3DKZeR4"
TELEGRAM_CHAT_ID = "865482105"  # This is your personal/user chat ID

# Optional: override with environment variables if they exist (good practice)
if os.getenv("TELEGRAM_BOT_TOKEN"):
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if os.getenv("TELEGRAM_CHAT_ID"):
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_message(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode="Markdown"
        )
    except Exception as e:
        print(f"Telegram error: {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ ProfitForge Pro connected.\nUse /status or /signals"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🟢 Bot is running\nMonitoring markets in real-time."
    )

# -----------------------------
# Telegram Bot
# -----------------------------

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "📊 *Latest Signals*\n\n"
    for symbol in TRADING_PAIRS[:5]:
        df = fetch_data("binance", symbol, "1h")
        if df.empty:
            continue
        signal, score, _ = generate_signal(df)
        msg += f"{symbol}: {signal} ({score:.1f})\n"
    await update.message.reply_text(msg, parse_mode="Markdown")


st.set_page_config(page_title="ProfitForge Pro v10", layout="wide")
st.title("🔥 ProfitForge Pro v10 - Backtesting + Multi-TF Optimization + Strategy Optimizer")

st.cache_data.clear()

TRADING_PAIRS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT",
    "ADA/USDT","LTC/USDT","DOGE/USDT","MATIC/USDT","AVAX/USDT"
]
TIMEFRAMES = ["5m","15m","30m","1h","4h","1d"]

# -----------------------------
# SESSION INFO
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
    def macd(df, fast=12, slow=26, signal=9):
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        return df

    @staticmethod
    def ichimoku(df, conv=9, base=26, span=52):
        high_conv = df['High'].rolling(conv).max()
        low_conv = df['Low'].rolling(conv).min()
        df['Conversion'] = (high_conv+low_conv)/2
        high_base = df['High'].rolling(base).max()
        low_base = df['Low'].rolling(base).min()
        df['Base'] = (high_base+low_base)/2
        df['SpanA'] = ((df['Conversion']+df['Base'])/2).shift(base)
        df['SpanB'] = ((df['High'].rolling(span).max()+df['Low'].rolling(span).min())/2).shift(base)
        return df

# -----------------------------
# DATA FETCHING
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_data(exchange_id, symbol, timeframe="1h", limit=500):
    exchanges = {
        "binance": ccxt.binance(),
        "bitget": ccxt.bitget(),
        "gateio": ccxt.gateio(),
        "xt": ccxt.xt()
    }
    exchange = exchanges.get(exchange_id)
    if not exchange: return pd.DataFrame()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        st.warning(f"{exchange_id.upper()} fetch error: {e}")
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

def generate_signal(df, params=None):
    if df.empty or len(df)<52: return "NO DATA",0,None
    params = params or {"atr":14,"rsi":14,"macd_fast":12,"macd_slow":26,"macd_signal":9,"ichimoku_conv":9,"ichimoku_base":26,"ichimoku_span":52}
    df = Indicators.atr(df.copy(), params["atr"])
    df = Indicators.rsi(df, params["rsi"])
    df = Indicators.macd(df, fast=params["macd_fast"], slow=params["macd_slow"], signal=params["macd_signal"])
    df = Indicators.ichimoku(df, conv=params["ichimoku_conv"], base=params["ichimoku_base"], span=params["ichimoku_span"])
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
# BACKTESTING ENGINE
# -----------------------------
def backtest(df, params=None):
    df = Indicators.atr(df.copy(), params.get("atr",14))
    df = Indicators.rsi(df, params.get("rsi",14))
    df = Indicators.macd(df, fast=params.get("macd_fast",12), slow=params.get("macd_slow",26), signal=params.get("macd_signal",9))
    df = Indicators.ichimoku(df, conv=params.get("ichimoku_conv",9), base=params.get("ichimoku_base",26), span=params.get("ichimoku_span",52))
    trades=[]
    for i in range(52,len(df)):
        signal,_,levels = generate_signal(df.iloc[:i+1], params)
        if signal in ["BUY","STRONG BUY","SELL","STRONG SELL"]:
            outcome = (df.iloc[i]["Close"]-levels["entry"]) if "BUY" in signal else (levels["entry"]-df.iloc[i]["Close"])
            trades.append(outcome)
    if trades:
        return {"total_trades":len(trades),"win_rate":np.mean([t>0 for t in trades])*100,"avg_pl":np.mean(trades),"total_pl":np.sum(trades)}
    return {"total_trades":0,"win_rate":0,"avg_pl":0,"total_pl":0}

# -----------------------------
# STRATEGY OPTIMIZER
# -----------------------------
def optimize_strategy(df, param_grid, top_n=3):
    results=[]
    for p in param_grid:
        res = backtest(df, p)
        res.update(p)
        results.append(res)
    results = sorted(results, key=lambda x: x["total_pl"], reverse=True)
    return results[:top_n]

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    exchange_id = st.selectbox("Exchange", ["binance","bitget","gateio","xt"])
    symbols = st.multiselect("Trading Pairs", TRADING_PAIRS, default=["BTC/USDT","ETH/USDT"])
    timeframes = st.multiselect("Timeframes", TIMEFRAMES, default=["1h","4h"])
    run_backtest = st.button("Run Backtest & Optimize")

# -----------------------------
# PARAM GRID (small subset for demo)
# -----------------------------
atr_range = [12,14,16]
rsi_range = [12,14,16]
macd_fast_range=[10,12]
macd_slow_range=[24,26]
param_grid = []
for atr,rsi,fast,slow in itertools.product(atr_range,rsi_range,macd_fast_range,macd_slow_range):
    param_grid.append({"atr":atr,"rsi":rsi,"macd_fast":fast,"macd_slow":slow,"macd_signal":9,"ichimoku_conv":9,"ichimoku_base":26,"ichimoku_span":52})

# -----------------------------
# DASHBOARD
# -----------------------------
st.subheader(f"Trading Dashboard - {session_name} ({session_note})")
st.markdown(f"**Current Time (UTC+1): {datetime.now(timezone.utc)+timedelta(hours=1):%Y-%m-%d %H:%M:%S}**")

for symbol in symbols:
    for tf in timeframes:
        df = fetch_data(exchange_id, symbol, tf)
        if df.empty: 
            st.warning(f"No data for {symbol} {tf}")
            continue
        live_price = df["Close"].iloc[-1]
        signal, score, levels = generate_signal(df)
        if signal in ["BUY", "SELL", "STRONG BUY", "STRONG SELL"]:
            msg = (
                f"📢 *{symbol} {tf}*\n"
                f"Signal: *{signal}*\n"
                f"Score: {score:.1f}\n"
                f"Entry: {levels['entry']:.2f}\n"
                f"SL: {levels['sl']:.2f}\n"
                f"TP1: {levels['tp1']:.2f}"
            )
            asyncio.run(send_telegram_message(msg))

        st.markdown(f"**{symbol} ({tf}) - Live Price: ${live_price:.2f} | Signal: {signal} ({score:.1f}%)**")
        if levels:
            st.markdown(f"Entry: {levels['entry']:.2f} | SL: {levels['sl']:.2f} | TP1: {levels['tp1']:.2f} | TP2: {levels['tp2']:.2f}")
        
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index[-100:], open=df["Open"][-100:], high=df["High"][-100:],
            low=df["Low"][-100:], close=df["Close"][-100:]
        )])
        if levels:
            fig.add_hline(y=levels["entry"], line_color="blue", line_dash="dot", annotation_text="Entry")
            fig.add_hline(y=levels["sl"], line_color="red", line_dash="dot", annotation_text="SL")
            fig.add_hline(y=levels["tp1"], line_color="green", line_dash="dot", annotation_text="TP1")
            fig.add_hline(y=levels["tp2"], line_color="green", line_dash="dot", annotation_text="TP2")
        fig.update_layout(height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Backtest + Optimize
        if run_backtest:
            st.info(f"Running Backtest & Optimization for {symbol} {tf}...")
            top_results = optimize_strategy(df, param_grid)
            for i,res in enumerate(top_results):
                st.markdown(f"**Top {i+1} Config:** ATR={res['atr']}, RSI={res['rsi']}, MACD={res['macd_fast']}/{res['macd_slow']}")
                st.markdown(f"Total Trades: {res['total_trades']}, Win Rate: {res['win_rate']:.1f}%, Avg P/L: {res['avg_pl']:.2f}, Total P/L: {res['total_pl']:.2f}")

# -----------------------------
# Start Telegram Bot (background)
# -----------------------------
def start_telegram_bot():
    if not TELEGRAM_TOKEN:
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("signals", signals))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(app.initialize())
    loop.create_task(app.start())
    loop.create_task(app.updater.start_polling())  # Fixed: added proper polling start
    loop.run_forever()

# Run bot in background thread (Streamlit compatible)
if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    import threading
    bot_thread = threading.Thread(target=start_telegram_bot, daemon=True)
    bot_thread.start()
