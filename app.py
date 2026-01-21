import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime
import plotly.express as px

# ============================= PAGE CONFIG =============================
st.set_page_config(
    page_title="UltimateCRT Bot - XT.com Live",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================= CONFIGURATION =============================
# --- API & TRADING SETTINGS ---
BASE_URL = "https://sapi.xt.com/v4/public/kline"
SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT"]

# --- TELEGRAM SETTINGS (Replace with your actual details) ---
TELEGRAM_TOKEN = "8367963721:AAH6B819_DevFNpZracbJ5EmHrDR3DKZeR4" 
TELEGRAM_CHAT_ID = "865482105"
ENABLE_TELEGRAM = True # Set to True after entering credentials

# --- DIRECTORIES ---
LOG_DIR = "performance_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

TRADE_LOG = os.path.join(LOG_DIR, "trades.json")
EQUITY_LOG = os.path.join(LOG_DIR, "equity.csv")

# --- INITIALIZE FILES ---
if not os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, 'w') as f:
        json.dump([], f)

if not os.path.exists(EQUITY_LOG):
    with open(EQUITY_LOG, 'w') as f:
        f.write("timestamp,equity,trades_count\n")

# --- SESSION STATE ---
if 'equity' not in st.session_state:
    st.session_state.equity = 100000.0
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_processed_ts' not in st.session_state:
    st.session_state.last_processed_ts = {}  # Track last candle TS per symbol

# ============================= TELEGRAM ENGINE =============================
def send_telegram_alert(message):
    """Sends a message to the configured Telegram bot."""
    if not ENABLE_TELEGRAM:
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ============================= DATA ENGINE =============================
@st.cache_data(ttl=15)
def fetch_klines(symbol: str, limit: int = 500):
    """Fetches 5m candles from XT.com"""
    params = {"symbol": symbol, "interval": "5min", "limit": limit}
    try:
        r = requests.get(BASE_URL, params=params, timeout=5)
        data = r.json()
        if data.get("rc") != 0 or not data.get("data"):
            return pd.DataFrame()
        
        df = pd.DataFrame(data["data"], columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit='ms')
        df.set_index("ts", inplace=True)
        df = df.astype(float)
        return df.sort_index()
    except Exception as e:
        st.error(f"API Error {symbol}: {e}")
        return pd.DataFrame()

# ============================= LOGIC ENGINE =============================
def calculate_indicators(df):
    """Calculates SMC indicators on the dataframe"""
    if df.empty: return df
    
    # 1. EMA 200 (Trend)
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 2. ATR (Volatility)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # 3. Volume Average
    df['vol_avg'] = df['volume'].rolling(20).mean()

    # 4. FVG Detection (3-Candle Pattern)
    # Bull FVG: Low of Candle[0] > High of Candle[-2] (Gap Up)
    # We shift future data to align with current row for "historical" tagging
    # Note: For live analysis, we look at completed patterns.
    
    # Calculate gap boundaries for the PREVIOUS COMPLETED candle structure
    # A gap formed at index `i` is defined by `i` (current), `i-1`, `i-2`.
    # The gap exists between `low[i]` and `high[i-2]` (Bullish).
    
    df['fvg_bull_top'] = df['low']
    df['fvg_bull_btm'] = df['high'].shift(2)
    df['is_bull_fvg'] = df['fvg_bull_top'] > df['fvg_bull_btm']
    
    df['fvg_bear_top'] = df['low'].shift(2)
    df['fvg_bear_btm'] = df['high']
    df['is_bear_fvg'] = df['fvg_bear_btm'] < df['fvg_bear_top']

    return df

def check_killzone(timestamp):
    h = timestamp.hour
    if 7 <= h < 10: return "London Open"
    if 12 <= h < 15: return "New York"
    if 15 <= h < 16: return "Silver Bullet"
    return None

def analyze_market(df, symbol):
    """
    Analyzes the LATEST COMPLETED candle (row -2) against history.
    """
    if len(df) < 201: return None

    # Get the completed candle (previous row, index -2)
    curr = df.iloc[-2]
    
    # Ensure we haven't processed this timestamp already
    last_ts = st.session_state.last_processed_ts.get(symbol)
    if last_ts == curr.name:
        return None
    
    kz = check_killzone(curr.name)
    if not kz: return None

    # --- STRATEGY PARAMETERS ---
    MIN_RR = 2.0
    RISK_PCT = 0.01

    # --- HTF CONTEXT (Simulated using 4H resampling) ---
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    if len(df_4h) < 2: return None
    htf_candle = df_4h.iloc[-2] 
    ph, pl = htf_candle['high'], htf_candle['low']

    # --- ENTRY LOGIC ---
    signal = None
    
    # 1. Trend & Volume Filter
    bull_trend = curr['close'] > curr['ema200']
    bear_trend = curr['close'] < curr['ema200']
    vol_spike = curr['volume'] > curr['vol_avg'] * 1.5

    # 2. FVG Confirmation (Did the completed candle CREATE or REJECT off an FVG?)
    # For this strategy, we enter if the completed candle CREATED a strong FVG in trend direction
    created_bull_fvg = curr['is_bull_fvg']
    created_bear_fvg = curr['is_bear_fvg']

    # LONG SETUP
    if bull_trend and vol_spike and created_bull_fvg:
        sl = curr['fvg_bull_btm'] # Stop loss below the gap
        tp = ph # Target HTF High
        entry = curr['close']
        
        if (entry - sl) > 0:
            rr = (tp - entry) / (entry - sl)
            if rr >= MIN_RR:
                size = st.session_state.equity * RISK_PCT / (entry - sl)
                signal = {
                    "dir": "BULLISH", "entry": entry, "sl": sl, 
                    "tp": tp, "rr": rr, "size": size, "desc": f"{kz} + FVG Create"
                }

    # SHORT SETUP
    if bear_trend and vol_spike and created_bear_fvg:
        sl = curr['fvg_bear_top'] # Stop loss above the gap
        tp = pl # Target HTF Low
        entry = curr['close']
        
        if (sl - entry) > 0:
            rr = (entry - tp) / (sl - entry)
            if rr >= MIN_RR:
                size = st.session_state.equity * RISK_PCT / (sl - entry)
                signal = {
                    "dir": "BEARISH", "entry": entry, "sl": sl, 
                    "tp": tp, "rr": rr, "size": size, "desc": f"{kz} + FVG Create"
                }
    
    # Update processed timestamp
    st.session_state.last_processed_ts[symbol] = curr.name
    return signal

def log_trade(symbol, sig):
    """Logs the trade, updates equity, and sends alert"""
    
    trade = {
        "time": datetime.now().isoformat(),
        "symbol": symbol,
        "direction": sig['dir'],
        "entry": round(sig['entry'], 2),
        "sl": round(sig['sl'], 2),
        "tp": round(sig['tp'], 2),
        "rr": round(sig['rr'], 2),
        "size": round(sig['size'], 4),
        "desc": sig['desc']
    }
    
    # 1. Save to JSON
    with open(TRADE_LOG, "r+") as f:
        try:
            trades = json.load(f)
        except:
            trades = []
        trades.append(trade)
        f.seek(0); f.truncate(); json.dump(trades, f, indent=2)
    
    # 2. Update Equity Log
    with open(EQUITY_LOG, "a") as f:
        f.write(f"{datetime.now().isoformat()},{st.session_state.equity:.0f},{len(trades)}\n")
    
    # 3. Send Telegram Alert
    msg = (
        f"🚨 *{sig['dir']} SIGNAL DETECTED* 🚨\n"
        f"Symbol: *{symbol}*\n"
        f"Entry: {sig['entry']}\n"
        f"SL: {sig['sl']} | TP: {sig['tp']}\n"
        f"R:R: {sig['rr']:.2f}\n"
        f"Reason: {sig['desc']}"
    )
    send_telegram_alert(msg)
        
    return trade

# ============================= DASHBOARD =============================
st.title("🚀 UltimateCRT Bot - Live Monitor")
st.markdown(f"**Status:** {'🟢 RUNNING' if st.session_state.running else '🔴 STOPPED'} | **Telegram:** {'✅ ON' if ENABLE_TELEGRAM else '❌ OFF'}")

# Control Panel
if st.button("▶ Start / ⏹ Stop", type="primary"):
    st.session_state.running = not st.session_state.running
    st.rerun()

# Metrics
try:
    with open(TRADE_LOG, 'r') as f:
        trade_history = json.load(f)
        count = len(trade_history)
except:
    count = 0
    trade_history = []

m1, m2, m3 = st.columns(3)
m1.metric("Account Balance", f"${st.session_state.equity:,.2f}")
m2.metric("Signals Generated", count)
m3.metric("Active Killzone", check_killzone(datetime.now()) or "None")

# Main Execution Loop
if st.session_state.running:
    status_container = st.container()
    
    with status_container:
        st.write("---")
        cols = st.columns(len(SYMBOLS))
        
        for idx, sym in enumerate(SYMBOLS):
            # 1. Fetch
            df = fetch_klines(sym)
            
            if not df.empty and len(df) > 200:
                # 2. Calculate
                df = calculate_indicators(df)
                
                # 3. Analyze
                sig = analyze_market(df, sym)
                
                # 4. Log if signal found
                if sig:
                    t = log_trade(sym, sig)
                    st.toast(f"🚀 SIGNAL: {sym} {sig['dir']}")
                
                # Display current status
                curr = df.iloc[-1]
                # Check if current price is inside previous candle's FVG
                prev = df.iloc[-2]
                in_bull_fvg = prev['is_bull_fvg'] and (prev['fvg_bull_btm'] <= curr['close'] <= prev['fvg_bull_top'])
                in_bear_fvg = prev['is_bear_fvg'] and (prev['fvg_bear_top'] <= curr['close'] <= prev['fvg_bear_btm'])
                
                status_fvg = "No FVG"
                if in_bull_fvg: status_fvg = "Inside BULL FVG"
                elif in_bear_fvg: status_fvg = "Inside BEAR FVG"

                trend_color = "green" if curr['close'] > curr['ema200'] else "red"
                
                cols[idx].metric(label=sym, value=f"${curr['close']:,.2f}", delta_color="off")
                cols[idx].caption(f"Trend: :{trend_color}[{'BULL' if trend_color=='green' else 'BEAR'}]")
                if "Inside" in status_fvg:
                     cols[idx].warning(f"⚠️ {status_fvg}")
                else:
                     cols[idx].info(f"Structure: {status_fvg}")
            
            else:
                cols[idx].warning("Loading...")

    # Auto-refresh logic
    time.sleep(10)
    st.rerun()

# Reporting Section
st.write("---")
st.subheader("Signal History")
if count > 0:
    hist_df = pd.DataFrame(trade_history)
    hist_df['time'] = pd.to_datetime(hist_df['time']).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(hist_df.sort_index(ascending=False), use_container_width=True)

    if os.path.exists(EQUITY_LOG):
        eq_df = pd.read_csv(EQUITY_LOG)
        if not eq_df.empty:
            fig = px.line(eq_df, x="timestamp", y="equity", title="Account Growth")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No signals generated yet.")
