import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime
import plotly.express as px
from backtesting import Strategy
from backtesting.lib import resample_apply

# ============================= CONFIGURATION =============================
# Replace with your actual Discord Webhook URL
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_HERE"

BASE_URL = "https://sapi.xt.com/v4/public/kline"
SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT"]
LOG_DIR = "performance_logs"
TRADE_LOG = os.path.join(LOG_DIR, "trades.json")
EQUITY_LOG = os.path.join(LOG_DIR, "equity.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# Initialize Files if missing
if not os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, 'w') as f: json.dump([], f)
if not os.path.exists(EQUITY_LOG):
    with open(EQUITY_LOG, 'w') as f: f.write("timestamp,equity,trades_count\n")

# Session State
if 'equity' not in st.session_state:
    st.session_state.equity = 100000.0
if 'running' not in st.session_state:
    st.session_state.running = False

# ============================= NOTIFICATIONS =============================
def send_discord_alert(content):
    if DISCORD_WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_HERE":
        return
    payload = {"content": content}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        st.error(f"Discord Alert Failed: {e}")

# ============================= TECHNICAL INDICATORS =============================
def ema(series, period): 
    return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_fvg(df):
    bullish = df['low'] > df['high'].shift(2)
    bearish = df['high'] < df['low'].shift(2)
    return df['low'].where(bullish).ffill(), df['high'].where(bearish).ffill()

def detect_swings(df, length=5):
    highs = df['high'].rolling(window=length*2+1, center=True).max() == df['high']
    lows = df['low'].rolling(window=length*2+1, center=True).min() == df['low']
    return df['high'][highs], df['low'][lows]

def detect_ob(df):
    bull_ob = (df['close'] > df['open']) & (df['close'].shift(-1) > df['high'])
    bear_ob = (df['close'] < df['open']) & (df['close'].shift(-1) < df['low'])
    return df['low'][bull_ob.shift(1).fillna(False)].ffill(), df['high'][bear_ob.shift(1).fillna(False)].ffill()

# ============================= DATA FETCH =============================
def fetch_klines(symbol: str, limit: int = 500):
    params = {"symbol": symbol, "interval": "5min", "limit": limit}
    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("rc") == 0:
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df.set_index("timestamp", inplace=True)
            return df.astype(float).sort_index()
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

# ============================= LOGGING =============================
def log_trade(direction, sym, entry, sl, tp, rr, size, session):
    trade = {
        "time": datetime.now().isoformat(),
        "symbol": sym,
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "rr": round(rr, 1),
        "size": round(size, 4),
        "session": session
    }
    
    # Save Trade
    with open(TRADE_LOG, "r") as f:
        trades = json.load(f)
    trades.append(trade)
    with open(TRADE_LOG, "w") as f:
        json.dump(trades, f, indent=2)
    
    # Update Equity Log (Based on risk amount)
    with open(EQUITY_LOG, "a") as f:
        f.write(f"{datetime.now().isoformat()},{st.session_state.equity:.0f},{len(trades)}\n")

    # Send Notification
    msg = f"🚀 **{direction} SIGNAL: {sym}**\nEntry: {entry}\nSL: {sl}\nTP: {tp}\nRR: {rr}\nSession: {session}"
    send_discord_alert(msg)

# ============================= STRATEGY ENGINE =============================
class UltimateCRT(Strategy):
    ema_period = 200
    risk_percent = 1.0
    min_rr = 2.0
    sl_buffer_pct = 0.1

    def init(self):
        self.ema200 = self.I(ema, self.data.Close, self.ema_period)
        self.fvg_up, self.fvg_down = self.I(detect_fvg, self.data.df)
        self.swing_h, self.swing_l = self.I(detect_swings, self.data.df)
        self.ob_bull, self.ob_bear = self.I(detect_ob, self.data.df)
        self.vol_avg = self.I(lambda x: pd.Series(x).rolling(20).mean(), self.data.volume)

    def killzone(self):
        h = self.data.index[-1].hour
        if 7 <= h < 10: return "London Open"
        if 12 <= h < 15: return "New York"
        if 15 <= h < 16: return "Silver Bullet"
        return "Outside Kill Zone"

    def next(self):
        if len(self.data) < 200: return
        
        p = self.data.Close[-1]
        l, h = self.data.Low[-1], self.data.High[-1]
        session = self.killzone()
        
        if session == "Outside Kill Zone": return

        # Confluence Signals
        vol_spike = self.data.volume[-1] > self.vol_avg[-1] * 1.5
        trend_up = p > self.ema200[-1]
        trend_down = p < self.ema200[-1]
        
        fvg_bull = not np.isnan(self.fvg_up[-1]) and p >= self.fvg_up[-1]
        ob_bull = not np.isnan(self.ob_bull[-1]) and p >= self.ob_bull[-1]
        
        fvg_bear = not np.isnan(self.fvg_down[-1]) and p <= self.fvg_down[-1]
        ob_bear = not np.isnan(self.ob_bear[-1]) and p <= self.ob_bear[-1]

        # BULLISH SETUP (Real Only)
        if trend_up and vol_spike and (fvg_bull and ob_bull):
            if not self.position:
                sl = l * (1 - self.sl_buffer_pct / 100)
                tp = p + (p - sl) * 2.5
                rr = (tp - p) / (p - sl)
                if rr >= self.min_rr:
                    size = (st.session_state.equity * 0.01) / (p - sl)
                    # self.buy(sl=sl, tp=tp, size=size) # Uncomment for auto-backtest execution
                    log_trade("BULLISH", "CRYPTO", p, sl, tp, rr, size, session)

        # BEARISH SETUP (Real Only)
        elif trend_down and vol_spike and (fvg_bear and ob_bear):
            if not self.position:
                sl = h * (1 + self.sl_buffer_pct / 100)
                tp = p - (sl - p) * 2.5
                rr = (p - tp) / (sl - p)
                if rr >= self.min_rr:
                    size = (st.session_state.equity * 0.01) / (sl - p)
                    log_trade("BEARISH", "CRYPTO", p, sl, tp, rr, size, session)

# ============================= DASHBOARD UI =============================
st.set_page_config(page_title="UltimateCRT - Institutional", layout="wide")
st.title("🚀 UltimateCRT Institutional Bot")

# Load real data for metrics
try:
    with open(TRADE_LOG, 'r') as f:
        all_trades = json.load(f)
except:
    all_trades = []

col1, col2, col3 = st.columns(3)
col1.metric("Account Equity", f"${st.session_state.equity:,.2f}")
col2.metric("Real Signals Found", len(all_trades))
col3.metric("Bot Status", "RUNNING" if st.session_state.running else "IDLE")

if st.button("START LIVE SCAN" if not st.session_state.running else "STOP SCAN", type="primary"):
    st.session_state.running = not st.session_state.running
    st.rerun()

placeholder = st.empty()

if st.session_state.running:
    while st.session_state.running:
        with placeholder.container():
            st.write(f"Last Scan: {datetime.now().strftime('%H:%M:%S')}")
            cols = st.columns(len(SYMBOLS))
            
            for idx, sym in enumerate(SYMBOLS):
                df = fetch_klines(sym)
                if not df.empty:
                    last_price = df['close'].iloc[-1]
                    # Check for signal on latest candle
                    strat = UltimateCRT(data=df)
                    strat.init()
                    strat.next()
                    
                    with cols[idx]:
                        st.metric(sym, f"${last_price:,.2f}")
                time.sleep(1) # Prevent API rate limit
            
            st.info("Searching for Institutional Confluence (EMA + FVG + OB + Vol Spike)...")
        
        time.sleep(60) # Wait for next 5m candle cycle
        st.rerun()

# Performance Table
if all_trades:
    st.markdown("---")
    st.subheader("Signal History")
    st.dataframe(pd.DataFrame(all_trades).tail(20), use_container_width=True)
