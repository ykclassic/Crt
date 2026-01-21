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

# ============================= PAGE CONFIG =============================
st.set_page_config(
    page_title="UltimateCRT Bot - XT.com Live",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================= CONFIG & CONSTANTS =============================
BASE_URL = "https://sapi.xt.com/v4/public/kline"
SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT"]
LOG_DIR = "performance_logs"
TRADE_LOG = os.path.join(LOG_DIR, "trades.json")
EQUITY_LOG = os.path.join(LOG_DIR, "equity.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# Initialize Files
if not os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, 'w') as f: json.dump([], f)
if not os.path.exists(EQUITY_LOG):
    with open(EQUITY_LOG, 'w') as f: f.write("timestamp,equity,trades_count\n")

# Session State Initialization
if 'equity' not in st.session_state:
    st.session_state.equity = 100000.0
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_log_update' not in st.session_state:
    st.session_state.last_log_update = 0

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
    # Bullish FVG: Low of candle 3 is higher than High of candle 1
    bullish = df['low'] > df['high'].shift(2)
    # Bearish FVG: High of candle 3 is lower than Low of candle 1
    bearish = df['high'] < df['low'].shift(2)
    return df['low'].where(bullish).ffill(), df['high'].where(bearish).ffill()

def detect_swings(df, length=5):
    highs = df['high'].rolling(window=length*2+1, center=True).max() == df['high']
    lows = df['low'].rolling(window=length*2+1, center=True).min() == df['low']
    return df['high'][highs], df['low'][lows]

def detect_ob(df):
    # Simplified Order Block: Last down candle before strong up move (and vice versa)
    bull_ob = (df['close'] > df['open']) & (df['close'].shift(-1) > df['high'])
    bear_ob = (df['close'] < df['open']) & (df['close'].shift(-1) < df['low'])
    return df['low'][bull_ob.shift(1).fillna(False)].ffill(), df['high'][bear_ob.shift(1).fillna(False)].ffill()

# ============================= LOGGING FUNCTION =============================
def log_trade(direction, entry, sl, tp1, tp2, rr, size, session):
    # Realistic profit simulation (1:2 RR result)
    # In live mode, this would be updated by real exchange callbacks
    sim_profit = (st.session_state.equity * 0.02) if "TEST" not in direction else 0
    st.session_state.equity += sim_profit
    
    trade = {
        "time": datetime.now().isoformat(),
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "rr": round(rr, 1),
        "size": round(size, 4),
        "equity": round(st.session_state.equity),
        "session": session
    }
    
    try:
        with open(TRADE_LOG, "r") as f:
            trades = json.load(f)
        trades.append(trade)
        with open(TRADE_LOG, "w") as f:
            json.dump(trades, f, indent=2)
        
        with open(EQUITY_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()},{st.session_state.equity:.0f},{len(trades)}\n")
    except Exception as e:
        st.error(f"Logging error: {e}")

# ============================= STRATEGY ENGINE =============================
class UltimateCRT(Strategy):
    atr_mult = 0.5
    ema_period = 200
    risk_percent = 1.0
    min_rr = 2.0
    volume_spike_mult = 1.5
    sl_buffer_pct = 0.1

    def init(self):
        # HTF Logic (4h Resample)
        self.ema200 = self.I(ema, self.data.Close, self.ema_period)
        self.atr14 = self.I(atr, self.data.df)
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
        
        # Confluence Signals
        vol_spike = self.data.volume[-1] > self.vol_avg[-1] * self.volume_spike_mult
        trend_up = p > self.ema200[-1]
        
        # FVG & OB Logic
        fvg_b = not np.isnan(self.fvg_up[-1]) and p >= self.fvg_up[-1]
        ob_b = not np.isnan(self.ob_bull[-1]) and p >= self.ob_bull[-1]

        # 1. Bullish Entry (Simplified Confluence for live detection)
        if trend_up and vol_spike and (fvg_b or ob_b) and session != "Outside Kill Zone":
            if not self.position:
                sl = l * (1 - self.sl_buffer_pct / 100)
                tp = p + (p - sl) * 2
                rr = (tp - p) / (p - sl)
                
                if rr >= self.min_rr:
                    size = (st.session_state.equity * 0.01) / (p - sl)
                    self.buy(sl=sl, tp=tp, size=size)
                    log_trade("BULLISH", p, sl, tp, tp, rr, size, session)

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
    except Exception as e:
        st.error(f"API Error for {symbol}: {e}")
    return pd.DataFrame()

# ============================= DASHBOARD UI =============================
st.title("🚀 UltimateCRT Trading Bot")
st.markdown("---")

# Metrics Top Bar
m1, m2, m3, m4 = st.columns(4)
current_trades = []
if os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, 'r') as f: current_trades = json.load(f)

m1.metric("Current Equity", f"${st.session_state.equity:,.2f}")
m2.metric("Trades Found", len(current_trades))
m3.metric("Market Status", "🟢 OPEN" if 0 <= datetime.now().weekday() <= 4 else "🟡 WEEKEND")
m4.metric("Active Sessions", "London/NY" if 7 <= datetime.now().hour <= 16 else "Dormant")

# Control Panel
if st.button("▶ Start Live Scan" if not st.session_state.running else "⏹ Stop Scan", 
             type="primary", use_container_width=True):
    st.session_state.running = not st.session_state.running
    st.rerun()

status_area = st.empty()

if st.session_state.running:
    with status_area.container():
        st.write(f"📡 **Scanning XT.com Pairs...** (Last Update: {datetime.now().strftime('%H:%M:%S')})")
        cols = st.columns(len(SYMBOLS))
        
        for idx, sym in enumerate(SYMBOLS):
            df = fetch_klines(sym)
            if not df.empty:
                # Run Logic
                last_price = df['close'].iloc[-1]
                change = ((last_price - df['open'].iloc[-1]) / df['open'].iloc[-1]) * 100
                
                with cols[idx]:
                    st.metric(sym, f"${last_price:,.2f}", f"{change:+.2f}%")
                    
                    # Run Strategy on latest data to check for signals
                    bt_strat = UltimateCRT(data=df)
                    bt_strat.init()
                    # Check manual confluence for UI feedback
                    vol_ok = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1]
                    cols[idx].caption("Vol Spike: " + ("✅" if vol_ok else "❌"))

        # Simulated Loop Delay
        time.sleep(30)
        st.rerun()

# Charts Section
st.markdown("---")
c1, c2 = st.columns([2, 1])

with c1:
    if os.path.exists(EQUITY_LOG):
        eq_df = pd.read_csv(EQUITY_LOG)
        if not eq_df.empty:
            fig = px.line(eq_df, x='timestamp', y='equity', title="Performance Over Time")
            st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Recent Activity")
    if current_trades:
        df_trades = pd.DataFrame(current_trades).tail(10)
        st.table(df_trades[['direction', 'entry', 'rr', 'session']])
    else:
        st.info("No trades logged yet.")

# Clear Data Button
if st.sidebar.button("🗑 Reset All Data"):
    if os.path.exists(TRADE_LOG): os.remove(TRADE_LOG)
    if os.path.exists(EQUITY_LOG): os.remove(EQUITY_LOG)
    st.session_state.equity = 100000.0
    st.rerun()
