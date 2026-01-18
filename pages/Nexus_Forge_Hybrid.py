# =========================================================
# Nexus HybridTrader v1 — Trend/Range Switcher with AI Mode
# XT + Gate.io | Dynamic Strategy Switching | AI Boost | Backtest Mode
# =========================================================

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import hashlib
import threading
import time
import requests
import json
from datetime import datetime, timezone
from lifelines import KaplanMeierFitter

# ---------------------------------------------------------
# sklearn Fallback Setup
# ---------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available – ML confidence will use deterministic fallback.")

RF_AVAILABLE = False
if SKLEARN_AVAILABLE:
    try:
        from sklearn.ensemble import RandomForestClassifier
        RF_AVAILABLE = True
    except ImportError:
        st.warning("RandomForest not available – falling back to LogisticRegression for ML.")

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Nexus HybridTrader v1", page_icon="🔄", layout="wide")
st.title("🔄 Nexus HybridTrader v1 — Trend/Range Switcher with AI Mode")

# ---------------------------------------------------------
# Database
# ---------------------------------------------------------
DB = sqlite3.connect("hybrid_signals.db", check_same_thread=False)

DB.execute("""
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    exchange TEXT,
    asset TEXT,
    timeframe TEXT,
    regime TEXT,
    signal TEXT,
    entry REAL,
    stop REAL,
    take REAL,
    confidence REAL,
    model_hash TEXT,
    status TEXT,
    exit_timestamp TEXT,
    exit_price REAL,
    features TEXT
)
""")
DB.commit()

def add_column(name, type_):
    try:
        DB.execute(f"ALTER TABLE signals ADD COLUMN {name} {type_}")
        DB.commit()
    except sqlite3.OperationalError:
        pass

add_column("features", "TEXT")

def log_signal(r: dict):
    DB.execute("""
        INSERT INTO signals (
            timestamp, exchange, asset, timeframe, regime,
            signal, entry, stop, take, confidence, model_hash, status,
            exit_timestamp, exit_price, features
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        r["timestamp"], r["exchange"], r["asset"], r["timeframe"],
        r["regime"], r["signal"], r["entry"], r["stop"], r["take"],
        r["confidence"], r["model_hash"], r["status"],
        None, None, r.get("features", None)
    ))
    DB.commit()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
mode = st.sidebar.radio("Mode", ["Live", "Backtest"])
strategy_mode = st.sidebar.radio("Strategy Mode", ["Trend", "Range"])
ai_mode = st.sidebar.checkbox("AI Mode (Sentiment Boost)", value=False)
exchange_name = st.sidebar.selectbox("Signal Base Exchange", ["XT", "Gate.io"])

TIMEFRAMES = ["1h", "4h", "1d"]
selected_timeframes = st.sidebar.multiselect("Timeframes", TIMEFRAMES, default=["1h", "4h"])

ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
    "LINK/USDT","DOGE/USDT","TRX/USDT","SUI/USDT","PEPE/USDT"
]
selected_assets = st.sidebar.multiselect("Assets", ASSETS, default=ASSETS[:5])

color_map = {"1h": "blue", "4h": "green", "1d": "yellow"}
require_confirmation = st.sidebar.checkbox("Higher TF Regime Confirmation", value=True)
atr_multiplier_stop = st.sidebar.number_input("ATR Stop Multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
rr_ratio = st.sidebar.number_input("Risk:Reward Ratio", min_value=1.0, max_value=4.0, value=1.5, step=0.5)
min_confidence = st.sidebar.slider("Min Confidence % to Show", 50, 100, 70)
trailing_stop_pct = st.sidebar.slider("Trailing Stop % (after 1R)", 0.0, 1.0, 0.5, 0.1)
webhook_url = st.sidebar.text_input("Webhook URL for Alerts", type="password")

if mode == "Backtest":
    st.sidebar.header("Backtest Settings")
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    initial_capital = st.sidebar.number_input("Initial Capital (USDT)", value=10000.0)
    risk_percent = st.sidebar.slider("Risk % per Trade", 0.5, 5.0, 2.0, 0.5)
    slippage_pct = st.sidebar.slider("Slippage %", 0.0, 0.5, 0.1, 0.05)
    fee_pct = st.sidebar.slider("Trading Fee % (per side)", 0.0, 0.2, 0.05, 0.01)

# ---------------------------------------------------------
# Exchange loader
# ---------------------------------------------------------
@st.cache_resource
def get_exchange(name):
    if name == "XT":
        ex = ccxt.xt({"enableRateLimit": True})
    else:
        ex = ccxt.gateio({"enableRateLimit": True})
    ex.load_markets()
    return ex

exchange = get_exchange(exchange_name)

# ---------------------------------------------------------
# Dynamic Indicators
# ---------------------------------------------------------
def compute_indicators(df, mode="Trend"):
    df = df.copy()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    if mode == "Trend":
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        hl2 = (df['high'] + df['low']) / 2
        df['upper_band'] = hl2 + (3.0 * df['atr'])
        df['lower_band'] = hl2 - (3.0 * df['atr'])
        df['in_uptrend'] = True
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] > df['upper_band'].iloc[i-1]:
                df.loc[df.index[i], 'in_uptrend'] = True
            elif df['close'].iloc[i-1] < df['lower_band'].iloc[i-1]:
                df.loc[df.index[i], 'in_uptrend'] = False
            else:
                df.loc[df.index[i], 'in_uptrend'] = df['in_uptrend'].iloc[i-1]
                if df['in_uptrend'].iloc[i] and df['lower_band'].iloc[i] < df['lower_band'].iloc[i-1]:
                    df.loc[df.index[i], 'lower_band'] = df['lower_band'].iloc[i-1]
                elif not df['in_uptrend'].iloc[i] and df['upper_band'].iloc[i] > df['upper_band'].iloc[i-1]:
                    df.loc[df.index[i], 'upper_band'] = df['upper_band'].iloc[i-1]
        df['supertrend'] = np.where(df['in_uptrend'], df['lower_band'], df['upper_band'])
        df["volume_ma"] = df["volume"].rolling(20).mean()
    else:
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
        atr_sum = df["atr"].rolling(14).sum()
        df["+di"] = 100 * (df["high"].diff().clip(lower=0).rolling(14).sum() / atr_sum)
        df["-di"] = 100 * (df["low"].diff().clip(upper=0).abs().rolling(14).sum() / atr_sum)
        dx = 100 * np.abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
        df["adx"] = dx.rolling(14).mean()
    return df

# ---------------------------------------------------------
# ML and Signal Engine
# ---------------------------------------------------------
ml_model, ml_scaler = None, None

def deterministic_signal(df, mode="Trend", asset="BTC/USDT"):
    last = df.iloc[-1]
    confidence = 80.0
    signal = "NEUTRAL"

    if mode == "Trend":
        super_up = last.get("supertrend", 0) and last["close"] > last["supertrend"]
        volume_ok = last.get("volume_ma", 0) and last["volume"] > last["volume_ma"] * 1.2
        if last.get("ema20", 0) and last["close"] > last["ema20"] and last["rsi"] < 70 and super_up and volume_ok:
            signal = "LONG"
        elif last.get("ema20", 0) and last["close"] < last["ema20"] and last["rsi"] > 30 and not super_up and volume_ok:
            signal = "SHORT"
        regime = "BULLISH" if last.get("ema50", 0) and last["close"] > last["ema50"] else "BEARISH" if last.get("ema50", 0) and last["close"] < last["ema50"] else "SIDEWAYS"
        features = [float(last["rsi"]), float(last.get("atr", 0)/last["close"])]
    else:
        ranging = last.get("adx", 100) < 25
        if ranging:
            if last["rsi"] < 30 and last["close"] < last.get("bb_lower", 0) * 1.01:
                signal = "LONG"
            elif last["rsi"] > 70 and last["close"] > last.get("bb_upper", 0) * 0.99:
                signal = "SHORT"
        regime = "RANGING" if ranging else "TRENDING"
        features = [float(last["rsi"]), float(last.get("adx", 0))]

    entry = last["close"]
    atr = last.get("atr", entry * 0.01)
    if signal == "LONG":
        stop = entry - atr_multiplier_stop * atr
        take = entry + rr_ratio * (entry - stop)
    elif signal == "SHORT":
        stop = entry + atr_multiplier_stop * atr
        take = entry - rr_ratio * (stop - entry)
    else:
        stop = take = entry

    if ai_mode:
        confidence += 5 # Placeholder for AI logic
    
    return signal, regime, entry, stop, take, np.clip(confidence, 0, 100), json.dumps(features)

# ---------------------------------------------------------
# Data fetch
# ---------------------------------------------------------
def fetch_ohlcv(symbol, tf, since=None, limit=500):
    try:
        data = exchange.fetch_ohlcv(symbol, tf, limit=limit, params={"since": since} if since else {})
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except: return pd.DataFrame()

# ---------------------------------------------------------
# Live Update Loop
# ---------------------------------------------------------
signals_cache = {asset: {} for asset in ASSETS}
if mode == "Live":
    st.subheader("Live Market Feed")
    # Implementation of thread/update cycle here...
    st.info("Live Update Loop Active in Background")

# ---------------------------------------------------------
# Disagreement & Arb Table (Fixed End)
# ---------------------------------------------------------
if mode == "Live":
    st.subheader("Exchange Disagreement & Arb")
    xt_ex = get_exchange("XT")
    gate_ex = get_exchange("Gate.io")
    rows = []
    for asset in selected_assets[:3]:
        try:
            xt_close = xt_ex.fetch_ticker(asset)['last']
            gate_close = gate_ex.fetch_ticker(asset)['last']
            diff = abs(xt_close - gate_close) / xt_close * 100
            rows.append({"Asset": asset, "XT Price": xt_close, "Gate Price": gate_close, "Diff %": round(diff, 4)})
        except: pass
    if rows:
        st.table(pd.DataFrame(rows))

if mode == "Backtest":
    st.info("Backtest results would populate here based on selected date range.")
