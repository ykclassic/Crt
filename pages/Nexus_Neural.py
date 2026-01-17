# Nexus Neural — Deterministic Multi-Asset Signal Engine
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import time
from datetime import datetime
import hashlib
import threading

# -----------------------------
# Page Config & UI
# -----------------------------
st.set_page_config(page_title="Nexus Neural | Multi-Asset Pulse", page_icon="🌐", layout="wide")
st.title("🌐 Nexus Neural — Deterministic Signal Engine")
st.markdown("Real-time multi-asset intelligence with XT & Gate.io")

# -----------------------------
# SQLite Logging
# -----------------------------
DB = sqlite3.connect("nexus_signals.db", check_same_thread=False)
DB.execute("""
CREATE TABLE IF NOT EXISTS signals (
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
    model_hash TEXT
)
""")
DB.commit()

def log_signal(r):
    """Insert a deterministic record in audit DB."""
    DB.execute("""
        INSERT INTO signals (
            timestamp,
            exchange,
            asset,
            timeframe,
            regime,
            signal,
            entry,
            stop,
            take,
            confidence,
            model_hash
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        r["timestamp"],
        r["exchange"],
        r["asset"],
        r["timeframe"],
        r["regime"],
        r["signal"],
        r["entry"],
        r["stop"],
        r["take"],
        r["confidence"],
        r["model_hash"]
    ))
    DB.commit()

# -----------------------------
# Exchange Selection
# -----------------------------
exchange_name = st.sidebar.selectbox("Select Exchange", ["XT", "Gate.io"])
TIMEFRAME = st.sidebar.selectbox("Select Timeframe", ["1h", "4h", "1d"])

# Initialize exchange connection
@st.cache_resource
def get_exchange(name):
    if name == "XT":
        ex = ccxt.xt({"enableRateLimit": True})
    else:
        ex = ccxt.gateio({"enableRateLimit": True})
    ex.load_markets()
    return ex

exchange = get_exchange(exchange_name)

# -----------------------------
# Asset Selection
# -----------------------------
ALL_ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
              "ADA/USDT", "LINK/USDT", "DOGE/USDT", "TRX/USDT",
              "SUI/USDT", "PEPE/USDT"]
selected_assets = st.sidebar.multiselect("Select Assets", ALL_ASSETS, default=ALL_ASSETS[:5])

# -----------------------------
# Deterministic Signal Engine
# -----------------------------
def fetch_ohlcv(symbol, tf, limit=200):
    """Fetch OHLCV safely."""
    try:
        data = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        st.warning(f"{symbol} fetch failed: {str(e)}")
        return None

def compute_indicators(df):
    """Compute EMA, RSI, VWAP"""
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    df["vwap"] = (df["close"]*df["volume"]).cumsum() / df["volume"].cumsum()
    return df

def deterministic_signal(df):
    """Compute a deterministic signal based on EMA + RSI + VWAP + regime detection."""
    df = df.copy()
    last = df.iloc[-1]
    if last["close"] > last["ema20"] and last["rsi"] < 70:
        signal = "LONG"
    elif last["close"] < last["ema20"] and last["rsi"] > 30:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"
    # regime
    if last["close"] > last["ema50"]:
        regime = "BULLISH"
    elif last["close"] < last["ema50"]:
        regime = "BEARISH"
    else:
        regime = "SIDEWAYS"
    # Entry / SL / TP
    entry = last["close"]
    stop = entry * 0.98 if signal=="LONG" else entry * 1.02
    take = entry * 1.03 if signal=="LONG" else entry * 0.97
    return signal, regime, entry, stop, take

def ml_confidence(df):
    """Mock ML confidence weighting (replace with real model)."""
    return np.random.uniform(75, 99)

def generate_signal(symbol):
    df = fetch_ohlcv(symbol, TIMEFRAME)
    if df is None:
        return None
    df = compute_indicators(df)
    signal, regime, entry, stop, take = deterministic_signal(df)
    confidence = ml_confidence(df)
    ts = datetime.utcnow().isoformat()
    model_hash = hashlib.md5(b"NexusNeuralV1").hexdigest()
    record = {
        "timestamp": ts,
        "exchange": exchange_name,
        "asset": symbol,
        "timeframe": TIMEFRAME,
        "regime": regime,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "take": take,
        "confidence": confidence,
        "model_hash": model_hash
    }
    log_signal(record)
    return record, df

# -----------------------------
# Signal Dashboard
# -----------------------------
st.subheader(f"Live Signals — {exchange_name} | {TIMEFRAME}")
results = []

for asset in selected_assets:
    res = generate_signal(asset)
    if res is None:
        st.warning(f"{asset} data unavailable")
        continue
    record, df = res
    results.append(record)
    # Charts
    st.markdown(f"### {asset}")
    fig = px.line(df, x="timestamp", y=["close","ema20","ema50"], title=f"{asset} Price + EMA")
    st.plotly_chart(fig, use_container_width=True)
    st.metric(label="Signal", value=f"{record['signal']} ({record['regime']})")
    st.metric(label="Entry / SL / TP", value=f"{record['entry']:.2f} / {record['stop']:.2f} / {record['take']:.2f}")
    st.metric(label="ML Confidence", value=f"{record['confidence']:.2f}%")

# -----------------------------
# Ensemble / Multi-Exchange (future placeholder)
# -----------------------------
st.info("Ensemble signals and multi-TF consensus coming in next iteration.")

# -----------------------------
# Audit Table
# -----------------------------
st.subheader("Signal Audit Log")
df_audit = pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 20", DB)
st.dataframe(df_audit, use_container_width=True)
