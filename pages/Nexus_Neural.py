# =========================================================
# Nexus Neural v3.2 — Regime-Aware Multi-TF Engine
# =========================================================

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import threading
import time
import hashlib
from datetime import datetime, timezone
import plotly.graph_objects as go

# =========================================================
# ENGINE CONFIG
# =========================================================
ENGINE_VERSION = "3.2.0"

TIMEFRAMES = {
    "1h": 300,
    "4h": 300,
    "1d": 300
}

ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT",
    "ADA/USDT","LINK/USDT","DOGE/USDT","TRX/USDT"
]

EXCHANGES = {
    "XT": "xt",
    "Gate.io": "gate"
}

# =========================================================
# MODEL REGISTRY (DETERMINISTIC)
# =========================================================
MODEL = {
    "weights": np.array([1.2, 1.0, 0.8, 0.6, 0.4]),
    "bias": -2.0
}

MODEL_HASH = hashlib.sha256(str(MODEL).encode()).hexdigest()

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Nexus Neural v3.2",
    layout="wide"
)

st.title("🌐 Nexus Neural — Regime-Aware Signal Engine")
st.caption("Multi-TF consensus • Deterministic logic • Production-safe")

# =========================================================
# DATABASE
# =========================================================
def init_db():
    conn = sqlite3.connect("nexus_audit.db", check_same_thread=False)
    conn.execute("""
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
    conn.commit()
    return conn

DB = init_db()

def log_signal(r):
    DB.execute(
        "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        tuple(r.values())
    )
    DB.commit()

# =========================================================
# EXCHANGE
# =========================================================
@st.cache_resource
def load_exchange(eid):
    ex = getattr(ccxt, eid)({"enableRateLimit": True})
    ex.load_markets()
    return ex

# =========================================================
# INDICATORS
# =========================================================
def indicators(df):
    df["EMA20"] = df.close.ewm(span=20).mean()
    df["EMA50"] = df.close.ewm(span=50).mean()
    delta = df.close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    df["ATR"] = (df.high - df.low).rolling(14).mean()
    df["VWAP"] = (df.close * df.volume).cumsum() / df.volume.cumsum()
    return df.dropna()

# =========================================================
# REGIME DETECTION
# =========================================================
def detect_regime(row):
    ema_slope = row.EMA20 - row.EMA50
    vol_ratio = row.ATR / row.close

    if vol_ratio > 0.025:
        return "VOLATILE"
    if abs(ema_slope) < 0.001 * row.close:
        return "RANGING"
    return "TRENDING"

# =========================================================
# SIGNAL ENGINE
# =========================================================
def signal_logic(row):
    features = np.array([
        row.EMA20 > row.EMA50,
        row.RSI > 55,
        row.close > row.VWAP,
        row.RSI < 75,
        row.ATR / row.close < 0.03
    ], dtype=float)

    z = np.dot(MODEL["weights"], features) + MODEL["bias"]
    prob = 1 / (1 + np.exp(-z))
    confidence = prob * 100

    if prob > 0.65:
        signal = "LONG"
    elif prob < 0.35:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    entry = row.close
    stop = entry - row.ATR if signal == "LONG" else entry + row.ATR
    take = entry + 2 * row.ATR if signal == "LONG" else entry - 2 * row.ATR

    return signal, round(confidence, 2), entry, stop, take

# =========================================================
# MULTI-TF CONSENSUS
# =========================================================
def consensus(signals):
    votes = [s for s in signals if s != "NEUTRAL"]
    if votes.count("LONG") >= 2:
        return "LONG"
    if votes.count("SHORT") >= 2:
        return "SHORT"
    return "NEUTRAL"

# =========================================================
# UI
# =========================================================
exchange_name = st.selectbox("Exchange", list(EXCHANGES))
exchange = load_exchange(EXCHANGES[exchange_name])

results = []

for asset in ASSETS:
    tf_signals = []
    tf_data = {}

    for tf, limit in TIMEFRAMES.items():
        if asset not in exchange.markets:
            continue

        ohlcv = exchange.fetch_ohlcv(asset, tf, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["ts","open","high","low","close","volume"]
        )
        df = indicators(df)
        last = df.iloc[-1]

        regime = detect_regime(last)
        sig, conf, entry, stop, take = signal_logic(last)
        tf_signals.append(sig)

        tf_data[tf] = (df, sig, regime, conf)

    final_signal = consensus(tf_signals)

    if final_signal != "NEUTRAL":
        avg_conf = np.mean([v[3] for v in tf_data.values()])
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange_name,
            "asset": asset,
            "timeframe": "1H/4H/1D",
            "regime": detect_regime(tf_data["1h"][0].iloc[-1]),
            "signal": final_signal,
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "take": round(take, 4),
            "confidence": round(avg_conf, 2),
            "model_hash": MODEL_HASH
        }
        log_signal(record)
        results.append(record)

# =========================================================
# DISPLAY
# =========================================================
st.subheader("📡 Regime-Aware Multi-TF Signals")
df_res = pd.DataFrame(results)
st.dataframe(df_res, use_container_width=True)

# =========================================================
# VISUALIZATION
# =========================================================
if not df_res.empty:
    st.subheader("📈 Price & Regime Visualization")

    asset = df_res.iloc[0]["asset"]
    df = tf_data["1h"][0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.ts, y=df.close, name="Price"))
    fig.add_trace(go.Scatter(x=df.ts, y=df.EMA20, name="EMA20"))
    fig.add_trace(go.Scatter(x=df.ts, y=df.EMA50, name="EMA50"))

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Nexus Neural v{ENGINE_VERSION} | "
    f"Model hash: {MODEL_HASH[:12]}… | "
    "Regime-aware • Multi-TF • Deterministic"
)
