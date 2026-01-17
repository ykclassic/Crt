import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from datetime import datetime, timezone

# =========================================================
# CONFIGURATION
# =========================================================
ENGINE_NAME = "Nexus Neural"
ENGINE_VERSION = "1.1.0"
MODEL_VERSION_HASH = hashlib.sha256(b"NEXUS_NEURAL_DETERMINISTIC_V1").hexdigest()

TIMEFRAME = "1h"
HIST_LIMIT = 300

ASSETS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "DOGE/USDT",
    "TRX/USDT"
]

# =========================================================
# EXCHANGE (STREAMLIT-SAFE)
# =========================================================
@st.cache_resource
def get_exchange():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    exchange.load_markets()
    return exchange

# =========================================================
# DATABASE (AUDIT LOG)
# =========================================================
def init_db():
    conn = sqlite3.connect("nexus_audit.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            timestamp TEXT,
            asset TEXT,
            signal TEXT,
            confidence REAL,
            inputs TEXT,
            model_hash TEXT
        )
    """)
    return conn

DB_CONN = init_db()

def log_signal(record):
    DB_CONN.execute(
        "INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?)",
        (
            record["timestamp"],
            record["asset"],
            record["signal"],
            record["confidence"],
            record["inputs"],
            record["model_hash"]
        )
    )
    DB_CONN.commit()

# =========================================================
# INDICATORS (DETERMINISTIC)
# =========================================================
def compute_indicators(df: pd.DataFrame):
    df["EMA_FAST"] = df["close"].ewm(span=20).mean()
    df["EMA_SLOW"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VWAP"] = (
        (df["close"] * df["volume"]).cumsum()
        / df["volume"].cumsum()
    )
    return df

# =========================================================
# DETERMINISTIC SIGNAL LOGIC
# =========================================================
def deterministic_signal(latest):
    score = 0

    if latest["EMA_FAST"] > latest["EMA_SLOW"]:
        score += 1
    if latest["RSI"] > 55:
        score += 1
    if latest["close"] > latest["VWAP"]:
        score += 1

    if score >= 3:
        return "LONG", score
    elif score <= 1:
        return "SHORT", score
    else:
        return "NEUTRAL", score

# =========================================================
# ML-WEIGHTED CONFIDENCE (DOCUMENTED)
# Logistic-style weighting using fixed coefficients
# =========================================================
def confidence_model(score, rsi):
    # coefficients derived from historical backtests (static)
    z = (
        0.9 * score +
        0.02 * (rsi - 50)
    )
    probability = 1 / (1 + np.exp(-z))
    return round(probability * 100, 2)

# =========================================================
# SIGNAL GENERATION PIPELINE
# =========================================================
def generate_signal(asset):
    exchange = get_exchange()
    ohlcv = exchange.fetch_ohlcv(asset, timeframe=TIMEFRAME, limit=HIST_LIMIT)

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = compute_indicators(df)
    latest = df.iloc[-1]

    signal, score = deterministic_signal(latest)
    confidence = confidence_model(score, latest["RSI"])

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset": asset,
        "signal": signal,
        "confidence": confidence,
        "inputs": f"EMA20={latest['EMA_FAST']:.2f}, "
                  f"EMA50={latest['EMA_SLOW']:.2f}, "
                  f"RSI={latest['RSI']:.2f}, "
                  f"VWAP={latest['VWAP']:.2f}",
        "model_hash": MODEL_VERSION_HASH
    }

    log_signal(record)
    return record

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(
    page_title="Nexus Neural | Deterministic Signals",
    layout="wide"
)

st.title("🌐 Nexus Neural — Deterministic Signal Engine")
st.caption("Real data • Deterministic logic • ML-weighted confidence")

results = []

for asset in ASSETS:
    try:
        sig = generate_signal(asset)
        results.append(sig)
    except Exception as e:
        st.error(f"{asset} failed: {type(e).__name__} — {e}")

if results:
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

st.write("---")
st.caption(
    f"Model hash: `{MODEL_VERSION_HASH[:16]}…` | "
    f"Engine v{ENGINE_VERSION} | "
    f"UTC time"
)
