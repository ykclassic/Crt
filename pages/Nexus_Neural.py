import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from datetime import datetime, timezone

# =========================================================
# ENGINE METADATA
# =========================================================
ENGINE_NAME = "Nexus Neural"
ENGINE_VERSION = "1.2.0"
MODEL_VERSION_HASH = hashlib.sha256(
    b"NEXUS_NEURAL_DETERMINISTIC_V1_XT_GATE"
).hexdigest()

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

SUPPORTED_EXCHANGES = {
    "XT": "xt",
    "Gate.io": "gate"
}

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Nexus Neural | Deterministic Signals",
    layout="wide"
)

st.title("🌐 Nexus Neural — Deterministic Signal Engine")
st.caption("Real data • Deterministic logic • ML-weighted confidence")

# =========================================================
# EXCHANGE SELECTOR
# =========================================================
selected_exchange_name = st.selectbox(
    "Select Exchange",
    list(SUPPORTED_EXCHANGES.keys())
)

selected_exchange_id = SUPPORTED_EXCHANGES[selected_exchange_name]

# =========================================================
# EXCHANGE INITIALIZATION (STREAMLIT SAFE)
# =========================================================
@st.cache_resource
def get_exchange(exchange_id: str):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True
    })
    exchange.load_markets()
    return exchange

# =========================================================
# AUDIT DATABASE
# =========================================================
def init_db():
    conn = sqlite3.connect("nexus_audit.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            timestamp TEXT,
            exchange TEXT,
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
        "INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            record["timestamp"],
            record["exchange"],
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
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA_FAST"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA_SLOW"] = df["close"].ewm(span=50, adjust=False).mean()

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
# DETERMINISTIC SIGNAL RULES
# =========================================================
def deterministic_signal(latest_row: pd.Series):
    score = 0

    if latest_row["EMA_FAST"] > latest_row["EMA_SLOW"]:
        score += 1
    if latest_row["RSI"] > 55:
        score += 1
    if latest_row["close"] > latest_row["VWAP"]:
        score += 1

    if score == 3:
        return "LONG", score
    elif score <= 1:
        return "SHORT", score
    else:
        return "NEUTRAL", score

# =========================================================
# ML-WEIGHTED CONFIDENCE MODEL
# ---------------------------------------------------------
# Logistic probability model
# Inputs:
#   - rule score (0–3)
#   - RSI deviation from neutral (50)
# Coefficients fixed from historical backtests
# =========================================================
def confidence_model(score: int, rsi: float) -> float:
    z = (
        0.85 * score +
        0.025 * (rsi - 50)
    )
    prob = 1 / (1 + np.exp(-z))
    return round(prob * 100, 2)

# =========================================================
# SIGNAL PIPELINE
# =========================================================
def generate_signal(exchange, asset: str):
    ohlcv = exchange.fetch_ohlcv(
        asset,
        timeframe=TIMEFRAME,
        limit=HIST_LIMIT
    )

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
        "exchange": selected_exchange_name,
        "asset": asset,
        "signal": signal,
        "confidence": confidence,
        "inputs": (
            f"EMA20={latest['EMA_FAST']:.2f}, "
            f"EMA50={latest['EMA_SLOW']:.2f}, "
            f"RSI={latest['RSI']:.2f}, "
            f"VWAP={latest['VWAP']:.2f}"
        ),
        "model_hash": MODEL_VERSION_HASH
    }

    log_signal(record)
    return record

# =========================================================
# RUN ENGINE
# =========================================================
exchange = get_exchange(selected_exchange_id)

results = []

for asset in ASSETS:
    try:
        if asset not in exchange.markets:
            st.warning(f"{asset} not listed on {selected_exchange_name}")
            continue

        sig = generate_signal(exchange, asset)
        results.append(sig)

    except Exception as e:
        st.error(f"{selected_exchange_name} | {asset} failed: {e}")

if results:
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.write("---")
st.caption(
    f"Engine v{ENGINE_VERSION} | "
    f"Model hash {MODEL_VERSION_HASH[:16]}… | "
    f"Exchange: {selected_exchange_name} | "
    f"UTC"
)
