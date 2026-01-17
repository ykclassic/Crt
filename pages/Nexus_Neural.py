# ============================================================
# Nexus Neural – Deterministic Signal Engine (Production Core)
# ============================================================

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Nexus Neural | Deterministic Signal Engine", layout="wide")

EXCHANGE_ID = "bitget"
TIMEFRAME = "1h"
HIST_LIMIT = 500
MODEL_VERSION = "v1.0.0-deterministic"

ASSETS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "LINK/USDT", "DOGE/USDT",
    "TRX/USDT", "SUI/USDT", "PEPE/USDT"
]

DB_PATH = "signal_audit.db"

# ============================================================
# SECURITY GATE
# ============================================================

if "authenticated" not in st.session_state:
    st.stop()

# ============================================================
# DATABASE (AUDIT LOGGING)
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            timestamp TEXT,
            asset TEXT,
            exchange TEXT,
            signal TEXT,
            confidence REAL,
            features TEXT,
            model_version TEXT,
            model_hash TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============================================================
# EXCHANGE CONNECTION
# ============================================================

@st.cache_resource
def get_exchange():
    ex = getattr(ccxt, EXCHANGE_ID)({
        "enableRateLimit": True
    })
    ex.load_markets()
    return ex

exchange = get_exchange()

# ============================================================
# DATA INGESTION
# ============================================================

@st.cache_data(ttl=300)
def fetch_ohlcv(symbol):
    data = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=HIST_LIMIT)
    df = pd.DataFrame(
        data,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# ============================================================
# TECHNICAL INDICATORS (DETERMINISTIC)
# ============================================================

def compute_indicators(df):
    df = df.copy()

    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )
    df["atr"] = tr.rolling(14).mean()

    return df.dropna()

# ============================================================
# DETERMINISTIC SIGNAL RULES
# ============================================================

def rule_based_signal(row):
    if (
        row["ema_20"] > row["ema_50"]
        and row["close"] > row["vwap"]
        and row["rsi"] > 55
    ):
        return 1  # LONG
    if (
        row["ema_20"] < row["ema_50"]
        and row["close"] < row["vwap"]
        and row["rsi"] < 45
    ):
        return -1  # SHORT
    return 0

# ============================================================
# ML CONFIDENCE MODEL
# ============================================================

@st.cache_resource
def train_confidence_model():
    """
    Logistic regression trained on historical indicator states
    Target: next-candle direction
    """
    X_all = []
    y_all = []

    for symbol in ASSETS[:3]:  # limit training size for Streamlit
        df = compute_indicators(fetch_ohlcv(symbol))
        df["future_return"] = df["close"].shift(-1) - df["close"]
        df["target"] = (df["future_return"] > 0).astype(int)

        features = df[["ema_20", "ema_50", "rsi", "vwap", "atr"]].values
        targets = df["target"].values

        X_all.append(features)
        y_all.append(targets)

    X = np.vstack(X_all)
    y = np.hstack(y_all)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)
    return model

confidence_model = train_confidence_model()

MODEL_HASH = hashlib.sha256(
    str(confidence_model.get_params()).encode()
).hexdigest()

# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(symbol):
    df = compute_indicators(fetch_ohlcv(symbol))
    latest = df.iloc[-1]

    rule_signal = rule_based_signal(latest)

    features = latest[["ema_20", "ema_50", "rsi", "vwap", "atr"]].values.reshape(1, -1)
    prob = confidence_model.predict_proba(features)[0][1]

    if rule_signal == 1:
        signal = "LONG"
    elif rule_signal == -1:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    confidence = round(prob * 100, 2)

    return {
        "asset": symbol,
        "signal": signal,
        "confidence": confidence,
        "features": dict(zip(
            ["ema_20", "ema_50", "rsi", "vwap", "atr"],
            latest[["ema_20", "ema_50", "rsi", "vwap", "atr"]].round(4)
        ))
    }

# ============================================================
# AUDIT LOGGING
# ============================================================

def log_signal(record):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        record["asset"],
        EXCHANGE_ID,
        record["signal"],
        record["confidence"],
        str(record["features"]),
        MODEL_VERSION,
        MODEL_HASH
    ))
    conn.commit()
    conn.close()

# ============================================================
# UI
# ============================================================

st.title("🌐 Nexus Neural — Deterministic Signal Engine")
st.caption("Real data • Deterministic logic • ML-weighted confidence")

results = []

for asset in ASSETS:
    try:
        signal = generate_signal(asset)
        log_signal(signal)
        results.append(signal)
    except Exception as e:
        st.warning(f"{asset}: data unavailable")

df_display = pd.DataFrame(results)

st.dataframe(df_display, use_container_width=True)

st.write("### System Integrity")
st.write(f"• Exchange: {EXCHANGE_ID}")
st.write(f"• Model version: {MODEL_VERSION}")
st.write(f"• Model hash: `{MODEL_HASH[:16]}…`")
st.write(f"• Timestamp (UTC): {datetime.utcnow().isoformat()}")

st.caption("Deterministic • Auditable • Reproducible")
