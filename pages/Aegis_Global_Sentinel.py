import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime, timezone

# =============================
# CONFIG
# =============================
DB_PATH = "aegis_signals.db"
TIMEFRAME = "15m"
LIMIT = 200
CONFIDENCE_THRESHOLD = 90

# =============================
# SECURITY: Secrets
# =============================
def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None

TELEGRAM_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")

# =============================
# DATABASE
# =============================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            timestamp TEXT,
            asset TEXT,
            signal TEXT,
            confidence REAL,
            price REAL
        )
        """)

def log_signal(asset, signal, confidence, price):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO signals VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), asset, signal, confidence, price)
        )

# =============================
# MARKET DATA
# =============================
@st.cache_data(ttl=60)
def get_ohlcv(symbol):
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(
        data, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    return df

# =============================
# INDICATORS
# =============================
def compute_indicators(df):
    df["ema_fast"] = df["close"].ewm(span=21).mean()
    df["ema_slow"] = df["close"].ewm(span=55).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )
    df["atr"] = tr.rolling(14).mean()

    return df

# =============================
# SIGNAL ENGINE
# =============================
def generate_signal(df):
    latest = df.iloc[-1]

    signal = "NO_TRADE"

    if latest["ema_fast"] > latest["ema_slow"] and latest["rsi"] > 55:
        signal = "LONG"
    elif latest["ema_fast"] < latest["ema_slow"] and latest["rsi"] < 45:
        signal = "SHORT"

    confidence = calculate_confidence(latest)
    return signal, confidence, latest["close"]

def calculate_confidence(row):
    score = 0

    score += min(abs(row["ema_fast"] - row["ema_slow"]) / row["close"] * 100, 40)
    score += abs(row["rsi"] - 50)
    score += min(row["atr"] / row["close"] * 100, 20)

    return round(min(score, 100), 2)

# =============================
# TELEGRAM
# =============================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# =============================
# STREAMLIT APP
# =============================
def main():
    st.set_page_config("Aegis Sentinel | Live Signal Engine", "📡", layout="wide")
    st.title("📡 Aegis Sentinel – Real-Time Signal Engine")

    init_db()

    assets = st.multiselect(
        "Select Assets",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        default=["BTC/USDT"]
    )

    if st.button("🚀 Run Signal Scan"):
        results = []

        for asset in assets:
            df = compute_indicators(get_ohlcv(asset))
            signal, confidence, price = generate_signal(df)

            if confidence >= CONFIDENCE_THRESHOLD and signal != "NO_TRADE":
                log_signal(asset, signal, confidence, price)

                msg = (
                    f"🔥 *AEGIS SIGNAL*\n"
                    f"*Asset:* {asset}\n"
                    f"*Signal:* {signal}\n"
                    f"*Confidence:* {confidence}%\n"
                    f"*Price:* {price}"
                )

                pushed = send_telegram(msg)
                status = "✅ SENT" if pushed else "❌ FAILED"
            else:
                status = "🔇 FILTERED"

            results.append({
                "Asset": asset,
                "Signal": signal,
                "Confidence": confidence,
                "Status": status
            })

        st.dataframe(pd.DataFrame(results))

    st.caption("Aegis Sentinel | Deterministic Signal Engine")

if __name__ == "__main__":
    main()
