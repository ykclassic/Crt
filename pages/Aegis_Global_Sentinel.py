import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime, timezone

# ============================================================
# CONFIGURATION
# ============================================================
DB_PATH = "aegis_signals.db"
TIMEFRAME = "15m"
LIMIT = 200
CONFIDENCE_THRESHOLD = 90

EXCHANGES = [
    ccxt.binance,
    ccxt.okx,
    ccxt.bybit
]

# ============================================================
# SECURITY — SECRETS
# ============================================================
def get_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return None

TELEGRAM_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")

# ============================================================
# DATABASE
# ============================================================
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
            (
                datetime.now(timezone.utc).isoformat(),
                asset,
                signal,
                confidence,
                price
            )
        )

# ============================================================
# MARKET DATA (HARDENED)
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_ohlcv(symbol: str) -> pd.DataFrame:
    last_error = None

    for exchange_cls in EXCHANGES:
        try:
            exchange = exchange_cls({
                "enableRateLimit": True,
                "timeout": 15000,
                "options": {"defaultType": "spot"}
            })

            exchange.load_markets()

            if symbol not in exchange.symbols:
                continue

            data = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=TIMEFRAME,
                limit=LIMIT
            )

            df = pd.DataFrame(
                data,
                columns=["ts", "open", "high", "low", "close", "volume"]
            )

            if len(df) < 50:
                raise ValueError("Insufficient OHLCV data")

            return df

        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"All exchanges failed for {symbol}: {last_error}")

# ============================================================
# INDICATORS
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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

# ============================================================
# SIGNAL ENGINE
# ============================================================
def calculate_confidence(row) -> float:
    score = 0.0

    ema_spread = abs(row["ema_fast"] - row["ema_slow"]) / row["close"]
    score += min(ema_spread * 100, 40)

    score += min(abs(row["rsi"] - 50), 30)

    atr_ratio = row["atr"] / row["close"]
    score += min(atr_ratio * 100, 30)

    return round(min(score, 100), 2)

def generate_signal(df: pd.DataFrame):
    row = df.iloc[-1]
    signal = "NO_TRADE"

    if row["ema_fast"] > row["ema_slow"] and row["rsi"] > 55:
        signal = "LONG"
    elif row["ema_fast"] < row["ema_slow"] and row["rsi"] < 45:
        signal = "SHORT"

    confidence = calculate_confidence(row)
    return signal, confidence, row["close"]

# ============================================================
# TELEGRAM ALERTS
# ============================================================
def send_telegram_alert(message: str) -> bool:
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

# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    st.set_page_config(
        page_title="Aegis Sentinel | Real-Time Signal Engine",
        page_icon="📡",
        layout="wide"
    )

    st.title("📡 Aegis Sentinel – Real-Time Signal Engine")
    st.caption("Deterministic signals • Multi-exchange fallback • Production hardened")

    init_db()

    assets = st.multiselect(
        "Select Assets",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        default=["BTC/USDT"]
    )

    if st.button("🚀 Run Signal Scan"):
        results = []

        for asset in assets:
            try:
                raw = get_ohlcv(asset)
                df = compute_indicators(raw)
                signal, confidence, price = generate_signal(df)

                if signal != "NO_TRADE" and confidence >= CONFIDENCE_THRESHOLD:
                    log_signal(asset, signal, confidence, price)

                    msg = (
                        f"🔥 *AEGIS SIGNAL*\n"
                        f"*Asset:* {asset}\n"
                        f"*Signal:* {signal}\n"
                        f"*Confidence:* {confidence}%\n"
                        f"*Price:* {price}"
                    )

                    sent = send_telegram_alert(msg)
                    status = "✅ SENT" if sent else "⚠️ NOT SENT"
                else:
                    status = "🔇 FILTERED"

            except Exception as e:
                signal = "N/A"
                confidence = 0.0
                status = "❌ DATA ERROR"

            results.append({
                "Asset": asset,
                "Signal": signal,
                "Confidence": confidence,
                "Status": status
            })

        st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.caption("Aegis Sentinel • Production Signal Engine • v1.0")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
