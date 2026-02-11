import ccxt
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from config import DB_FILE, TIMEFRAMES, ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP

# -----------------------------
# Exchange (NO XT)
# -----------------------------
ex = ccxt.gateio({'enableRateLimit': True})

# -----------------------------
# Ensure DB Schema
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine TEXT,
            asset TEXT,
            timeframe TEXT,
            signal TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            confidence REAL,
            rsi REAL,
            vol_change REAL,
            dist_ema REAL,
            reason TEXT,
            status TEXT,
            ts TEXT
        )
    """)
    conn.commit()
    conn.close()

# -----------------------------
# Indicators
# -----------------------------
def calculate_indicators(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df, 14)
    df["vol_change"] = df["volume"].pct_change()
    df["dist_ema"] = (df["close"] - df["ema20"]) / df["ema20"]
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# Signal Logic
# -----------------------------
def generate_signal(df):
    latest = df.iloc[-1]

    if latest["ema20"] > latest["ema50"] and latest["rsi"] < 70:
        return "LONG"
    if latest["ema20"] < latest["ema50"] and latest["rsi"] > 30:
        return "SHORT"
    return None

# -----------------------------
# Store Signal
# -----------------------------
def save_signal(asset, timeframe, signal, df):
    latest = df.iloc[-1]
    entry = latest["close"]
    atr = latest["atr"]

    sl = entry - ATR_MULTIPLIER_SL * atr if signal == "LONG" else entry + ATR_MULTIPLIER_SL * atr
    tp = entry + ATR_MULTIPLIER_TP * atr if signal == "LONG" else entry - ATR_MULTIPLIER_TP * atr

    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO signals (
            engine, asset, timeframe, signal,
            entry, sl, tp, confidence,
            rsi, vol_change, dist_ema,
            reason, status, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "AI_ENGINE",
        asset,
        timeframe,
        signal,
        float(entry),
        float(sl),
        float(tp),
        0.75,
        float(latest["rsi"]),
        float(latest["vol_change"]),
        float(latest["dist_ema"]),
        "EMA+RSI strategy",
        "ACTIVE",
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

# -----------------------------
# Main Run
# -----------------------------
def run():
    init_db()
    markets = ex.load_markets()

    for symbol in list(markets.keys())[:10]:
        if not symbol.endswith("/USDT"):
            continue

        for tf in TIMEFRAMES:
            try:
                ohlcv = ex.fetch_ohlcv(symbol, tf, limit=200)
                df = pd.DataFrame(ohlcv, columns=[
                    "timestamp", "open", "high", "low", "close", "volume"
                ])
                df = calculate_indicators(df)
                signal = generate_signal(df)

                if signal:
                    save_signal(symbol, tf, signal, df)

            except Exception:
                continue

if __name__ == "__main__":
    run()
