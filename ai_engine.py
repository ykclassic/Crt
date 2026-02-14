import ccxt
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
datetime.now(timezone.utc).isoformat()
from config import DB_FILE, TIMEFRAMES, ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP
from db_manager import initialize_database

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# Exchange
# ----------------------------
ex = ccxt.gateio({
    "enableRateLimit": True,
})

# Limit to stable, liquid pairs
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT"
]

# ----------------------------
# Indicators
# ----------------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_indicators(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["atr"] = compute_atr(df)
    df["vol_change"] = df["volume"].pct_change()
    df["dist_ema"] = (df["close"] - df["ema20"]) / df["ema20"]
    return df


# ----------------------------
# Signal Logic
# ----------------------------

def generate_signal(df):
    latest = df.iloc[-1]

    if np.isnan(latest["atr"]) or np.isnan(latest["rsi"]):
        return None

    if latest["ema20"] > latest["ema50"] and latest["rsi"] < 70:
        return "LONG"

    if latest["ema20"] < latest["ema50"] and latest["rsi"] > 30:
        return "SHORT"

    return None


# ----------------------------
# Save Signal
# ----------------------------

def save_signal(asset, timeframe, signal, df):
    latest = df.iloc[-1]
    entry = float(latest["close"])
    atr = float(latest["atr"])

    if atr <= 0:
        return

    if signal == "LONG":
        sl = entry - ATR_MULTIPLIER_SL * atr
        tp = entry + ATR_MULTIPLIER_TP * atr
    else:
        sl = entry + ATR_MULTIPLIER_SL * atr
        tp = entry - ATR_MULTIPLIER_TP * atr

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
        entry,
        sl,
        tp,
        0.75,
        float(latest["rsi"]),
        float(latest["vol_change"]),
        float(latest["dist_ema"]),
        "EMA+RSI",
        "ACTIVE",
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

    logging.info(f"Signal saved: {asset} {timeframe} {signal}")


# ----------------------------
# Engine Runner
# ----------------------------

def run():
    logging.info("Starting AI Engine")
    initialize_database()

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                ohlcv = ex.fetch_ohlcv(symbol, tf, limit=200)

                if len(ohlcv) < 60:
                    continue

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp","open","high","low","close","volume"]
                )

                df = calculate_indicators(df)
                signal = generate_signal(df)

                if signal:
                    save_signal(symbol, tf, signal, df)

            except Exception as e:
                logging.error(f"{symbol} {tf} error: {str(e)}")

    logging.info("AI Engine cycle complete")


if __name__ == "__main__":
    run()
