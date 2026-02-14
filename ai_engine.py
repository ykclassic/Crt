import ccxt
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

from config import DB_FILE, TIMEFRAMES, RISK_PERCENT, REWARD_PERCENT, TRADING_PAIRS
from db_manager import initialize_database

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | AI_ENGINE | %(levelname)s | %(message)s"
)

# ----------------------------
# Exchange
# ----------------------------
ex = ccxt.gateio({"enableRateLimit": True})

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


def calculate_indicators(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["vol_change"] = df["volume"].pct_change()
    df["dist_ema"] = (df["close"] - df["ema20"]) / df["ema20"]
    return df


# ----------------------------
# Signal Logic
# ----------------------------
def generate_signal(df):
    latest = df.iloc[-1]

    if np.isnan(latest["rsi"]):
        return None

    if latest["ema20"] > latest["ema50"] and latest["rsi"] < 70:
        return "LONG"

    if latest["ema20"] < latest["ema50"] and latest["rsi"] > 30:
        return "SHORT"

    return None


# ----------------------------
# Save Signal
# ----------------------------
def save_signal(pair, timeframe, direction, df):
    latest = df.iloc[-1]
    entry_price = float(latest["close"])

    if direction == "LONG":
        stop_loss = entry_price * (1 - RISK_PERCENT)
        take_profit = entry_price * (1 + REWARD_PERCENT)
    else:
        stop_loss = entry_price * (1 + RISK_PERCENT)
        take_profit = entry_price * (1 - REWARD_PERCENT)

    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO signals (
            engine, pair, timeframe, direction,
            entry, stop_loss, take_profit,
            confidence, rsi, vol_change, dist_ema,
            reason, status, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "AI_ENGINE",
        pair,
        timeframe,
        direction,
        entry_price,
        stop_loss,
        take_profit,
        0.75,
        float(latest["rsi"]),
        float(latest["vol_change"]),
        float(latest["dist_ema"]),
        "EMA+RSI",
        "ACTIVE",
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()

    logging.info(f"{pair} {timeframe} {direction} saved")


# ----------------------------
# Engine Runner
# ----------------------------
def run():
    logging.info("Starting AI Engine")
    initialize_database()

    for pair in TRADING_PAIRS:
        for tf in TIMEFRAMES:
            try:
                ohlcv = ex.fetch_ohlcv(pair, tf, limit=200)
                if len(ohlcv) < 60:
                    continue

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

                df = calculate_indicators(df)
                signal = generate_signal(df)

                if signal:
                    save_signal(pair, tf, signal, df)

            except Exception as e:
                logging.error(f"{pair} {tf} error: {e}")

    logging.info("AI Engine cycle complete")


if __name__ == "__main__":
    run()
