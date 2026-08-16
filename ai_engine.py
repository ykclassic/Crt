import ccxt
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from config import *
from db_utils import init_db, insert_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s | AI_ENGINE | %(levelname)s | %(message)s")

ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})

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

def generate_signal(df):
    latest = df.iloc[-1]
    if np.isnan(latest["rsi"]):
        return None
    if latest["ema20"] > latest["ema50"] and latest["rsi"] < 70:
        return "LONG"
    if latest["ema20"] < latest["ema50"] and latest["rsi"] > 30:
        return "SHORT"
    return None

def save_signal(pair, timeframe, direction, df):
    latest = df.iloc[-1]
    entry_price = float(latest["close"])

    if direction == "LONG":
        stop_loss = entry_price * (1 - RISK_PERCENT)
        take_profit = entry_price * (1 + REWARD_PERCENT)
    else:
        stop_loss = entry_price * (1 + RISK_PERCENT)
        take_profit = entry_price * (1 - REWARD_PERCENT)

    insert_signal(DB_FILE, {
        "engine_id": "AI_PREDICT",
        "symbol": pair,
        "timeframe": timeframe,
        "direction": direction,
        "entry": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": 0.75,
        "rsi": float(latest["rsi"]),
        "vol_change": float(latest["vol_change"]),
        "dist_ema": float(latest["dist_ema"]),
        "reason": "EMA+RSI",
        "status": "ACTIVE",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    })
    logging.info(f"{pair} {timeframe} {direction} saved")

def run():
    logging.info("Starting AI Engine")
    init_db(DB_FILE)

    for pair in TRADING_PAIRS:
        for tf in TIMEFRAMES:
            try:
                ohlcv = ex.fetch_ohlcv(pair, tf, limit=200)
                if len(ohlcv) < 60:
                    continue

                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = calculate_indicators(df)
                signal = generate_signal(df)

                if signal:
                    save_signal(pair, tf, signal, df)

            except Exception as e:
                logging.error(f"{pair} {tf} error: {e}")
    logging.info("AI Engine cycle complete")

if __name__ == "__main__":
    run()
