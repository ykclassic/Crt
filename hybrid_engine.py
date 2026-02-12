import ccxt
import pandas as pd
import sqlite3
import logging
import json
import os
import requests
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, ENGINES, PERFORMANCE_FILE

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

STRATEGY_ID = "hybrid_v1"
APP_NAME = ENGINES.get(STRATEGY_ID, "Nexus Hybrid")

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT"
]

# ----------------------------
# Engine Status
# ----------------------------

def is_engine_enabled():
    if not os.path.exists(PERFORMANCE_FILE):
        return True
    try:
        with open(PERFORMANCE_FILE, "r") as f:
            perf = json.load(f)
            return perf.get(STRATEGY_ID, {}).get("status", "LIVE") == "LIVE"
    except:
        return True

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

# ----------------------------
# Engine
# ----------------------------

def run_hybrid_engine():

    if not is_engine_enabled():
        logging.warning(f"{APP_NAME} in RECOVERY mode. Skipping.")
        return

    ex = ccxt.gateio({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in SYMBOLS:
        try:
            ohlcv = ex.fetch_ohlcv(asset, "1h", limit=120)
            if len(ohlcv) < 50:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])

            df["ema8"] = df["c"].ewm(span=8).mean()
            df["ema21"] = df["c"].ewm(span=21).mean()
            df["vol_sma"] = df["v"].rolling(20).mean()

            last = df.iloc[-1]
            prev = df.iloc[-2]

            signal = None
            reason = ""

            if (
                last["ema8"] > last["ema21"]
                and prev["ema8"] <= prev["ema21"]
                and last["v"] > last["vol_sma"]
            ):
                signal = "LONG"
                reason = "Bullish EMA Cross + Volume Surge"

            elif (
                last["ema8"] < last["ema21"]
                and prev["ema8"] >= prev["ema21"]
                and last["v"] > last["vol_sma"]
            ):
                signal = "SHORT"
                reason = "Bearish EMA Cross + Volume Surge"

            if signal:
                price = float(last["c"])
                sl = price * 0.98 if signal == "LONG" else price * 1.02
                tp = price * 1.05 if signal == "LONG" else price * 0.95

                cursor.execute("""
                    INSERT INTO signals (
                        engine, asset, timeframe, signal,
                        entry, sl, tp, confidence,
                        reason, status, ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    STRATEGY_ID,
                    asset,
                    "1h",
                    signal,
                    price,
                    sl,
                    tp,
                    0.65,
                    reason,
                    "ACTIVE",
                    datetime.utcnow().isoformat()
                ))

                conn.commit()
                logging.info(f"{asset} {signal} signal generated")
                notify(f"⚡ {signal} {asset} | {reason}")

        except Exception as e:
            logging.error(f"Hybrid Engine Error ({asset}): {e}")

    conn.close()

if __name__ == "__main__":
    run_hybrid_engine()
