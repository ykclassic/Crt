import ccxt
import pandas as pd
import sqlite3
import logging
import json
import os
import requests
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, ENGINES, PERFORMANCE_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

STRATEGY_ID = "rangemaster"
APP_NAME = ENGINES.get(STRATEGY_ID, "Nexus Rangemaster")

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT"
]

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

def run_range_engine():

    if not is_engine_enabled():
        logging.warning(f"{APP_NAME} in RECOVERY mode. Skipping.")
        return

    ex = ccxt.gateio({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in SYMBOLS:
        try:
            ohlcv = ex.fetch_ohlcv(asset, "15m", limit=80)
            if len(ohlcv) < 30:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])

            df["sma"] = df["c"].rolling(20).mean()
            df["std"] = df["c"].rolling(20).std()
            df["upper"] = df["sma"] + (2 * df["std"])
            df["lower"] = df["sma"] - (2 * df["std"])

            last = df.iloc[-1]

            signal = None
            reason = ""

            if last["c"] <= last["lower"]:
                signal = "LONG"
                reason = "Bollinger Lower Band Touch"

            elif last["c"] >= last["upper"]:
                signal = "SHORT"
                reason = "Bollinger Upper Band Touch"

            if signal:
                price = float(last["c"])
                sl = float(last["lower"] * 0.995) if signal == "LONG" else float(last["upper"] * 1.005)
                tp = float(last["sma"])

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
                    "15m",
                    signal,
                    price,
                    sl,
                    tp,
                    0.60,
                    reason,
                    "ACTIVE",
                    datetime.utcnow().isoformat()
                ))

                conn.commit()
                logging.info(f"{asset} {signal} range signal")
                notify(f"↔️ {signal} {asset} | {reason}")

        except Exception as e:
            logging.error(f"Range Engine Error ({asset}): {e}")

    conn.close()

if __name__ == "__main__":
    run_range_engine()
