import ccxt
import pandas as pd
import sqlite3
import logging
from datetime import datetime
from config import DB_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT"
]

TIMEFRAME = "1h"

def run_engine():

    logging.info(f"🚀 Core Engine Pulse: {datetime.utcnow()}")

    ex = ccxt.gateio({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    found_any = False

    for symbol in SYMBOLS:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
            if len(ohlcv) < 50:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])

            df["ema20"] = df["c"].ewm(span=20).mean()

            last_price = df["c"].iloc[-1]
            prev_ema = df["ema20"].iloc[-2]
            curr_ema = df["ema20"].iloc[-1]

            signal = None
            reason = ""

            if last_price > curr_ema and last_price > prev_ema:
                signal = "LONG"
                reason = "Price Above EMA20"

            elif last_price < curr_ema and last_price < prev_ema:
                signal = "SHORT"
                reason = "Price Below EMA20"

            if signal:
                cursor.execute("""
                    INSERT INTO signals (
                        engine, asset, timeframe, signal,
                        entry, sl, tp, confidence,
                        reason, status, ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    "nexus_core",
                    symbol,
                    TIMEFRAME,
                    signal,
                    float(last_price),
                    None,
                    None,
                    0.70,
                    reason,
                    "ACTIVE",
                    datetime.utcnow().isoformat()
                ))

                conn.commit()
                logging.info(f"{symbol} {signal} signal")
                found_any = True

        except Exception as e:
            logging.error(f"Core Engine Error ({symbol}): {e}")

    if not found_any:
        logging.info("😴 No signals met criteria this cycle.")

    conn.close()

if __name__ == "__main__":
    run_engine()
