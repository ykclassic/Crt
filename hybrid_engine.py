import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
import logging
from datetime import datetime
from config import (
    DB_FILE, WEBHOOK_URL, ENGINES, 
    ATR_MULTIPLIER_SL, RR_RATIO, ASSETS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

STRATEGY_ID = "hybrid_v1"
APP_NAME = ENGINES.get("hybrid", "Nexus Hybrid")

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

def run_hybrid():
    # Using Gate.io for better volume analysis
    ex = ccxt.gateio({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            # Fetch 1h timeframe for Hybrid strategy
            data = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            
            # Indicators: EMA Cross + Volume Surge
            df['ema8'] = df['close'].ewm(span=8).mean()
            df['ema21'] = df['close'].ewm(span=21).mean()
            df['vol_sma'] = df['volume'].rolling(20).mean()
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            signal = "NEUTRAL"
            reason = "NONE"

            # Logic: EMA Cross + Volume confirmation
            if last['ema8'] > last['ema21'] and prev['ema8'] <= prev['ema21'] and last['volume'] > last['vol_sma']:
                signal = "LONG"
                reason = "Bullish EMA Cross + Vol Spike"
            elif last['ema8'] < last['ema21'] and prev['ema8'] >= prev['ema21'] and last['volume'] > last['vol_sma']:
                signal = "SHORT"
                reason = "Bearish EMA Cross + Vol Spike"

            if signal != "NEUTRAL":
                entry = last['close']
                # Standardized SL/TP logic
                sl = entry * 0.98 if signal == "LONG" else entry * 1.02
                tp = entry * 1.05 if signal == "LONG" else entry * 0.95

                cursor.execute("""
                    INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (STRATEGY_ID, asset, '1h', signal, entry, sl, tp, 65.0, reason, datetime.now().isoformat()))
                
                conn.commit()
                notify(f"⚡ **Hybrid Signal**: {signal} {asset}\nReason: {reason}")

        except Exception as e:
            logging.error(f"Hybrid Engine Error ({asset}): {e}")
    
    conn.close()

if __name__ == "__main__":
    run_hybrid()
