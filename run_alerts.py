import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
import logging
import time
from datetime import datetime
from config import (
    DB_FILE, WEBHOOK_URL, ENGINES, 
    ATR_MULTIPLIER_SL, RR_RATIO, 
    ASSETS, TIMEFRAMES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

STRATEGY_ID = "nexus_core"

def fetch_multi_exchange_data(asset, tf, limit=100):
    # Updated to use your specific exchanges
    exchanges = [ccxt.bitget(), ccxt.gateio(), ccxt.xt()]
    all_data = []
    
    for ex in exchanges:
        try:
            logging.info(f"Fetching {asset} from {ex.id}...")
            data = ex.fetch_ohlcv(asset, tf, limit=limit)
            if data:
                df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                all_data.append(df)
        except Exception as e:
            logging.warning(f"Fetch failed on {ex.id}: {e}")
            
    if not all_data:
        raise ValueError(f"Could not fetch data for {asset} from any exchange.")
        
    # Merge and average data from all 3 exchanges for accuracy
    df_avg = pd.concat(all_data).groupby(level=0).mean()
    return df_avg

def compute_indicators(df):
    df = df.copy()
    # RSI
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    # ATR for SL/TP
    tr = pd.concat([(df['high']-df['low']), abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df

def run_alerts():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine TEXT, asset TEXT, timeframe TEXT, signal TEXT, 
            entry REAL, sl REAL, tp REAL, confidence REAL, 
            reason TEXT, ts TEXT
        )
    """)
    conn.commit()

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            try:
                df = fetch_multi_exchange_data(asset, tf)
                df = compute_indicators(df)
                last = df.iloc[-1]
                
                signal = "NEUTRAL"
                if last["rsi"] < 30: signal = "LONG"
                elif last["rsi"] > 70: signal = "SHORT"

                if signal != "NEUTRAL":
                    entry = last["close"]
                    atr = last["atr"] if not np.isnan(last["atr"]) else entry * 0.02
                    sl = (entry - ATR_MULTIPLIER_SL * atr) if signal == "LONG" else (entry + ATR_MULTIPLIER_SL * atr)
                    tp = (entry + RR_RATIO * abs(entry - sl)) if signal == "LONG" else (entry - RR_RATIO * abs(entry - sl))
                    
                    cursor.execute("""
                        INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (STRATEGY_ID, asset, tf, signal, entry, sl, tp, 70.0, "RSI Extreme", datetime.now().isoformat()))
                    conn.commit()
                    logging.info(f"Saved {signal} for {asset}")

            except Exception as e:
                logging.error(f"Error on {asset}: {e}")
    conn.close()

if __name__ == "__main__":
    run_alerts()
