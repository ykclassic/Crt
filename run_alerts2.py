import ccxt
import pandas as pd
import numpy as np
import sqlite3
import os
import requests
import json
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS RANGEMASTER"
DB_FILE = "rangemaster.db"
PERFORMANCE_FILE = "performance.json"
STRATEGY_ID = "rangemaster"

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def get_learned_confidence():
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                data = json.load(f)
                stats = data.get(STRATEGY_ID, {"win_rate": 50.0, "status": "LIVE"})
                if stats.get("status") == "RECOVERY": return None
                return stats["win_rate"]
    except: pass
    return 50.0

def run_rangemaster():
    ex = ccxt.xt({"enableRateLimit": True})
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT,
            timeframe TEXT,
            signal TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            confidence REAL,
            reason TEXT,
            ts TEXT
        )
    """)
    
    cursor.execute("PRAGMA table_info(signals)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'timeframe' not in columns:
        cursor.execute("ALTER TABLE signals ADD COLUMN timeframe TEXT")
    
    conn.commit()

    current_conf = get_learned_confidence()
    if current_conf is None: 
        conn.close()
        return

    for asset in ["BTC/USDT", "XRP/USDT", "ETH/USDT"]:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
            
            # BB Calculation
            df['sma20'] = df['c'].rolling(20).mean()
            df['std'] = df['c'].rolling(20).std()
            df['upper'] = df['sma20'] + (2 * df['std'])
            df['lower'] = df['sma20'] - (2 * df['std'])
            
            last = df.iloc[-1]
            price = last['c']
            signal = None
            reason = "NONE"

            if price <= last['lower']:
                signal = "LONG"; reason = "BB LOWER TOUCH"
            elif price >= last['upper']:
                signal = "SHORT"; reason = "BB UPPER TOUCH"
            
            if signal:
                sl = price * (0.98 if signal == "LONG" else 1.02)
                tp = price * (1.04 if signal == "LONG" else 0.96)
                
                cursor.execute("""
                    INSERT INTO signals (asset, timeframe, signal, entry, sl, tp, confidence, reason, ts) 
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (asset, '1h', price, sl, tp, current_conf, reason, datetime.now().isoformat()))
                conn.commit()
                
                notify(f"⚖️ **Range Alert**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {current_conf}%** (📊 {reason})\n---\nEntry: {price:.4f}\nSL: {sl:.4f} | TP: {tp:.4f}")
        except Exception as e:
            print(f"Error: {e}")
    conn.close()

if __name__ == "__main__":
    run_rangemaster()
