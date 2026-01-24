import ccxt
import pandas as pd
import os
import requests
import sqlite3
import json
import sys
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS HYBRID V1"
EXCHANGE_NAME = "XT" 
DB_FILE = "hybrid_v1.db"
PERFORMANCE_FILE = "performance.json"
STRATEGY_ID = "hybrid_v1"

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def get_learned_confidence():
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                data = json.load(f)
                stats = data.get(STRATEGY_ID, {"win_rate": 50.0, "status": "LIVE", "sample_size": 0})
                if stats.get("status") == "RECOVERY" or (stats.get("sample_size", 0) > 5 and stats["win_rate"] < 40.0):
                    print(f"🛑 {STRATEGY_ID} is in RECOVERY/KILLED mode. Skipping alerts.")
                    return None
                return stats["win_rate"]
    except: pass
    return 50.0

def run_hybrid():
    ex = ccxt.xt({"enableRateLimit": True})
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Standard schema
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
    
    # Migration: add timeframe column if missing (safe on old DBs)
    cursor.execute("PRAGMA table_info(signals)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'timeframe' not in columns:
        cursor.execute("ALTER TABLE signals ADD COLUMN timeframe TEXT")
    
    conn.commit()

    current_conf = get_learned_confidence()
    if current_conf is None: 
        conn.close()
        return

    for asset in ["SOL/USDT", "BTC/USDT", "ETH/USDT"]:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
            
            df['sma50'] = df['c'].rolling(50).mean()
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            signal = None
            reason = "NONE"

            if last['c'] > last['sma50'] and prev['c'] <= prev['sma50']:
                signal = "LONG"; reason = "TREND BREAKOUT"
            elif last['c'] < last['sma50'] and prev['c'] >= prev['sma50']:
                signal = "SHORT"; reason = "TREND BREAKDOWN"
            elif last['c'] > last['sma50']:
                signal = "LONG"; reason = "BULLISH MOMENTUM"
            else:
                signal = "SHORT"; reason = "BEARISH MOMENTUM"

            entry = last['c']
            sl = entry * (0.97 if signal == "LONG" else 1.03)
            tp = entry * (1.06 if signal == "LONG" else 0.94)

            cursor.execute("""
                INSERT INTO signals (asset, timeframe, signal, entry, sl, tp, confidence, reason, ts) 
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (asset, '1h', signal, entry, sl, tp, current_conf, reason, datetime.now().isoformat()))
            conn.commit()
            
            notify(f"🔄 **Hybrid Alert**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {current_conf}%** (📊 {reason})\n---\nEntry: {entry:.2f}\nSL: {sl:.2f} | TP: {tp:.2f}")
        except Exception as e: 
            print(f"Error: {e}")
            
    conn.close()

if __name__ == "__main__":
    run_hybrid()
