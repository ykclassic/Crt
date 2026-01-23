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
EXCHANGE_NAME = "XT" 
DB_FILE = "rangemaster.db"
PERFORMANCE_FILE = "performance.json"
ASSETS = ["BTC/USDT", "XRP/USDT"]

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def get_exchange(name):
    name = name.upper()
    if name == "XT": return ccxt.xt({"enableRateLimit": True})
    elif name == "GATE": return ccxt.gateio({"enableRateLimit": True})
    elif name == "BITGET": return ccxt.bitget({"enableRateLimit": True})
    else: raise ValueError(f"Unknown exchange: {name}")

def get_learned_confidence():
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                data = json.load(f)
                stats = data.get("rangemaster", {"win_rate": 50.0})
                return stats["win_rate"]
    except: pass
    return 50.0

def run_rangemaster():
    ex = get_exchange(EXCHANGE_NAME)
    ex.load_markets()
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Updated schema to match the other brains for the auditor
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, conf REAL, ts TEXT
        )
    """)
    conn.commit()

    current_win_rate = get_learned_confidence()

    for asset in ASSETS:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
            
            # RSI 14
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            df['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            
            # BB (20, 2)
            df['sma20'] = df['c'].rolling(20).mean()
            df['std'] = df['c'].rolling(20).std()
            df['upper'] = df['sma20'] + (2 * df['std'])
            df['lower'] = df['sma20'] - (2 * df['std'])
            
            last = df.iloc[-1]
            price = last['c']
            signal = None

            if last['rsi'] < 30 and price <= last['lower']:
                signal = "LONG"
                sl = price * 0.98
                tp = price * 1.04
            elif last['rsi'] > 70 and price >= last['upper']:
                signal = "SHORT"
                sl = price * 1.02
                tp = price * 0.96

            if signal:
                conf = current_win_rate
                cursor.execute("INSERT INTO signals (asset, signal, entry, sl, tp, conf, ts) VALUES (?,?,?,?,?,?,?)",
                               (asset, signal, price, sl, tp, conf, datetime.now().isoformat()))
                conn.commit()
                
                notify(f"⚖️ **Mean Reversion Alert**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {conf}%**\n**Price:** {price:.4f} | **RSI:** {last['rsi']:.1f}")
                
        except Exception as e:
            print(f"Error: {e}")

    conn.close()

if __name__ == "__main__":
    run_rangemaster()
