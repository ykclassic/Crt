import ccxt
import pandas as pd
import numpy as np
import os
import requests
import sqlite3
import json
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS AI PREDICT"
EXCHANGE_NAME = "XT" 
DB_FILE = "nexus_ai.db"
PERFORMANCE_FILE = "performance.json"
ASSETS = ["BTC/USDT", "ETH/USDT"]

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def get_exchange(name):
    name = name.upper()
    if name == "XT": return ccxt.xt({"enableRateLimit": True})
    elif name == "GATE": return ccxt.gateio({"enableRateLimit": True})
    elif name == "BITGET": return ccxt.bitget({"enableRateLimit": True})
    else: raise ValueError(f"Unknown exchange: {name}")

def get_historical_winrate():
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                data = json.load(f)
                stats = data.get("nexus_ai", {"win_rate": 50.0})
                return stats["win_rate"]
    except: pass
    return 50.0

def run_ai_logic():
    ex = get_exchange(EXCHANGE_NAME)
    ex.load_markets()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, asset TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, conf REAL, ts TEXT)")
    conn.commit()

    hist_wr = get_historical_winrate()

    for asset in ASSETS:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=100)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])

            delta = df['c'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain/loss)))
            
            rsi = df['rsi'].iloc[-1]
            price = df['c'].iloc[-1]
            signal = "LONG" if rsi > 50 else "SHORT"
            
            # --- WEIGHTED AI CONFIDENCE ---
            # 1. Math Probability (How extreme is the RSI?)
            math_prob = 50 + (abs(rsi - 50) / 50.0) * 49.0
            # 2. Hybrid Score: (Current Math + Historical Performance) / 2
            conf_score = round((math_prob + hist_wr) / 2, 2)

            sl = price * (0.98 if signal == "LONG" else 1.02)
            tp = price * (1.04 if signal == "LONG" else 0.96)

            cursor.execute("INSERT INTO signals (asset, signal, entry, sl, tp, conf, ts) VALUES (?,?,?,?,?,?,?)",
                           (asset, signal, price, sl, tp, conf_score, datetime.now().isoformat()))
            conn.commit()
            
            notify(f"🤖 **AI Analysis**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {conf_score}%**\n*(Hist: {hist_wr}% | Math: {math_prob:.1f}%)*")
            
        except Exception as e:
            print(f"Error: {e}")

    conn.close()

if __name__ == "__main__":
    run_ai_logic()
