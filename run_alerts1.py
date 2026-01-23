import ccxt
import pandas as pd
import os
import requests
import sqlite3
import json
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS HYBRID V1"
EXCHANGE_NAME = "XT" 
DB_FILE = "hybrid_v1.db"
PERFORMANCE_FILE = "performance.json"

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
                stats = data.get("hybrid_v1", {"win_rate": 50.0})
                return stats["win_rate"]
    except: pass
    return 50.0

def run_hybrid():
    ex = get_exchange(EXCHANGE_NAME)
    ex.load_markets()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, asset TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, conf REAL, ts TEXT)")
    conn.commit()

    current_win_rate = get_learned_confidence()

    for asset in ["SOL/USDT", "BTC/USDT"]:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])

            df['hl2'] = (df['h'] + df['l']) / 2
            last_price = df['c'].iloc[-1]
            
            # Use learned win rate as confidence
            conf = current_win_rate

            signal = "LONG" if last_price > df['hl2'].mean() else "SHORT"
            sl = last_price * (0.97 if signal == "LONG" else 1.03)
            tp = last_price * (1.06 if signal == "LONG" else 0.94)

            cursor.execute("INSERT INTO signals (asset, signal, entry, sl, tp, conf, ts) VALUES (?,?,?,?,?,?,?)",
                           (asset, signal, last_price, sl, tp, conf, datetime.now().isoformat()))
            conn.commit()
            
            emoji = "🔄"
            status = "⭐ PROVEN" if conf > 55 else "🧪 TESTING"
            notify(f"{emoji} **Hybrid Alert** ({status})\nAsset: {asset}\nSignal: {signal}\n**Confidence: {conf}%**\n---\nEntry: {last_price:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}")
        except Exception as e: 
            print(f"Error: {e}")
            
    conn.close()

if __name__ == "__main__":
    run_hybrid()
