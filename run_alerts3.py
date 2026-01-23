import ccxt
import pandas as pd
import numpy as np
import os
import requests
import sqlite3
import json
import pickle
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS AI PREDICT"
DB_FILE = "nexus_ai.db"
MODEL_FILE = "nexus_brain.pkl"
PERFORMANCE_FILE = "performance.json"
STRATEGY_ID = "nexus_ai"

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def get_ai_prediction(rsi, price, ema):
    try:
        if not os.path.exists(MODEL_FILE): return None
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        dist_ema = (price - ema) / price
        feat = np.array([[rsi, 0.0, dist_ema]]) # vol_change set to 0 for simplicity
        feat_scaled = scaler.transform(feat)
        return round(model.predict_proba(feat_scaled)[0][1] * 100, 2)
    except: return None

def run_ai_logic():
    ex = ccxt.xt({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, 
            confidence REAL, reason TEXT, ts TEXT
        )
    """)
    conn.commit()

    # Learning Check
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, "r") as f:
            perf = json.load(f).get(STRATEGY_ID, {"status": "LIVE"})
            if perf["status"] == "RECOVERY": return

    for asset in ["BTC/USDT", "ETH/USDT"]:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=100)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
            df['ema20'] = df['c'].rolling(20).mean()
            
            # RSI
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            df['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            
            last = df.iloc[-1]
            ai_conf = get_ai_prediction(last['rsi'], last['c'], last['ema20'])
            
            reason = "DEEP NETWORK" if ai_conf else "HEURISTIC FALLBACK"
            final_conf = ai_conf if ai_conf else 52.0
            signal = "LONG" if final_conf > 50 else "SHORT"
            
            price = last['c']
            sl = price * (0.98 if signal == "LONG" else 1.02)
            tp = price * (1.04 if signal == "LONG" else 0.96)

            cursor.execute("""
                INSERT INTO signals (asset, signal, entry, sl, tp, confidence, reason, ts) 
                VALUES (?,?,?,?,?,?,?,?)""",
                (asset, signal, price, sl, tp, final_conf, reason, datetime.now().isoformat()))
            conn.commit()
            
            notify(f"🤖 **AI Prediction**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {final_conf}%** (📊 {reason})")
        except Exception as e:
            print(f"Error: {e}")
    conn.close()

if __name__ == "__main__":
    run_ai_logic()
