import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
import logging
import pickle
import os
import json
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, MODEL_FILE, PERFORMANCE_FILE, ENGINES, ASSETS

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

STRATEGY_ID = "ai"
APP_NAME = ENGINES.get("ai", "Nexus AI")

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

def is_engine_enabled():
    """Checks performance.json to see if the engine is allowed to trade."""
    if not os.path.exists(PERFORMANCE_FILE): return True
    try:
        with open(PERFORMANCE_FILE, "r") as f:
            perf = json.load(f)
            return perf.get(STRATEGY_ID, {}).get("status", "LIVE") == "LIVE"
    except: return True

def get_ai_prediction(rsi, vol_change, dist_ema):
    """Loads the .pkl brain to predict signal confidence."""
    try:
        if not os.path.exists(MODEL_FILE): return None
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        feat = np.array([[rsi, vol_change, dist_ema]])
        feat_scaled = scaler.transform(feat)
        # Probability of a '1' (Win)
        return round(model.predict_proba(feat_scaled)[0][1] * 100, 2)
    except Exception as e:
        logging.error(f"AI Prediction Error: {e}")
        return None

def run_ai_engine():
    if not is_engine_enabled():
        logging.warning("AI Engine is in RECOVERY mode. Execution skipped.")
        return

    # Using XT.com for high-speed AI data
    ex = ccxt.xt({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # Feature Calculation
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            df['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            df['ema20'] = df['c'].ewm(span=20).mean()
            dist_ema = (df['c'].iloc[-1] - df['ema20'].iloc[-1]) / df['c'].iloc[-1]
            vol_change = df['v'].pct_change().iloc[-1]
            rsi_last = df['rsi'].iloc[-1]

            # AI Prediction
            ai_conf = get_ai_prediction(rsi_last, vol_change, dist_ema)
            
            # Sentiment Placeholder (Simplified for stability)
            sentiment_score = 0.5 # Neutral
            
            reason = "DEEP NETWORK" if ai_conf else "HEURISTIC FALLBACK"
            final_conf = ai_conf if ai_conf else 52.0
            
            # Signal Logic
            signal = "LONG" if final_conf > 50 else "SHORT"
            price = df['c'].iloc[-1]
            sl = price * (0.98 if signal == "LONG" else 1.02)
            tp = price * (1.04 if signal == "LONG" else 0.96)

            cursor.execute("""
                INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (STRATEGY_ID, asset, '1h', signal, price, sl, tp, final_conf, reason, datetime.now().isoformat()))
            
            conn.commit()
            notify(f"🤖 **AI Prediction**: {signal} {asset}\nConfidence: `{final_conf}%` | Logic: `{reason}`")

        except Exception as e:
            logging.error(f"AI Engine Error ({asset}): {e}")

    conn.close()

if __name__ == "__main__":
    run_ai_engine()
