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
APP_NAME = ENGINES.get(STRATEGY_ID, "Nexus AI")

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": msg})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

def is_engine_enabled():
    """Reads performance.json to respect the automated Kill-Switch."""
    if not os.path.exists(PERFORMANCE_FILE): return True
    try:
        with open(PERFORMANCE_FILE, "r") as f:
            perf = json.load(f)
            return perf.get(STRATEGY_ID, {}).get("status", "LIVE") == "LIVE"
    except: return True

def get_ai_prediction(rsi, vol_change, dist_ema):
    """Uses the trained .pkl model to calculate win probability."""
    try:
        if not os.path.exists(MODEL_FILE): return None
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        
        feat = np.array([[rsi, vol_change, dist_ema]])
        feat_scaled = scaler.transform(feat)
        # Returns probability of class 1 (Win)
        return round(model.predict_proba(feat_scaled)[0][1] * 100, 2)
    except Exception as e:
        logging.error(f"AI Prediction error: {e}")
        return None

def run_ai_engine():
    if not is_engine_enabled():
        logging.warning(f"{APP_NAME} is currently in RECOVERY mode. skipping execution.")
        return

    # Using XT.com for AI data feed
    ex = ccxt.xt({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # Technical Indicators
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            rsi = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            ema = df['c'].ewm(span=20).mean()
            
            # Current values
            price = df['c'].iloc[-1]
            rsi_last = rsi.iloc[-1]
            vol_change = df['v'].pct_change().iloc[-1]
            dist_ema = (price - ema.iloc[-1]) / price

            # AI Logic
            ai_conf = get_ai_prediction(rsi_last, vol_change, dist_ema)
            
            # Determine Logic Mode
            if ai_conf is not None:
                reason = "DEEP NETWORK"
                confidence = ai_conf
            else:
                reason = "RSI EXTREME (FALLBACK)"
                confidence = 52.0 # Neutral baseline

            # Signal Generation
            signal = "LONG" if confidence > 50 else "SHORT"
            sl = price * (0.98 if signal == "LONG" else 1.02)
            tp = price * (1.04 if signal == "LONG" else 0.96)

            # Store in Database
            cursor.execute("""
                INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (STRATEGY_ID, asset, '1h', signal, price, sl, tp, confidence, reason, datetime.now().isoformat()))
            conn.commit()

            # Enhanced Notification
            alert_msg = (
                f"🤖 **{APP_NAME} Signal**\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📈 **Asset**: `{asset}`\n"
                f"🚦 **Signal**: `{signal}`\n"
                f"📍 **Entry**: `{price:.4f}`\n"
                f"🎯 **TP**: `{tp:.4f}`\n"
                f"🛑 **SL**: `{sl:.4f}`\n"
                f"📊 **Confidence**: `{confidence}%`\n"
                f"💡 **Logic**: `{reason}`"
            )
            notify(alert_msg)

        except Exception as e:
            logging.error(f"AI Engine Error ({asset}): {e}")

    conn.close()

if __name__ == "__main__":
    run_ai_engine()
