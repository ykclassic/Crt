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
from config import (
    DB_FILE, WEBHOOK_URL, MODEL_FILE, PERFORMANCE_FILE, 
    ENGINES, ASSETS, RISK_PER_TRADE, TOTAL_CAPITAL, ATR_MULTIPLIER, MIN_CONFIDENCE_FOR_SIZE_BOOST
)

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
    if not os.path.exists(PERFORMANCE_FILE): return True
    try:
        with open(PERFORMANCE_FILE, "r") as f:
            perf = json.load(f)
            engine_data = perf.get(STRATEGY_ID, {})
            if engine_data.get("status") == "RECOVERY":
                # Check if it was a Guardian stop
                reason = engine_data.get("reason", "Low Win Rate")
                logging.warning(f"Engine Disabled. Reason: {reason}")
                return False
            return True
    except: return True

def get_ai_prediction(rsi, vol_change, dist_ema):
    try:
        if not os.path.exists(MODEL_FILE): return None
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        feat = np.array([[rsi, vol_change, dist_ema]])
        feat_scaled = scaler.transform(feat)
        return round(model.predict_proba(feat_scaled)[0][1] * 100, 2)
    except Exception as e:
        logging.error(f"AI Prediction error: {e}")
        return None

def calculate_dynamic_risk(df, current_price, signal, confidence):
    high_low = df['h'] - df['l']
    high_close = np.abs(df['h'] - df['c'].shift())
    low_close = np.abs(df['l'] - df['c'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]

    sl_distance = atr * ATR_MULTIPLIER
    sl_price = current_price - sl_distance if signal == "LONG" else current_price + sl_distance
    tp_price = current_price + (sl_distance * 2) if signal == "LONG" else current_price - (sl_distance * 2)

    risk_amount = TOTAL_CAPITAL * RISK_PER_TRADE
    if confidence >= MIN_CONFIDENCE_FOR_SIZE_BOOST:
        risk_amount *= 1.5 
        
    sl_pct = sl_distance / current_price
    suggested_size = risk_amount / sl_pct

    return round(sl_price, 4), round(tp_price, 4), round(suggested_size, 2)

def run_ai_engine():
    if not is_engine_enabled():
        return

    ex = ccxt.xt({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=100)
            df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            rsi = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            ema = df['c'].ewm(span=20).mean()
            
            price = df['c'].iloc[-1]
            confidence = get_ai_prediction(rsi.iloc[-1], df['v'].pct_change().iloc[-1], (price - ema.iloc[-1])/price) or 52.0
            signal = "LONG" if confidence > 50 else "SHORT"

            sl, tp, size = calculate_dynamic_risk(df, price, signal, confidence)

            cursor.execute("""
                INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (STRATEGY_ID, asset, '1h', signal, price, sl, tp, confidence, "PHASE 1: DYNAMIC RISK", datetime.now().isoformat()))
            conn.commit()

            alert_msg = (
                f"🤖 **{APP_NAME}**\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📈 **Asset**: `{asset}` | **Signal**: `{signal}`\n"
                f"📍 **Entry**: `{price:.4f}`\n"
                f"🎯 **TP**: `{tp:.4f}` | 🛑 **SL**: `{sl:.4f}`\n"
                f"💰 **Suggested Size**: `${size}`\n"
                f"📊 **AI Confidence**: `{confidence}%`"
            )
            notify(alert_msg)

        except Exception as e:
            logging.error(f"AI Engine Error ({asset}): {e}")

    conn.close()

if __name__ == "__main__":
    run_ai_engine()
