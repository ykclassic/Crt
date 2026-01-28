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
    ENGINES, ASSETS, RISK_PER_TRADE, TOTAL_CAPITAL, ATR_MULTIPLIER, 
    MIN_CONFIDENCE_FOR_SIZE_BOOST, MIN_ENSEMBLE_CONFIDENCE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

STRATEGY_ID = "ai"
APP_NAME = ENGINES.get(STRATEGY_ID, "Nexus AI")

def notify(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg})

def is_engine_enabled():
    if not os.path.exists(PERFORMANCE_FILE): return True
    try:
        with open(PERFORMANCE_FILE, "r") as f:
            perf = json.load(f)
            status = perf.get(STRATEGY_ID, {}).get("status", "LIVE")
            return status == "LIVE"
    except: return True

def get_ensemble_prediction(rsi, vol_change, dist_ema):
    try:
        if not os.path.exists(MODEL_FILE): return None
        with open(MODEL_FILE, "rb") as f:
            ensemble, scaler = pickle.load(f)
        feat_scaled = scaler.transform(np.array([[rsi, vol_change, dist_ema]]))
        # Average probability across XGB, RF, and GBT
        return round(ensemble.predict_proba(feat_scaled)[0][1] * 100, 2)
    except Exception as e:
        logging.error(f"Ensemble Prediction error: {e}")
        return None

def calculate_dynamic_risk(df, current_price, signal, confidence):
    high_low = df['h'] - df['l']
    atr = high_low.rolling(14).mean().iloc[-1]
    sl_dist = atr * ATR_MULTIPLIER
    sl = current_price - sl_dist if signal == "LONG" else current_price + sl_dist
    tp = current_price + (sl_dist * 2) if signal == "LONG" else current_price - (sl_dist * 2)
    
    risk_amt = TOTAL_CAPITAL * RISK_PER_TRADE
    if confidence >= MIN_CONFIDENCE_FOR_SIZE_BOOST: risk_amt *= 1.5 
    
    size = risk_amt / (sl_dist / current_price)
    return round(sl, 4), round(tp, 4), round(size, 2)

def run_ai_engine():
    if not is_engine_enabled(): return

    ex = ccxt.xt()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=100)
            df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # Indicator Calculations
            price = df['c'].iloc[-1]
            rsi = (100 - (100 / (1 + df['c'].diff().clip(lower=0).rolling(14).mean() / -df['c'].diff().clip(upper=0).rolling(14).mean()))).iloc[-1]
            vol_change = df['v'].pct_change().iloc[-1]
            dist_ema = (price - df['c'].ewm(span=20).mean().iloc[-1]) / price

            # THE ENSEMBLE VOTE
            confidence = get_ensemble_prediction(rsi, vol_change, dist_ema)
            
            if confidence is None or (50 < confidence < MIN_ENSEMBLE_CONFIDENCE):
                logging.info(f"Skipping {asset}: Low Committee Consensus ({confidence}%)")
                continue

            signal = "LONG" if confidence > 50 else "SHORT"
            sl, tp, size = calculate_dynamic_risk(df, price, signal, confidence)

            cursor.execute("""
                INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (STRATEGY_ID, asset, '1h', signal, price, sl, tp, confidence, "PHASE 3: ENSEMBLE CONSENSUS", datetime.now().isoformat()))
            conn.commit()

            notify(f"🤖 **{APP_NAME}**\n━━━━━━━━━━━━━━━\n📈 **Asset**: `{asset}` | **Signal**: `{signal}`\n📍 **Entry**: `{price:.4f}` | 💰 **Size**: `${size}`\n🎯 **TP**: `{tp:.4f}` | 🛑 **SL**: `{sl:.4f}`\n📊 **Committee Consensus**: `{confidence}%`")

        except Exception as e:
            logging.error(f"AI Error ({asset}): {e}")
    conn.close()

if __name__ == "__main__":
    run_ai_engine()
