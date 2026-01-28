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
                reason = engine_data.get("reason", "Low Win Rate")
                logging.warning(f"Engine {STRATEGY_ID} Disabled: {reason}")
                return False
            return True
    except: return True

def get_ensemble_prediction(rsi, vol_change, dist_ema):
    try:
        if not os.path.exists(MODEL_FILE): return None
        with open(MODEL_FILE, "rb") as f:
            ensemble, scaler = pickle.load(f)
        feat_scaled = scaler.transform(np.array([[rsi, vol_change, dist_ema]]))
        # Soft voting returns probability: [loss_prob, win_prob]
        return round(ensemble.predict_proba(feat_scaled)[0][1] * 100, 2)
    except Exception as e:
        logging.error(f"Ensemble Prediction error: {e}")
        return None

def calculate_dynamic_risk(df, current_price, signal, confidence):
    # Standard ATR Calculation
    high_low = df['h'] - df['l']
    high_close = np.abs(df['h'] - df['c'].shift())
    low_close = np.abs(df['l'] - df['c'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]

    sl_dist = atr * ATR_MULTIPLIER
    sl = current_price - sl_dist if signal == "LONG" else current_price + sl_dist
    tp = current_price + (sl_dist * 2) if signal == "LONG" else current_price - (sl_dist * 2)

    risk_amt = TOTAL_CAPITAL * RISK_PER_TRADE
    if confidence >= MIN_CONFIDENCE_FOR_SIZE_BOOST:
        risk_amt *= 1.5 
        
    sl_pct = sl_dist / current_price
    suggested_size = risk_amt / sl_pct

    return round(sl, 4), round(tp, 4), round(suggested_size, 2)

def run_ai_engine():
    if not is_engine_enabled():
        return

    # FIX: Using 'gateio' or 'binance' is more stable for testing, 
    # but for XT.com the correct call is usually 'xtcom' or 'xt' 
    # depending on CCXT version. Let's use gateio as a reliable default 
    # if xt isn't working, or fix the XT call:
    try:
        ex = ccxt.xtcom({"enableRateLimit": True})
    except AttributeError:
        logging.warning("XT.com attribute not found, falling back to Gate.io for data fetch")
        ex = ccxt.gateio({"enableRateLimit": True})

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=100)
            df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # Technical Indicators
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            rsi = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            ema = df['c'].ewm(span=20).mean()
            
            price = df['c'].iloc[-1]
            vol_change = df['v'].pct_change().iloc[-1]
            dist_ema = (price - ema.iloc[-1]) / price
            
            # THE ENSEMBLE VOTE (Phase 3)
            confidence = get_ensemble_prediction(rsi.iloc[-1], vol_change, dist_ema) or 52.0
            
            # Filter by Committee Consensus
            if 50 < confidence < MIN_ENSEMBLE_CONFIDENCE:
                logging.info(f"Skipping {asset}: Low Consensus ({confidence}%)")
                continue

            signal = "LONG" if confidence > 50 else "SHORT"
            sl, tp, size = calculate_dynamic_risk(df, price, signal, confidence)

            cursor.execute("""
                INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (STRATEGY_ID, asset, '1h', signal, price, sl, tp, confidence, "PHASE 3 ENSEMBLE", datetime.now().isoformat()))
            conn.commit()

            alert_msg = (
                f"🤖 **{APP_NAME}**\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📈 **Asset**: `{asset}` | **Signal**: `{signal}`\n"
                f"📍 **Entry**: `{price:.4f}`\n"
                f"🎯 **TP**: `{tp:.4f}` | 🛑 **SL**: `{sl:.4f}`\n"
                f"💰 **Suggested Size**: `${size}`\n"
                f"📊 **Committee Consensus**: `{confidence}%`"
            )
            notify(alert_msg)

        except Exception as e:
            logging.error(f"AI Engine Error ({asset}): {e}")

    conn.close()

if __name__ == "__main__":
    run_ai_engine()
