import ccxt
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
import logging
from datetime import datetime
from config import (
    DB_FILE, MODEL_FILE, ASSETS, ENGINES, 
    MIN_ENSEMBLE_CONFIDENCE, TOTAL_CAPITAL, RISK_PER_TRADE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def get_ensemble_prediction(rsi, vol_change, dist_ema):
    if not os.path.exists(MODEL_FILE):
        return None # Signal for Fallback Mode
    try:
        with open(MODEL_FILE, "rb") as f:
            pipeline = pickle.load(f)
        feat_df = pd.DataFrame([[rsi, vol_change, dist_ema]], columns=['rsi', 'vol_change', 'dist_ema'])
        probs = pipeline.predict_proba(feat_df)
        return round(probs[0][1] * 100, 2)
    except:
        return None

def run_ai_engine():
    logging.info("--- 🚨 EXECUTING AI ENGINE (STABILITY PATCH) ---")
    
    # Priority: Gate.io (Higher uptime for OHLCV)
    try:
        ex = ccxt.gateio({'enableRateLimit': True})
    except:
        ex = ccxt.xt()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            ohlcv = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # --- Feature Engineering with NaN Protection ---
            price = df['c'].iloc[-1]
            delta = df['c'].diff()
            rsi = (100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean() / -delta.clip(upper=0).rolling(14).mean()))).iloc[-1]
            vol_change = df['v'].pct_change().iloc[-1]
            dist_ema = (price - df['c'].ewm(span=20).mean().iloc[-1]) / price

            # Handle NaNs (Replace with 0 or neutral)
            rsi = rsi if not np.isnan(rsi) else 50.0
            vol_change = vol_change if not np.isnan(vol_change) else 0.0

            conf = get_ensemble_prediction(rsi, vol_change, dist_ema)
            
            # --- FALLBACK LOGIC ---
            # If AI is training/missing, use Technical Baseline (RSI + EMA)
            if conf is None:
                logging.info(f"AI Brain missing. Using Core Baseline for {asset}.")
                if rsi < 30 and dist_ema < -0.02: conf = 80.0 # High confidence LONG
                elif rsi > 70 and dist_ema > 0.02: conf = 20.0 # High confidence SHORT
                else: conf = 50.0 # Neutral

            if conf >= MIN_ENSEMBLE_CONFIDENCE or conf <= (100 - MIN_ENSEMBLE_CONFIDENCE):
                signal = "LONG" if conf > 50 else "SHORT"
                sl = price * 0.98 if signal == "LONG" else price * 1.02
                tp = price * 1.04 if signal == "LONG" else price * 0.96
                
                cursor.execute("""
                    INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, rsi, vol_change, dist_ema, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, ('ai', asset, '1h', signal, price, sl, tp, conf, rsi, vol_change, dist_ema, datetime.now().isoformat()))
                logging.info(f"✅ SIGNAL SAVED: {asset} {signal} ({conf}%)")
            else:
                logging.info(f"⏸️ {asset} Filtered: Confidence {conf}% below threshold.")

        except Exception as e:
            logging.error(f"Critical Error on {asset}: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    run_ai_engine()
