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
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            pipeline = pickle.load(f)
        feat_df = pd.DataFrame([[rsi, vol_change, dist_ema]], columns=['rsi', 'vol_change', 'dist_ema'])
        probs = pipeline.predict_proba(feat_df)
        return round(probs[0][1] * 100, 2)
    except Exception as e:
        # SELF-HEALING: If the brain is incompatible, remove it so the next run can retrain
        logging.error(f"Brain Incompatibility Detected: {e}. Purging legacy brain.")
        try:
            os.remove(MODEL_FILE)
        except:
            pass
        return None

def run_ai_engine():
    logging.info("--- STARTING NEXUS AI ENGINE (SELF-HEALING MODE) ---")
    
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
            
            # --- Technical Logic ---
            price = df['c'].iloc[-1]
            delta = df['c'].diff()
            rsi = (100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean() / -delta.clip(upper=0).rolling(14).mean()))).iloc[-1]
            vol_change = df['v'].pct_change().iloc[-1]
            dist_ema = (price - df['c'].ewm(span=20).mean().iloc[-1]) / price

            # Data Cleaning
            rsi = 50.0 if np.isnan(rsi) else rsi
            vol_change = 0.0 if np.isnan(vol_change) else vol_change

            conf = get_ensemble_prediction(rsi, vol_change, dist_ema)
            
            # --- FALLBACK: Alert even if AI is dead ---
            if conf is None:
                logging.info(f"Using Technical Fallback for {asset}")
                if rsi < 35: conf = 75.0  # Simple oversold logic
                elif rsi > 65: conf = 25.0 # Simple overbought logic
                else: conf = 50.0

            if conf >= MIN_ENSEMBLE_CONFIDENCE or conf <= (100 - MIN_ENSEMBLE_CONFIDENCE):
                signal = "LONG" if conf > 50 else "SHORT"
                # Save Signal
                cursor.execute("""
                    INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, rsi, vol_change, dist_ema, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, ('ai', asset, '1h', signal, price, price*0.98, price*1.04, conf, rsi, vol_change, dist_ema, datetime.now().isoformat()))
                logging.info(f"✅ ALERT GENERATED: {asset} {signal} ({conf}%)")

        except Exception as e:
            logging.error(f"Asset Error {asset}: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    run_ai_engine()
