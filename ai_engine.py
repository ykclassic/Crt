import ccxt
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
import logging
from datetime import datetime
from config import DB_FILE, MODEL_FILE, ASSETS, ENGINES, MIN_ENSEMBLE_CONFIDENCE

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def get_ensemble_prediction(rsi, vol_change, dist_ema):
    try:
        if not os.path.exists(MODEL_FILE): return None
        with open(MODEL_FILE, "rb") as f:
            pipeline = pickle.load(f)
        
        # We pass raw data; the pipeline handles imputation and scaling internally
        feat = pd.DataFrame([[rsi, vol_change, dist_ema]], columns=['rsi', 'vol_change', 'dist_ema'])
        probs = pipeline.predict_proba(feat)
        return round(probs[0][1] * 100, 2)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

def run_ai_engine():
    # ... [Same initialization as previous version] ...
    try:
        ex = ccxt.gateio() # High reliability fallback
    except: return

    conn = sqlite3.connect(DB_FILE)
    for asset in ASSETS:
        try:
            ohlcv = ex.fetch_ohlcv(asset, '1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # Feature calculation
            price = df['c'].iloc[-1]
            rsi = (100 - (100 / (1 + df['c'].diff().clip(lower=0).rolling(14).mean() / -df['c'].diff().clip(upper=0).rolling(14).mean()))).iloc[-1]
            vol_change = df['v'].pct_change().iloc[-1]
            dist_ema = (price - df['c'].ewm(span=20).mean().iloc[-1]) / price

            conf = get_ensemble_prediction(rsi, vol_change, dist_ema)
            
            if conf and (conf >= MIN_ENSEMBLE_CONFIDENCE or conf <= (100 - MIN_ENSEMBLE_CONFIDENCE)):
                signal = "LONG" if conf > 50 else "SHORT"
                # ... [Rest of signal saving logic from previous version] ...
                logging.info(f"Signal Generated: {asset} {signal} ({conf}%)")
        except Exception as e:
            logging.error(f"Asset Error {asset}: {e}")
    conn.close()

if __name__ == "__main__":
    run_ai_engine()
