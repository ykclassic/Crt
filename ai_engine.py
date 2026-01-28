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

# Set up clean logging for GitHub Actions
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def get_ensemble_prediction(rsi, vol_change, dist_ema):
    """Predicts signal probability using the saved Scikit-Learn Pipeline."""
    try:
        if not os.path.exists(MODEL_FILE):
            logging.warning("AI Brain file not found. Skipping AI prediction.")
            return None
            
        with open(MODEL_FILE, "rb") as f:
            pipeline = pickle.load(f)
        
        # Create a DataFrame to match the feature names used during training
        # This allows the Imputer to handle NaNs if technical indicators fail
        feat_df = pd.DataFrame(
            [[rsi, vol_change, dist_ema]], 
            columns=['rsi', 'vol_change', 'dist_ema']
        )
        
        # predict_proba returns [prob_loss, prob_win]
        probs = pipeline.predict_proba(feat_df)
        return round(probs[0][1] * 100, 2)
    except Exception as e:
        logging.error(f"Ensemble Prediction Error: {e}")
        return None

def run_ai_engine():
    logging.info("--- STARTING NEXUS AI ENGINE (PHASE 3) ---")
    
    # Correcting the XT.com initialization to prevent AttributeError
    try:
        # ccxt.xt() is the standard ID for XT.com
        ex = ccxt.xt({'enableRateLimit': True})
    except AttributeError:
        logging.warning("XT attribute error. Falling back to Gate.io for market data.")
        ex = ccxt.gateio({'enableRateLimit': True})

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            # Fetch data (1h timeframe for AI features)
            ohlcv = ex.fetch_ohlcv(asset, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # --- Feature Engineering ---
            price = df['c'].iloc[-1]
            
            # RSI Calculation
            delta = df['c'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Volume and Trend features
            vol_change = df['v'].pct_change().iloc[-1]
            ema_20 = df['c'].ewm(span=20).mean().iloc[-1]
            dist_ema = (price - ema_20) / price

            # --- AI Decision Logic ---
            confidence = get_ensemble_prediction(rsi, vol_change, dist_ema)
            
            # If no model exists, we default to a baseline or skip
            if confidence is None:
                continue

            # Determine if consensus meets the threshold
            if confidence >= MIN_ENSEMBLE_CONFIDENCE or confidence <= (100 - MIN_ENSEMBLE_CONFIDENCE):
                signal = "LONG" if confidence > 50 else "SHORT"
                
                # Placeholder Dynamic Risk (Can be replaced by ATR logic)
                sl = price * 0.98 if signal == "LONG" else price * 1.02
                tp = price * 1.04 if signal == "LONG" else price * 0.96
                
                # Save to Database with full feature set for future retraining
                cursor.execute("""
                    INSERT INTO signals (
                        engine, asset, timeframe, signal, entry, sl, tp, 
                        confidence, rsi, vol_change, dist_ema, ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'ai', asset, '1h', signal, price, sl, tp, 
                    confidence, rsi, vol_change, dist_ema, datetime.now().isoformat()
                ))
                
                logging.info(f"🚀 Signal Generated: {asset} {signal} ({confidence}%)")
            else:
                logging.info(f"Skipping {asset}: Low Confidence ({confidence}%)")

        except Exception as e:
            logging.error(f"Error processing {asset}: {e}")

    conn.commit()
    conn.close()
    logging.info("--- AI ENGINE CYCLE COMPLETE ---")

if __name__ == "__main__":
    run_ai_engine()
