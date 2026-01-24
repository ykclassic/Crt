import ccxt
import pandas as pd
import numpy as np
import os
import sqlite3
import requests
import logging
import pickle
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, MODEL_FILE, PERFORMANCE_FILE, ENGINES, ATR_PERIOD, ATR_MULTIPLIER_SL, RR_RATIO, DEFAULT_TIMEFRAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

APP_NAME = ENGINES["ai"]
STRATEGY_ID = "ai"

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

def get_ai_prediction(features):
    """
    features: dict with keys:
    'rsi', 'vol_change', 'dist_ema', 'atr_norm',
    'return_lag_1', 'return_lag_3', 'return_lag_6'
    """
    try:
        if not os.path.exists(MODEL_FILE):
            logging.warning("Model file missing - falling back to heuristic")
            return None
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        
        feat_array = np.array([[
            features['rsi'],
            features['vol_change'],
            features['dist_ema'],
            features['atr_norm'],
            features['return_lag_1'],
            features['return_lag_3'],
            features['return_lag_6']
        ]])
        feat_scaled = scaler.transform(feat_array)
        prob_up = model.predict_proba(feat_scaled)[0][1]
        return round(prob_up * 100, 2)
    except Exception as e:
        logging.error(f"AI prediction error: {e}")
        return None

def compute_atr(df):
    tr = pd.concat([
        (df['h'] - df['l']),
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()

def should_insert_signal(cursor, asset, timeframe, new_signal):
    cursor.execute("""
        SELECT signal FROM signals 
        WHERE asset = ? AND timeframe = ? AND engine = ? 
        ORDER BY ts DESC LIMIT 1
    """, (asset, timeframe, STRATEGY_ID))
    last = cursor.fetchone()
    return not last or last[0] != new_signal

def run_ai():
    ex = ccxt.xt({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine TEXT,
            asset TEXT,
            timeframe TEXT,
            signal TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            confidence REAL,
            reason TEXT,
            ts TEXT
        )
    """)
    conn.commit()

    # Recovery check
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                perf = json.load(f).get(STRATEGY_ID, {"status": "LIVE"})
                if perf.get("status") == "RECOVERY":
                    logging.info("AI engine in RECOVERY mode - skipping run")
                    conn.close()
                    return
        except Exception as e:
            logging.error(f"Performance file read error: {e}")

    for asset in ["BTC/USDT", "ETH/USDT"]:
        try:
            data = ex.fetch_ohlcv(asset, DEFAULT_TIMEFRAME, limit=100)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
            
            # Indicators
            df['ema20'] = df['c'].ewm(span=20).mean()
            
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            df['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            
            df['vol_change'] = df['v'].pct_change()
            df['dist_ema'] = (df['c'] - df['ema20']) / df['c'] * 100
            
            df['atr'] = compute_atr(df)
            df['atr_norm'] = df['atr'] / df['c'] * 100
            
            # Lagged returns
            for lag in [1, 3, 6]:
                df[f'return_lag_{lag}'] = df['c'].pct_change(lag) * 100
            
            last = df.iloc[-1]
            
            # Fill any NaN with safe defaults
            atr_norm = last['atr_norm'] if not np.isnan(last['atr_norm']) else 0.0
            return_lag_1 = last.get('return_lag_1', 0.0) if not np.isnan(last.get('return_lag_1', 0.0)) else 0.0
            return_lag_3 = last.get('return_lag_3', 0.0) if not np.isnan(last.get('return_lag_3', 0.0)) else 0.0
            return_lag_6 = last.get('return_lag_6', 0.0) if not np.isnan(last.get('return_lag_6', 0.0)) else 0.0
            
            ai_conf = get_ai_prediction({
                'rsi': last['rsi'],
                'vol_change': last['vol_change'],
                'dist_ema': last['dist_ema'],
                'atr_norm': atr_norm,
                'return_lag_1': return_lag_1,
                'return_lag_3': return_lag_3,
                'return_lag_6': return_lag_6
            })
            
            reason = "DEEP NETWORK" if ai_conf is not None else "HEURISTIC FALLBACK"
            final_conf = ai_conf if ai_conf is not None else 52.0
            signal = "LONG" if final_conf > 50 else "SHORT"
            
            price = last['c']
            atr = last['atr'] if not np.isnan(last['atr']) else price * 0.01
            sl = (price - ATR_MULTIPLIER_SL * atr) if signal == "LONG" else (price + ATR_MULTIPLIER_SL * atr)
            tp = (price + RR_RATIO * abs(price - sl)) if signal == "LONG" else (price - RR_RATIO * abs(price - sl))

            if should_insert_signal(cursor, asset, DEFAULT_TIMEFRAME, signal):
                cursor.execute("""
                    INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts) 
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (STRATEGY_ID, asset, DEFAULT_TIMEFRAME, signal, price, sl, tp, final_conf, reason, datetime.now().isoformat()))
                conn.commit()
                
                notify(f"🤖 **AI Prediction**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {final_conf}%** (📊 {reason})")
                logging.info(f"AI signal inserted: {signal} {asset} (Conf: {final_conf}%)")
        except Exception as e:
            logging.error(f"AI error {asset}: {e}")
    
    conn.close()
    logging.info("AI engine run complete.")

if __name__ == "__main__":
    run_ai()
