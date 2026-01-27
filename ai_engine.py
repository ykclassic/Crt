import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
import logging
import pickle
import os    # Added missing import
import json  # Added missing import
from datetime import datetime
from config import (
    DB_FILE, WEBHOOK_URL, MODEL_FILE, PERFORMANCE_FILE, 
    ENGINES, ATR_PERIOD, ATR_MULTIPLIER_SL, RR_RATIO, DEFAULT_TIMEFRAME
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

APP_NAME = ENGINES.get("ai", "NEXUS AI")
STRATEGY_ID = "ai"

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

def get_ai_prediction(rsi, price, vol_change, dist_ema):
    try:
        if not os.path.exists(MODEL_FILE):
            logging.warning(f"Model file {MODEL_FILE} not found. Using heuristic fallback.")
            return None
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        feat = np.array([[rsi, vol_change, dist_ema]])
        feat_scaled = scaler.transform(feat)
        return round(model.predict_proba(feat_scaled)[0][1] * 100, 2)
    except Exception as e:
        logging.error(f"AI prediction error: {e}")
        return None

def compute_atr(df):
    # Using 'high', 'low', 'close' column naming convention from CCXT
    tr = pd.concat([
        (df['high'] - df['low']), 
        abs(df['high'] - df['close'].shift()), 
        abs(df['low'] - df['close'].shift())
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
    # Primary data source for AI (XT Exchange)
    ex = ccxt.xt({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine TEXT, asset TEXT, timeframe TEXT, signal TEXT,
            entry REAL, sl REAL, tp REAL, confidence REAL,
            reason TEXT, ts TEXT
        )
    """)
    conn.commit()

    # Recovery/Kill-switch check
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                perf = json.load(f).get(STRATEGY_ID, {"status": "LIVE"})
                if perf.get("status") == "RECOVERY":
                    logging.info("AI Engine in RECOVERY mode. Skipping execution.")
                    conn.close()
                    return
        except Exception as e:
            logging.warning(f"Could not read performance file: {e}")

    for asset in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        try:
            data = ex.fetch_ohlcv(asset, DEFAULT_TIMEFRAME, limit=100)
            df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
            
            # Feature Engineering
            df['ema20'] = df['close'].ewm(span=20).mean()
            delta = df['close'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            df['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            df['vol_change'] = df['volume'].pct_change()
            df['dist_ema'] = (df['close'] - df['ema20']) / df['close']
            df['atr'] = compute_atr(df)
            
            last = df.iloc[-1]
            if np.isnan(last['rsi']): continue

            ai_conf = get_ai_prediction(last['rsi'], last['close'], last['vol_change'], last['dist_ema'])
            
            reason = "DEEP NETWORK" if ai_conf is not None else "HEURISTIC FALLBACK"
            final_conf = ai_conf if ai_conf is not None else 50.0
            signal = "LONG" if final_conf >= 50.0 else "SHORT"
            
            entry_price = last['close']
            atr = last['atr'] if not np.isnan(last['atr']) else entry_price * 0.02
            
            sl = (entry_price - ATR_MULTIPLIER_SL * atr) if signal == "LONG" else (entry_price + ATR_MULTIPLIER_SL * atr)
            tp = (entry_price + RR_RATIO * abs(entry_price - sl)) if signal == "LONG" else (entry_price - RR_RATIO * abs(entry_price - sl))

            if should_insert_signal(cursor, asset, DEFAULT_TIMEFRAME, signal):
                cursor.execute("""
                    INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts) 
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (STRATEGY_ID, asset, DEFAULT_TIMEFRAME, signal, entry_price, sl, tp, final_conf, reason, datetime.now().isoformat()))
                conn.commit()
                
                notify(f"🤖 **AI Prediction**\nAsset: {asset}\nSignal: {signal}\nConfidence: `{final_conf}%`\nReason: *{reason}*")
                logging.info(f"AI signal inserted: {signal} {asset}")
        except Exception as e:
            logging.error(f"AI error {asset}: {e}")
            
    conn.close()
    logging.info("AI engine run complete.")

if __name__ == "__main__":
    run_ai()
