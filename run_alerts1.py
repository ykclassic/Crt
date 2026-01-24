import ccxt
import pandas as pd
import sqlite3
import requests
import logging
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, PERFORMANCE_FILE, ENGINES, ATR_PERIOD, ATR_MULTIPLIER_SL, RR_RATIO, DEFAULT_TIMEFRAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

APP_NAME = ENGINES["hybrid"]
STRATEGY_ID = "hybrid"

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

def get_learned_confidence():
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                data = json.load(f)
                stats = data.get(STRATEGY_ID, {"win_rate": 50.0, "status": "LIVE", "sample_size": 0})
                if stats.get("status") == "RECOVERY" or (stats.get("sample_size", 0) > 5 and stats["win_rate"] < 40.0):
                    logging.warning(f"{STRATEGY_ID} in RECOVERY/KILLED mode. Skipping.")
                    return None
                return stats["win_rate"]
    except Exception as e:
        logging.error(f"Learning error: {e}")
    return 50.0

def compute_atr(df):
    tr = pd.concat([(df['h']-df['l']), abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()

def should_insert_signal(cursor, asset, timeframe, new_signal):
    cursor.execute("""
        SELECT signal FROM signals 
        WHERE asset = ? AND timeframe = ? AND engine = ? 
        ORDER BY ts DESC LIMIT 1
    """, (asset, timeframe, STRATEGY_ID))
    last = cursor.fetchone()
    return not last or last[0] != new_signal

def run_hybrid():
    ex = ccxt.xt({"enableRateLimit": True})
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Table creation is now centralized – just ensure it exists
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

    current_conf = get_learned_confidence()
    if current_conf is None:
        conn.close()
        return

    for asset in ["SOL/USDT", "BTC/USDT", "ETH/USDT"]:
        try:
            data = ex.fetch_ohlcv(asset, DEFAULT_TIMEFRAME, limit=100)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
            
            df['sma50'] = df['c'].rolling(50).mean()
            df['atr'] = compute_atr(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            atr = last['atr'] if not np.isnan(last['atr']) else last['c'] * 0.01
            
            signal = None
            reason = "NONE"

            if last['c'] > last['sma50'] and prev['c'] <= prev['sma50']:
                signal = "LONG"; reason = "TREND BREAKOUT"
            elif last['c'] < last['sma50'] and prev['c'] >= prev['sma50']:
                signal = "SHORT"; reason = "TREND BREAKDOWN"
            elif last['c'] > last['sma50']:
                signal = "LONG"; reason = "BULLISH MOMENTUM"
            else:
                signal = "SHORT"; reason = "BEARISH MOMENTUM"

            if signal and should_insert_signal(cursor, asset, DEFAULT_TIMEFRAME, signal):
                entry = last['c']
                sl = (entry - ATR_MULTIPLIER_SL * atr) if signal == "LONG" else (entry + ATR_MULTIPLIER_SL * atr)
                tp = (entry + RR_RATIO * abs(entry - sl)) if signal == "LONG" else (entry - RR_RATIO * abs(entry - sl))

                cursor.execute("""
                    INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts) 
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (STRATEGY_ID, asset, DEFAULT_TIMEFRAME, signal, entry, sl, tp, current_conf, reason, datetime.now().isoformat()))
                conn.commit()
                
                notify(f"🔄 **Hybrid Alert**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {current_conf}%** (📊 {reason})\n---\nEntry: {entry:.2f}\nSL: {sl:.2f} | TP: {tp:.2f}")
                logging.info(f"Hybrid signal inserted: {signal} {asset}")
        except Exception as e: 
            logging.error(f"Hybrid error {asset}: {e}")
            
    conn.close()
    logging.info("Hybrid engine run complete.")

if __name__ == "__main__":
    run_hybrid()
