import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
import logging
import sys
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, PERFORMANCE_FILE, ENGINES, ATR_PERIOD, ATR_MULTIPLIER_SL, RR_RATIO, DEFAULT_TIMEFRAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

APP_NAME = ENGINES["core"]
STRATEGY_ID = "core"

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
                if stats.get("sample_size", 0) > 5 and stats["win_rate"] < 40.0:
                    logging.warning(f"KILL SWITCH TRIGGERED for {STRATEGY_ID}: Win Rate {stats['win_rate']}%")
                    sys.exit(0)
                return stats["win_rate"]
    except Exception as e:
        logging.error(f"Learning error: {e}")
    return 50.0

def compute_indicators(df):
    df = df.copy()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    tr = pd.concat([(df['high']-df['low']), abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df

def should_insert_signal(cursor, asset, timeframe, new_signal):
    cursor.execute("""
        SELECT signal FROM signals 
        WHERE asset = ? AND timeframe = ? AND engine = ? 
        ORDER BY ts DESC LIMIT 1
    """, (asset, timeframe, STRATEGY_ID))
    last = cursor.fetchone()
    return not last or last[0] != new_signal

def run_alerts():
    ex = ccxt.xt({"enableRateLimit": True})
    ex.load_markets()

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

    current_win_rate = get_learned_confidence()

    for asset in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT", "DOGE/USDT", "TRX/USDT", "SUI/USDT", "PEPE/USDT"]:
        for tf in ["1h", "4h"]:
            try:
                data = ex.fetch_ohlcv(asset, tf, limit=100)
                df = compute_indicators(pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"]))

                last = df.iloc[-1]
                entry = last["close"]
                atr = last["atr"] if not np.isnan(last["atr"]) else entry * 0.01

                signal = "NEUTRAL"
                reason = "NONE"
                
                if last["rsi"] > 70:
                    signal = "SHORT"; reason = "OVERBOUGHT REJECTION"
                elif last["rsi"] < 30:
                    signal = "LONG"; reason = "OVERSOLD REVERSAL"
                elif last["close"] > last["ema20"] and df["close"].iloc[-2] <= df["ema20"].iloc[-2]:
                    signal = "LONG"; reason = "EMA20 CROSSOVER"
                elif last["close"] < last["ema20"] and df["close"].iloc[-2] >= df["ema20"].iloc[-2]:
                    signal = "SHORT"; reason = "EMA20 BREAKDOWN"
                elif last["close"] > last["ema20"]:
                    signal = "LONG"; reason = "MEAN REVERSION"
                elif last["close"] < last["ema20"]:
                    signal = "SHORT"; reason = "MEAN REVERSION"

                if signal != "NEUTRAL" and should_insert_signal(cursor, asset, tf, signal):
                    sl = (entry - ATR_MULTIPLIER_SL * atr) if signal == "LONG" else (entry + ATR_MULTIPLIER_SL * atr)
                    tp = (entry + RR_RATIO * abs(entry - sl)) if signal == "LONG" else (entry - RR_RATIO * abs(entry - sl))
                    
                    confidence = current_win_rate

                    cursor.execute("""
                        INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (STRATEGY_ID, asset, tf, signal, entry, sl, tp, confidence, reason, datetime.now().isoformat()))
                    conn.commit()

                    emoji = "🟢" if signal == "LONG" else "🔴"
                    notify(
                        f"{emoji} **{signal}** {asset} ({tf})\n"
                        f"**Confidence: {confidence}%** (📊 {reason})\n"
                        f"---\n"
                        f"Entry: {entry:.4f}\n"
                        f"SL: {sl:.4f} | TP: {tp:.4f}"
                    )
                    logging.info(f"Signal inserted: {signal} {asset} ({tf})")

                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error processing {asset} {tf}: {e}")

    conn.close()
    logging.info("Core engine run complete.")

if __name__ == "__main__":
    run_alerts()
