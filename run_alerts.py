# Updated core engine (run_alerts.py) with MACD, OBV, regime detection, multi-exchange, sentiment, on-chain
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
import logging
import sys
import time
import json
import os
from datetime import datetime
from config import (
    DB_FILE, WEBHOOK_URL, PERFORMANCE_FILE, ENGINES, 
    ATR_PERIOD, ATR_MULTIPLIER_SL, RR_RATIO, 
    DEFAULT_TIMEFRAME, DRY_RUN, ASSETS, TIMEFRAMES
)

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
        logging.error(f"Learning Error: {e}")
    return 50.0

def fetch_multi_exchange_data(asset, tf, limit=100):
    exchanges = [ccxt.binance(), ccxt.gateio()]
    all_data = []
    
    for ex in exchanges:
        try:
            data = ex.fetch_ohlcv(asset, tf, limit=limit)
            if data:
                df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                all_data.append(df)
        except Exception as e:
            logging.warning(f"Fetch failed on {ex.id}: {e}")
            
    if not all_data:
        raise ValueError(f"No data fetched for {asset}")
        
    df_avg = pd.concat(all_data).groupby(level=0).mean()
    return df_avg

def calculate_adx(df, period=14):
    df = df.copy()
    alpha = 1/period
    df['h-l'] = df['high'] - df['low']
    df['h-c'] = abs(df['high'] - df['close'].shift(1))
    df['l-c'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-c', 'l-c']].max(axis=1)
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['pdm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['ndm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['tr_ema'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['pdm_ema'] = df['pdm'].ewm(alpha=alpha, adjust=False).mean()
    df['ndm_ema'] = df['ndm'].ewm(alpha=alpha, adjust=False).mean()
    df['pdi'] = 100 * (df['pdm_ema'] / df['tr_ema'])
    df['ndi'] = 100 * (df['ndm_ema'] / df['tr_ema'])
    df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'])
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    return df['adx']

def compute_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    tr = pd.concat([(df['high']-df['low']), abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['adx'] = calculate_adx(df)
    return df

def get_sentiment_score(asset):
    return 0.6 # Mock sentiment

def get_on_chain_data(asset):
    return 'normal'

def detect_regime(df):
    if len(df) < 50: return "range_bound"
    last_adx = df['adx'].iloc[-1]
    volatility = df['close'].pct_change().rolling(14).std().iloc[-1]
    if last_adx > 25 and volatility > 0.02:
        return "bull_trend" if df['close'].iloc[-1] > df['ema50'].iloc[-1] else "bear_trend"
    return "range_bound"

def run_alerts():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Added 'engine' column to the schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine TEXT,
            asset TEXT, timeframe TEXT, signal TEXT, 
            entry REAL, sl REAL, tp REAL, confidence REAL, 
            reason TEXT, ts TEXT
        )
    """)
    conn.commit()

    current_win_rate = get_learned_confidence()
    logging.info(f"Analyzing {len(ASSETS)} assets...")

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            try:
                df = fetch_multi_exchange_data(asset, tf)
                if df.empty: continue
                df = compute_indicators(df)
                last = df.iloc[-1]
                entry = last["close"]
                atr = last["atr"] if not np.isnan(last["atr"]) else entry * 0.01

                signal = "NEUTRAL"
                reason = "NONE"
                regime = detect_regime(df)
                sentiment = get_sentiment_score(asset)
                
                if regime == "bull_trend" and sentiment > 0.5:
                    if last["rsi"] < 40 or (last["macd"] > last["macd_signal"] and last["obv"] > df['obv'].mean()):
                        signal = "LONG"; reason = "Bull Regime + Tech Alignment"
                elif regime == "bear_trend":
                    if last["rsi"] > 60 or (last["macd"] < last["macd_signal"] and last["obv"] < df['obv'].mean()):
                        signal = "SHORT"; reason = "Bear Regime + Tech Breakdown"
                
                if signal == "NEUTRAL" and regime == "range_bound":
                    if last["rsi"] < 30: signal = "LONG"; reason = "Range Bottom"
                    elif last["rsi"] > 70: signal = "SHORT"; reason = "Range Top"

                if signal != "NEUTRAL":
                    cursor.execute("SELECT signal FROM signals WHERE asset=? AND timeframe=? ORDER BY id DESC LIMIT 1", (asset, tf))
                    last_saved = cursor.fetchone()

                    if not last_saved or last_saved[0] != signal:
                        sl = (entry - ATR_MULTIPLIER_SL * atr) if signal == "LONG" else (entry + ATR_MULTIPLIER_SL * atr)
                        tp = (entry + RR_RATIO * abs(entry - sl)) if signal == "LONG" else (entry - RR_RATIO * abs(entry - sl))
                        
                        if not DRY_RUN:
                            cursor.execute("""
                                INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (STRATEGY_ID, asset, tf, signal, entry, sl, tp, current_win_rate, reason, datetime.now().isoformat()))
                            conn.commit()

                        notify(f"{'🟢' if signal == 'LONG' else '🔴'} **{signal}** {asset} ({tf})\n"
                               f"Confidence: {current_win_rate}% | {reason}")
                        logging.info(f"Signal Generated: {signal} {asset}")

                time.sleep(0.5) 
            except Exception as e:
                logging.error(f"Error processing {asset}: {e}")

    conn.close()

if __name__ == "__main__":
    run_alerts()
