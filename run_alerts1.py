import ccxt
import pandas as pd
import numpy as np
import sqlite3
import os
import requests
import json
from datetime import datetime, timezone

# Configuration
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT"]
TIMEFRAMES = ["1h", "4h"]
EXCHANGE_NAME = "XT"
APP_NAME = "Nexus Hybrid V1"

DB = sqlite3.connect("hybrid_signals.db")
DB.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, asset TEXT, signal TEXT, confidence REAL, status TEXT)")
DB.commit()

def send_discord(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def compute_indicators(df):
    df = df.copy()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    df["ema20"] = df["close"].ewm(span=20).mean()
    return df

ex = ccxt.xt() if EXCHANGE_NAME == "XT" else ccxt.gateio()

for asset in ASSETS:
    for tf in TIMEFRAMES:
        try:
            ohlcv = ex.fetch_ohlcv(asset, tf, limit=50)
            df = compute_indicators(pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v']).rename(columns={'c':'close'}))
            last = df.iloc[-1]
            
            signal = "NEUTRAL"
            if last['close'] > last['ema20'] and last['rsi'] < 70: signal = "LONG"
            elif last['close'] < last['ema20'] and last['rsi'] > 30: signal = "SHORT"
            
            if signal != "NEUTRAL":
                # Only alert if new
                check = DB.execute("SELECT id FROM signals WHERE asset=? AND signal=? ORDER BY id DESC LIMIT 1", (asset, signal)).fetchone()
                if not check:
                    DB.execute("INSERT INTO signals (timestamp, asset, signal, status) VALUES (?,?,?,?)", 
                               (datetime.now().isoformat(), asset, signal, "OPEN"))
                    DB.commit()
                    send_discord(f"🚀 New Signal: {signal} on {asset} ({tf}) | Price: {last['close']}")
        except Exception as e:
            print(f"Error: {e}")
