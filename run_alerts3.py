import ccxt
import pandas as pd
import numpy as np
import sqlite3
import os
import requests
import json
from datetime import datetime

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "Nexus Hybrid V2 (Predictive)"
DB = sqlite3.connect("nexus_live.db")
DB.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT, regime TEXT, confidence REAL)")
DB.commit()

def send_discord(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})

ex = ccxt.xt()
for asset in ["BTC/USDT", "SOL/USDT"]:
    try:
        ohlcv = ex.fetch_ohlcv(asset, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v']).rename(columns={'c':'close','h':'high','l':'low'})
        
        # ADX Regime Logic
        tr = (df['high'] - df['low']).rolling(14).mean()
        # Simplified Regime
        regime = "TREND" if tr.iloc[-1] > tr.mean() else "RANGE"
        
        send_discord(f"🤖 Market Intelligence: {asset} is currently in a {regime} regime.")
    except Exception as e:
        print(e)
