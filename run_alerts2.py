import ccxt
import pandas as pd
import numpy as np
import sqlite3
import os
import requests
from datetime import datetime

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "Nexus RangeMaster"
DB = sqlite3.connect("rangemaster_signals.db")
DB.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT, signal TEXT, timestamp TEXT)")
DB.commit()

def send_discord(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def get_range_signal(df):
    last = df.iloc[-1]
    # BB Math
    mid = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    upper, lower = mid + (2*std), mid - (2*std)
    
    sig = "NEUTRAL"
    if last['close'] < lower.iloc[-1]: sig = "LONG"
    elif last['close'] > upper.iloc[-1]: sig = "SHORT"
    return sig

ex = ccxt.gateio()
for asset in ["BTC/USDT", "ETH/USDT"]:
    try:
        ohlcv = ex.fetch_ohlcv(asset, '1h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v']).rename(columns={'c':'close'})
        signal = get_range_signal(df)
        
        if signal != "NEUTRAL":
            send_discord(f"📉 Range Edge Detected: {signal} on {asset} | Mean Reversion Play")
    except Exception as e:
        print(e)
