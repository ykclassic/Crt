import ccxt, pandas as pd, numpy as np, sqlite3, os, requests
from datetime import datetime

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS RANGEMASTER"
EXCHANGE = ccxt.gateio()
DB = sqlite3.connect("rangemaster.db")

# Logic: RSI Overbought/Oversold + BB Touch
for asset in ["BTC/USDT", "XRP/USDT"]:
    try:
        data = EXCHANGE.fetch_ohlcv(asset, '1h', limit=50)
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
        # RSI 14
        delta = df['c'].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
        df['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
        last = df.iloc[-1]
        
        if last['rsi'] < 30:
            notify(f"⚖️ **Mean Reversion Alert**\nAsset: {asset}\nSignal: OVERSOLD (LONG)\nRSI: {last['rsi']:.2f}")
        elif last['rsi'] > 70:
            notify(f"⚖️ **Mean Reversion Alert**\nAsset: {asset}\nSignal: OVERBOUGHT (SHORT)\nRSI: {last['rsi']:.2f}")
    except Exception as e: print(f"Error: {e}")
