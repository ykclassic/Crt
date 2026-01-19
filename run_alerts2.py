import ccxt, pandas as pd, numpy as np, sqlite3, os, requests, json
from datetime import datetime

# --- CONFIG ---
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS HYBRID V1"
EXCHANGE = ccxt.xt()
DB = sqlite3.connect("hybrid_v1.db")
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

# Database Init
DB.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, asset TEXT, signal TEXT, ts TEXT)")

for asset in ASSETS:
    try:
        data = EXCHANGE.fetch_ohlcv(asset, '1h', limit=100)
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
        df['ema20'] = df['c'].ewm(span=20).mean()
        last = df.iloc[-1]
        
        signal = "LONG" if last['c'] > last['ema20'] else "SHORT"
        
        # Prevent duplicate alerts
        last_sig = DB.execute("SELECT signal FROM signals WHERE asset=? ORDER BY id DESC LIMIT 1", (asset,)).fetchone()
        if not last_sig or last_sig[0] != signal:
            DB.execute("INSERT INTO signals (asset, signal, ts) VALUES (?,?,?)", (asset, signal, datetime.now().isoformat()))
            DB.commit()
            notify(f"🚀 **Trend Change Detected**\nAsset: {asset}\nDirection: {signal}\nPrice: {last['c']}")
    except Exception as e: print(f"Error {asset}: {e}")
