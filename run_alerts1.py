import ccxt, pandas as pd, numpy as np, sqlite3, os, requests
from datetime import datetime

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS HYBRID V1"
EXCHANGE = ccxt.xt()
DB_FILE = "hybrid_v1.db"

def notify(msg):
    if WEBHOOK: 
        requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

# Connect and Init
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, asset TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, confidence REAL, ts TEXT)")
conn.commit()

for asset in ["BTC/USDT", "ETH/USDT"]:
    try:
        data = EXCHANGE.fetch_ohlcv(asset, '1h', limit=50)
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
        
        # Simple Logic
        last_price = df['c'].iloc[-1]
        sma = df['c'].rolling(20).mean().iloc[-1]
        atr = (df['h'] - df['l']).rolling(14).mean().iloc[-1]
        
        signal = "LONG" if last_price > sma else "SHORT"
        conf = 75.0 # Static base for V1
        
        sl = last_price - (atr * 2) if signal == "LONG" else last_price + (atr * 2)
        tp = last_price + (atr * 3) if signal == "LONG" else last_price - (atr * 3)

        # Check for duplicates
        cursor.execute("SELECT signal FROM signals WHERE asset=? ORDER BY id DESC LIMIT 1", (asset,))
        row = cursor.fetchone()
        
        if not row or row[0] != signal:
            cursor.execute("INSERT INTO signals (asset, signal, entry, sl, tp, confidence, ts) VALUES (?,?,?,?,?,?,?)", 
                           (asset, signal, last_price, sl, tp, conf, datetime.now().isoformat()))
            conn.commit()
            
            notify(f"🚀 **New Signal**\nAsset: {asset}\nDirection: {signal}\nConfidence: {conf}%\n---\nEntry: {last_price:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}")
            
    except Exception as e:
        print(f"Error {asset}: {e}")

conn.close()
