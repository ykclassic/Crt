import ccxt, pandas as pd, numpy as np, os, requests, sqlite3
from datetime import datetime

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS AI PREDICT"
EXCHANGE = ccxt.xt()
DB_FILE = "nexus_ai.db"

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, asset TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, conf REAL, ts TEXT)")
conn.commit()

for asset in ["BTC/USDT", "ETH/USDT"]:
    try:
        data = EXCHANGE.fetch_ohlcv(asset, '1h', limit=100)
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
        
        # RSI Confidence Calculation
        delta = df['c'].diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain/loss))).iloc[-1]
        
        price = df['c'].iloc[-1]
        signal = "LONG" if rsi > 50 else "SHORT"
        # The further from 50, the higher the confidence
        conf_score = round(abs(rsi - 50) * 2, 2) + 50 
        
        sl = price * (0.98 if signal == "LONG" else 1.02)
        tp = price * (1.04 if signal == "LONG" else 0.96)

        cursor.execute("INSERT INTO signals (asset, signal, entry, sl, tp, conf, ts) VALUES (?,?,?,?,?,?,?)",
                       (asset, signal, price, sl, tp, conf_score, datetime.now().isoformat()))
        conn.commit()
        notify(f"🤖 **AI Analysis**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {conf_score}%**\n---\nEntry: {price:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}")
    except Exception as e: print(e)
conn.close()
