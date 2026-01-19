import ccxt, pandas as pd, numpy as np, sqlite3, os, requests
from datetime import datetime

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS AI PREDICT"
EXCHANGE = ccxt.xt()
DB_FILE = "nexus_ai.db"

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, asset TEXT, signal TEXT, confidence REAL, ts TEXT)")
conn.commit()

for asset in ["SOL/USDT", "BTC/USDT"]:
    try:
        data = EXCHANGE.fetch_ohlcv(asset, '1h', limit=100)
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
        
        # Calculate RSI for Confidence
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Confidence logic: Higher RSI in an uptrend = Higher Confidence
        price = df['c'].iloc[-1]
        signal = "LONG" if rsi > 50 else "SHORT"
        confidence_score = round(rsi if signal == "LONG" else (100 - rsi), 2)
        
        sl = price * 0.98 if signal == "LONG" else price * 1.02
        tp = price * 1.05 if signal == "LONG" else price * 0.95

        cursor.execute("INSERT INTO signals (asset, signal, confidence, ts) VALUES (?,?,?,?)", 
                       (asset, signal, confidence_score, datetime.now().isoformat()))
        conn.commit()
        
        notify(f"🤖 **AI Analysis**\nAsset: {asset}\nSignal: {signal}\n**Confidence Score: {confidence_score}%**\n---\nEntry: {price:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}")
        
    except Exception as e:
        print(f"Error {asset}: {e}")

conn.close()
