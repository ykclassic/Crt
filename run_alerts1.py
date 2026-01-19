import ccxt, pandas as pd, os, requests, sqlite3
from datetime import datetime

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS HYBRID V1"
EXCHANGE = ccxt.gateio()
DB_FILE = "hybrid_v1.db"

def notify(msg):
    if WEBHOOK: requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, asset TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, conf REAL, ts TEXT)")
conn.commit()

for asset in ["SOL/USDT", "BTC/USDT"]:
    try:
        data = EXCHANGE.fetch_ohlcv(asset, '1h', limit=50)
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
        
        # Supertrend-style logic
        df['hl2'] = (df['h'] + df['l']) / 2
        last_price = df['c'].iloc[-1]
        conf = 75.0
        
        signal = "LONG" if last_price > df['hl2'].mean() else "SHORT"
        sl = last_price * (0.97 if signal == "LONG" else 1.03)
        tp = last_price * (1.06 if signal == "LONG" else 0.94)

        cursor.execute("INSERT INTO signals (asset, signal, entry, sl, tp, conf, ts) VALUES (?,?,?,?,?,?,?)",
                       (asset, signal, last_price, sl, tp, conf, datetime.now().isoformat()))
        conn.commit()
        notify(f"🔄 **Hybrid Alert**\nAsset: {asset}\nSignal: {signal}\nConfidence: {conf}%\n---\nEntry: {last_price:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}")
    except Exception as e: print(e)
conn.close()
