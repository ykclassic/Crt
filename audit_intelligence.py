import sqlite3
import pandas as pd
import ccxt
import json
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FILES = ["nexus_core.db", "nexus_ai.db", "hybrid_v1.db", "rangemaster.db"]
EXCHANGE_NAME = "XT" 
PERFORMANCE_FILE = "performance.json"

def get_exchange(name):
    name = name.upper()
    if name == "XT": return ccxt.xt({"enableRateLimit": True})
    elif name == "GATE": return ccxt.gateio({"enableRateLimit": True})
    elif name == "BITGET": return ccxt.bitget({"enableRateLimit": True})
    else: raise ValueError(f"Unknown exchange: {name}")

def audit_all():
    ex = get_exchange(EXCHANGE_NAME)
    ex.load_markets()
    performance_report = {}

    for db_file in DB_FILES:
        if not os.path.exists(db_file): continue
        
        conn = sqlite3.connect(db_file)
        try:
            df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 50", conn)
            if df.empty: continue

            strategy_name = db_file.replace(".db", "")
            wins, losses = 0, 0

            for _, row in df.iterrows():
                try:
                    asset, entry, sl, tp, signal = row['asset'], row['entry'], row['sl'], row['tp'], row['signal']
                    ts_str = row['ts'].split('.')[0]
                    signal_time = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                    
                    if datetime.now() - signal_time < timedelta(hours=1): continue

                    since = int(signal_time.timestamp() * 1000)
                    ohlcv = ex.fetch_ohlcv(asset, '1h', since=since, limit=24)
                    price_data = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])

                    for _, candle in price_data.iterrows():
                        if signal == "LONG":
                            if candle['h'] >= tp: wins += 1; break
                            elif candle['l'] <= sl: losses += 1; break
                        elif signal == "SHORT":
                            if candle['l'] <= tp: wins += 1; break
                            elif candle['h'] >= sl: losses += 1; break
                except: continue

            total = wins + losses
            wr = round((wins / total * 100), 2) if total > 0 else 50.0
            performance_report[strategy_name] = {"win_rate": wr, "sample_size": total}
            
        finally:
            conn.close()

    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance_report, f)
    print(f"✅ Intelligence Audit Complete. Performance saved to {PERFORMANCE_FILE}")

if __name__ == "__main__":
    audit_all()
