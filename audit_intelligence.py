import sqlite3
import pandas as pd
import ccxt
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FILES = ["nexus_core.db", "nexus_ai.db", "hybrid_v1.db", "rangemaster.db"]
EXCHANGE_NAME = "XT"  # Use the same exchange you use for alerts

def get_exchange(name):
    name = name.upper()
    if name == "XT": return ccxt.xt({"enableRateLimit": True})
    elif name == "GATE": return ccxt.gateio({"enableRateLimit": True})
    elif name == "BITGET": return ccxt.bitget({"enableRateLimit": True})
    else: raise ValueError(f"Unknown exchange: {name}")

def audit_database(db_file, exchange):
    print(f"\n🔎 AUDITING BRAIN: {db_file}")
    conn = sqlite3.connect(db_file)
    try:
        # Load signals that haven't been audited yet (or all of them)
        # Note: In a production system, we would add a 'result' column to the DB 
        # to avoid re-checking old signals. For now, we check the last 20.
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 20", conn)
        
        if df.empty:
            print("   -> No signals found.")
            return

        wins = 0
        losses = 0
        pending = 0
        
        print(f"   -> Analyzing last {len(df)} signals...")

        for index, row in df.iterrows():
            asset = row['asset']
            entry = row['entry']
            # Handle different schema naming (sl/tp might not exist in all DBs)
            try:
                sl = row['sl']
                tp = row['tp']
                signal = row['signal']
                # Clean timestamp format if necessary (remove 'T' or milliseconds)
                ts_str = row['ts'].split('.')[0] 
                signal_time = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
            except Exception as e:
                # Skip rows that don't fit standard schema (like Rangemaster might differ)
                continue

            # If signal is too recent (less than 1 hour), skip
            if datetime.now() - signal_time < timedelta(hours=1):
                pending += 1
                continue

            # Fetch price history AFTER the signal to see what happened
            # We fetch 1h candles starting from the signal time
            since = int(signal_time.timestamp() * 1000)
            try:
                ohlcv = exchange.fetch_ohlcv(asset, '1h', since=since, limit=24) # Check next 24 hours
                price_data = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
            except:
                print(f"   -> API Error fetching data for {asset}")
                continue

            outcome = "PENDING"
            
            # Replay the market
            for _, candle in price_data.iterrows():
                high = candle['h']
                low = candle['l']
                
                if signal == "LONG":
                    if high >= tp:
                        outcome = "WIN"
                        wins += 1
                        break # Stop checking this trade
                    elif low <= sl:
                        outcome = "LOSS"
                        losses += 1
                        break
                elif signal == "SHORT":
                    if low <= tp:
                        outcome = "WIN"
                        wins += 1
                        break
                    elif high >= sl:
                        outcome = "LOSS"
                        losses += 1
                        break
            
            if outcome == "PENDING":
                pending += 1

        # --- REPORT CARD ---
        total_closed = wins + losses
        if total_closed > 0:
            win_rate = (wins / total_closed) * 100
            print(f"   🏆 WINS: {wins} | 💀 LOSSES: {losses} | ⏳ PENDING: {pending}")
            print(f"   📊 REAL WIN RATE: {win_rate:.2f}%")
            
            # Simple "Learning" Feedback
            if win_rate < 40:
                print("   ⚠️ CRITICAL: Strategy is failing. Reduce confidence or stop trading.")
            elif win_rate > 60:
                print("   ✅ STRONG: Strategy is performing well.")
            else:
                print("   ⚖️ NEUTRAL: Strategy is average.")
        else:
            print("   -> Not enough closed trades to calculate Win Rate.")

    except Exception as e:
        print(f"   -> Error reading DB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    ex = get_exchange(EXCHANGE_NAME)
    ex.load_markets()
    
    for db in DB_FILES:
        if "sqlite" in db or ".db" in db:
            audit_database(db, ex)
