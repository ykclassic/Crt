import pandas as pd
import sqlite3
import ccxt
from datetime import datetime
import os

# --- CONFIG ---
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT']
TIMEFRAME = '1h'
DB_NAME = "nexus_core.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute('''CREATE TABLE IF NOT EXISTS signals 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     asset TEXT, signal TEXT, confidence REAL, 
                     reason TEXT, ts TEXT)''')
    conn.close()

def fetch_data(symbol):
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        return df
    except Exception as e:
        print(f"⚠️ Exchange Fetch Error ({symbol}): {e}")
        return None

def run_engine():
    init_db()
    print(f"🚀 Core Engine Pulse: {datetime.now()}")
    found_any = False

    for symbol in SYMBOLS:
        df = fetch_data(symbol)
        if df is None or len(df) < 20: continue
        
        # Simple Logic: EMA Cross + RSI
        df['ema20'] = df['c'].ewm(span=20).mean()
        df['rsi'] = 50 # Placeholder for TA library or manual calc
        
        last_price = df['c'].iloc[-1]
        prev_ema = df['ema20'].iloc[-2]
        curr_ema = df['ema20'].iloc[-1]
        
        signal = None
        reason = ""
        
        if last_price > curr_ema and last_price > prev_ema:
            signal = "LONG"
            reason = "Price Above EMA20"
        elif last_price < curr_ema and last_price < prev_ema:
            signal = "SHORT"
            reason = "Price Below EMA20"

        if signal:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("INSERT INTO signals (asset, signal, confidence, reason, ts) VALUES (?, ?, ?, ?, ?)",
                         (symbol, signal, 70.0, reason, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            print(f"✅ {signal} Signal Found for {symbol}")
            found_any = True
            
    if not found_any:
        print("😴 No signals met criteria this cycle.")

if __name__ == "__main__":
    run_engine()
