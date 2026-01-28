import ccxt
import pandas as pd
import sqlite3
import logging
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, ASSETS

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

STRATEGY_ID = "rangemaster"

def run_range_engine():
    # Using Bitget for Range/Scalp data
    ex = ccxt.bitget({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for asset in ASSETS:
        try:
            data = ex.fetch_ohlcv(asset, '15m', limit=50) # Lower timeframe for range play
            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            
            # Bollinger Bands
            df['sma'] = df['close'].rolling(20).mean()
            df['std'] = df['close'].rolling(20).std()
            df['upper'] = df['sma'] + (2 * df['std'])
            df['lower'] = df['sma'] - (2 * df['std'])
            
            last = df.iloc[-1]
            signal = "NEUTRAL"
            
            if last['close'] <= last['lower']:
                signal = "LONG"
                reason = "Range Bottom (Bollinger)"
            elif last['close'] >= last['upper']:
                signal = "SHORT"
                reason = "Range Top (Bollinger)"

            if signal != "NEUTRAL":
                entry = last['close']
                sl = last['lower'] * 0.99 if signal == "LONG" else last['upper'] * 1.01
                tp = last['sma'] # Target the mean (middle band)

                cursor.execute("""
                    INSERT INTO signals (engine, asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (STRATEGY_ID, asset, '15m', signal, entry, sl, tp, 60.0, reason, datetime.now().isoformat()))
                conn.commit()
                logging.info(f"Range Signal: {signal} {asset}")

        except Exception as e:
            logging.error(f"Range Engine Error ({asset}): {e}")

    conn.close()

if __name__ == "__main__":
    run_range_engine()
