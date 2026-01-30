import sqlite3
import pandas as pd
import ccxt
import logging
import json
import os
from datetime import datetime
from config import DB_FILE, PERFORMANCE_FILE, KILL_THRESHOLD, RECOVERY_THRESHOLD

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def get_market_outcome(ex, asset, timeframe, start_ts, tp, sl, signal_type):
    """Checks if price hit TP or SL first using historical OHLCV data."""
    try:
        since = int(datetime.fromisoformat(start_ts).timestamp() * 1000)
        # Fetch up to 100 candles following the signal to check outcome
        ohlcv = ex.fetch_ohlcv(asset, timeframe, since=since, limit=100)
        
        for candle in ohlcv:
            # candle format: [timestamp, open, high, low, close, volume]
            high, low = candle[2], candle[3]
            
            if signal_type == "LONG":
                if high >= tp: return "WIN"
                if low <= sl: return "LOSS"
            elif signal_type == "SHORT":
                if low <= tp: return "WIN"
                if high >= sl: return "LOSS"
        return "PENDING"
    except Exception as e:
        logging.error(f"Audit error for {asset}: {e}")
        return "ERROR"

def run_audit():
    logging.info("--- STARTING PERFORMANCE AUDIT ---")
    if not os.path.exists(DB_FILE):
        logging.error("No database found for audit.")
        return

    conn = sqlite3.connect(DB_FILE)
    # Pull signals from the last 7 days for a rolling win-rate
    query = "SELECT * FROM signals WHERE ts > datetime('now', '-7 days')"
    df = pd.read_sql_query(query, conn)
    
    # Initialize Gate.io for outcome checking
    ex = ccxt.gateio()
    performance = {}

    # Get current status if file exists to prevent status flickering
    current_perf = {}
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                current_perf = json.load(f)
        except: pass

    for engine in df['engine'].unique():
        engine_df = df[df['engine'] == engine]
        outcomes = []

        for _, row in engine_df.iterrows():
            res = get_market_outcome(ex, row['asset'], row['timeframe'], row['ts'], row['tp'], row['sl'], row['signal'])
            outcomes.append(res)

        # Metrics Calculation
        completed = [o for o in outcomes if o in ["WIN", "LOSS"]]
        wins = completed.count("WIN")
        total = len(completed)
        win_rate = (wins / total * 100) if total > 0 else 0.0

        # Kill-Switch Logic
        prev_status = current_perf.get(engine, {}).get("status", "LIVE")
        status = prev_status

        # Only toggle status if we have enough data (min 5 trades)
        if total >= 5:
            if win_rate < KILL_THRESHOLD:
                status = "RECOVERY"
                logging.warning(f"🚨 Engine {engine} moved to RECOVERY (Win Rate: {win_rate}%)")
            elif prev_status == "RECOVERY" and win_rate >= RECOVERY_THRESHOLD:
                status = "LIVE"
                logging.info(f"✅ Engine {engine} restored to LIVE (Win Rate: {win_rate}%)")

        performance[engine] = {
            "win_rate": round(win_rate, 2),
            "total_trades": total,
            "status": status,
            "last_updated": datetime.now().isoformat()
        }

    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance, f, indent=4)
    
    conn.close()
    logging.info(f"Audit complete. Performance saved to {PERFORMANCE_FILE}")

if __name__ == "__main__":
    run_audit()
