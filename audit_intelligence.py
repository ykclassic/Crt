import sqlite3
import pandas as pd
import ccxt
import logging
import json
import os
from datetime import datetime, timedelta
from config import DB_FILE, PERFORMANCE_FILE, KILL_THRESHOLD, RECOVERY_THRESHOLD

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def get_market_outcome(ex, asset, timeframe, start_ts, tp, sl, signal_type):
    """Checks OHLCV data to see if TP or SL was hit first."""
    try:
        # Fetch 100 candles following the signal to check outcome
        since = int(datetime.fromisoformat(start_ts).timestamp() * 1000)
        ohlcv = ex.fetch_ohlcv(asset, timeframe, since=since, limit=100)
        
        for candle in ohlcv:
            high, low = candle[2], candle[3]
            
            if signal_type == "LONG":
                if high >= tp: return "WIN"
                if low <= sl: return "LOSS"
            elif signal_type == "SHORT":
                if low <= tp: return "WIN"
                if high >= sl: return "LOSS"
        return "PENDING"
    except Exception as e:
        logging.error(f"Error fetching outcome for {asset}: {e}")
        return "ERROR"

def run_audit():
    logging.info("--- STARTING PERFORMANCE AUDIT ---")
    conn = sqlite3.connect(DB_FILE)
    
    # Load all signals from the last 7 days
    query = "SELECT * FROM signals WHERE ts > datetime('now', '-7 days')"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        logging.info("No signals found to audit.")
        return

    # Use Gate.io as the reference for auditing (high fidelity)
    ex = ccxt.gateio()
    performance = {}

    # Group by engine to calculate individual "Alpha"
    for engine in df['engine'].unique():
        engine_df = df[df['engine'] == engine]
        results = []

        for _, row in engine_df.iterrows():
            outcome = get_market_outcome(
                ex, row['asset'], row['timeframe'], 
                row['ts'], row['tp'], row['sl'], row['signal']
            )
            results.append(outcome)

        # Calculate Metrics
        total = len([r for r in results if r in ["WIN", "LOSS"]])
        wins = results.count("WIN")
        win_rate = (wins / total * 100) if total > 0 else 0.0

        # Load existing status to check for recovery
        current_status = "LIVE"
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                old_perf = json.load(f)
                current_status = old_perf.get(engine, {}).get("status", "LIVE")

        # Kill-Switch / Recovery Logic
        new_status = current_status
        if win_rate < KILL_THRESHOLD and total >= 5:
            new_status = "RECOVERY"
            logging.warning(f"🚨 ENGINE {engine} KILLED: Win rate {win_rate}% is too low.")
        elif current_status == "RECOVERY" and win_rate >= RECOVERY_THRESHOLD:
            new_status = "LIVE"
            logging.info(f"✅ ENGINE {engine} RECOVERED: Win rate back to {win_rate}%.")

        performance[engine] = {
            "win_rate": round(win_rate, 2),
            "total_audited": total,
            "status": new_status,
            "last_audit": datetime.now().isoformat()
        }

    # Save to performance.json
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance, f, indent=4)
    
    conn.close()
    logging.info("--- AUDIT COMPLETE: PERFORMANCE.JSON UPDATED ---")

if __name__ == "__main__":
    run_audit()
