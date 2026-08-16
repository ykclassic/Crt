import sqlite3
import pandas as pd
import ccxt
import logging
import json
import os
from datetime import datetime, timezone
from config import (
    DB_FILE,
    PERFORMANCE_FILE,
    KILL_THRESHOLD,
    RECOVERY_THRESHOLD,
    EXCHANGE_ID,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | AUDIT | %(levelname)s | %(message)s'
)

MAX_SIGNALS_PER_ENGINE = 30 

def get_market_outcome(ex, cache, symbol, timeframe, start_ts, tp, sl, direction):
    try:
        if not tp or not sl:
            return "PENDING"

        since = int(datetime.fromisoformat(start_ts).timestamp() * 1000)
        cache_key = (symbol, timeframe)

        if cache_key not in cache:
            cache[cache_key] = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=100)

        ohlcv = cache[cache_key]

        for candle in ohlcv:
            high, low = candle[2], candle[3]
            if direction.upper() == "LONG":
                if high >= tp: return "WIN"
                if low <= sl: return "LOSS"
            elif direction.upper() == "SHORT":
                if low <= tp: return "WIN"
                if high >= sl: return "LOSS"

        return "PENDING"
    except Exception as e:
        logging.error(f"Market check error for {symbol}: {e}")
        return "ERROR"

def run_audit():
    logging.info("--- STARTING PERFORMANCE AUDIT ---")

    if not os.path.exists(DB_FILE):
        logging.error(f"Database {DB_FILE} not found. Skipping audit.")
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        # Aligned with unified master schema column names (engine_id, symbol, timestamp, etc.)
        query = "SELECT * FROM signals WHERE timestamp > datetime('now', '-7 days')"
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        logging.error(f"Failed to query database: {e}")
        conn.close()
        return

    if df.empty:
        logging.info("No signals found in the last 7 days to audit.")
        conn.close()
        return

    # Exchange validation and initialization matching core architecture
    if EXCHANGE_ID not in ccxt.exchanges:
        raise EnvironmentError(f"Exchange '{EXCHANGE_ID}' is not supported by the current CCXT version.")

    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True, "timeout": 15000})
    performance = {}
    cache = {}

    current_perf = {}
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                current_perf = json.load(f)
        except:
            current_perf = {}

    # Iterate using unified column names ('engine_id', 'symbol', 'take_profit', 'stop_loss', 'direction')
    for engine in df['engine_id'].dropna().unique():
        engine_df = df[df['engine_id'] == engine].tail(MAX_SIGNALS_PER_ENGINE)
        outcomes = [
            get_market_outcome(
                ex, cache, r['symbol'], r['timeframe'], r['timestamp'], 
                r['take_profit'], r['stop_loss'], r['direction']
            ) for _, r in engine_df.iterrows()
        ]

        completed = [o for o in outcomes if o in ["WIN", "LOSS"]]
        wins = completed.count("WIN")
        total = len(completed)
        win_rate = (wins / total * 100) if total > 0 else 0.0

        prev_status = current_perf.get(engine, {}).get("status", "LIVE")
        status = prev_status

        if total >= 5:
            if win_rate < KILL_THRESHOLD:
                status = "RECOVERY"
            elif prev_status == "RECOVERY" and win_rate >= RECOVERY_THRESHOLD:
                status = "LIVE"

        performance[engine] = {
            "win_rate": round(win_rate, 2),
            "total_trades": total,
            "status": status,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance, f, indent=4)

    conn.close()
    logging.info(f"Audit complete. Results saved to {PERFORMANCE_FILE}")

if __name__ == "__main__":
    run_audit()
