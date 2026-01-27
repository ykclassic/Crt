import sqlite3
import pandas as pd
import ccxt
import json
import os
import logging
from datetime import datetime
from config import DB_FILE, PERFORMANCE_FILE, ENGINES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Performance Thresholds
KILL_THRESHOLD = 40.0
RECOVERY_THRESHOLD = 50.0
MAX_SIGNALS_PER_ENGINE = 50
MAX_CANDLES_LOOKFORWARD = 500

def determine_outcome(row, ex):
    try:
        # Standardize timestamp format
        ts_str = row['ts'].replace('Z', '+00:00') if 'Z' in row['ts'] else row['ts']
        signal_ts = datetime.fromisoformat(ts_str)
        since_ms = int(signal_ts.timestamp() * 1000) + 1

        ohlcv = ex.fetch_ohlcv(
            row['asset'],
            timeframe=row['timeframe'],
            since=since_ms,
            limit=MAX_CANDLES_LOOKFORWARD
        )
        if not ohlcv:
            return "ongoing"

        df_candles = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])

        entry = float(row['entry'])
        sl = float(row['sl'])
        tp = float(row['tp'])
        direction = row['signal'].upper()

        for _, candle in df_candles.iterrows():
            if direction == "LONG":
                if candle['low'] <= sl: return "loss"
                if candle['high'] >= tp: return "win"
            else:
                if candle['high'] >= sl: return "loss"
                if candle['low'] <= tp: return "win"

        return "ongoing"
    except Exception as e:
        logging.warning(f"Outcome check failed for {row['asset']} {row['ts']}: {e}")
        return "skipped"

def audit_all():
    # Using Gate.io for auditing data (high precision)
    ex = ccxt.gateio({"enableRateLimit": True})
    performance_report = {}

    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                performance_report = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load performance file: {e}")

    conn = sqlite3.connect(DB_FILE)
    try:
        # Audit every engine defined in config.py
        for strat_id in ENGINES.keys():
            try:
                df = pd.read_sql_query("""
                    SELECT * FROM signals 
                    WHERE engine = ? 
                    ORDER BY ts DESC LIMIT ?
                """, conn, params=(strat_id, MAX_SIGNALS_PER_ENGINE))

                if df.empty:
                    continue

                # Ensure numeric types
                df['entry'] = pd.to_numeric(df['entry'], errors='coerce')
                df['sl'] = pd.to_numeric(df['sl'], errors='coerce')
                df['tp'] = pd.to_numeric(df['tp'], errors='coerce')
                df = df.dropna(subset=['entry', 'sl', 'tp', 'signal'])

                wins = losses = ongoing = skipped = 0

                for _, row in df.iterrows():
                    outcome = determine_outcome(row, ex)
                    if outcome == "win": wins += 1
                    elif outcome == "loss": losses += 1
                    elif outcome == "ongoing": ongoing += 1
                    else: skipped += 1

                closed_trades = wins + losses
                wr = round((wins / closed_trades * 100), 2) if closed_trades > 0 else 50.0

                # Kill Switch Logic
                current_status = performance_report.get(strat_id, {}).get("status", "LIVE")
                if current_status == "LIVE" and wr < KILL_THRESHOLD and closed_trades >= 5:
                    new_status = "RECOVERY"
                    logging.warning(f"!!! KILL SWITCH TRIGGERED FOR {strat_id} !!!")
                elif current_status == "RECOVERY" and wr >= RECOVERY_THRESHOLD and closed_trades >= 5:
                    new_status = "LIVE"
                    logging.info(f"Engine {strat_id} recovered and is now back LIVE.")
                else:
                    new_status = current_status

                performance_report[strat_id] = {
                    "win_rate": wr,
                    "status": new_status,
                    "closed_sample": closed_trades,
                    "wins": wins,
                    "losses": losses,
                    "ongoing": ongoing,
                    "last_audit": datetime.now().isoformat()
                }

                logging.info(f"AUDIT {strat_id.upper()}: {wr}% WR | Status: {new_status}")

            except Exception as e:
                logging.error(f"Error auditing {strat_id}: {e}")

    finally:
        conn.close()

    # Save the updated performance journal
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance_report, f, indent=4)

    logging.info("Audit cycle complete.")

if __name__ == "__main__":
    audit_all()
