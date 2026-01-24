import sqlite3
import pandas as pd
import ccxt
import json
import os
import logging
from datetime import datetime
from config import DB_FILE, PERFORMANCE_FILE, ENGINES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Thresholds
KILL_THRESHOLD = 40.0
RECOVERY_THRESHOLD = 50.0

MAX_SIGNALS_PER_ENGINE = 100
MAX_CANDLES_LOOKFORWARD = 500

def determine_outcome(row, ex):
    try:
        signal_ts = datetime.fromisoformat(row['ts'].replace('Z', '+00:00') if 'Z' in row['ts'] else row['ts'])
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

        entry = row['entry']
        sl = row['sl']
        tp = row['tp']
        direction = row['signal'].upper()

        for _, candle in df_candles.iterrows():
            if direction == "LONG":
                if candle['low'] <= sl:
                    return "loss"
                if candle['high'] >= tp:
                    return "win"
            else:
                if candle['high'] >= sl:
                    return "loss"
                if candle['low'] <= tp:
                    return "win"

        return "ongoing"
    except Exception as e:
        logging.warning(f"Outcome check failed for {row['asset']} {row['engine']} {row['ts']}: {e}")
        return "skipped"

def audit_all():
    ex = ccxt.xt({"enableRateLimit": True})
    performance_report = {}

    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                performance_report = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load performance file: {e}")

    conn = sqlite3.connect(DB_FILE)
    try:
        for strat_id in ENGINES.keys():
            try:
                df = pd.read_sql_query("""
                    SELECT * FROM signals 
                    WHERE engine = ? 
                    ORDER BY id DESC LIMIT ?
                """, conn, params=(strat_id, MAX_SIGNALS_PER_ENGINE))

                if df.empty:
                    logging.info(f"No signals for {strat_id}")
                    continue

                df['entry'] = pd.to_numeric(df['entry'], errors='coerce')
                df['sl'] = pd.to_numeric(df['sl'], errors='coerce')
                df['tp'] = pd.to_numeric(df['tp'], errors='coerce')
                df = df.dropna(subset=['entry', 'sl', 'tp', 'timeframe', 'signal'])

                wins = losses = ongoing = skipped = 0

                for _, row in df.iterrows():
                    outcome = determine_outcome(row, ex)
                    if outcome == "win": wins += 1
                    elif outcome == "loss": losses += 1
                    elif outcome == "ongoing": ongoing += 1
                    else: skipped += 1

                closed_trades = wins + losses
                wr = round((wins / closed_trades * 100), 2) if closed_trades > 0 else 50.0

                current_status = performance_report.get(strat_id, {}).get("status", "LIVE")
                if current_status == "LIVE" and wr < KILL_THRESHOLD and closed_trades >= 5:
                    new_status = "RECOVERY"
                elif current_status == "RECOVERY" and wr >= RECOVERY_THRESHOLD and closed_trades >= 5:
                    new_status = "LIVE"
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

                logging.info(f"{strat_id.upper()}: WR={wr}% ({wins}W/{losses}L, {ongoing} ongoing) → {new_status}")

            except Exception as e:
                logging.error(f"Error auditing {strat_id}: {e}")

    finally:
        conn.close()

    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance_report, f, indent=4)

    logging.info(f"Audit complete. {len(performance_report)} engines updated.")

if __name__ == "__main__":
    audit_all()
