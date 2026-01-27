import sqlite3
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from config import DB_FILE, WEBHOOK_URL, ENGINES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def send_to_discord(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": msg})
        except Exception as e:
            logging.error(f"Discord post failed: {e}")

def dispatch_alerts():
    logging.info(f"Dispatcher active: {datetime.now()}")
    
    # Look back 65 minutes to ensure we catch everything from the hourly run
    lookback_time = (datetime.now() - timedelta(minutes=65)).isoformat()

    conn = sqlite3.connect(DB_FILE)
    try:
        # Check if table exists first to avoid crash on fresh install
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals';")
        if not cursor.fetchone():
            logging.warning("Signals table does not exist yet. No alerts to dispatch.")
            return

        df = pd.read_sql_query("SELECT * FROM signals WHERE ts > ? ORDER BY ts DESC", conn, params=(lookback_time,))
    except Exception as e:
        logging.error(f"Database error: {e}")
        return
    finally:
        conn.close()

    if df.empty:
        logging.info("No new signals found in the last hour.")
        return

    # --- SCHEMA SAFEGUARD ---
    # If the database was created by the previous run_alerts.py, it might lack the 'engine' column.
    # We default to 'core' to prevent the dispatcher from crashing.
    if 'engine' not in df.columns:
        logging.warning("Column 'engine' missing in DB. Defaulting to 'core'.")
        df['engine'] = 'core'
    # ------------------------

    # Deduplicate: Avoid sending the exact same signal twice if the job overlaps
    # We use 'asset', 'timeframe', 'signal', 'ts' as the unique key
    df = df.drop_duplicates(subset=['asset', 'timeframe', 'signal', 'ts'])

    count = 0
    for _, row in df.iterrows():
        try:
            emoji = "🟢" if row['signal'].upper() == "LONG" else "🔴"
            conf = row.get('confidence', 50.0) # Default to 50 if missing
            reason = row.get('reason', 'Technical Assessment')
            
            # Map internal engine key to display name
            engine_key = row.get('engine', 'core')
            engine_name = ENGINES.get(engine_key, engine_key.upper())
            
            # Formatting the message
            msg = (f"🛡️ **NEXUS SIGNAL ALERT**\n"
                   f"----------------------------\n"
                   f"Asset: **{row['asset']}** ({row['timeframe']})\n"
                   f"Signal: {emoji} **{row['signal']}**\n"
                   f"Engine: `{engine_name}`\n"
                   f"Confidence: `{conf}%`\n"
                   f"Reason: *{reason}*")
            
            send_to_discord(msg)
            logging.info(f"Dispatched {row['signal']} {row['asset']} ({engine_name})")
            count += 1
        except Exception as e:
            logging.error(f"Failed to dispatch row: {e}")

    logging.info(f"Dispatch complete. Sent {count} alerts.")

if __name__ == "__main__":
    dispatch_alerts()
