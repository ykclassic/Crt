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
    
    lookback_time = (datetime.now() - timedelta(minutes=65)).isoformat()

    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT * FROM signals WHERE ts > ? ORDER BY ts DESC", conn, params=(lookback_time,))
    finally:
        conn.close()

    if df.empty:
        logging.info("No new signals")
        return

    # Deduplicate by exact row (in case of re-runs)
    df = df.drop_duplicates(subset=['engine', 'asset', 'timeframe', 'signal', 'ts'])

    for _, row in df.iterrows():
        emoji = "🟢" if row['signal'].upper() == "LONG" else "🔴"
        conf = row.get('confidence', 0)
        reason = row.get('reason', 'Technical Assessment')
        engine_name = ENGINES.get(row['engine'], row['engine'])
        
        msg = (f"🛡️ **NEXUS SIGNAL ALERT**\n"
               f"----------------------------\n"
               f"Asset: **{row['asset']}** ({row['timeframe']})\n"
               f"Signal: {emoji} **{row['signal']}**\n"
               f"Engine: `{engine_name}`\n"
               f"Confidence: `{conf}%`\n"
               f"Reason: *{reason}*")
        send_to_discord(msg)
        logging.info(f"Dispatched {row['signal']} {row['asset']} from {row['engine']}")

if __name__ == "__main__":
    dispatch_alerts()
