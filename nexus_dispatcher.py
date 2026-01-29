import sqlite3
import pandas as pd
import requests
import logging
import os
import json
from datetime import datetime, timedelta
from config import DB_FILE, WEBHOOK_URL, ENGINES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def send_to_discord(msg):
    # 1. Debug: Check if URL exists
    if not WEBHOOK_URL:
        logging.error("❌ CRITICAL: WEBHOOK_URL is missing! Check GitHub Secrets.")
        return False

    try:
        # 2. explicit conversion to string to prevent JSON errors
        payload = {"content": str(msg)}
        
        response = requests.post(WEBHOOK_URL, json=payload)
        
        # 3. Debug: Print Status Code
        if response.status_code == 204:
            logging.info("✅ Discord API Success (204)")
            return True
        else:
            logging.error(f"❌ Discord Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Discord Connection Error: {e}")
        return False

def dispatch_alerts():
    logging.info(f"--- DISPATCHER STARTED: {datetime.now()} ---")
    
    # Debug: Print first 5 chars of webhook to verify it's loaded (securely)
    if WEBHOOK_URL:
        logging.info(f"Webhook URL loaded: {WEBHOOK_URL[:5]}...")
    else:
        logging.warning("⚠️ Webhook URL not found in environment variables.")

    # Look back 2 hours to be safe
    lookback_time = (datetime.now() - timedelta(hours=2)).isoformat()

    if not os.path.exists(DB_FILE):
        logging.warning("Database not found.")
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        # Check for table existence
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals';")
        if not cursor.fetchone():
            logging.warning("Signals table missing.")
            return

        df = pd.read_sql_query("SELECT * FROM signals WHERE ts > ? ORDER BY ts DESC", conn, params=(lookback_time,))
    except Exception as e:
        logging.error(f"Database Read Error: {e}")
        return
    finally:
        conn.close()

    if df.empty:
        logging.info("No signals found in the last 2 hours.")
        return

    # Deduplication logic (Asset + Timeframe + Signal + Timestamp)
    df = df.drop_duplicates(subset=['asset', 'timeframe', 'signal', 'ts'])

    count = 0
    success_count = 0
    
    for _, row in df.iterrows():
        try:
            # Safe Data Extraction
            asset = row.get('asset', 'Unknown')
            signal = row.get('signal', 'N/A')
            tf = row.get('timeframe', '1h')
            conf = row.get('confidence', 0)
            engine_key = row.get('engine', 'core')
            engine_name = ENGINES.get(engine_key, engine_key.upper())
            
            # Emoji Selection
            emoji = "🟢" if str(signal).upper() == "LONG" else "🔴"
            
            msg = (f"🛡️ **NEXUS ALERT**\n"
                   f"-------------------\n"
                   f"**{asset}** ({tf})\n"
                   f"{emoji} **{signal}**\n"
                   f"Engine: `{engine_name}`\n"
                   f"Confidence: `{conf}%`\n"
                   f"Time: `{row['ts']}`")
            
            # Send
            if send_to_discord(msg):
                success_count += 1
            count += 1
            
        except Exception as e:
            logging.error(f"Row Processing Error: {e}")

    logging.info(f"Dispatch Summary: Found {count}, Sent {success_count}.")

if __name__ == "__main__":
    dispatch_alerts()
