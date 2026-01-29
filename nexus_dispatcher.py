import sqlite3
import pandas as pd
import requests
import logging
import os
import time
from datetime import datetime, timedelta
from config import DB_FILE, WEBHOOK_URL, ENGINES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def send_to_discord(msg):
    if not WEBHOOK_URL:
        logging.error("❌ WEBHOOK_URL missing.")
        return False
    try:
        payload = {"content": str(msg)}
        response = requests.post(WEBHOOK_URL, json=payload)
        
        if response.status_code == 204:
            return True
        elif response.status_code == 429:
            retry_after = response.json().get('retry_after', 1)
            logging.warning(f"⚠️ Rate limited. Sleeping for {retry_after}s")
            time.sleep(retry_after)
            # One-time retry
            return requests.post(WEBHOOK_URL, json=payload).status_code == 204
        else:
            logging.error(f"❌ Discord Failed: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"❌ Connection Error: {e}")
        return False

def dispatch_alerts():
    logging.info(f"--- DISPATCHER: ANTI-SPAM MODE ---")
    lookback_time = (datetime.now() - timedelta(minutes=65)).isoformat()

    if not os.path.exists(DB_FILE): return

    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM signals WHERE ts > ? ORDER BY ts DESC", conn, params=(lookback_time,))
    conn.close()

    if df.empty: return

    df = df.drop_duplicates(subset=['asset', 'signal', 'ts'])
    success_count = 0
    
    for _, row in df.iterrows():
        emoji = "🟢" if str(row['signal']).upper() == "LONG" else "🔴"
        msg = (f"🛡️ **NEXUS ALERT**\n**{row['asset']}** | {emoji} {row['signal']}\n"
               f"Conf: `{row['confidence']}%` | Time: `{row['ts']}`")
        
        if send_to_discord(msg):
            success_count += 1
            time.sleep(0.5) # The "Breath" to prevent 429 errors
            
    logging.info(f"Dispatch Summary: Sent {success_count}/{len(df)}")

if __name__ == "__main__":
    dispatch_alerts()
