import sqlite3
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DB_FILES = ["nexus_core.db", "hybrid_v1.db", "rangemaster.db", "nexus_ai.db"]

def send_to_discord(msg):
    if not WEBHOOK_URL: return
    try:
        requests.post(WEBHOOK_URL, json={"content": msg})
    except Exception as e:
        print(f"Failed to post to Discord: {e}")

def dispatch_alerts():
    print(f"📡 Nexus Dispatcher Active: {datetime.now()}")
    new_signals = []
    
    # Check for signals generated in the last 65 minutes (to cover the hourly trigger)
    lookback_time = (datetime.now() - timedelta(minutes=65)).isoformat()

    for db_file in DB_FILES:
        if not os.path.exists(db_file): continue
        try:
            conn = sqlite3.connect(db_file)
            query = f"SELECT * FROM signals WHERE ts > '{lookback_time}'"
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['engine'] = db_file.replace(".db", "")
                new_signals.append(df)
            conn.close()
        except Exception as e:
            print(f"Error reading {db_file}: {e}")

    if new_signals:
        master = pd.concat(new_signals)
        # Drop duplicates in case multiple runs happen
        master = master.drop_duplicates(subset=['asset', 'signal', 'ts'])
        
        for _, row in master.iterrows():
            emoji = "🟢" if row['signal'].upper() == "LONG" else "🔴"
            conf = row.get('confidence', 0)
            reason = row.get('reason', 'Technical Assessment')
            
            msg = (f"🛡️ **NEXUS SIGNAL ALERT**\n"
                   f"----------------------------\n"
                   f"Asset: **{row['asset']}**\n"
                   f"Signal: {emoji} **{row['signal']}**\n"
                   f"Engine: `{row['engine']}`\n"
                   f"Confidence: `{conf}%`\n"
                   f"Reason: *{reason}*")
            send_to_discord(msg)
            print(f"📢 Dispatch sent for {row['asset']}")
    else:
        print("📭 No new signals to report in this window.")

if __name__ == "__main__":
    dispatch_alerts()
