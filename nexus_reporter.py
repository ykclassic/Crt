import sqlite3
import pandas as pd
import requests
import os
import json
from datetime import datetime

# --- CONFIGURATION ---
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DB_FILES = {
    "nexus_core": "nexus_core.db",
    "hybrid_v1": "hybrid_v1.db",
    "rangemaster": "rangemaster.db",
    "nexus_ai": "nexus_ai.db",
    "journal": "nexus_journal.db"
}

def send_to_discord(filename, title):
    if not WEBHOOK_URL:
        print("Error: No Discord Webhook URL found.")
        return

    try:
        with open(filename, "rb") as f:
            payload = {"content": f"📊 **{title}**\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
            files = {"file": (filename, f, "text/csv")}
            response = requests.post(WEBHOOK_URL, data=payload, files=files)
            
        if response.status_code in [200, 204]:
            print(f"Successfully forwarded {filename} to Discord.")
            return True
        else:
            print(f"Discord error: {response.status_code}")
    except Exception as e:
        print(f"Failed to send report: {e}")
    return False

def generate_consolidated_report(report_type="Nexus System Report"):
    all_data = []
    for engine, db_path in DB_FILES.items():
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                table_name = "journal" if "journal" in engine else "signals"
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                df['source_engine'] = engine
                all_data.append(df)
                conn.close()
            except Exception as e:
                print(f"Error reading {engine}: {e}")

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        filename = f"{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        master_df.to_csv(filename, index=False)
        
        # Forward to Discord
        send_to_discord(filename, report_type)
        
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
    else:
        print("No data available to generate report.")

if __name__ == "__main__":
    # If run directly, generate a general system report
    generate_consolidated_report()
