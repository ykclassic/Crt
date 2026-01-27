import sqlite3
import pandas as pd
import requests
from config import DB_FILE, WEBHOOK_URL

def send_to_discord(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg})

def generate_report():
    conn = sqlite3.connect(DB_FILE)
    try:
        # Pull the 5 most recent signals from the database
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC LIMIT 5", conn)
        
        if df.empty:
            send_to_discord("📊 **Nexus Report**: No new signals found in the database.")
            return

        report = "📊 **NEXUS SYSTEM REPORT**\n----------------------------\n"
        for _, row in df.iterrows():
            emoji = "🟢" if row['signal'] == "LONG" else "🔴"
            report += (f"{emoji} **{row['asset']}** ({row['timeframe']})\n"
                       f"Type: {row['signal']} | Entry: {row['entry']:.4f}\n"
                       f"Engine: `{row['engine']}` | Time: {row['ts'][:16]}\n\n")
        
        send_to_discord(report)
        # Also save a CSV backup for your GitHub records
        df.to_csv(f"Nexus_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
        
    except Exception as e:
        print(f"Report Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    generate_report()
