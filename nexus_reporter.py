import sqlite3
import pandas as pd
import requests
import logging
import argparse
from datetime import datetime, timedelta
from config import DB_FILE, WEBHOOK_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def send_report(msg):
    if WEBHOOK_URL:
        try:
            # Discord has a 2000 character limit per message
            if len(msg) > 1900:
                parts = [msg[i:i+1900] for i in range(0, len(msg), 1900)]
                for p in parts:
                    requests.post(WEBHOOK_URL, json={"content": p})
            else:
                requests.post(WEBHOOK_URL, json={"content": msg})
        except Exception as e:
            logging.error(f"Reporting failed: {e}")

def generate_report(full=False):
    conn = sqlite3.connect(DB_FILE)
    try:
        if full:
            query = "SELECT * FROM signals ORDER BY ts DESC"
            title = "📊 NEXUS FULL SYSTEM REPORT"
        else:
            # Only report the last 24 hours by default
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            query = "SELECT * FROM signals WHERE ts > ? ORDER BY ts DESC"
            title = "📅 NEXUS DAILY PERFORMANCE REPORT"
            
        df = pd.read_sql_query(query, conn, params=(None if full else (yesterday,)))
        
        if df.empty:
            send_report(f"{title}\nNo signals found for the selected period.")
            return

        # Basic Stats
        total = len(df)
        longs = len(df[df['signal'] == 'LONG'])
        shorts = len(df[df['signal'] == 'SHORT'])
        top_asset = df['asset'].mode()[0] if not df['asset'].empty else "N/A"

        report_body = (
            f"**{title}**\n"
            f"Total Signals: `{total}` (🟢 {longs} | 🔴 {shorts})\n"
            f"Most Active Asset: `{top_asset}`\n"
            f"Latest Timestamp: `{df['ts'].max()}`\n"
            f"----------------------------\n"
        )

        # Add last 5 signals for preview
        report_body += "**Recent Activity Preview:**\n"
        for _, row in df.head(5).iterrows():
            emoji = "🟢" if row['signal'] == "LONG" else "🔴"
            report_body += f"{emoji} {row['asset']} | Entry: {row['entry']:.4f} | {row['ts'][:16]}\n"

        send_report(report_body)
        logging.info("Report sent successfully.")

    except Exception as e:
        logging.error(f"Error generating report: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Generate a full history report")
    args = parser.parse_args()
    
    generate_report(full=args.full)
