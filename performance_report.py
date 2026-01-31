import sqlite3
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FILE = "nexus.db"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

def generate_weekly_report():
    if not os.path.exists(DB_FILE): return
    
    conn = sqlite3.connect(DB_FILE)
    # Fetch trades from the last 7 days
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    df = pd.read_sql_query(f"SELECT * FROM signals WHERE ts >= '{one_week_ago}'", conn)
    conn.close()

    if df.empty:
        return "No trades recorded this week."

    # Calculation Logic
    total_trades = len(df)
    wins = len(df[df['status'] == 'SUCCESS'])
    losses = len(df[df['status'] == 'FAILED'])
    active = len(df[df['status'].isin(['ACTIVE', 'SCANNING', None])])
    
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    # AI Performance Grade Logic
    grade = "C"
    if win_rate >= 80: grade = "A+"
    elif win_rate >= 70: grade = "A"
    elif win_rate >= 60: grade = "B"
    elif win_rate < 40: grade = "D"

    # Best Performing Node
    best_node = df[df['status'] == 'SUCCESS']['exchange'].mode().iloc[0] if wins > 0 else "N/A"

    # Discord Embed Payload
    payload = {
        "username": "Nexus Performance Auditor",
        "embeds": [{
            "title": "📊 AEGIS WEEKLY PERFORMANCE REPORT",
            "description": f"Performance summary for the week ending {datetime.now().strftime('%Y-%m-%d')}",
            "color": 0xbc8cff,
            "fields": [
                {"name": "Neural Grade", "value": f"**{grade}**", "inline": True},
                {"name": "Win Rate", "value": f"**{win_rate:.1f}%**", "inline": True},
                {"name": "Top Node", "value": f"**{best_node.upper()}**", "inline": True},
                {"name": "Total Signals", "value": str(total_trades), "inline": True},
                {"name": "Wins/Losses", "value": f"✅ {wins} / ❌ {losses}", "inline": True},
                {"name": "Currently Active", "value": str(active), "inline": True}
            ],
            "footer": {"text": "Aegis Wealth Management • Performance Verified"}
        }]
    }
    
    if DISCORD_WEBHOOK_URL:
        requests.post(DISCORD_WEBHOOK_URL, json=payload)
    print(f"Weekly Report Sent. Win Rate: {win_rate}%")

if __name__ == "__main__":
    generate_weekly_report()
