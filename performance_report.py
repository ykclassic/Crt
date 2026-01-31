import sqlite3
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# --- TESTING PHASE CONFIGURATION ---
DB_FILE = "nexus.db"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")
TEST_START_BALANCE = 100.0  
FIXED_RISK = 5.0            # Fixed $5 per trade for the 3-month test
REWARD_MULTIPLIER = 1.5     # Standardizing 1:1.5 Risk/Reward for the audit

def run_weekly_wealth_audit():
    if not os.path.exists(DB_FILE): return
    
    conn = sqlite3.connect(DB_FILE)
    # Looking at the full testing history to track the $100 balance accurately
    df = pd.read_sql_query("SELECT * FROM signals WHERE status IN ('SUCCESS', 'FAILED')", conn)
    conn.close()

    # Calculate Current Balance from Start of Testing
    current_balance = TEST_START_BALANCE
    for _, trade in df.iterrows():
        if trade['status'] == 'SUCCESS':
            current_balance += (FIXED_RISK * REWARD_MULTIPLIER)
        else:
            current_balance -= FIXED_RISK

    # Filter for just THIS week's performance metrics
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    df['ts'] = pd.to_datetime(df['ts'])
    weekly_df = df[df['ts'] >= one_week_ago]
    
    wins = len(weekly_df[weekly_df['status'] == 'SUCCESS'])
    losses = len(weekly_df[weekly_df['status'] == 'FAILED'])
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    # --- DISCORD ALERT: THE WEEKLY BALANCE ---
    # This is the separate alert you requested for the remaining balance
    balance_payload = {
        "username": "Nexus Wealth Auditor",
        "content": f"📝 **WEEKLY RECONCILIATION COMPLETE**\n"
                   f"The theoretical fund started at **$100.00**.\n"
                   f"After this week's activity, the remaining balance is: **${current_balance:.2f}**\n"
                   f"Testing Period: Week {((datetime.now() - datetime(2026, 1, 31)).days // 7) + 1} of 12."
    }

    # --- DISCORD ALERT: PERFORMANCE DATA ---
    metrics_payload = {
        "username": "Nexus Performance Auditor",
        "embeds": [{
            "title": "🛡️ Aegis Weekly Audit Trail",
            "color": 0x58a6ff,
            "fields": [
                {"name": "Weekly Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
                {"name": "Weekly Outcome", "value": f"{wins} Wins / {losses} Losses", "inline": True},
                {"name": "Theoretical Fund", "value": f"${current_balance:.2f}", "inline": True}
            ],
            "footer": {"text": "3-Month Testing Phase | Data Verified by Recon Engine"}
        }]
    }

    if DISCORD_WEBHOOK_URL:
        requests.post(DISCORD_WEBHOOK_URL, json=metrics_payload)
        requests.post(DISCORD_WEBHOOK_URL, json=balance_payload)

if __name__ == "__main__":
    run_weekly_wealth_audit()
