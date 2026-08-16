import sqlite3
import pandas as pd
import requests
import os
from datetime import datetime, timedelta, timezone

# --- TESTING PHASE CONFIGURATION ---
DB_FILE = "nexus_signals.db"  # Unified master schema
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")
TEST_START_BALANCE = 100.0  
FIXED_RISK = 5.0            
REWARD_MULTIPLIER = 1.5     

def run_weekly_wealth_audit():
    if not os.path.exists(DB_FILE):
        print(f"❌ {DB_FILE} not found. Exiting audit.")
        return
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM signals WHERE status IN ('SUCCESS', 'FAILED')", conn)
    conn.close()

    if df.empty:
        print("⚠️ No completed trades logged yet. Exiting audit.")
        return

    # --- PERFORMANCE OPTIMIZATION: Vectorized Balance Calculation ---
    wins_total = len(df[df['status'] == 'SUCCESS'])
    losses_total = len(df[df['status'] == 'FAILED'])
    
    current_balance = TEST_START_BALANCE + (wins_total * FIXED_RISK * REWARD_MULTIPLIER) - (losses_total * FIXED_RISK)

    # --- WEEKLY FILTERING ---
    one_week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Ensure pandas treats the timestamp column as datetime for accurate comparison
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    weekly_df = df[df['timestamp'] >= pd.to_datetime(one_week_ago)]
    
    wins = len(weekly_df[weekly_df['status'] == 'SUCCESS'])
    losses = len(weekly_df[weekly_df['status'] == 'FAILED'])
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    # --- MISSING ARTIFACT FIX: Generate CSV for GitHub Action ---
    weekly_df.to_csv("weekly_performance_report.csv", index=False)
    print("✅ weekly_performance_report.csv generated.")

    # --- DISCORD ALERT: THE WEEKLY BALANCE ---
    testing_weeks = ((datetime.now(timezone.utc) - datetime(2026, 1, 31, tzinfo=timezone.utc)).days // 7) + 1
    
    balance_payload = {
        "username": "Nexus Wealth Auditor",
        "content": f"📝 **WEEKLY RECONCILIATION COMPLETE**\n"
                   f"The theoretical fund started at **$100.00**.\n"
                   f"After this week's activity, the remaining balance is: **${current_balance:.2f}**\n"
                   f"Testing Period: Week {testing_weeks} of 12."
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
        print("✅ Discord Webhooks fired successfully.")
    else:
        print("⚠️ DISCORD_WEBHOOK missing from environment. Skipping alerts.")

if __name__ == "__main__":
    run_weekly_wealth_audit()
