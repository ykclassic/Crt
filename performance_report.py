import sqlite3
import pandas as pd
import requests
import os
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FILE = "nexus.db"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")
STARTING_BALANCE = 100.0  # Your theoretical $100
RISK_PER_TRADE = 5.0      # Theoretical $5 risk per signal
REWARD_RATIO = 1.5        # Assuming 1:1.5 Risk/Reward if not specified

def generate_weekly_report():
    if not os.path.exists(DB_FILE): return
    
    conn = sqlite3.connect(DB_FILE)
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    df = pd.read_sql_query(f"SELECT * FROM signals WHERE ts >= '{one_week_ago}'", conn)
    conn.close()

    if df.empty:
        print("No trades found for this week.")
        return

    # --- BALANCE CALCULATION ---
    current_balance = STARTING_BALANCE
    balance_history = [STARTING_BALANCE]
    
    closed_trades = df[df['status'].isin(['SUCCESS', 'FAILED'])]
    
    for _, trade in closed_trades.iterrows():
        if trade['status'] == 'SUCCESS':
            current_balance += (RISK_PER_TRADE * REWARD_RATIO)
        else:
            current_balance -= RISK_PER_TRADE
        balance_history.append(current_balance)

    # --- EQUITY CURVE GENERATION ---
    fig = px.line(x=list(range(len(balance_history))), y=balance_history, 
                 title="Weekly Equity Curve", template="plotly_dark")
    fig.update_layout(xaxis_title="Trades", yaxis_title="Balance ($)")
    fig.write_image("equity_curve.png") # Requires 'kaleido' package

    # --- PERFORMANCE METRICS ---
    total_closed = len(closed_trades)
    wins = len(df[df['status'] == 'SUCCESS'])
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    profit_loss = current_balance - STARTING_BALANCE
    status_emoji = "📈" if profit_loss >= 0 else "📉"

    # --- DISCORD ALERT 1: METRICS ---
    payload_metrics = {
        "username": "Nexus Wealth Auditor",
        "embeds": [{
            "title": f"📊 WEEKLY PERFORMANCE AUDIT",
            "color": 0xbc8cff,
            "fields": [
                {"name": "Starting Fund", "value": f"${STARTING_BALANCE}", "inline": True},
                {"name": "Closing Fund", "value": f"**${current_balance:.2f}**", "inline": True},
                {"name": "Net P/L", "value": f"{status_emoji} ${profit_loss:.2f}", "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
                {"name": "Closed Trades", "value": str(total_closed), "inline": True}
            ],
            "footer": {"text": f"Audit Period: {one_week_ago} to Present"}
        }]
    }

    # --- DISCORD ALERT 2: BALANCE FINAL ---
    payload_balance = {
        "username": "Nexus Wealth Auditor",
        "content": f"🚨 **FINAL WEEKLY BALANCE ALERT** 🚨\nYour theoretical fund currently stands at: **${current_balance:.2f}**"
    }
    
    if DISCORD_WEBHOOK_URL:
        requests.post(DISCORD_WEBHOOK_URL, json=payload_metrics)
        requests.post(DISCORD_WEBHOOK_URL, json=payload_balance)
        # Note: Sending the equity_curve.png would require a multipart/form-data request
        print("Weekly Performance & Balance alerts sent.")

if __name__ == "__main__":
    generate_weekly_report()
