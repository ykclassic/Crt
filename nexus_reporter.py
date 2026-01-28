import sqlite3
import pandas as pd
import requests
import json
import os
import pickle
from datetime import datetime, timedelta
from config import DB_FILE, WEBHOOK_URL, MODEL_FILE, PERFORMANCE_FILE

def get_ai_narrative():
    if not os.path.exists(MODEL_FILE):
        return "AI Brain: Training required."
    try:
        with open(MODEL_FILE, "rb") as f:
            model, _ = pickle.load(f)
        importances = model.feature_importances_
        traits = ["Momentum", "Volatility", "Trend"]
        top = traits[importances.argmax()]
        return f"The AI is currently prioritized on **{top}** patterns."
    except: return "AI Brain: Error reading logic."

def run_weekly_report():
    conn = sqlite3.connect(DB_FILE)
    
    # 1. Total Signals vs Diamond Consensus
    df_all = pd.read_sql_query("SELECT * FROM signals WHERE ts > datetime('now', '-7 days')", conn)
    
    # Logic to count 'Diamond' hits (where 1h and 4h aligned)
    # We can approximate this by looking for signals on same asset/direction within same hour
    diamond_count = 0
    assets = df_all['asset'].unique()
    for asset in assets:
        asset_df = df_all[df_all['asset'] == asset]
        for _, row in asset_df[asset_df['timeframe'] == '1h'].iterrows():
            match = asset_df[(asset_df['timeframe'].isin(['4h', '1d'])) & 
                             (asset_df['signal'] == row['signal']) &
                             (pd.to_datetime(asset_df['ts']) <= pd.to_datetime(row['ts']))]
            if not match.empty:
                diamond_count += 1

    # 2. Performance Stats
    perf_summary = "No performance data yet."
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, "r") as f:
            perf = json.load(f)
            perf_summary = "\n".join([f"• **{k}**: {v['win_rate']}% Win Rate ({v['status']})" for k, v in perf.items()])

    ai_bio = get_ai_narrative()

    report_msg = (
        f"📊 **NEXUS CTO WEEKLY BRIEF**\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📡 **Network Activity (7D)**\n"
        f"• Total Raw Signals: `{len(df_all)}`\n"
        f"• Diamond Consensus: `{diamond_count}` 💎\n\n"
        f"🧠 **AI Brain State**\n"
        f"{ai_bio}\n\n"
        f"🏆 **Engine Standings**\n"
        f"{perf_summary}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🚀 *System Status: Fully Operational*"
    )

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": report_msg})
    
    conn.close()

if __name__ == "__main__":
    run_weekly_report()
