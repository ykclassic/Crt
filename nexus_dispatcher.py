import sqlite3
import pandas as pd
import requests
import logging
import os
import time
from datetime import datetime, timedelta
from config import DB_FILE, WEBHOOK_URL, ENGINES, TOTAL_CAPITAL, RISK_PER_TRADE

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
            time.sleep(retry_after)
            return requests.post(WEBHOOK_URL, json=payload).status_code == 204
        return False
    except Exception as e:
        logging.error(f"❌ Connection Error: {e}")
        return False

def dispatch_alerts():
    logging.info(f"--- DISPATCHER: RISK & TIER UPDATE ---")
    # Extended lookback to 70 mins to ensure no signals are missed during workflow lag
    lookback_time = (datetime.now() - timedelta(minutes=70)).isoformat()

    if not os.path.exists(DB_FILE):
        logging.error("Database file not found!")
        return

    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT asset, signal, timeframe, entry, sl, tp, confidence, 
               rsi, vol_change, dist_ema, ts, engine 
        FROM signals WHERE ts > ? 
        ORDER BY ts DESC
    """
    df = pd.read_sql_query(query, conn, params=(lookback_time,))
    conn.close()

    if df.empty:
        logging.info("Zero signals found in lookback period.")
        return

    # Remove duplicates to avoid spamming the same entry
    df = df.drop_duplicates(subset=['asset', 'signal', 'ts'])
    
    for _, row in df.iterrows():
        # 1. Tier Identification
        conf = row['confidence']
        if conf >= 90 or conf <= 10:
            tier = "💎 **DIAMOND CONSENSUS** 💎"
        elif conf >= 80 or conf <= 20:
            tier = "🥇 **GOLD CONSENSUS** 🥇"
        else:
            tier = "🛡️ **NEXUS STANDARD**"

        # 2. Risk Calculation (Position Sizing in Tens of Dollars)
        # Formula: Capital * Risk% (e.g., $1000 * 0.02 = $20 Risk)
        risk_amt = TOTAL_CAPITAL * RISK_PER_TRADE
        # Clamp between $10 and $100 as requested
        risk_display = max(10, min(100, round(risk_amt / 10) * 10))
        
        # 3. Risk/Reward Calculation
        risk_dist = abs(row['entry'] - row['sl'])
        reward_dist = abs(row['tp'] - row['entry'])
        rr_ratio = round(reward_dist / risk_dist, 2) if risk_dist != 0 else 0

        # 4. Reason Formulation
        reason = "Oversold + EMA Mean Reversion" if row['signal'] == "LONG" else "Overbought + EMA Mean Reversion"
        emoji = "🟢" if str(row['signal']).upper() == "LONG" else "🔴"
        
        # 5. The "Elite" Alert Format
        msg = (
            f"{tier}\n"
            f"**Asset:** {emoji} `{row['asset']}` | **Signal:** `{row['signal']}`\n"
            f"**Timeframe:** ⏱️ `{row['timeframe']}`\n"
            f"**Engine:** ⚙️ `{ENGINES.get(row['engine'], 'AI-CORE')}`\n"
            f"------------------------------------\n"
            f"**Entry:** 📥 `{row['entry']:.5f}`\n"
            f"**Stop Loss:** 🛑 `{row['sl']:.5f}`\n"
            f"**Take Profit:** 🎯 `{row['tp']:.5f}`\n"
            f"------------------------------------\n"
            f"**Risk per Trade:** 💵 `${risk_display}`\n"
            f"**Risk/Reward:** 📈 `1:{rr_ratio}`\n"
            f"**Confidence:** ` {conf}%`\n"
            f"**Reason:** 💡 `{reason}`\n"
            f"------------------------------------\n"
            f"🕒 `{row['ts'][:16]}`"
        )
        
        if send_to_discord(msg):
            logging.info(f"Sent: {row['asset']}")
            time.sleep(1) # Prevent rate limit and preserve order

if __name__ == "__main__":
    dispatch_alerts()
