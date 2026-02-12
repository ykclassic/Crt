import sqlite3
import pandas as pd
import requests
import logging
import os
import time
from datetime import datetime, timedelta
from config import DB_FILE, ENGINES, TOTAL_CAPITAL, RISK_PER_TRADE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------------------------------
# Discord Webhook (Environment Secret)
# -------------------------------------------------
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# -------------------------------------------------
# Discord Sender
# -------------------------------------------------
def send_to_discord(msg):

    if not DISCORD_WEBHOOK_URL:
        logging.error("❌ DISCORD_WEBHOOK_URL not configured.")
        return False

    try:
        payload = {"content": str(msg)}
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)

        if response.status_code == 204:
            return True

        # Rate limit handling
        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 1)
            logging.warning(f"Rate limited. Retrying in {retry_after}s")
            time.sleep(retry_after)
            retry = requests.post(DISCORD_WEBHOOK_URL, json=payload)
            return retry.status_code == 204

        logging.error(f"Discord error: {response.status_code} | {response.text}")
        return False

    except Exception as e:
        logging.error(f"❌ Discord connection error: {e}")
        return False


# -------------------------------------------------
# Dispatcher
# -------------------------------------------------
def dispatch_alerts():

    logging.info("--- DISPATCHER: RISK & TIER UPDATE ---")

    lookback_time = (datetime.utcnow() - timedelta(minutes=70)).isoformat()

    if not os.path.exists(DB_FILE):
        logging.error("Database file not found.")
        return

    conn = sqlite3.connect(DB_FILE)

    query = """
        SELECT asset, signal, timeframe, entry, sl, tp, confidence,
               rsi, vol_change, dist_ema, ts, engine
        FROM signals
        WHERE ts > ?
        ORDER BY ts DESC
    """

    df = pd.read_sql_query(query, conn, params=(lookback_time,))
    conn.close()

    if df.empty:
        logging.info("Zero signals found in lookback period.")
        return

    # Remove duplicates
    df = df.drop_duplicates(subset=["asset", "signal", "timeframe", "ts"])

    for _, row in df.iterrows():

        conf = float(row["confidence"]) if row["confidence"] is not None else 0

        # -------------------------------------------------
        # Tier Logic
        # -------------------------------------------------
        if conf >= 0.90:
            tier = "💎 **DIAMOND CONSENSUS** 💎"
        elif conf >= 0.80:
            tier = "🥇 **GOLD CONSENSUS** 🥇"
        else:
            tier = "🛡️ **NEXUS STANDARD**"

        # -------------------------------------------------
        # Risk Calculation
        # -------------------------------------------------
        risk_amt = TOTAL_CAPITAL * RISK_PER_TRADE
        risk_display = max(10, min(100, round(risk_amt / 10) * 10))

        entry = row["entry"]
        sl = row["sl"]
        tp = row["tp"]

        rr_ratio = 0

        if entry and sl and tp:
            risk_dist = abs(entry - sl)
            reward_dist = abs(tp - entry)
            if risk_dist > 0:
                rr_ratio = round(reward_dist / risk_dist, 2)

        # -------------------------------------------------
        # Reason
        # -------------------------------------------------
        reason = row.get("reason", "Signal Generated")
        emoji = "🟢" if str(row["signal"]).upper() == "LONG" else "🔴"

        # -------------------------------------------------
        # Alert Format
        # -------------------------------------------------
        msg = (
            f"{tier}\n"
            f"**Asset:** {emoji} `{row['asset']}` | **Signal:** `{row['signal']}`\n"
            f"**Timeframe:** ⏱️ `{row['timeframe']}`\n"
            f"**Engine:** ⚙️ `{ENGINES.get(row['engine'], 'AI-CORE')}`\n"
            f"------------------------------------\n"
            f"**Entry:** 📥 `{entry}`\n"
            f"**Stop Loss:** 🛑 `{sl}`\n"
            f"**Take Profit:** 🎯 `{tp}`\n"
            f"------------------------------------\n"
            f"**Risk per Trade:** 💵 `${risk_display}`\n"
            f"**Risk/Reward:** 📈 `1:{rr_ratio}`\n"
            f"**Confidence:** `{round(conf * 100 if conf <= 1 else conf, 2)}%`\n"
            f"**Reason:** 💡 `{reason}`\n"
            f"------------------------------------\n"
            f"🕒 `{str(row['ts'])[:16]} UTC`"
        )

        if send_to_discord(msg):
            logging.info(f"Sent: {row['asset']}")
            time.sleep(1)

    logging.info("Dispatcher cycle complete.")


if __name__ == "__main__":
    dispatch_alerts()
