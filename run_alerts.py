import asyncio
import ccxt.async_support as ccxt
import sqlite3
import requests
from config import DB_FILE, WEBHOOK_URL
from db_manager import initialize_database

# --- Exchanges (async) ---
exchanges = [
    ccxt.gateio({'enableRateLimit': True}),
    ccxt.bitget({'enableRateLimit': True})
]

# --- Send a batch alert to Discord ---
def send_alert_batch(messages: list[str]):
    if not WEBHOOK_URL or not messages:
        print("WEBHOOK_URL not set or no messages. Skipping alert.")
        return
    payload = {"content": "\n\n".join(messages)}
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print("Webhook error:", e)

# --- Update trade status in DB ---
def update_status(conn, trade_id, new_status):
    conn.execute("UPDATE signals SET status = ? WHERE id = ?", (new_status, trade_id))
    conn.commit()

# --- Fetch current price async for one asset/exchange ---
async def fetch_price(ex, asset):
    try:
        ticker = await ex.fetch_ticker(asset)
        return ticker["last"]
    except Exception:
        return None

# --- Process a single trade asynchronously ---
async def process_trade(trade, conn):
    asset = trade["asset"]
    if trade["tp"] is None or trade["sl"] is None:
        return None  # Skip invalid trades

    outcome = None
    for ex in exchanges:
        price = await fetch_price(ex, asset)
        if price is None:
            continue

        if trade["signal"] == "LONG":
            if price >= trade["tp"]:
                outcome = "SUCCESS"
            elif price <= trade["sl"]:
                outcome = "FAILED"
        else:  # SHORT
            if price <= trade["tp"]:
                outcome = "SUCCESS"
            elif price >= trade["sl"]:
                outcome = "FAILED"

        if outcome:
            update_status(conn, trade["id"], outcome)
            msg = (
                f"📊 **Nexus Alert**\n"
                f"Asset: {asset}\n"
                f"Signal: {trade['signal']}\n"
                f"Entry: {trade['entry']}\n"
                f"TP: {trade['tp']}\n"
                f"SL: {trade['sl']}\n"
                f"Result: {outcome}"
            )
            return msg
    return None

# --- Main async engine ---
async def check_alerts_async():
    initialize_database()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM signals WHERE status = 'ACTIVE'")
    trades = cursor.fetchall()

    tasks = [process_trade(trade, conn) for trade in trades]
    results = await asyncio.gather(*tasks)
    conn.close()

    # Filter out None results and send batch alert
    messages = [msg for msg in results if msg]
    send_alert_batch(messages)

if __name__ == "__main__":
    asyncio.run(check_alerts_async())
