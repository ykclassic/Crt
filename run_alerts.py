import ccxt
import sqlite3
import requests
from config import DB_FILE, WEBHOOK_URL
from db_manager import initialize_database

# Enable rate limit
exchanges = [
    ccxt.gateio({'enableRateLimit': True}),
    ccxt.bitget({'enableRateLimit': True})
]

def send_alert(message):
    if not WEBHOOK_URL:
        print("WEBHOOK_URL not set. Skipping alert.")
        return
    try:
        requests.post(WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception as e:
        print("Webhook error:", e)

def update_status(conn, trade, new_status):
    conn.execute(
        "UPDATE signals SET status = ? WHERE id = ?",
        (new_status, trade["id"])
    )
    conn.commit()

    message = (
        f"📊 **Nexus Alert**\n"
        f"Asset: {trade['asset']}\n"
        f"Signal: {trade['signal']}\n"
        f"Entry: {trade['entry']}\n"
        f"TP: {trade['tp']}\n"
        f"SL: {trade['sl']}\n"
        f"Result: {new_status}"
    )
    send_alert(message)

def check_alerts():
    initialize_database()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM signals WHERE status = 'ACTIVE'")
    trades = cursor.fetchall()

    if not trades:
        print("No active trades found.")
        return

    # --- Build unique asset list ---
    assets = set(trade["asset"] for trade in trades)

    # --- Fetch latest prices per asset ---
    asset_prices = {}
    for asset in assets:
        for ex in exchanges:
            try:
                ticker = ex.fetch_ticker(asset)
                price = ticker.get("last")
                if price is not None:
                    asset_prices[asset] = price
                    break  # Use first successful exchange
            except Exception:
                continue

    # --- Update trades based on fetched prices ---
    for trade in trades:
        price = asset_prices.get(trade["asset"])
        if price is None or trade["tp"] is None or trade["sl"] is None:
            continue  # Skip if price or TP/SL missing

        outcome = None
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
            update_status(conn, trade, outcome)

    conn.close()

if __name__ == "__main__":
    check_alerts()
