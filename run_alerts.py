import ccxt
import sqlite3
import requests
from config import DB_FILE, WEBHOOK_URL
from db_manager import initialize_database

# Exchanges with rate limit enabled
exchanges = [
    ccxt.gateio({'enableRateLimit': True}),
    ccxt.bitget({'enableRateLimit': True})
]

def send_alert(trade, outcome):
    if not WEBHOOK_URL:
        print("WEBHOOK_URL not set. Skipping alert.")
        return

    emoji = "🟢" if trade['signal'] == "LONG" else "🔴"
    message = (
        f"📊 **Nexus Alert**\n"
        f"{emoji} Asset: {trade['asset']}\n"
        f"Signal: {trade['signal']}\n"
        f"Entry: {trade['entry']}\n"
        f"TP: {trade['tp']}\n"
        f"SL: {trade['sl']}\n"
        f"Result: {outcome}"
    )
    try:
        requests.post(WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception as e:
        print("Webhook error:", e)

def check_alerts():
    initialize_database()

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM signals WHERE status = 'ACTIVE'")
    trades = cursor.fetchall()
    if not trades:
        print("No active trades.")
        conn.close()
        return

    # --- Fetch latest price per asset once ---
    assets = {trade['asset'] for trade in trades}
    asset_prices = {}
    for asset in assets:
        for ex in exchanges:
            try:
                ticker = ex.fetch_ticker(asset)
                price = ticker.get("last")
                if price is not None:
                    asset_prices[asset] = price
                    break
            except Exception:
                continue

    # --- Process trades ---
    for trade in trades:
        price = asset_prices.get(trade['asset'])
        if price is None or trade['tp'] is None or trade['sl'] is None:
            continue  # Skip if price missing or TP/SL invalid

        # Skip invalid TP/SL
        if trade['tp'] <= 0 or trade['sl'] <= 0:
            continue

        outcome = None
        if trade['signal'] == "LONG":
            if price >= trade['tp']:
                outcome = "SUCCESS"
            elif price <= trade['sl']:
                outcome = "FAILED"
        else:  # SHORT
            if price <= trade['tp']:
                outcome = "SUCCESS"
            elif price >= trade['sl']:
                outcome = "FAILED"

        if outcome:
            cursor.execute(
                "UPDATE signals SET status = ? WHERE id = ?",
                (outcome, trade['id'])
            )
            send_alert(trade, outcome)

    conn.commit()
    conn.close()
    print(f"Processed {len(trades)} trades.")

if __name__ == "__main__":
    check_alerts()
