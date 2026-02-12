import ccxt
import sqlite3
import requests
from config import DB_FILE, WEBHOOK_URL
from db_manager import initialize_database

exchanges = [
    ccxt.gateio({'enableRateLimit': True}),
    ccxt.bitget({'enableRateLimit': True})
]

def send_alert(message):
    if not WEBHOOK_URL:
        print("WEBHOOK_URL not set. Skipping alert.")
        return

    try:
        requests.post(
            WEBHOOK_URL,
            json={"content": message},
            timeout=10
        )
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

    for trade in trades:
        for ex in exchanges:
            try:
                ticker = ex.fetch_ticker(trade["asset"])
                price = ticker["last"]

                outcome = None

                if trade["signal"] == "LONG":
                    if price >= trade["tp"]:
                        outcome = "SUCCESS"
                    elif price <= trade["sl"]:
                        outcome = "FAILED"
                else:
                    if price <= trade["tp"]:
                        outcome = "SUCCESS"
                    elif price >= trade["sl"]:
                        outcome = "FAILED"

                if outcome:
                    update_status(conn, trade, outcome)

                break

            except Exception:
                continue

    conn.close()

if __name__ == "__main__":
    check_alerts()
