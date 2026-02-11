import ccxt
import sqlite3
from config import DB_FILE
from db_manager import initialize_database

exchanges = [
    ccxt.gateio({'enableRateLimit': True}),
    ccxt.bitget({'enableRateLimit': True})
]

def update_status(conn, trade_id, status):
    conn.execute(
        "UPDATE signals SET status = ? WHERE id = ?",
        (status, trade_id)
    )
    conn.commit()

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

                if trade["signal"] == "LONG":
                    if price >= trade["tp"]:
                        update_status(conn, trade["id"], "SUCCESS")
                    elif price <= trade["sl"]:
                        update_status(conn, trade["id"], "FAILED")
                else:
                    if price <= trade["tp"]:
                        update_status(conn, trade["id"], "SUCCESS")
                    elif price >= trade["sl"]:
                        update_status(conn, trade["id"], "FAILED")

                break

            except Exception:
                continue

    conn.close()

if __name__ == "__main__":
    check_alerts()
