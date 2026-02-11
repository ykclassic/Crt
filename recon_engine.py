import ccxt
import sqlite3
from config import DB_FILE
from db_manager import initialize_database

ex = ccxt.gateio({'enableRateLimit': True})

def run_recon():
    initialize_database()

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM signals WHERE status = 'ACTIVE'")
    trades = cursor.fetchall()

    for trade in trades:
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
                cursor.execute(
                    "UPDATE signals SET status = ? WHERE id = ?",
                    (outcome, trade["id"])
                )
                conn.commit()

        except Exception:
            continue

    conn.close()

if __name__ == "__main__":
    run_recon()
