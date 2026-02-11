import ccxt
import sqlite3
from datetime import datetime
from config import DB_FILE

ex = ccxt.gateio({'enableRateLimit': True})

def run_recon():
    conn = sqlite3.connect(DB_FILE)
    df = conn.execute("""
        SELECT * FROM signals
        WHERE status = 'ACTIVE'
    """).fetchall()

    columns = [desc[0] for desc in conn.execute("PRAGMA table_info(signals)")]

    for row in df:
        trade = dict(zip(columns, row))
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
                conn.execute("""
                    UPDATE signals
                    SET status = ?
                    WHERE id = ?
                """, (outcome, trade["id"]))
                conn.commit()

        except Exception:
            continue

    conn.close()

if __name__ == "__main__":
    run_recon()
