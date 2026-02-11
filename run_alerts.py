import ccxt
import sqlite3
from config import DB_FILE

exchanges = [
    ccxt.gateio({'enableRateLimit': True}),
    ccxt.bitget({'enableRateLimit': True})
]

def check_alerts():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""
        SELECT * FROM signals
        WHERE status = 'ACTIVE'
    """).fetchall()

    columns = [desc[0] for desc in conn.execute("PRAGMA table_info(signals)")]

    for row in rows:
        trade = dict(zip(columns, row))

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

def update_status(conn, trade_id, status):
    conn.execute("""
        UPDATE signals
        SET status = ?
        WHERE id = ?
    """, (status, trade_id))
    conn.commit()

if __name__ == "__main__":
    check_alerts()
