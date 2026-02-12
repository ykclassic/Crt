import ccxt
import sqlite3
import logging
from config import DB_FILE
from db_manager import initialize_database

logging.basicConfig(level=logging.INFO)

# Safe exchange config
ex = ccxt.gateio({
    "enableRateLimit": True,
    "timeout": 15000
})


def run_recon():
    initialize_database()

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM signals WHERE status = 'ACTIVE'")
    trades = cursor.fetchall()

    if not trades:
        conn.close()
        return

    # ============================================================
    # 1️⃣ Collect Unique Assets
    # ============================================================
    assets = list({trade["asset"] for trade in trades})

    # ============================================================
    # 2️⃣ Fetch All Tickers Once
    # ============================================================
    prices = {}

    for asset in assets:
        try:
            ticker = ex.fetch_ticker(asset)
            prices[asset] = ticker.get("last")
        except Exception as e:
            logging.warning(f"Ticker fetch failed for {asset}: {e}")
            prices[asset] = None

    # ============================================================
    # 3️⃣ Evaluate Trades
    # ============================================================
    updates = []

    for trade in trades:
        price = prices.get(trade["asset"])

        if not price:
            continue

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
            updates.append((outcome, trade["id"]))

    # ============================================================
    # 4️⃣ Batch Update
    # ============================================================
    if updates:
        cursor.executemany(
            "UPDATE signals SET status = ? WHERE id = ?",
            updates
        )
        conn.commit()
        logging.info(f"{len(updates)} trades updated.")

    conn.close()


if __name__ == "__main__":
    run_recon()
