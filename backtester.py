import sqlite3
import ccxt
import os
import logging
from nexus_dispatcher import update_signal_performance, ROOT_DB_PATH

# Initialize Gate.io instead of Binance
# No API keys are needed just for fetching public price tickers
exchange = ccxt.gateio({
    'enableRateLimit': True,
})

def fetch_signals_to_grade():
    """Retrieves all pending signals from the local root database."""
    try:
        conn = sqlite3.connect(ROOT_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dispatched_alerts WHERE outcome = 'PENDING'")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logging.error(f"Database access error: {e}")
        return []

def backtest_pending_signals():
    """Checks current Gate.io prices against pending signals to grade them."""
    signals = fetch_signals_to_grade()
    if not signals:
        print("No pending signals to grade.")
        return

    print(f"Checking {len(signals)} pending signals on Gate.io...")

    for sig in signals:
        try:
            # Gate.io uses the same 'pair' format (e.g., BTC/USDT)
            ticker = exchange.fetch_ticker(sig['pair'])
            current_price = ticker['last']
            
            outcome = "PENDING"
            pnl = 0.0

            # Logic to determine if TP or SL was hit
            if sig['direction'].upper() == "LONG":
                if current_price >= sig['take_profit']:
                    outcome = "HIT_TP"
                    pnl = ((sig['take_profit'] / sig['entry']) - 1) * 100
                elif current_price <= sig['stop_loss']:
                    outcome = "HIT_SL"
                    pnl = ((sig['stop_loss'] / sig['entry']) - 1) * 100
            
            elif sig['direction'].upper() == "SHORT":
                if current_price <= sig['take_profit']:
                    outcome = "HIT_TP"
                    pnl = (1 - (sig['take_profit'] / sig['entry'])) * 100
                elif current_price >= sig['stop_loss']:
                    outcome = "HIT_SL"
                    pnl = (1 - (sig['stop_loss'] / sig['entry'])) * 100

            # If the trade reached a conclusion, update the DB so the AI can learn
            if outcome != "PENDING":
                update_signal_performance(sig['id'], outcome, round(pnl, 2))
                print(f"✅ Signal {sig['id']} ({sig['pair']}): {outcome} | PnL: {pnl:.2f}%")
            else:
                print(f"⏳ Signal {sig['id']} ({sig['pair']}): Still active. Price: {current_price}")
                
        except Exception as e:
            logging.error(f"Error checking {sig['pair']} on Gate.io: {e}")

if __name__ == "__main__":
    backtest_pending_signals()
