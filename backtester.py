import sqlite3
import ccxt
import os
import logging
from nexus_dispatcher import update_signal_performance, ROOT_DB_PATH

exchange = ccxt.binance()

def fetch_signals_to_grade():
    conn = sqlite3.connect(ROOT_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Find signals that haven't been graded yet
    cursor.execute("SELECT * FROM dispatched_alerts WHERE outcome = 'PENDING'")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def backtest_pending_signals():
    signals = fetch_signals_to_grade()
    if not signals:
        print("No pending signals to grade.")
        return

    for sig in signals:
        try:
            # Fetch current price for the pair
            ticker = exchange.fetch_ticker(sig['pair'])
            current_price = ticker['last']
            
            outcome = "PENDING"
            pnl = 0.0

            if sig['direction'].upper() == "LONG":
                if current_price >= sig['take_profit']:
                    outcome, pnl = "HIT_TP", ((sig['take_profit'] / sig['entry']) - 1) * 100
                elif current_price <= sig['stop_loss']:
                    outcome, pnl = "HIT_SL", ((sig['stop_loss'] / sig['entry']) - 1) * 100
            else: # SHORT
                if current_price <= sig['take_profit']:
                    outcome, pnl = "HIT_TP", (1 - (sig['take_profit'] / sig['entry'])) * 100
                elif current_price >= sig['stop_loss']:
                    outcome, pnl = "HIT_SL", (1 - (sig['stop_loss'] / sig['entry'])) * 100

            if outcome != "PENDING":
                update_signal_performance(sig['id'], outcome, round(pnl, 2))
                print(f"Signal {sig['id']} ({sig['pair']}) closed as {outcome} with {pnl:.2f}% PnL")
                
        except Exception as e:
            logging.error(f"Error backtesting {sig['pair']}: {e}")

if __name__ == "__main__":
    backtest_pending_signals()
