import sqlite3
from collections import Counter
import logging
from config import TRADING_PAIRS, DB_FILE, EXECUTION_TF
from nexus_dispatcher import dispatch_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s | CONSENSUS | %(levelname)s | %(message)s")

def classify_tier(count):
    if count >= 4:
        return "💎 Diamond"
    if count == 3:
        return "🥇 Gold"
    if count == 2:
        return "🥈 Silver"
    return None

def process_pair(pair):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Fetch signals from the last 2 hours 
    cursor.execute("""
        SELECT engine_id, direction, entry, stop_loss, take_profit
        FROM signals
        WHERE symbol = ? AND timeframe = ? AND timestamp >= datetime('now', '-2 hours')
    """, (pair, EXECUTION_TF))
    
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return

    # Isolate the latest signal per engine to avoid duplicate weight
    engine_signals = {}
    for r in rows:
        engine_id, direction, entry, sl, tp = r
        engine_signals[engine_id] = (direction, entry, sl, tp)

    # Require minimum of 2 engines agreeing
    if len(engine_signals) < 2:
        return

    directions = [s[0] for s in engine_signals.values()]
    majority_direction, majority_count = Counter(directions).most_common(1)[0]

    tier = classify_tier(majority_count)
    
    if not tier or majority_count < 2:
        return

    aligned = [s for s in engine_signals.values() if s[0] == majority_direction]

    avg_entry = sum(s[1] for s in aligned) / len(aligned)
    avg_stop_loss = sum(s[2] for s in aligned) / len(aligned)
    avg_take_profit = sum(s[3] for s in aligned) / len(aligned)
    confidence = majority_count / 4.0

    logging.info(f"{pair} | {tier} | {majority_direction} calculated")
    dispatch_signal(pair, majority_direction, tier, confidence, avg_entry, avg_stop_loss, avg_take_profit)

def run():
    logging.info("Starting Consensus Engine")
    for pair in TRADING_PAIRS:
        try:
            process_pair(pair)
        except Exception as e:
            logging.error(f"{pair} error: {e}")
    logging.info("Consensus cycle complete")

if __name__ == "__main__":
    run()
