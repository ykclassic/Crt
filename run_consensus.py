import sqlite3
from collections import Counter
from datetime import datetime, timezone
import logging

from config import TRADING_PAIRS
from nexus_dispatcher import dispatch_signal

DB_FILE = "signals.db"
CONSENSUS_DB = "consensus_signals.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | CONSENSUS | %(levelname)s | %(message)s"
)

def initialize_database():
    conn = sqlite3.connect(CONSENSUS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consensus (
            pair TEXT,
            direction TEXT,
            tier TEXT,
            confidence REAL,
            entry REAL,
            stop_loss REAL,
            take_profit REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def fetch_latest_signals(pair):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT direction, entry, stop_loss, take_profit
        FROM signals
        WHERE pair = ?
        ORDER BY timestamp DESC
    """, (pair,))

    rows = cursor.fetchall()
    conn.close()

    return rows


def classify_tier(count):
    if count >= 4:
        return "QUANTUM_ALIGNMENT", 0.95
    if count == 3:
        return "TRINITY_SYNC", 0.80
    if count == 2:
        return "DUAL_CONVERGENCE", 0.60
    return None, 0


def process_pair(pair):
    signals = fetch_latest_signals(pair)

    if len(signals) < 2:
        return

    directions = [s[0] for s in signals]
    majority_direction, majority_count = Counter(directions).most_common(1)[0]

    tier, confidence = classify_tier(majority_count)

    if not tier or confidence < 0.60:
        return

    aligned = [s for s in signals if s[0] == majority_direction]

    entry = sum(s[1] for s in aligned) / len(aligned)
    stop_loss = sum(s[2] for s in aligned) / len(aligned)
    take_profit = sum(s[3] for s in aligned) / len(aligned)

    save_consensus(pair, majority_direction, tier,
                   confidence, entry, stop_loss, take_profit)

    dispatch_signal(pair, majority_direction, tier,
                    confidence, entry, stop_loss, take_profit)

    logging.info(f"{pair} | {tier} | {majority_direction} dispatched")


def save_consensus(pair, direction, tier,
                   confidence, entry, stop_loss, take_profit):

    conn = sqlite3.connect(CONSENSUS_DB)
    conn.execute("""
        INSERT INTO consensus (
            pair, direction, tier,
            confidence, entry,
            stop_loss, take_profit,
            timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pair,
        direction,
        tier,
        confidence,
        entry,
        stop_loss,
        take_profit,
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()


def run():
    logging.info("Starting Consensus Engine")
    initialize_database()

    for pair in TRADING_PAIRS:
        try:
            process_pair(pair)
        except Exception as e:
            logging.error(f"{pair} error: {e}")

    logging.info("Consensus cycle complete")


if __name__ == "__main__":
    run()
