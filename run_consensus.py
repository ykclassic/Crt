import sqlite3
from collections import Counter
from datetime import datetime
import logging

from nexus_dispatcher import dispatch_signal

# ==========================================
# DATABASE FILES
# ==========================================

TREND_DB = "trending_signals.db"
RANGE_DB = "ranging_signals.db"
HYBRID_DB = "hybrid_signals.db"
AI_DB = "ai_engine_signals.db"

CONSENSUS_DB = "consensus_signals.db"

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | CONSENSUS | %(levelname)s | %(message)s"
)

# ==========================================
# DATABASE INIT
# ==========================================

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

# ==========================================
# FETCH LATEST SIGNAL
# ==========================================

def fetch_latest_signal(db_file, pair):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pair, direction, entry, stop_loss, take_profit, timestamp
        FROM signals
        WHERE pair = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (pair,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "pair": row[0],
            "direction": row[1],
            "entry": row[2],
            "stop_loss": row[3],
            "take_profit": row[4],
            "timestamp": row[5]
        }

    return None

# ==========================================
# TIER CLASSIFICATION
# ==========================================

def classify_tier(count):
    if count == 4:
        return "QUANTUM_ALIGNMENT", 0.95
    elif count == 3:
        return "TRINITY_SYNC", 0.80
    elif count == 2:
        return "DUAL_CONVERGENCE", 0.60
    else:
        return "DISSONANCE_STATE", 0.30

# ==========================================
# CONSENSUS CORE
# ==========================================

def process_pair(pair):

    trend = fetch_latest_signal(TREND_DB, pair)
    range_ = fetch_latest_signal(RANGE_DB, pair)
    hybrid = fetch_latest_signal(HYBRID_DB, pair)
    ai = fetch_latest_signal(AI_DB, pair)

    signals = [s for s in [trend, range_, hybrid, ai] if s]

    if len(signals) < 2:
        return

    directions = [s["direction"] for s in signals]
    direction_counts = Counter(directions)

    majority_direction, majority_count = direction_counts.most_common(1)[0]

    tier, base_confidence = classify_tier(majority_count)

    # --------------------------------------
    # AI BOOST / FILTER LOGIC
    # --------------------------------------

    ai_signal = ai["direction"] if ai else None

    final_confidence = base_confidence
    final_tier = tier

    # Case: DISSONANCE → Block immediately
    if tier == "DISSONANCE_STATE":
        logging.info(f"{pair} | Dissonance - No Dispatch")
        return

    if ai_signal:

        if ai_signal == majority_direction:
            final_confidence += 0.05

        else:
            # AI conflict cases
            if majority_count == 3:
                # Downgrade one tier
                final_tier = "DUAL_CONVERGENCE"
                final_confidence = 0.60
                final_confidence -= 0.15

            elif majority_count == 2:
                # AI blocks weak majority
                logging.info(f"{pair} | AI Filtered Dual Convergence")
                return

    if final_confidence < 0.60:
        logging.info(f"{pair} | Confidence below threshold")
        return

    # --------------------------------------
    # Aggregate Entry / SL / TP
    # --------------------------------------

    aligned_signals = [s for s in signals if s["direction"] == majority_direction]

    entry = sum(s["entry"] for s in aligned_signals) / len(aligned_signals)
    stop_loss = sum(s["stop_loss"] for s in aligned_signals) / len(aligned_signals)
    take_profit = sum(s["take_profit"] for s in aligned_signals) / len(aligned_signals)

    save_consensus(pair, majority_direction, final_tier,
                   final_confidence, entry, stop_loss, take_profit)

    dispatch_signal(
        pair=pair,
        direction=majority_direction,
        tier=final_tier,
        confidence=final_confidence,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    logging.info(f"{pair} | {final_tier} | {majority_direction} dispatched")

# ==========================================
# SAVE CONSENSUS
# ==========================================

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
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

# ==========================================
# RUN
# ==========================================

def run(trading_pairs):
    logging.info("Starting Consensus Engine")
    initialize_database()

    for pair in trading_pairs:
        try:
            process_pair(pair)
        except Exception as e:
            logging.error(f"{pair} error: {str(e)}")

    logging.info("Consensus cycle complete")

if __name__ == "__main__":
    from config import TRADING_PAIRS
    run(TRADING_PAIRS)
