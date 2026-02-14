import sqlite3
from config import DB_FILE

def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Drop old schema to avoid column mismatch
    cursor.execute("DROP TABLE IF EXISTS signals")

    cursor.execute("""
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine TEXT,
            pair TEXT,
            timeframe TEXT,
            direction TEXT,
            entry REAL,
            stop_loss REAL,
            take_profit REAL,
            confidence REAL,
            rsi REAL,
            vol_change REAL,
            dist_ema REAL,
            reason TEXT,
            status TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()
