import sqlite3
from config import DB_FILE

UNIFIED_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    engine TEXT,
    asset TEXT,
    timeframe TEXT,
    signal TEXT,
    entry REAL,
    sl REAL,
    tp REAL,
    confidence REAL,
    rsi REAL,
    vol_change REAL,
    dist_ema REAL,
    reason TEXT,
    status TEXT,
    ts TEXT
);
"""

REQUIRED_COLUMNS = {
    "rsi": "REAL",
    "vol_change": "REAL",
    "dist_ema": "REAL",
    "reason": "TEXT",
    "status": "TEXT"
}

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(UNIFIED_SCHEMA)
    conn.commit()
    conn.close()

def migrate_schema():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(signals)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    for column, col_type in REQUIRED_COLUMNS.items():
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE signals ADD COLUMN {column} {col_type}")
            print(f"[DB MIGRATION] Added column: {column}")

    conn.commit()
    conn.close()

def initialize_database():
    init_db()
    migrate_schema()
