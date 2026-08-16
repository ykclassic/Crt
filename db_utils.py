import sqlite3
import logging

def init_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Master unified table for all engine signals
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            confidence REAL,
            rsi REAL,
            vol_change REAL,
            dist_ema REAL,
            reason TEXT,
            status TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_signal(db_name, data_dict):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        columns = ', '.join(data_dict.keys())
        placeholders = ', '.join('?' * len(data_dict))
        query = f"INSERT INTO signals ({columns}) VALUES ({placeholders})"
        
        cursor.execute(query, tuple(data_dict.values()))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Database Insert Error: {e}")
