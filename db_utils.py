import sqlite3

def init_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            entry REAL,
            stop_loss REAL,
            take_profit REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_signal(db_name, data):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO signals
        (symbol, direction, entry, stop_loss, take_profit, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()


def get_latest_signal(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT symbol, direction, entry, stop_loss, take_profit
        FROM signals
        WHERE id IN (
            SELECT MAX(id) FROM signals GROUP BY symbol
        )
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows
