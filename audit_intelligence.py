import sqlite3
import pandas as pd

def check_learning_progress():
    conn = sqlite3.connect("hybrid_v1.db")
    try:
        # Load the last 100 signals
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 100", conn)
        
        if df.empty:
            print("No data collected yet. The 'Brain' is still empty.")
            return

        # Calculate Win Rate if you've been tracking outcomes
        # In a real setup, we compare 'entry' vs 'current_price'
        total_signals = len(df)
        unique_assets = df['asset'].nunique()
        
        print(f"--- Intelligence Audit ---")
        print(f"Total Experience Points (Signals): {total_signals}")
        print(f"Market Coverage: {unique_assets} assets")
        print(f"Learning Status: {'Initial Collection' if total_signals < 50 else 'Optimizing'}")
        
    except Exception as e:
        print(f"Audit Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_learning_progress()
