import sqlite3
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

DB_FILE = "nexus.db"

def generate_dashboard():
    """
    Generates the Aegis Wealth Command dashboard.
    Includes dynamic column mapping to prevent KeyErrors and ensures non-hardcoded metrics.
    """
    if not os.path.exists(DB_FILE):
        print(f"❌ Error: {DB_FILE} not found.")
        return
        
    conn = sqlite3.connect(DB_FILE)
    try:
        # Fetch all signals, ordering by most recent
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
    except Exception as e:
        print(f"❌ Database Query Error: {e}")
        return
    finally:
        conn.close()
    
    if df.empty:
        print("⚠️ Dashboard: No data found in signals table.")
        return

    # --- DYNAMIC SCHEMA NORMALIZATION ---
    # Detect the correct column name for 'status' (case-insensitive search)
    # This prevents the KeyError: 'status' if the DB uses 'Status', 'STATUS', etc.
    col_map = {col.lower(): col for col in df.columns}
    status_key = col_map.get('status')

    if not status_key:
        print(f"❌ Critical Error: 'status' column not found. Available: {df.columns.tolist()}")
        return

    # --- CALCULATE METRICS (APP TRUTH) ---
    total_signals = len(df)
    # Dynamically filter based on the detected status_key
    wins = len(df[df[status_key].astype(str).str.upper() == 'SUCCESS'])
    losses = len(df[df[status_key].astype(str).str.upper() == 'FAILED'])
    
    # Mathematical calculation of win rate
    closed_trades = wins + losses
    win_rate = round((wins / closed_trades * 100)) if closed_trades > 0 else 0

    # --- DATA FORMATTING FOR UI ---
    display_df = df.head(25).copy()
    
    outcomes = []
    for _, row in display_df.iterrows():
        # Use the dynamic status_key to fetch row value
        val = str(row.get(status_key, "")).upper()
        if val == 'SUCCESS':
            outcomes.append('<span style="color: #3fb950; font-weight: bold;">✅ HIT TP</span>')
        elif val == 'FAILED':
            outcomes.append('<span style="color: #f85149; font-weight: bold;">❌ HIT SL</span>')
        else:
            outcomes.append('<span style="color: #8b949e;">📡 SCANNING</span>')
    
    display_df['Outcome'] = outcomes
    
    # Ensure 'signal' column exists before applying lambda
    signal_key = col_map.get('signal', 'signal')
    if signal_key in display_df.columns:
        display_df['signal_fmt'] = display_df[signal_key].apply(
            lambda x: f'<b style="color:{"#3fb950" if str(x).upper()=="LONG" else "#f85149"}">{x}</b>'
        )
    else:
        display_df['signal_fmt'] = "N/A"

    # Define columns for the final HTML table
    # Mapping table headers to the actual detected database columns
    table_cols = {
        'ts': col_map.get('ts', 'ts'),
        'asset': col_map.get('asset', 'asset'),
        'signal': 'signal_fmt',
        'entry': col_map.get('entry', 'entry'),
        'tp': col_map.get('tp', 'tp'),
        'sl': col_map.get('sl', 'sl'),
        'Outcome': 'Outcome'
    }
    
    # Filter only available columns to prevent further KeyErrors
    final_cols = [v for v in table_cols.values() if v in display_df.columns]
    table_html = display_df[final_cols].to_html(escape=False, index=False, border=0)

    # --- UI HTML TEMPLATE ---
    HTML_CONTENT = f"""
    <html>
    <head>
        <title>Aegis Wealth Command</title>
        <meta http-equiv="refresh" content="300">
        <style>
            body {{ background: #0d1117; color: #e6edf3; font-family: sans-serif; padding: 20px; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
            .metric {{ font-size: 32px; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; padding: 10px; text-transform: uppercase; font-size: 11px; }}
            td {{ padding: 10px; border-bottom: 1px solid #21262d; font-size: 13px; }}
        </style>
    </head>
    <body>
        <h1>🛡️ Aegis Wealth Command</h1>
        <div style="display: flex; gap: 20px;">
            <div class="card" style="flex: 1;">Signals Processed<br><span class="metric">{total_signals}</span></div>
            <div class="card" style="flex: 1;">App Win Rate<br><span class="metric" style="color:#3fb950;">{win_rate}%</span></div>
            <div class="card" style="flex: 1;">Sync Time<br><span class="metric" style="font-size:18px;">{datetime.now().strftime('%H:%M UTC')}</span></div>
        </div>
        <div class="card">
            <h3>Performance Audit Trail (Verified from App)</h3>
            {table_html}
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)
    print("✅ Dashboard generated successfully in index.html")

if __name__ == "__main__":
    generate_dashboard()
