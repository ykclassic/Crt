import sqlite3
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

DB_FILE = "nexus.db"

def generate_dashboard():
    if not os.path.exists(DB_FILE): return
    conn = sqlite3.connect(DB_FILE)
    # Fetch all signals, ordering by most recent
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
    conn.close()
    
    if df.empty: return

    # Calculate Metrics based on APP TRUTH
    total_signals = len(df)
    wins = len(df[df['status'] == 'SUCCESS'])
    losses = len(df[df['status'] == 'FAILED'])
    win_rate = round((wins / (wins + losses) * 100)) if (wins + losses) > 0 else 0

    # Data Formatting for UI
    display_df = df.head(25).copy()
    
    outcomes = []
    for _, row in display_df.iterrows():
        status = row.get('status')
        if status == 'SUCCESS':
            outcomes.append('<span style="color: #3fb950; font-weight: bold;">✅ HIT TP</span>')
        elif status == 'FAILED':
            outcomes.append('<span style="color: #f85149; font-weight: bold;">❌ HIT SL</span>')
        else:
            outcomes.append('<span style="color: #8b949e;">📡 SCANNING</span>')
    
    display_df['Outcome'] = outcomes
    display_df['signal'] = display_df['signal'].apply(lambda x: f'<b style="color:{"#3fb950" if x=="LONG" else "#f85149"}">{x}</b>')
    
    table_html = display_df[['ts', 'asset', 'signal', 'entry', 'tp', 'sl', 'Outcome']].to_html(escape=False, index=False, border=0)

    # UI HTML Template
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
            th {{ text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; padding: 10px; }}
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

if __name__ == "__main__":
    generate_dashboard()
