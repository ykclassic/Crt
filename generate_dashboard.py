import sqlite3
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# --- CONFIGURATION ---
DB_FILE = "nexus.db"
TOTAL_CAPITAL = 1000      
RISK_PER_TRADE = 0.02     

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aegis Wealth Command</title>
    <meta http-equiv="refresh" content="300"> 
    <style>
        body {{ background-color: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 24px; color: #58a6ff; }}
        .timestamp {{ color: #8b949e; font-size: 14px; }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        
        .metric-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; margin-bottom: 8px; }}
        .metric-value {{ font-size: 28px; font-weight: 700; color: #ffffff; }}
        .metric-sub {{ font-size: 13px; color: #3fb950; margin-top: 5px; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 0; font-size: 14px; }}
        th {{ text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; padding: 12px; }}
        td {{ padding: 12px; border-bottom: 1px solid #21262d; }}
        tr:hover {{ background-color: #21262d; }}
        
        .tag {{ padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold; }}
        .tag-long {{ background-color: rgba(63, 185, 80, 0.15); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.4); }}
        .tag-short {{ background-color: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }}
        
        .download-btn {{ display: inline-block; background-color: #238636; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600; margin-top: 30px; transition: 0.2s; }}
        .download-btn:hover {{ background-color: #2ea043; }}
        
        .progress-bg {{ background: #30363d; border-radius: 10px; height: 8px; width: 100px; display: inline-block; }}
        .progress-fill {{ background: #58a6ff; height: 8px; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🛡️ Aegis Wealth Command</h1>
            <div style="font-size: 13px; color: #8b949e; margin-top: 5px;">SATELLITE INTELLIGENCE NODE</div>
        </div>
        <div class="timestamp">LAST SYNC: {update_time}</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="metric-label">Total Signals Captured</div>
            <div class="metric-value">{total_signals}</div>
            <div class="metric-sub">Scanning active markets</div>
        </div>
        <div class="card">
            <div class="metric-label">Elite Opportunities</div>
            <div class="metric-value" style="color: {elite_color};">{elite_signals}</div>
            <div class="metric-sub">Confidence &ge; 80%</div>
        </div>
        <div class="card">
            <div class="metric-label">Latest Target</div>
            <div class="metric-value">{latest_asset}</div>
            <div class="metric-sub">Latest Intelligence Hit</div>
        </div>
    </div>

    <div class="card">
        <div class="metric-label">CONFIDENCE TIMELINE</div>
        {plot_div}
    </div>

    <div class="card" style="margin-top: 30px;">
        <div class="metric-label" style="margin-bottom: 15px;">RECENT MARKET LOGS</div>
        {table_html}
    </div>
    
    <center>
        <a href="nexus_signals.csv" class="download-btn">📥 DOWNLOAD FULL CSV REPORT</a>
    </center>
</body>
</html>
"""

def generate_dashboard():
    if not os.path.exists(DB_FILE):
        return

    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
    conn.close()

    if df.empty:
        return

    total = len(df)
    elite = len(df[(df['confidence'] >= 80) | (df['confidence'] <= 20)])
    elite_color = "#ffffff" if elite == 0 else "#f1e05a"
    latest_asset = df['asset'].iloc[0] if total > 0 else "N/A"

    # Fix Chart Colors
    fig = px.scatter(df, x="ts", y="confidence", color="signal", 
                     color_discrete_map={"LONG": "#3fb950", "SHORT": "#f85149"},
                     template="plotly_dark")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=350)
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Table with visual confidence gauge
    display_df = df.head(15).copy()
    display_df['signal'] = display_df['signal'].apply(lambda x: f'<span class="tag tag-{x.lower()}">{x}</span>')
    
    def make_gauge(val):
        return f'<div class="progress-bg"><div class="progress-fill" style="width: {val}%;"></div></div> {val}%'
    
    display_df['confidence'] = display_df['confidence'].apply(make_gauge)
    
    table_html = display_df[['ts', 'asset', 'signal', 'confidence', 'entry']].to_html(escape=False, index=False, border=0)
    table_html = table_html.replace('class="dataframe"', '')

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.format(
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            total_signals=total, elite_signals=elite, elite_color=elite_color,
            latest_asset=latest_asset, plot_div=plot_html, table_html=table_html
        ))
    df.to_csv("nexus_signals.csv", index=False)

if __name__ == "__main__":
    generate_dashboard()
