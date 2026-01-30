import sqlite3
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# --- CONFIGURATION ---
# We hardcode these here to avoid importing config.py and causing path issues
DB_FILE = "nexus.db" 

# HTML TEMPLATE (Professional Dark Mode)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aegis Wealth Command</title>
    <meta http-equiv="refresh" content="300"> <style>
        body {{ background-color: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 24px; color: #58a6ff; }}
        .timestamp {{ color: #8b949e; font-size: 14px; }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; }}
        .metric-label {{ font-size: 14px; color: #8b949e; margin-bottom: 5px; }}
        .metric-value {{ font-size: 28px; font-weight: 600; color: #ffffff; }}
        .metric-sub {{ font-size: 12px; color: #3fb950; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
        th {{ text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; padding: 10px; }}
        td {{ padding: 10px; border-bottom: 1px solid #21262d; }}
        tr:hover {{ background-color: #21262d; }}
        
        .tag {{ padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .tag-long {{ background-color: rgba(63, 185, 80, 0.15); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.4); }}
        .tag-short {{ background-color: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }}
        
        .download-btn {{ display: inline-block; background-color: #238636; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: bold; margin-top: 20px; }}
        .download-btn:hover {{ background-color: #2ea043; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🛡️ Aegis Wealth Command</h1>
            <div style="font-size: 12px; color: #8b949e; margin-top: 5px;">SERVERLESS INTELLIGENCE NODE</div>
        </div>
        <div class="timestamp">Last Update: {update_time}</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="metric-label">TOTAL INTELLIGENCE SIGNALS</div>
            <div class="metric-value">{total_signals}</div>
            <div class="metric-sub">Lifetime Database Count</div>
        </div>
        <div class="card">
            <div class="metric-label">ELITE OPPORTUNITIES</div>
            <div class="metric-value">{elite_signals}</div>
            <div class="metric-sub">High Confidence (>80%)</div>
        </div>
        <div class="card">
            <div class="metric-label">LATEST TARGET</div>
            <div class="metric-value">{latest_asset}</div>
            <div class="metric-sub">{latest_conf}% Confidence</div>
        </div>
    </div>

    <div class="card">
        <div class="metric-label">CONFIDENCE TIMELINE</div>
        {plot_div}
    </div>

    <div class="card">
        <div class="header" style="border:none; padding:0; margin:0;">
            <div class="metric-label">RECENT MARKET LOGS</div>
        </div>
        {table_html}
    </div>
    
    <center>
        <a href="nexus_signals.csv" class="download-btn">📥 Download Full CSV Database</a>
    </center>
</body>
</html>
"""

def generate_dashboard():
    if not os.path.exists(DB_FILE):
        print("❌ DB not found.")
        return

    # 1. READ DATA
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
    conn.close()

    if df.empty:
        print("⚠️ DB Empty.")
        return

    # 2. CALCULATE METRICS
    total = len(df)
    elite = len(df[df['confidence'] >= 80])
    latest_asset = df['asset'].iloc[0]
    latest_conf = df['confidence'].iloc[0]

    # 3. GENERATE PLOT
    fig = px.scatter(df, x="ts", y="confidence", color="signal", 
                     color_discrete_map={"LONG": "#3fb950", "SHORT": "#f85149"},
                     template="plotly_dark")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'))
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # 4. GENERATE TABLE
    # Format the table for display
    display_df = df.head(20).copy()
    display_df['signal'] = display_df['signal'].apply(
        lambda x: f'<span class="tag tag-{x.lower()}">{x}</span>'
    )
    table_html = display_df[['ts', 'asset', 'signal', 'confidence', 'entry']].to_html(escape=False, index=False, border=0)
    table_html = table_html.replace('class="dataframe"', '')

    # 5. RENDER HTML
    html = HTML_TEMPLATE.format(
        update_time=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        total_signals=total,
        elite_signals=elite,
        latest_asset=latest_asset,
        latest_conf=latest_conf,
        plot_div=plot_html,
        table_html=table_html
    )

    # 6. SAVE OUTPUTS
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    # Save CSV for the download button
    df.to_csv("nexus_signals.csv", index=False)
    
    print("✅ Dashboard & CSV Generated.")

if __name__ == "__main__":
    generate_dashboard()
