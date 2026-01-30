import sqlite3
import pandas as pd
import plotly.express as px
import os
import requests
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FILE = "nexus.db"
TOTAL_CAPITAL = 1000      
RISK_PER_TRADE = 0.02     

def get_live_prices():
    """Fetches real-time prices to audit TP/SL outcomes."""
    try:
        # Fetching top 100 symbols to cover most traded assets
        response = requests.get("https://api.binance.com/api/3/ticker/price", timeout=5)
        data = response.json()
        return {item['symbol']: float(item['price']) for item in data}
    except Exception as e:
        print(f"Price Fetch Error: {e}")
        return {}

def calculate_outcome(row, live_prices):
    """Audits signal performance and calculates duration."""
    asset = row['asset'].replace("/", "")
    current_price = live_prices.get(asset)
    
    # Duration Logic
    start_time = pd.to_datetime(row['ts'])
    now = datetime.now()
    duration_delta = now - start_time
    duration_str = str(duration_delta).split('.')[0]

    if not current_price:
        return "📡 SCANNING", "color: #8b949e", duration_str, None

    is_long = row['signal'].upper() == 'LONG'
    tp = float(row['tp'])
    sl = float(row['sl'])

    if is_long:
        if current_price >= tp:
            return "✅ HIT TP", "color: #3fb950; font-weight: bold;", duration_str, 1
        if current_price <= sl:
            return "❌ HIT SL", "color: #f85149; font-weight: bold;", duration_str, 0
    else: # SHORT
        if current_price <= tp:
            return "✅ HIT TP", "color: #3fb950; font-weight: bold;", duration_str, 1
        if current_price >= sl:
            return "❌ HIT SL", "color: #f85149; font-weight: bold;", duration_str, 0

    return "🔵 ACTIVE", "color: #58a6ff;", duration_str, None

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
        
        .status-container {{ display: flex; gap: 15px; font-size: 12px; color: #8b949e; }}
        .indicator {{ display: flex; align-items: center; gap: 5px; }}
        .dot {{ height: 10px; width: 10px; border-radius: 50%; display: inline-block; }}
        .pulse {{ animation: pulse-animation 2s infinite; }}
        @keyframes pulse-animation {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} 100% {{ opacity: 1; }} }}

        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        
        .sentiment-container {{ margin-top: 10px; height: 10px; width: 100%; background: #30363d; border-radius: 5px; display: flex; overflow: hidden; }}
        .long-bar {{ background: #3fb950; height: 100%; }}
        .short-bar {{ background: #f85149; height: 100%; }}

        .metric-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px; color: #8b949e; margin-bottom: 5px; }}
        .metric-value {{ font-size: 28px; font-weight: 800; color: #ffffff; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
        th {{ text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; padding: 10px; }}
        td {{ padding: 12px 10px; border-bottom: 1px solid #21262d; }}
        
        .tag {{ padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; }}
        .tag-long {{ background: rgba(63, 185, 80, 0.1); color: #3fb950; border: 1px solid #3fb950; }}
        .tag-short {{ background: rgba(248, 81, 73, 0.1); color: #f85149; border: 1px solid #f85149; }}
        
        .conf-bg {{ background: #30363d; border-radius: 3px; height: 5px; width: 50px; display: inline-block; vertical-align: middle; }}
        .conf-fill {{ height: 100%; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🛡️ Aegis Wealth Command</h1>
            <div class="status-container">
                <div class="indicator"><span class="dot pulse" style="background-color: #3fb950;"></span> ENGINE LIVE</div>
                <div class="indicator"><span class="dot" style="background-color: {vol_color};"></span> VOLATILITY: {vol_status}</div>
            </div>
        </div>
        <div class="timestamp">LAST SYNC: {update_time}</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="metric-label">Signal Intensity</div>
            <div class="metric-value">{total_signals}</div>
        </div>
        <div class="card">
            <div class="metric-label">Win Rate (Recent)</div>
            <div class="metric-value" style="color: #3fb950;">{win_rate}%</div>
            <div class="metric-label" style="margin-top:5px;">Based on resolved trades</div>
        </div>
        <div class="card">
            <div class="metric-label">Sentiment Bias</div>
            <div class="sentiment-container"><div class="long-bar" style="width: {long_pct}%"></div><div class="short-bar" style="width: {short_pct}%"></div></div>
            <div style="font-size:11px; margin-top:5px;">{long_pct}% Bullish / {short_pct}% Bearish</div>
        </div>
        <div class="card">
            <div class="metric-label">Elite Alpha / Risk</div>
            <div class="metric-value" style="color: #f1e05a;">{elite_signals} <span style="font-size:14px; color:#8b949e;">/ ${risk_val}</span></div>
        </div>
    </div>

    <div class="grid" style="grid-template-columns: 1.8fr 1.2fr;">
        <div class="card">
            <div class="metric-label">Multi-Asset Confidence Matrix</div>
            {history_plot}
        </div>
        <div class="card">
            <div class="metric-label">Asset Concentration</div>
            {concentration_plot}
        </div>
    </div>

    <div class="card">
        <div class="metric-label">Live Performance Audit Trail</div>
        {table_html}
    </div>
</body>
</html>
"""

def generate_dashboard():
    if not os.path.exists(DB_FILE): return
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
    conn.close()
    if df.empty: return

    live_prices = get_live_prices()
    
    # Process Table and Outcomes
    display_df = df.head(25).copy()
    outcomes, styles, durations, wins = [], [], [], []
    
    for _, row in display_df.iterrows():
        status, style, dur, win = calculate_outcome(row, live_prices)
        outcomes.append(f'<span style="{style}">{status}</span>')
        durations.append(dur)
        if win is not None: wins.append(win)
    
    display_df['Outcome'] = outcomes
    display_df['Duration'] = durations
    
    # Win Rate Calculation
    win_rate = round((sum(wins) / len(wins) * 100)) if wins else 0
    
    # UI Formatting
    display_df['signal'] = display_df['signal'].apply(lambda x: f'<span class="tag tag-{x.lower()}">{x}</span>')
    def render_conf(val):
        color = "#58a6ff"
        if val >= 75: color = "#3fb950"
        if val >= 85: color = "#f1e05a"
        return f'<div class="conf-bg"><div class="conf-fill" style="width: {val}%; background: {color};"></div></div> {val}%'
    display_df['Confidence'] = display_df['confidence'].apply(render_conf)

    # General Metrics
    total = len(df)
    elite = len(df[df['confidence'] >= 80])
    long_pct = round((len(df[df['signal'] == 'LONG']) / total * 100)) if total > 0 else 50
    short_pct = 100 - long_pct
    risk_val = max(10, min(100, round((1000 * 0.02) / 10) * 10))
    
    # Volatility Check
    recent_count = len(df[pd.to_datetime(df['ts']) > (datetime.now() - timedelta(hours=4))])
    vol_status, vol_color = ("STABLE", "#3fb950") if recent_count < 30 else ("HIGH", "#f1e05a")
    if recent_count > 60: vol_status, vol_color = ("EXTREME", "#f85149")

    # Plots
    fig_hist = px.line(df.head(50), x="ts", y="confidence", color="asset", template="plotly_dark", markers=True)
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'), height=350, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
    
    asset_counts = df['asset'].value_counts().head(5).reset_index()
    fig_conc = px.pie(asset_counts, values='count', names='asset', hole=.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.Ice)
    fig_conc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, height=350, margin=dict(l=20,r=20,t=20,b=20))

    # HTML Table Generation
    cols = ['ts', 'asset', 'signal', 'Confidence', 'entry', 'tp', 'sl', 'Outcome', 'Duration']
    table_html = display_df[cols].to_html(escape=False, index=False, border=0).replace('class="dataframe"', '')

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.format(
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            vol_status=vol_status, vol_color=vol_color, win_rate=win_rate,
            total_signals=total, long_pct=long_pct, short_pct=short_pct,
            elite_signals=elite, risk_val=risk_val,
            history_plot=fig_hist.to_html(full_html=False, include_plotlyjs='cdn'),
            concentration_plot=fig_conc.to_html(full_html=False, include_plotlyjs='cdn'),
            table_html=table_html
        ))
    df.to_csv("nexus_signals.csv", index=False)

if __name__ == "__main__":
    generate_dashboard()
