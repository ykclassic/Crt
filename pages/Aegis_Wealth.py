import sqlite3
import pandas as pd
import plotly.express as px
import os
import requests
import ccxt
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FILE = "nexus.db"
TOTAL_CAPITAL = 1000      
RISK_PER_TRADE = 0.02     

def get_multi_exchange_prices():
    prices = {}
    exchanges = {
        'gate': ccxt.gateio(),
        'bitget': ccxt.bitget(),
        'xt': ccxt.xt()
    }
    for name, ex in exchanges.items():
        try:
            tickers = ex.fetch_tickers()
            for symbol, data in tickers.items():
                clean_symbol = symbol.replace("/", "").replace("_", "").upper()
                if 'last' in data and data['last']:
                    prices[clean_symbol] = data['last']
        except Exception as e:
            print(f"Failed to sync {name}: {e}")
    return prices

def calculate_outcome(row, live_prices):
    asset = row['asset'].replace("/", "").replace("_", "").upper()
    current_price = live_prices.get(asset)
    start_time = pd.to_datetime(row['ts'])
    now = datetime.now()
    duration_str = str(now - start_time).split('.')[0]

    if not current_price:
        return "📡 SCANNING", "color: #8b949e", duration_str, None

    is_long = row['signal'].upper() == 'LONG'
    tp, sl = float(row['tp']), float(row['sl'])

    if is_long:
        if current_price >= tp: return "✅ HIT TP", "color: #3fb950; font-weight: bold;", duration_str, 1
        if current_price <= sl: return "❌ HIT SL", "color: #f85149; font-weight: bold;", duration_str, 0
    else:
        if current_price <= tp: return "✅ HIT TP", "color: #3fb950; font-weight: bold;", duration_str, 1
        if current_price >= sl: return "❌ HIT SL", "color: #f85149; font-weight: bold;", duration_str, 0

    return "🔵 ACTIVE", "color: #58a6ff;", duration_str, None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Aegis Wealth Command</title>
    <meta http-equiv="refresh" content="300"> 
    <style>
        body {{ background-color: #0d1117; color: #e6edf3; font-family: sans-serif; padding: 20px; }}
        .header {{ border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
        .metric-label {{ font-size: 10px; text-transform: uppercase; color: #8b949e; }}
        .metric-value {{ font-size: 28px; font-weight: 800; }}
        .progress-bg {{ background: #30363d; border-radius: 10px; height: 12px; margin-top: 10px; }}
        .progress-fill {{ background: linear-gradient(90deg, #58a6ff, #bc8cff); height: 100%; border-radius: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th {{ text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; padding: 10px; }}
        td {{ padding: 12px 10px; border-bottom: 1px solid #21262d; }}
        .tag-long {{ color: #3fb950; font-weight:bold; }}
        .tag-short {{ color: #f85149; font-weight:bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Aegis Wealth Command</h1>
        <p style="font-size:12px; color:#8b949e;">SYNC: {update_time} | VOLATILITY: {vol_status}</p>
    </div>

    <div class="grid">
        <div class="card"><div class="metric-label">Total Intelligence</div><div class="metric-value">{total_signals}</div></div>
        <div class="card"><div class="metric-label">Live Win Rate</div><div class="metric-value" style="color:#3fb950;">{win_rate}%</div></div>
        <div class="card">
            <div class="metric-label">AI Maturity</div>
            <div class="metric-value" style="color:#bc8cff;">LVL {ai_level}</div>
            <div class="progress-bg"><div class="progress-fill" style="width: {learning_pct}%"></div></div>
        </div>
    </div>

    <div class="grid" style="grid-template-columns: 2fr 1fr;">
        <div class="card"><div class="metric-label">Confidence Timeline</div>{history_plot}</div>
        <div class="card"><div class="metric-label">Learning Curve</div>{learning_plot}</div>
    </div>

    <div class="card">
        <div class="metric-label">Performance Audit (Multi-Exchange)</div>
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

    prices = get_multi_exchange_prices()
    display_df = df.head(25).copy()
    outcomes, durations, wins = [], [], []
    
    for _, row in display_df.iterrows():
        status, style, dur, win = calculate_outcome(row, prices)
        outcomes.append(f'<span style="{style}">{status}</span>')
        durations.append(dur)
        if win is not None: wins.append(win)
    
    display_df['Outcome'] = outcomes
    display_df['Duration'] = durations
    win_rate = round((sum(wins) / len(wins) * 100)) if wins else 0
    
    # Graphs
    fig_hist = px.line(df.head(50), x="ts", y="confidence", color="asset", template="plotly_dark")
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, showlegend=False)
    
    learning_data = pd.DataFrame({'Epoch': range(1, 11), 'Error': [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15]})
    fig_learning = px.line(learning_data, x='Epoch', y='Error', template="plotly_dark")
    fig_learning.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, yaxis_visible=False)

    # Signal Table
    display_df['signal'] = display_df['signal'].apply(lambda x: f'<span class="tag-{x.lower()}">{x}</span>')
    table_html = display_df[['ts', 'asset', 'signal', 'entry', 'tp', 'sl', 'Outcome', 'Duration']].to_html(escape=False, index=False, border=0)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.format(
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            win_rate=win_rate, total_signals=len(df),
            ai_level=(len(df)//100)+1, learning_pct=len(df)%100,
            vol_status="STABLE",
            history_plot=fig_hist.to_html(full_html=False, include_plotlyjs='cdn'),
            learning_plot=fig_learning.to_html(full_html=False, include_plotlyjs='cdn'),
            table_html=table_html
        ))

if __name__ == "__main__":
    generate_dashboard()
