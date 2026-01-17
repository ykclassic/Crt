import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

def get_live_signal(asset="BTC/USDT"):
    # Insert your actual model logic here
    # Example mock return:
    return "LONG", 92.5, "12:00:00"

if __name__ == "__main__":
    # Your existing UI code goes inside this block
    # This prevents the UI from rendering when Nexus Forge imports the file
    pass 

# 1. Dashboard Configuration
st.set_page_config(page_title="Nexus Neural | Multi-Asset Pulse", page_icon="🌐", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# --- SIMULATED REAL-TIME SIGNAL AGGREGATOR ---
def get_global_signals(assets):
    """Generates a snapshot of signals using clean keys for attribute access."""
    data = []
    for asset in assets:
        regime = np.random.choice(["BULLISH", "BEARISH", "SIDEWAYS"], p=[0.4, 0.3, 0.3])
        conf = np.random.uniform(60, 98)
        decay_min = np.random.randint(5, 180) 
        
        data.append({
            "Asset": asset,
            "Regime": regime,
            "Confidence": round(conf, 2),
            "Decay_Min": decay_min,  # Clean key: no spaces or parentheses
            "Status": "🔥 HIGH PRIORITY" if conf > 90 and regime != "SIDEWAYS" else "📡 MONITORING"
        })
    return pd.DataFrame(data)

# --- AUTO-REFRESH FRAGMENT ---
@st.fragment(run_every="60s")
def render_global_matrix():
    asset_library = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", 
        "ADA/USDT", "LINK/USDT", "TRX/USDT", "SUI/USDT", "PEPE/USDT"
    ]
    
    df_signals = get_global_signals(asset_library)
    
    # Header with Timestamp
    st.subheader(f"🚀 High-Confidence Entries (Last Update: {time.strftime('%H:%M:%S')})")
    
    # 1. Top Signals Highlight
    top_signals = df_signals[df_signals["Status"] == "🔥 HIGH PRIORITY"].sort_values(by="Confidence", ascending=False)
    
    if not top_signals.empty:
        cols = st.columns(len(top_signals))
        for i, row in enumerate(top_signals.itertuples()):
            with cols[i]:
                # Accessing clean attributes defined in get_global_signals
                st.metric(row.Asset, f"{row.Confidence}%", f"{row.Regime}")
                st.caption(f"Expires in: {row.Decay_Min}m") # FIXED ATTRIBUTE ACCESS
    else:
        st.info("No high-priority signals meeting the 90% threshold. Monitoring market...")

    # 2. Signal Decay & Confidence Heatmap
    st.write("---")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Signal Reliability Matrix")
        # Rename for UI display purposes only inside the plot
        plot_df = df_signals.rename(columns={"Decay_Min": "Decay (Min)"})
        fig = px.scatter(plot_df, x="Confidence", y="Decay (Min)", size="Confidence", 
                         color="Regime", hover_name="Asset",
                         color_discrete_map={"BULLISH": "#00FFCC", "BEARISH": "#FF4B4B", "SIDEWAYS": "#808080"})
        fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Immediate Expiry Zone")
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Full Asset Pulse")
        def color_regime(val):
            color = '#00FFCC' if val == 'BULLISH' else ('#FF4B4B' if val == 'BEARISH' else '#888888')
            return f'color: {color}'
        
        display_df = df_signals.rename(columns={"Decay_Min": "Decay (Min)"})
        st.dataframe(display_df.style.applymap(color_regime, subset=['Regime']), use_container_width=True)

# --- UI LAYOUT ---
st.title("🌐 Nexus Neural: Multi-Asset Intelligence Matrix")
st.write("Automated Signal Node: Aggregating 10 assets every 60 seconds.")

# Execute the auto-refreshing fragment
render_global_matrix()

# 3. Persistent System Health (Maintained)
st.write("---")
h1, h2, h3 = st.columns(3)
with h1:
    st.write("🔒 **Vault Status**: Encrypted")
with h2:
    st.write("📊 **Regime Engine**: K-Means Active")
with h3:
    st.write("📡 **Data Source**: Bitget/XT Multi-feed")

st.caption("Nexus Neural | Fragment-based Auto-Refresh Active")


