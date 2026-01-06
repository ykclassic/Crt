import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def get_live_signal(asset="BTC/USDT"):
    # Insert your actual model logic here
    # Example mock return:
    return "LONG", 92.5, "12:00:00"

if __name__ == "__main__":
    # Your existing UI code goes inside this block
    # This prevents the UI from rendering when Nexus Forge imports the file
    pass 

# 1. Page Configuration
st.set_page_config(page_title="Aegis Wealth | Risk Shield", page_icon="🛡️", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# 3. Header
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title("🛡️ Aegis Wealth: Risk Shield")
    st.write("Function: **Autonomous Rebalancing & Capital Preservation**")
with col_h2:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

st.write("---")

# --- MOCK DATA FOR THE SHIELD ENGINE ---
# In production, this would pull from your live Exchange API
if 'portfolio_assets' not in st.session_state:
    st.session_state.portfolio_assets = {
        "BTC": {"qty": 0.5, "price": 95000, "target_weight": 0.40},
        "ETH": {"qty": 5.0, "price": 2800, "target_weight": 0.30},
        "SOL": {"qty": 100, "price": 240, "target_weight": 0.20},
        "USDT": {"qty": 5000, "price": 1.0, "target_weight": 0.10}
    }

def calculate_wealth_state():
    data = []
    total_val = 0
    for asset, info in st.session_state.portfolio_assets.items():
        val = info['qty'] * info['price']
        total_val += val
        data.append({"Asset": asset, "Value": val, "Target": info['target_weight']})
    
    df = pd.DataFrame(data)
    df['Current_Weight'] = df['Value'] / total_val
    df['Deviation'] = df['Current_Weight'] - df['Target']
    return df, total_val

df_wealth, total_nav = calculate_wealth_state()

# 4. Shield Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Net Asset Value (NAV)", f"${total_nav:,.2f}")
max_dev = df_wealth['Deviation'].abs().max()
m2.metric("Drift Severity", f"{max_dev*100:.1f}%", delta="Action Required" if max_dev > 0.05 else "Stable")
m3.metric("Shield Status", "ACTIVE", delta_color="normal")

st.write("---")

# 5. Rebalancing Logic
col_viz, col_actions = st.columns([2, 1])

with col_viz:
    st.subheader("📊 Target vs. Current Allocation")
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Current', x=df_wealth['Asset'], y=df_wealth['Current_Weight'], marker_color='#4F8BF9'))
    fig.add_trace(go.Bar(name='Target', x=df_wealth['Asset'], y=df_wealth['Target'], marker_color='#00ff00'))
    fig.update_layout(template="plotly_dark", barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_actions:
    st.subheader("⚡ Shield Actions")
    drift_assets = df_wealth[df_wealth['Deviation'].abs() > 0.05]
    
    if not drift_assets.empty:
        st.warning("⚠️ Significant Portfolio Drift Detected")
        for _, row in drift_assets.iterrows():
            action = "SELL" if row['Deviation'] > 0 else "BUY"
            amount = abs(row['Deviation'] * total_nav)
            st.write(f"**{action}** ${amount:,.2f} of **{row['Asset']}**")
        
        if st.button("Execute Auto-Rebalance", use_container_width=True):
            st.success("Rebalance Orders Sent to API...")
    else:
        st.success("✅ Portfolio weights are within safety parameters.")

# 6. Capital Preservation Mode
st.write("---")
st.subheader("🚨 Emergency Capital Preservation")
risk_level = st.slider("Market Panic Index (via Aegis Risk)", 0, 100, 30)

if risk_level > 70:
    st.error("!!! CRITICAL RISK DETECTED !!!")
    st.markdown("""
    **Shield Recommendation:**
    - Liquidate 50% of volatile assets (BTC, ETH, SOL)
    - Convert to USDT / USDC
    - Pause all Nexus Signal auto-entries.
    """)
    if st.button("ACTIVATE CAPITAL SHIELD"):
        st.warning("Moving Assets to Stablecoins...")
else:
    st.info("Market conditions stable. No emergency liquidation required.")
