import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import importlib.util
import os
import time
import requests
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────
MODULES_TO_POLL = [
    "Nexus_Neural",    # Weight: 0.50 (ML regime/predictions)
    "Nexus_Signal",    # Weight: 0.30 (Multi-TF confluence)
    "Neural_Profit",   # Weight: 0.20 (Profit estimation)
    "Aegis_Risk",      # Weight: 0.15 (Risk adjustment - new)
    # Add more here if they implement get_live_signal, e.g., "Aegis_Wealth"
]
WEIGHTS = {"Nexus_Neural": 0.50, "Nexus_Signal": 0.30, "Neural_Profit": 0.20, "Aegis_Risk": 0.15}

PAGES_DIR = os.path.dirname(__file__)

st.set_page_config(page_title="Nexus Forge", layout="wide")
st.title("🔗 Nexus Forge: Executive Decision Node")
st.markdown("Aggregates live signals from all tools for **profitable, risk-adjusted** trading decisions.")

# Authentication check (keep your existing if any)
if not st.session_state.get("authenticated"):
    st.warning("Redirecting to Home...")
    st.switch_page("Home.py")

# ─── HELPERS ──────────────────────────────────────────────────────────────
def fetch_market_context():
    try:
        fng = requests.get("https://api.alternative.me/fng/").json()["data"][0]
        return {"fear_greed": int(fng["value"]), "sentiment": fng["value_classification"]}
    except:
        return {"fear_greed": 50, "sentiment": "Neutral"}

def poll_module(module_name, asset):
    module_path = os.path.join(PAGES_DIR, f"{module_name}.py")
    if not os.path.exists(module_path):
        return {"status": "OFFLINE", "reason": "Module file not found"}
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, "get_live_signal"):
            result = module.get_live_signal(asset)  # Expect dict now
            if isinstance(result, dict):
                result["status"] = "ONLINE"
                result["module"] = module_name
                return result
            else:
                return {"status": "ERROR", "reason": "Invalid return format"}
        else:
            return {"status": "UNKNOWN", "reason": "No get_live_signal function"}
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}

def compute_executive_decision(results, market_context):
    valid_results = [r for r in results if r["status"] == "ONLINE" and "direction" in r]
    if not valid_results:
        return {"decision": "STANDBY / INSUFFICIENT DATA", "confidence": 0, "color": "grey", "reasons": ["No valid signals"]}

    # Weighted direction votes
    directions = {"LONG": 0.0, "SHORT": 0.0, "NEUTRAL": 0.0}
    total_weight = 0
    compiled_reasons = []
    expected_profits = []
    rrs = []
    risk_score = 0.5  # Default neutral

    for res in valid_results:
        mod = res["module"]
        weight = WEIGHTS.get(mod, 0.1)
        conf = res.get("confidence", 0.5)
        dir = res.get("direction", "NEUTRAL")
        
        directions[dir] += weight * conf
        total_weight += weight
        
        if "reason" in res:
            compiled_reasons.append(f"{mod}: {res['reason']}")
        if "expected_profit_pct" in res:
            expected_profits.append(res["expected_profit_pct"])
        if "rr_ratio" in res:
            rrs.append(res["rr_ratio"])
        if mod == "Aegis_Risk" and "risk_score" in res:
            risk_score = res["risk_score"]  # Assume 0-1, higher = riskier

    # Consensus direction
    master_dir = max(directions, key=directions.get)
    weighted_conf = directions[master_dir] / total_weight if total_weight > 0 else 0
    
    # Risk-adjusted confidence
    adjusted_conf = weighted_conf * (1 - risk_score)
    
    # Profitability metrics
    avg_profit = sum(expected_profits) / len(expected_profits) if expected_profits else 0
    avg_rr = sum(rrs) / len(rrs) if rrs else 1.0
    
    # Decision logic (tuned for profitability)
    if adjusted_conf > 0.85 and master_dir != "NEUTRAL" and avg_rr > 1.5 and avg_profit > 1.0:
        decision = f"STRONG {master_dir} EXECUTION"
        color = "#00ff00"  # Neon green
    elif adjusted_conf > 0.65 and master_dir != "NEUTRAL" and avg_rr > 1.2:
        decision = f"ACCUMULATE {master_dir} (CAUTION)"
        color = "#ffa500"  # Orange
    else:
        decision = "STANDBY / NEUTRAL"
        color = "grey"
    
    summary = {
        "decision": decision,
        "confidence": adjusted_conf,
        "color": color,
        "direction": master_dir,
        "expected_profit_pct": round(avg_profit, 2),
        "risk_reward": round(avg_rr, 2),
        "reasons": compiled_reasons or ["Consensus based on weighted signals"],
        "market_sentiment": market_context["sentiment"]
    }
    return summary

# ─── UI ───────────────────────────────────────────────────────────────────
asset_options = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]  # Expand as needed
target_asset = st.selectbox("Select Target Asset", asset_options)

if st.button("🚀 Run Nexus Forge Analysis", type="primary"):
    with st.spinner(f"Polling modules for {target_asset}..."):
        results = []
        for mod in MODULES_TO_POLL:
            res = poll_module(mod, target_asset)
            results.append(res)
        
        market_context = fetch_market_context()
        
        decision = compute_executive_decision(results, market_context)
    
    # Master Decision Display
    st.markdown(
        f"<h1 style='color:{decision['color']}; text-align:center;'>{decision['decision']}</h1>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Adjusted Confidence", f"{decision['confidence']:.2%}")
    col2.metric("Expected Profit", f"{decision['expected_profit_pct']}%")
    col3.metric("Risk-Reward Ratio", decision['risk_reward'])
    
    st.info(f"Market Sentiment: {decision['market_sentiment']} (F&G Index context)")
    
    # Insights & Reasons
    st.subheader("📊 Decision Insights")
    st.write("\n".join(f"- {r}" for r in decision['reasons']))
    
    # Module Breakdown
    st.subheader("🔍 Individual Module Results")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    
    # Chart (enhanced)
    fig = go.Figure()
    for r in results:
        if r["status"] == "ONLINE" and "confidence" in r:
            color = "#00ff00" if r["direction"] == "LONG" else "#ff0000" if r["direction"] == "SHORT" else "grey"
            fig.add_trace(go.Bar(x=[r["module"]], y=[r["confidence"]], marker_color=color, name=r["module"]))
    fig.update_layout(title="Module Confidence Contributions", yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Focus on high RR & confidence for profits.")
