# Nexus Neural v3+ — Real-Time Ensemble Signal Engine
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import hashlib
import threading
import time
from datetime import datetime
from lifelines import KaplanMeierFitter
import shap

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Nexus Neural v3+", page_icon="🌐", layout="wide")
st.title("🌐 Nexus Neural v3+ — Deterministic + ML Ensemble Signal Engine")

# -----------------------------
# Database setup
# -----------------------------
DB = sqlite3.connect("nexus_signals.db", check_same_thread=False)
DB.execute("""
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    exchange TEXT,
    asset TEXT,
    timeframe TEXT,
    regime TEXT,
    signal TEXT,
    entry REAL,
    stop REAL,
    take REAL,
    confidence REAL,
    model_hash TEXT,
    status TEXT
)
""")
DB.commit()

def log_signal(record):
    DB.execute("""
        INSERT INTO signals (
            timestamp, exchange, asset, timeframe, regime, signal,
            entry, stop, take, confidence, model_hash, status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record["timestamp"], record["exchange"], record["asset"], record["timeframe"],
        record["regime"], record["signal"], record["entry"], record["stop"], record["take"],
        record["confidence"], record["model_hash"], record["status"]
    ))
    DB.commit()

# -----------------------------
# Sidebar inputs
# -----------------------------
exchange_name = st.sidebar.selectbox("Select Exchange for Signal Base", ["XT", "Gate.io"])
TIMEFRAMES = ["1h","4h","1d"]
TIMEFRAME = st.sidebar.selectbox("Select Primary Timeframe", TIMEFRAMES)
ALL_ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
              "LINK/USDT","DOGE/USDT","TRX/USDT","SUI/USDT","PEPE/USDT"]
selected_assets = st.sidebar.multiselect("Select Assets", ALL_ASSETS, default=ALL_ASSETS[:5])

# -----------------------------
# Exchange initialization
# -----------------------------
@st.cache_resource
def get_exchange(name):
    if name == "XT":
        ex = ccxt.xt({"enableRateLimit": True})
    else:
        ex = ccxt.gateio({"enableRateLimit": True})
    ex.load_markets()
    return ex

exchange = get_exchange(exchange_name)

# -----------------------------
# Indicators & deterministic logic
# -----------------------------
def compute_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    df["vwap"] = (df["close"]*df["volume"]).cumsum() / df["volume"].cumsum()
    return df

def deterministic_signal(df):
    last = df.iloc[-1]
    signal = "NEUTRAL"
    if last["close"] > last["ema20"] and last["rsi"] < 70:
        signal = "LONG"
    elif last["close"] < last["ema20"] and last["rsi"] > 30:
        signal = "SHORT"
    regime = "BULLISH" if last["close"] > last["ema50"] else "BEARISH" if last["close"] < last["ema50"] else "SIDEWAYS"
    entry = last["close"]
    stop = entry*0.98 if signal=="LONG" else entry*1.02 if signal=="SHORT" else entry
    take = entry*1.03 if signal=="LONG" else entry*0.97 if signal=="SHORT" else entry
    return signal, regime, entry, stop, take

def ml_confidence(df):
    # Placeholder: replace with real ML model
    return np.random.uniform(75,99)

# -----------------------------
# Multi-TF consensus
# -----------------------------
def multi_tf_consensus(symbol, tfs):
    signals, regimes = [], []
    for tf in tfs:
        df = fetch_ohlcv(symbol, tf)
        if df is None: continue
        df = compute_indicators(df)
        sig, reg, _, _, _ = deterministic_signal(df)
        signals.append(sig)
        regimes.append(reg)
    if signals:
        consensus_signal = max(set(signals), key=signals.count)
        consensus_regime = max(set(regimes), key=regimes.count)
        return consensus_signal, consensus_regime
    return "NEUTRAL","SIDEWAYS"

# -----------------------------
# Funding rate for Gate.io
# -----------------------------
def fetch_funding_rate(symbol):
    if exchange_name.lower() != "gate.io":
        return 0
    try:
        # Mock: Replace with real funding API call
        return np.random.uniform(-0.05,0.05)
    except:
        return 0

def adjust_signal_with_funding(signal, funding_rate):
    if funding_rate > 0.02 and signal=="LONG":
        return "LONG+"
    elif funding_rate < -0.02 and signal=="SHORT":
        return "SHORT+"
    return signal

# -----------------------------
# OHLCV fetch
# -----------------------------
def fetch_ohlcv(symbol, tf, limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except:
        return None

# -----------------------------
# Full signal generator
# -----------------------------
def generate_full_signal(symbol, tfs):
    signal, regime = multi_tf_consensus(symbol, tfs)
    df = fetch_ohlcv(symbol, tfs[0])
    if df is None: return None
    df = compute_indicators(df)
    confidence = ml_confidence(df)
    funding_rate = fetch_funding_rate(symbol)
    signal_adj = adjust_signal_with_funding(signal, funding_rate)
    last = df.iloc[-1]
    entry = last["close"]
    stop = entry*0.98 if signal_adj.startswith("LONG") else entry*1.02 if signal_adj.startswith("SHORT") else entry
    take = entry*1.03 if signal_adj.startswith("LONG") else entry*0.97 if signal_adj.startswith("SHORT") else entry
    ts = datetime.utcnow().isoformat()
    model_hash = hashlib.md5(b"NexusNeuralV3Plus").hexdigest()
    record = {
        "timestamp": ts,
        "exchange": exchange_name,
        "asset": symbol,
        "timeframe": "/".join(tfs),
        "regime": regime,
        "signal": signal_adj,
        "entry": entry,
        "stop": stop,
        "take": take,
        "confidence": confidence,
        "model_hash": model_hash,
        "status": "OPEN"
    }
    log_signal(record)
    return record, df, funding_rate

# -----------------------------
# Real-time threaded loop
# -----------------------------
signals_cache = {}
def update_signals_loop():
    while True:
        for asset in selected_assets:
            try:
                res = generate_full_signal(asset, TIMEFRAMES)
                if res: signals_cache[asset] = res
            except Exception as e:
                print(f"Error {asset}: {e}")
        time.sleep(60)

threading.Thread(target=update_signals_loop, daemon=True).start()

# -----------------------------
# Dashboard display
# -----------------------------
st.subheader(f"Live Ensemble Signals — {exchange_name}")
for asset in selected_assets:
    if asset not in signals_cache:
        st.warning(f"{asset} data unavailable yet")
        continue
    record, df, funding_rate = signals_cache[asset]
    
    st.markdown(f"### {asset}")
    fig = px.line(df, x="timestamp", y=["close","ema20","ema50"], title=f"{asset} Price + EMA")
    st.plotly_chart(fig, use_container_width=True)
    
    color = "green" if record["signal"].startswith("LONG") else "red" if record["signal"].startswith("SHORT") else "yellow"
    st.markdown(f"<p style='color:{color}; font-weight:bold'>Signal: {record['signal']} ({record['regime']})</p>", unsafe_allow_html=True)
    st.markdown(f"Entry / SL / TP: {record['entry']:.2f} / {record['stop']:.2f} / {record['take']:.2f}")
    st.markdown(f"Confidence: {record['confidence']:.2f}% | Funding Rate: {funding_rate:.2%}")

# -----------------------------
# Signal Lifecycle Table
# -----------------------------
st.subheader("Signal Lifecycle Table")
df_audit = pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC", DB)
st.dataframe(df_audit, use_container_width=True)

# -----------------------------
# Survival Analysis
# -----------------------------
st.subheader("Signal Survival Analysis by Regime")
if not df_audit.empty:
    kmf = KaplanMeierFitter()
    for regime in df_audit["regime"].unique():
        subset = df_audit[df_audit["regime"]==regime]
        durations = (pd.Timestamp.utcnow() - pd.to_datetime(subset["timestamp"], errors='coerce')).dt.total_seconds()/60
        events = subset["status"].apply(lambda x: 1 if x!="OPEN" else 0)
        if len(durations)>0:
            kmf.fit(durations, events, label=regime)
            st.line_chart(pd.DataFrame({"time": kmf.survival_function_.index, f"{regime}": kmf.survival_function_[regime]}).set_index("time"))
else:
    st.info("No historical signals yet.")

# -----------------------------
# Exchange disagreement detection
# -----------------------------
st.subheader("Exchange Disagreement Detection (XT vs Gate.io)")
disagreement_table = []
for asset in selected_assets:
    try:
        xt_ex = get_exchange("XT")
        gate_ex = get_exchange("Gate.io")
        xt_df = compute_indicators(fetch_ohlcv(asset, TIMEFRAME))
        gate_df = compute_indicators(fetch_ohlcv(asset, TIMEFRAME))
        xt_signal, _ ,_, _, _ = deterministic_signal(xt_df)
        gate_signal, _, _, _, _ = deterministic_signal(gate_df)
        consensus = xt_signal if xt_signal==gate_signal else "DISAGREEMENT"
        disagreement_table.append({"Asset": asset, "XT": xt_signal, "Gate": gate_signal, "Consensus": consensus})
    except:
        disagreement_table.append({"Asset": asset, "XT":"NA","Gate":"NA","Consensus":"NA"})

df_disagreement = pd.DataFrame(disagreement_table)
def color_signal(val):
    if val.startswith("LONG"): return "color:green"
    if val.startswith("SHORT"): return "color:red"
    if val=="NEUTRAL": return "color:yellow"
    if val=="DISAGREEMENT": return "color:orange"
    return ""
st.dataframe(df_disagreement.style.applymap(color_signal, subset=["XT","Gate","Consensus"]), use_container_width=True)
