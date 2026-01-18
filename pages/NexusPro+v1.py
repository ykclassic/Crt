# =========================================================
# Nexus Neural v4 — Deterministic Ensemble Signal Engine
# XT + Gate.io | Deterministic Logic | Survival Analysis
# =========================================================

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

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Nexus Neural v4", page_icon="🌐", layout="wide")
st.title("🌐 Nexus Neural v4 — Deterministic Ensemble Signal Engine")

# ---------------------------------------------------------
# Database
# ---------------------------------------------------------
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

def log_signal(r: dict):
    DB.execute("""
        INSERT INTO signals (
            timestamp, exchange, asset, timeframe, regime,
            signal, entry, stop, take, confidence, model_hash, status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        r["timestamp"], r["exchange"], r["asset"], r["timeframe"],
        r["regime"], r["signal"], r["entry"], r["stop"], r["take"],
        r["confidence"], r["model_hash"], r["status"]
    ))
    DB.commit()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
exchange_name = st.sidebar.selectbox("Signal Base Exchange", ["XT", "Gate.io"])
TIMEFRAME = st.sidebar.selectbox("Primary Timeframe", ["1h", "4h", "1d"])

ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
    "LINK/USDT","DOGE/USDT","TRX/USDT","SUI/USDT","PEPE/USDT"
]
selected_assets = st.sidebar.multiselect(
    "Assets", ASSETS, default=ASSETS[:5]
)

# ---------------------------------------------------------
# Exchange loader
# ---------------------------------------------------------
@st.cache_resource
def get_exchange(name):
    if name == "XT":
        ex = ccxt.xt({"enableRateLimit": True})
    else:
        ex = ccxt.gateio({"enableRateLimit": True})
    ex.load_markets()
    return ex

exchange = get_exchange(exchange_name)

# ---------------------------------------------------------
# Indicators
# ---------------------------------------------------------
def compute_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))

    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    return df

# ---------------------------------------------------------
# Deterministic signal engine
# ---------------------------------------------------------
def deterministic_signal(df):
    last = df.iloc[-1]

    if last["close"] > last["ema20"] and last["rsi"] < 70:
        signal = "LONG"
    elif last["close"] < last["ema20"] and last["rsi"] > 30:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    if last["close"] > last["ema50"]:
        regime = "BULLISH"
    elif last["close"] < last["ema50"]:
        regime = "BEARISH"
    else:
        regime = "SIDEWAYS"

    entry = last["close"]
    stop = entry * (0.98 if signal == "LONG" else 1.02)
    take = entry * (1.03 if signal == "LONG" else 0.97)

    return signal, regime, entry, stop, take

def ml_confidence(df):
    # deterministic placeholder (replace with real model later)
    return float(np.clip(70 + df["rsi"].iloc[-1] / 2, 70, 95))

# ---------------------------------------------------------
# Data fetch
# ---------------------------------------------------------
def fetch_ohlcv(symbol, tf, limit=200):
    data = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(
        data, columns=["timestamp","open","high","low","close","volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def generate_signal(symbol):
    df = fetch_ohlcv(symbol, TIMEFRAME)
    df = compute_indicators(df)

    signal, regime, entry, stop, take = deterministic_signal(df)
    confidence = ml_confidence(df)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "exchange": exchange_name,
        "asset": symbol,
        "timeframe": TIMEFRAME,
        "regime": regime,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "take": take,
        "confidence": confidence,
        "model_hash": hashlib.md5(b"NexusNeuralV4").hexdigest(),
        "status": "OPEN"
    }

    log_signal(record)
    return record, df

# ---------------------------------------------------------
# INITIAL SYNC FETCH  ✅ FIXES “data unavailable”
# ---------------------------------------------------------
signals_cache = {}

for asset in selected_assets:
    try:
        signals_cache[asset] = generate_signal(asset)
    except Exception as e:
        print(f"Initial fetch failed {asset}: {e}")

# ---------------------------------------------------------
# Background update loop
# ---------------------------------------------------------
def update_loop():
    while True:
        for asset in selected_assets:
            try:
                signals_cache[asset] = generate_signal(asset)
            except Exception as e:
                print(f"Update error {asset}: {e}")
        time.sleep(60)

threading.Thread(target=update_loop, daemon=True).start()

# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------
st.subheader(f"Live Signals — {exchange_name} | {TIMEFRAME}")

for asset, (record, df) in signals_cache.items():
    st.markdown(f"### {asset}")

    fig = px.line(
        df, x="timestamp", y=["close","ema20","ema50"],
        title=f"{asset} Price & EMAs"
    )
    st.plotly_chart(fig, use_container_width=True)

    color = (
        "green" if record["signal"] == "LONG"
        else "red" if record["signal"] == "SHORT"
        else "yellow"
    )

    st.markdown(
        f"<b style='color:{color}'>"
        f"{record['signal']} | {record['regime']}</b>",
        unsafe_allow_html=True
    )
    st.write(
        f"Entry: {record['entry']:.2f} | "
        f"SL: {record['stop']:.2f} | "
        f"TP: {record['take']:.2f}"
    )
    st.write(f"Confidence: {record['confidence']:.2f}%")

# ---------------------------------------------------------
# Signal lifecycle table
# ---------------------------------------------------------
st.subheader("Signal Lifecycle Table")
df_audit = pd.read_sql_query(
    "SELECT * FROM signals ORDER BY timestamp DESC", DB
)
st.dataframe(df_audit, use_container_width=True)

# ---------------------------------------------------------
# Survival analysis (FIXED)
# ---------------------------------------------------------
st.subheader("Signal Survival Analysis by Regime")

if not df_audit.empty:
    df_audit["timestamp"] = pd.to_datetime(
        df_audit["timestamp"], errors="coerce"
    )
    df_audit = df_audit.dropna(subset=["timestamp"])

    kmf = KaplanMeierFitter()

    for regime in df_audit["regime"].unique():
        subset = df_audit[df_audit["regime"] == regime]
        if len(subset) < 2:
            continue

        ts = subset["timestamp"]
        now = pd.Series(pd.Timestamp.utcnow(), index=ts.index)

        durations = (now - ts).dt.total_seconds() / 60
        events = (subset["status"] != "OPEN").astype(int)

        kmf.fit(durations, events, label=regime)
        st.line_chart(kmf.survival_function_)

else:
    st.info("No historical signals yet.")

# ---------------------------------------------------------
# Exchange disagreement detection
# ---------------------------------------------------------
st.subheader("Exchange Disagreement (XT vs Gate.io)")

xt = get_exchange("XT")
gate = get_exchange("Gate.io")

rows = []

for asset in selected_assets:
    try:
        xt_df = compute_indicators(
            pd.DataFrame(
                xt.fetch_ohlcv(asset, TIMEFRAME),
                columns=["timestamp","open","high","low","close","volume"]
            )
        )
        gate_df = compute_indicators(
            pd.DataFrame(
                gate.fetch_ohlcv(asset, TIMEFRAME),
                columns=["timestamp","open","high","low","close","volume"]
            )
        )

        xt_sig, *_ = deterministic_signal(xt_df)
        gate_sig, *_ = deterministic_signal(gate_df)

        consensus = xt_sig if xt_sig == gate_sig else "DISAGREEMENT"

        rows.append({
            "Asset": asset,
            "XT": xt_sig,
            "Gate": gate_sig,
            "Consensus": consensus
        })
    except:
        rows.append({
            "Asset": asset,
            "XT": "NA", "Gate": "NA", "Consensus": "NA"
        })

df_dis = pd.DataFrame(rows)

def color(val):
    if val == "LONG": return "color:green"
    if val == "SHORT": return "color:red"
    if val == "NEUTRAL": return "color:yellow"
    if val == "DISAGREEMENT": return "color:orange"
    return ""

st.dataframe(
    df_dis.style.applymap(color),
    use_container_width=True
)
