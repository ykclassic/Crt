# =========================================================
# Nexus Neural v4 — Deterministic Ensemble Signal Engine
# XT + Gate.io | Deterministic Logic | Survival Analysis
# =========================================================

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    status TEXT,
    exit_timestamp TEXT,
    exit_price REAL
)
""")
DB.commit()

def add_column(name, type_):
    try:
        DB.execute(f"ALTER TABLE signals ADD COLUMN {name} {type_}")
        DB.commit()
    except sqlite3.OperationalError:
        pass

add_column("exit_timestamp", "TEXT")
add_column("exit_price", "REAL")

def log_signal(r: dict):
    DB.execute("""
        INSERT INTO signals (
            timestamp, exchange, asset, timeframe, regime,
            signal, entry, stop, take, confidence, model_hash, status,
            exit_timestamp, exit_price
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        r["timestamp"], r["exchange"], r["asset"], r["timeframe"],
        r["regime"], r["signal"], r["entry"], r["stop"], r["take"],
        r["confidence"], r["model_hash"], r["status"],
        None, None
    ))
    DB.commit()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
exchange_name = st.sidebar.selectbox("Signal Base Exchange", ["XT", "Gate.io"])

TIMEFRAMES = ["1h", "4h", "1d"]
selected_timeframes = st.sidebar.multiselect(
    "Timeframes", TIMEFRAMES, default=["1h", "4h"]
)

ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
    "LINK/USDT","DOGE/USDT","TRX/USDT","SUI/USDT","PEPE/USDT"
]
selected_assets = st.sidebar.multiselect(
    "Assets", ASSETS, default=ASSETS[:5]
)

color_map = {"1h": "blue", "4h": "green", "1d": "yellow"}

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

def generate_and_log_signal(asset, tf, df):
    signal, regime, entry, stop, take = deterministic_signal(df)
    confidence = ml_confidence(df)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "exchange": exchange_name,
        "asset": asset,
        "timeframe": tf,
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
# INITIAL SYNC FETCH
# ---------------------------------------------------------
signals_cache = {asset: {} for asset in selected_assets}

for asset in selected_assets:
    for tf in selected_timeframes:
        try:
            df = fetch_ohlcv(asset, tf)
            df = compute_indicators(df)
            record, df = generate_and_log_signal(asset, tf, df)
            signals_cache[asset][tf] = (record, df)
        except Exception as e:
            print(f"Initial fetch failed {asset} {tf}: {e}")

# ---------------------------------------------------------
# Background update loop WITH CLOSURE LOGIC
# ---------------------------------------------------------
def update_loop():
    while True:
        for asset in selected_assets:
            for tf in selected_timeframes:
                try:
                    df = fetch_ohlcv(asset, tf)
                    df = compute_indicators(df)
                    current_price = df.iloc[-1]["close"]

                    # === SIGNAL CLOSURE LOGIC ===
                    open_signals = DB.execute("""
                        SELECT id, signal, stop, take FROM signals
                        WHERE asset = ? AND timeframe = ? AND exchange = ? AND status = 'OPEN'
                    """, (asset, tf, exchange_name)).fetchall()

                    for sig_id, sig, stop, take in open_signals:
                        status_new = None
                        if sig == "LONG":
                            if current_price <= stop:
                                status_new = "CLOSED_LOSS"
                            elif current_price >= take:
                                status_new = "CLOSED_WIN"
                        elif sig == "SHORT":
                            if current_price >= stop:
                                status_new = "CLOSED_LOSS"
                            elif current_price <= take:
                                status_new = "CLOSED_WIN"

                        if status_new:
                            DB.execute("""
                                UPDATE signals
                                SET status = ?, exit_price = ?, exit_timestamp = ?
                                WHERE id = ?
                            """, (status_new, current_price, datetime.utcnow().isoformat(), sig_id))

                    DB.commit()

                    # === Generate & log new signal ===
                    record, _ = generate_and_log_signal(asset, tf, df)
                    signals_cache[asset][tf] = (record, df)

                except Exception as e:
                    print(f"Update error {asset} {tf}: {e}")

        time.sleep(60)

threading.Thread(target=update_loop, daemon=True).start()

# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------
st.subheader(f"Live Signals — {exchange_name} | {', '.join(selected_timeframes)}")

for asset in selected_assets:
    st.markdown(f"### {asset}")

    if asset not in signals_cache or not signals_cache[asset]:
        st.info("No data available yet.")
        continue

    # === Combined chart across timeframes ===
    df_list = []
    for tf in selected_timeframes:
        if tf in signals_cache[asset]:
            _, df = signals_cache[asset][tf]
            df = df.copy()
            df["tf"] = tf
            df_list.append(df)

    if df_list:
        df_all = pd.concat(df_list)

        fig = px.line(
            df_all,
            x="timestamp",
            y="close",
            color="tf",
            title=f"{asset} Closing Prices & EMAs Across Timeframes",
            color_discrete_map=color_map
        )

        for tf in selected_timeframes:
            color = color_map.get(tf, "black")
            tf_df = df_all[df_all["tf"] == tf]
            if not tf_df.empty:
                fig.add_scatter(x=tf_df["timestamp"], y=tf_df["ema20"], mode="lines",
                                name=f"EMA20 {tf}", line=dict(color=color, dash="dash"))
                fig.add_scatter(x=tf_df["timestamp"], y=tf_df["ema50"], mode="lines",
                                name=f"EMA50 {tf}", line=dict(color=color, dash="dot"))

        st.plotly_chart(fig, use_container_width=True)

    # === Signals per timeframe ===
    for tf in selected_timeframes:
        if tf not in signals_cache[asset]:
            continue

        record, _ = signals_cache[asset][tf]
        tf_color = color_map.get(tf, "grey")
        sig_color = "green" if record["signal"] == "LONG" else "red" if record["signal"] == "SHORT" else "orange"

        st.markdown(
            f"#### <span style='color:{tf_color}'>{tf.upper()} Signal</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<b style='color:{sig_color}; font-size:1.2em'>{record['signal']} | {record['regime']}</b>",
            unsafe_allow_html=True
        )
        st.write(
            f"**Entry:** {record['entry']:.2f} | "
            f"**SL:** {record['stop']:.2f} | "
            f"**TP:** {record['take']:.2f} | "
            f"**Confidence:** {record['confidence']:.2f}%"
        )
        st.divider()

# ---------------------------------------------------------
# Signal lifecycle table
# ---------------------------------------------------------
st.subheader("Signal Lifecycle Table")
df_audit = pd.read_sql_query(
    "SELECT * FROM signals ORDER BY timestamp DESC", DB
)
st.dataframe(df_audit, use_container_width=True)

# ---------------------------------------------------------
# Survival analysis
# ---------------------------------------------------------
st.subheader("Signal Survival Analysis by Regime & Timeframe")

if not df_audit.empty:
    df_audit = df_audit.copy()
    df_audit["timestamp"] = pd.to_datetime(df_audit["timestamp"], utc=True, errors="coerce")
    df_audit["exit_timestamp"] = pd.to_datetime(df_audit["exit_timestamp"], utc=True, errors="coerce")
    df_audit = df_audit.dropna(subset=["timestamp"])

    now = pd.Timestamp.utcnow()

    open_mask = df_audit["status"] == "OPEN"

    df_audit["duration_min"] = 0.0
    df_audit.loc[open_mask, "duration_min"] = (now - df_audit.loc[open_mask, "timestamp"]).dt.total_seconds() / 60
    df_audit.loc[~open_mask, "duration_min"] = (df_audit.loc[~open_mask, "exit_timestamp"] - df_audit.loc[~open_mask, "timestamp"]).dt.total_seconds() / 60

    df_audit["event_observed"] = (~open_mask).astype(int)

    df_audit = df_audit.dropna(subset=["duration_min"])

    fig = go.Figure()

    kmf = KaplanMeierFitter()
    has_data = False

    for regime in df_audit["regime"].unique():
        for tf in selected_timeframes:
            subset = df_audit[(df_audit["regime"] == regime) & (df_audit["timeframe"] == tf)]
            if len(subset) < 2:
                continue

            kmf.fit(subset["duration_min"], subset["event_observed"])
            surv = kmf.survival_function_
            color = color_map.get(tf, "black")
            fig.add_scatter(
                x=surv.index,
                y=surv.iloc[:, 0],
                name=f"{regime} ({tf})",
                line=dict(color=color)
            )
            has_data = True

    if has_data:
        fig.update_layout(
            title="Survival Probability by Regime & Timeframe",
            xaxis_title="Minutes Since Signal Open",
            yaxis_title="Survival Probability",
            yaxis_range=[0, 1.05]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for survival analysis yet.")

else:
    st.info("No historical signals yet.")

# ---------------------------------------------------------
# Exchange disagreement detection
# ---------------------------------------------------------
st.subheader("Exchange Disagreement (XT vs Gate.io) Across Timeframes")

xt_ex = get_exchange("XT")
gate_ex = get_exchange("Gate.io")

rows = []

for tf in selected_timeframes:
    for asset in selected_assets:
        try:
            xt_df = compute_indicators(
                pd.DataFrame(
                    xt_ex.fetch_ohlcv(asset, tf, limit=200),
                    columns=["timestamp","open","high","low","close","volume"]
                )
            )
            gate_df = compute_indicators(
                pd.DataFrame(
                    gate_ex.fetch_ohlcv(asset, tf, limit=200),
                    columns=["timestamp","open","high","low","close","volume"]
                )
            )

            xt_sig, *_ = deterministic_signal(xt_df)
            gate_sig, *_ = deterministic_signal(gate_df)

            consensus = "CONSENSUS" if xt_sig == gate_sig else "DISAGREEMENT"

            rows.append({
                "Timeframe": tf,
                "Asset": asset,
                "XT": xt_sig,
                "Gate.io": gate_sig,
                "Consensus": consensus
            })
        except Exception:
            rows.append({
                "Timeframe": tf,
                "Asset": asset,
                "XT": "NA",
                "Gate.io": "NA",
                "Consensus": "NA"
            })

df_dis = pd.DataFrame(rows).sort_values(["Timeframe", "Asset"])

def highlight_sig(val):
    if val == "LONG": return "color:green"
    if val == "SHORT": return "color:red"
    if val == "NEUTRAL": return "color:orange"
    return ""

def highlight_cons(val):
    if val == "DISAGREEMENT": return "background-color:#ffcccb"
    if val == "CONSENSUS": return "background-color:#90ee90"
    return ""

styled = (
    df_dis.style
    .applymap(highlight_sig, subset=["XT", "Gate.io"])
    .applymap(highlight_cons, subset=["Consensus"])
)

st.dataframe(styled, use_container_width=True)
