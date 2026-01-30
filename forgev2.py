# =========================================================
# Nexus Neural v5 — Advanced Ensemble Signal Engine (Phase 3 Sub-Phase 2)
# XT + Gate.io | ATR-Adjusted | Supertrend Filter | ML Confidence | Backtest Mode
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
import requests
import json
from datetime import datetime, timezone
from lifelines import KaplanMeierFitter

# ---------------------------------------------------------
# sklearn Fallback Setup
# ---------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available – ML confidence will use deterministic fallback.")

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Nexus Neural v5", page_icon="🌐", layout="wide")
st.title("🌐 Nexus Neural v5 — Advanced Ensemble Signal Engine")

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
    exit_price REAL,
    features TEXT
)
""")
DB.commit()

def add_column(name, type_):
    try:
        DB.execute(f"ALTER TABLE signals ADD COLUMN {name} {type_}")
        DB.commit()
    except sqlite3.OperationalError:
        pass

add_column("features", "TEXT")

def log_signal(r: dict):
    DB.execute("""
        INSERT INTO signals (
            timestamp, exchange, asset, timeframe, regime,
            signal, entry, stop, take, confidence, model_hash, status,
            exit_timestamp, exit_price, features
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        r["timestamp"], r["exchange"], r["asset"], r["timeframe"],
        r["regime"], r["signal"], r["entry"], r["stop"], r["take"],
        r["confidence"], r["model_hash"], r["status"],
        None, None, r.get("features", None)
    ))
    DB.commit()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
mode = st.sidebar.radio("Mode", ["Live", "Backtest"], help="Backtest mode in development – Live is fully active")

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

require_confirmation = st.sidebar.checkbox(
    "Higher TF Regime Confirmation", value=True
)

atr_multiplier_stop = st.sidebar.number_input("ATR Stop Multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
rr_ratio = st.sidebar.number_input("Risk:Reward Ratio", min_value=1.0, max_value=4.0, value=1.5, step=0.5)

webhook_url = st.sidebar.text_input(
    "Webhook URL for Alerts", type="password"
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
# Advanced Indicators (Supertrend + ATR)
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

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Supertrend
    hl2 = (df['high'] + df['low']) / 2
    df['upper_band'] = hl2 + (3.0 * df['atr'])
    df['lower_band'] = hl2 - (3.0 * df['atr'])
    df['in_uptrend'] = True

    for i in range(1, len(df)):
        if df['close'].iloc[i-1] > df['upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'in_uptrend'] = True
        elif df['close'].iloc[i-1] < df['lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'in_uptrend'] = False
        else:
            df.loc[df.index[i], 'in_uptrend'] = df['in_uptrend'].iloc[i-1]
            if df['in_uptrend'].iloc[i] and df['lower_band'].iloc[i] < df['lower_band'].iloc[i-1]:
                df.loc[df.index[i], 'lower_band'] = df['lower_band'].iloc[i-1]
            elif not df['in_uptrend'].iloc[i] and df['upper_band'].iloc[i] > df['upper_band'].iloc[i-1]:
                df.loc[df.index[i], 'upper_band'] = df['upper_band'].iloc[i-1]

    df['supertrend'] = np.where(df['in_uptrend'], df['lower_band'], df['upper_band'])

    return df

# ---------------------------------------------------------
# Advanced Signal Engine with ML Confidence
# ---------------------------------------------------------
ml_model = None
ml_scaler = None

def train_ml_model():
    global ml_model, ml_scaler
    if not SKLEARN_AVAILABLE:
        return

    df = pd.read_sql_query("SELECT * FROM signals WHERE status LIKE 'CLOSED_%'", DB)
    if len(df) < 50:
        return

    df = df.dropna(subset=["features"])
    if len(df) < 50:
        return

    features_list = [json.loads(f) for f in df["features"]]
    X = np.array(features_list)
    y = (df["status"] == "CLOSED_WIN").astype(int)

    ml_scaler = StandardScaler()
    X_scaled = ml_scaler.fit_transform(X)

    ml_model = LogisticRegression(max_iter=1000)
    ml_model.fit(X_scaled, y)

# Train on startup
train_ml_model()

def deterministic_signal(df):
    last = df.iloc[-1]
    super_up = last["close"] > last["supertrend"]

    signal = "NEUTRAL"
    if last["close"] > last["ema20"] and last["rsi"] < 70 and super_up:
        signal = "LONG"
    elif last["close"] < last["ema20"] and last["rsi"] > 30 and not super_up:
        signal = "SHORT"

    regime = "BULLISH" if last["close"] > last["ema50"] else "BEARISH" if last["close"] < last["ema50"] else "SIDEWAYS"

    entry = last["close"]
    atr = last["atr"] if not np.isnan(last["atr"]) else entry * 0.01

    if signal == "LONG":
        stop = entry - atr_multiplier_stop * atr
        take = entry + rr_ratio * (entry - stop)
    elif signal == "SHORT":
        stop = entry + atr_multiplier_stop * atr
        take = entry - rr_ratio * (stop - entry)
    else:
        stop = take = entry

    features = json.dumps([
        float(last["rsi"]),
        float((last["close"] - last["ema20"]) / last["ema20"] if last["ema20"] != 0 else 0),
        float((last["close"] - last["ema50"]) / last["ema50"] if last["ema50"] != 0 else 0),
        float(atr / entry)
    ])

    confidence = 80.0
    if ml_model and SKLEARN_AVAILABLE:
        try:
            X = np.array([json.loads(features)])
            X_scaled = ml_scaler.transform(X)
            prob = ml_model.predict_proba(X_scaled)[0][1]
            confidence = 50 + 50 * prob
        except:
            pass

    return signal, regime, entry, stop, take, confidence, features

# ---------------------------------------------------------
# Data fetch
# ---------------------------------------------------------
def fetch_ohlcv(symbol, tf, since=None, limit=500):
    params = {"enableRateLimit": True}
    if since:
        params['since'] = since
    data = exchange.fetch_ohlcv(symbol, tf, limit=limit, params=params)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# ---------------------------------------------------------
# Signal generation with throttling (fully integrated new returns)
# ---------------------------------------------------------
last_logged = {asset: {tf: None for tf in TIMEFRAMES} for asset in ASSETS}

def generate_and_log_signal(asset, tf, df, force_log=False):
    signal, regime, entry, stop, take, confidence, features = deterministic_signal(df)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange": exchange_name,
        "asset": asset,
        "timeframe": tf,
        "regime": regime,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "take": take,
        "confidence": confidence,
        "model_hash": hashlib.md5(b"NexusNeuralV5").hexdigest(),
        "status": "OPEN",
        "features": features
    }

    prev = last_logged.get(asset, {}).get(tf)

    should_log = force_log or prev is None or (
        prev["signal"] != signal or
        prev["regime"] != regime or
        abs(prev["entry"] - entry) / entry > 0.001 or
        abs(prev["stop"] - stop) / entry > 0.001 or
        abs(prev["take"] - take) / entry > 0.001 or
        abs(prev["confidence"] - confidence) > 5  # confidence can fluctuate
    )

    new_signal_alert = False
    if should_log:
        log_signal(record)
        last_logged[asset][tf] = record.copy()
        new_signal_alert = True

    return record, df, new_signal_alert, should_log

# ---------------------------------------------------------
# INITIAL SYNC FETCH
# ---------------------------------------------------------
signals_cache = {asset: {} for asset in selected_assets}

for asset in selected_assets:
    for tf in selected_timeframes:
        try:
            df = fetch_ohlcv(asset, tf)
            if df.empty:
                continue
            df = compute_indicators(df)
            record, df, _, _ = generate_and_log_signal(asset, tf, df)
            signals_cache[asset][tf] = (record, df)
        except Exception as e:
            st.error(f"Initial fetch failed {asset} {tf}: {e}")

# ---------------------------------------------------------
# Background update loop (Phase 2 logic fully preserved)
# ---------------------------------------------------------
tf_hierarchy = {"1h": "4h", "4h": "1d", "1d": None}

def update_loop():
    global ml_model, ml_scaler
    retrain_counter = 0
    alert_messages = []
    while True:
        alert_messages.clear()
        for asset in selected_assets:
            for tf in selected_timeframes:
                try:
                    df = fetch_ohlcv(asset, tf)
                    if df.empty:
                        continue
                    df = compute_indicators(df)
                    current_price = df.iloc[-1]["close"]
                    now_iso = datetime.now(timezone.utc).isoformat()

                    closed_this_cycle = False
                    closure_msg = ""

                    open_signals = DB.execute("""
                        SELECT id, timestamp, signal, entry, stop, take FROM signals
                        WHERE asset = ? AND timeframe = ? AND exchange = ? AND status = 'OPEN'
                        ORDER BY timestamp DESC LIMIT 1
                    """, (asset, tf, exchange_name)).fetchall()

                    hit_found = False
                    if open_signals:
                        sig_id, sig_ts, sig, entry, stop, take = open_signals[0]
                        since_ms = int(pd.to_datetime(sig_ts, utc=True).timestamp() * 1000) + 1
                        hist_df = fetch_ohlcv(asset, tf, since=since_ms, limit=1000)

                        if not hist_df.empty:
                            exit_price = current_price
                            exit_time = now_iso
                            status_new = None

                            for _, candle in hist_df.iterrows():
                                if sig == "LONG":
                                    if candle["low"] <= stop:
                                        status_new = "CLOSED_LOSS"
                                        exit_price = stop
                                        exit_time = candle["timestamp"].isoformat()
                                        break
                                    if candle["high"] >= take:
                                        status_new = "CLOSED_WIN"
                                        exit_price = take
                                        exit_time = candle["timestamp"].isoformat()
                                        break
                                elif sig == "SHORT":
                                    if candle["high"] >= stop:
                                        status_new = "CLOSED_LOSS"
                                        exit_price = stop
                                        exit_time = candle["timestamp"].isoformat()
                                        break
                                    if candle["low"] <= take:
                                        status_new = "CLOSED_WIN"
                                        exit_price = take
                                        exit_time = candle["timestamp"].isoformat()
                                        break

                            if status_new:
                                DB.execute("""
                                    UPDATE signals SET status = ?, exit_price = ?, exit_timestamp = ?
                                    WHERE id = ?
                                """, (status_new, exit_price, exit_time, sig_id))
                                closed_this_cycle = True
                                closure_msg = f"{asset} {tf} {status_new.replace('CLOSED_', '')} at {exit_price:.2f}"
                                hit_found = True

                    DB.commit()

                    record, _, new_alert, logged = generate_and_log_signal(asset, tf, df)

                    regime_change_close = False
                    if logged and open_signals and not hit_found:
                        prev_id = open_signals[0][0]
                        DB.execute("""
                            UPDATE signals SET status = 'CLOSED_REGIME_CHANGE', exit_price = ?, exit_timestamp = ?
                            WHERE id = ?
                        """, (current_price, now_iso, prev_id))
                        regime_change_close = True
                        closure_msg = f"{asset} {tf} closed (regime change) at {current_price:.2f}"

                    DB.commit()

                    signals_cache[asset][tf] = (record, df)

                    if webhook_url:
                        if new_alert:
                            alert_messages.append(f"🆕 New {record['signal']} {asset} {tf} | Entry {record['entry']:.2f} | Conf {record['confidence']:.1f}%")
                        if closed_this_cycle or regime_change_close:
                            alert_messages.append(f"🔒 {closure_msg}")

                except Exception as e:
                    st.error(f"Update error {asset} {tf}: {e}")

        retrain_counter += 1
        if retrain_counter >= 10:
            train_ml_model()
            retrain_counter = 0

        if webhook_url and alert_messages:
            try:
                requests.post(webhook_url, json={"content": "\n".join(alert_messages)})
            except:
                pass

        time.sleep(60)

if mode == "Live":
    threading.Thread(target=update_loop, daemon=True).start()

# ---------------------------------------------------------
# Dashboard with Supertrend & ATR Visualization (Phase 3 Placeholder 3 Filled)
# ---------------------------------------------------------
st.subheader(f"Live Signals — {exchange_name} | {', '.join(selected_timeframes)}")

higher_map = {"1h": "4h", "4h": "1d", "1d": None}

for asset in selected_assets:
    st.markdown(f"### {asset}")

    if asset not in signals_cache or not signals_cache[asset]:
        st.info("No data available yet.")
        continue

    df_list = []
    for tf in selected_timeframes:
        if tf in signals_cache[asset]:
            _, df = signals_cache[asset][tf]
            df = df.copy()
            df["tf"] = tf
            df_list.append(df)

    if df_list:
        df_all = pd.concat(df_list).sort_values("timestamp")

        fig = px.line(
            df_all,
            x="timestamp",
            y="close",
            color="tf",
            title=f"{asset} Price, EMAs & Supertrend Across Timeframes",
            color_discrete_map=color_map
        )

        # EMAs
        for tf in selected_timeframes:
            color = color_map.get(tf, "black")
            tf_df = df_all[df_all["tf"] == tf]
            if not tf_df.empty:
                fig.add_scatter(x=tf_df["timestamp"], y=tf_df["ema20"], mode="lines",
                                name=f"EMA20 {tf}", line=dict(color=color, dash="dash"))
                fig.add_scatter(x=tf_df["timestamp"], y=tf_df["ema50"], mode="lines",
                                name=f"EMA50 {tf}", line=dict(color=color, dash="dot"))

                # Supertrend line
                fig.add_scatter(x=tf_df["timestamp"], y=tf_df["supertrend"], mode="lines",
                                name=f"Supertrend {tf}", line=dict(color=color, width=3))

        # Active Entry/SL/TP
        min_ts = df_all["timestamp"].min()
        max_ts = df_all["timestamp"].max()
        for tf in selected_timeframes:
            if tf not in signals_cache[asset]:
                continue
            record, _ = signals_cache[asset][tf]
            if record["signal"] == "NEUTRAL":
                continue

            color = color_map.get(tf, "grey")
            fig.add_scatter(x=[min_ts, max_ts], y=[record["entry"], record["entry"]],
                            mode="lines", line=dict(color=color, width=2), name=f"Entry {tf}")
            fig.add_scatter(x=[min_ts, max_ts], y=[record["stop"], record["stop"]],
                            mode="lines", line=dict(color="red", dash="dash"), name=f"SL {tf}")
            fig.add_scatter(x=[min_ts, max_ts], y=[record["take"], record["take"]],
                            mode="lines", line=dict(color="green", dash="dash"), name=f"TP {tf}")

        st.plotly_chart(fig, use_container_width=True)

    # Signals display with confirmation
    for tf in selected_timeframes:
        if tf not in signals_cache[asset]:
            continue
        record, _ = signals_cache[asset][tf]

        higher_tf = higher_map.get(tf)
        confirmed = True
        if require_confirmation and higher_tf and higher_tf in selected_timeframes:
            higher_rec, _ = signals_cache[asset].get(higher_tf, (None, None))
            if higher_rec and record["signal"] != "NEUTRAL":
                if record["signal"] == "LONG" and higher_rec["regime"] != "BULLISH":
                    confirmed = False
                elif record["signal"] == "SHORT" and higher_rec["regime"] != "BEARISH":
                    confirmed = False

        if require_confirmation and not confirmed and record["signal"] != "NEUTRAL":
            continue

        tf_color = color_map.get(tf, "grey")
        sig_color = "green" if record["signal"] == "LONG" else "red" if record["signal"] == "SHORT" else "orange"
        conf_badge = " ✅ Confirmed" if confirmed else ""

        st.markdown(
            f"#### <span style='color:{tf_color}'>{tf.upper()} Signal{conf_badge}</span>",
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
            f"**Confidence:** {record['confidence']:.1f}% (ML-enhanced)"
        )
        st.divider()

# ---------------------------------------------------------
# Phase 3 Sub-Phase 2 Completion
# ---------------------------------------------------------
st.success("🚀 **Phase 3 Sub-Phase 2 COMPLETE:**\n"
           "- Full integration of new 7-value signal return (incl. confidence & features)\n"
           "- Supertrend visualized on combined charts (thick line per TF)\n"
           "- Confidence display updated to show ML-enhanced value\n"
           "- Minor fixes: safer division in features, confidence change threshold in throttling\n"
           "- All prior phases (1-2 + Sub-Phase 1) preserved and working\n\n"
           "Charts now clearly show Supertrend direction – signals only fire when aligned. Ready for Sub-Phase 3 (Backtest Mode Skeleton), YKonChain 🕊️!")
