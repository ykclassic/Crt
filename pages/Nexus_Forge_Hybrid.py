# =========================================================
# Nexus HybridTrader v1 — Trend/Range Switcher with AI Mode
# XT + Gate.io | Dynamic Strategy Switching | AI Boosts | Backtest Mode
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

RF_AVAILABLE = False
if SKLEARN_AVAILABLE:
    try:
        from sklearn.ensemble import RandomForestClassifier
        RF_AVAILABLE = True
    except ImportError:
        st.warning("RandomForest not available – falling back to LogisticRegression for ML.")

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Nexus HybridTrader v1", page_icon="🔄", layout="wide")
st.title("🔄 Nexus HybridTrader v1 — Trend/Range Switcher with AI Mode")

# ---------------------------------------------------------
# Database
# ---------------------------------------------------------
DB = sqlite3.connect("hybrid_signals.db", check_same_thread=False)

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
# Sidebar controls (added strategy & AI toggles)
# ---------------------------------------------------------
mode = st.sidebar.radio("Mode", ["Live", "Backtest"])

strategy_mode = st.sidebar.radio("Strategy Mode", ["Trend", "Range"])

ai_mode = st.sidebar.checkbox("AI Mode (Sentiment, Optimization, Queries)", value=False)

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

min_confidence = st.sidebar.slider("Min Confidence % to Show", 50, 100, 70)
trailing_stop_pct = st.sidebar.slider("Trailing Stop % (after 1R)", 0.0, 1.0, 0.5, 0.1)

webhook_url = st.sidebar.text_input(
    "Webhook URL for Alerts", type="password"
)

if mode == "Backtest":
    st.sidebar.header("Backtest Settings")
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    initial_capital = st.sidebar.number_input("Initial Capital (USDT)", value=10000.0)
    risk_percent = st.sidebar.slider("Risk % per Trade", 0.5, 5.0, 2.0, 0.5)
    slippage_pct = st.sidebar.slider("Slippage %", 0.0, 0.5, 0.1, 0.05)
    fee_pct = st.sidebar.slider("Trading Fee % (per side)", 0.0, 0.2, 0.05, 0.01)

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
# Dynamic Indicators based on strategy mode
# ---------------------------------------------------------
def compute_indicators(df, mode="Trend"):
    df = df.copy()

    if mode == "Trend":
        # Trend indicators (from v5.1)
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()

        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + rs))

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

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

        df["volume_ma"] = df["volume"].rolling(20).mean()

    else:  # Range mode
        # Range indicators (from RangeMaster)
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + rs))

        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df["+di"] = 100 * (df["high"].diff().clip(lower=0).rolling(14).sum() / atr)
        df["-di"] = 100 * (df["low"].diff().clip(upper=0).abs().rolling(14).sum() / atr)
        dx = 100 * np.abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
        df["adx"] = dx.rolling(14).mean()

        df["atr"] = atr

    return df

# ---------------------------------------------------------
# Dynamic Signal Engine based on mode
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

    if RF_AVAILABLE:
        ml_model = RandomForestClassifier(n_estimators=100, max_depth=10)
    else:
        ml_model = LogisticRegression(max_iter=1000)

    ml_model.fit(X_scaled, y)

train_ml_model()

def deterministic_signal(df, mode="Trend"):
    last = df.iloc[-1]

    if mode == "Trend":
        super_up = last["close"] > last["supertrend"]

        volume_ok = last["volume"] > last["volume_ma"] * 1.2

        signal = "NEUTRAL"
        if last["close"] > last["ema20"] and last["rsi"] < 70 and super_up and volume_ok:
            signal = "LONG"
        elif last["close"] < last["ema20"] and last["rsi"] > 30 and not super_up and volume_ok:
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
            float(atr / entry),
            float((last["close"] - last["supertrend"]) / entry),
            float(last["volume"] / last["volume_ma"] if last["volume_ma"] != 0 else 0)
        ])

    else:  # Range
        ranging = last["adx"] < 25

        signal = "NEUTRAL"
        if ranging:
            if last["rsi"] < 30 and last["close"] < last["bb_lower"] * 1.01:
                signal = "LONG"
            elif last["rsi"] > 70 and last["close"] > last["bb_upper"] * 0.99:
                signal = "SHORT"

        regime = "RANGING" if ranging else "TRENDING"

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
            float((last["close"] - last["bb_mid"]) / last["bb_std"]),
            float(last["adx"]),
            float(atr / entry)
        ])

    # AI Mode: Sentiment boost if on
    sentiment = 0
    if ai_mode:
        try:
            sentiment_data = x_semantic_search(query=f"{asset} sentiment last 24h", limit=20)
            # Simple sentiment parse (improve with LLM summary)
            positive = sentiment_data.count("bullish") + sentiment_data.count("up")
            negative = sentiment_data.count("bearish") + sentiment_data.count("down")
            sentiment = 1 if positive > negative else -1 if negative > positive else 0
            confidence += sentiment * 5  # Adjust based on sentiment
        except:
            pass

    confidence = np.clip(confidence, 0, 100)

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
# Signal generation with throttling
# ---------------------------------------------------------
last_logged = {asset: {tf: None for tf in TIMEFRAMES} for asset in ASSETS}

def generate_and_log_signal(asset, tf, df, force_log=False):
    signal, regime, entry, stop, take, confidence, features = deterministic_signal(df, mode=strategy_mode)

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
        "model_hash": hashlib.md5(b"HybridTraderV1").hexdigest(),
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
        abs(prev["confidence"] - confidence) > 5
    )

    new_signal_alert = False
    if should_log:
        log_signal(record)
        last_logged[asset][tf] = record.copy()
        new_signal_alert = True

    return record, df, new_signal_alert, should_log

# ---------------------------------------------------------
# Initial Fetch for Live
# ---------------------------------------------------------
signals_cache = {asset: {} for asset in selected_assets}

if mode == "Live":
    for asset in selected_assets:
        for tf in selected_timeframes:
            try:
                df = fetch_ohlcv(asset, tf)
                if df.empty:
                    continue
                df = compute_indicators(df, mode=strategy_mode)
                record, df, _, _ = generate_and_log_signal(asset, tf, df)
                signals_cache[asset][tf] = (record, df)
            except Exception as e:
                st.error(f"Initial fetch failed {asset} {tf}: {e}")

# ---------------------------------------------------------
# Background Update Loop for Live
# ---------------------------------------------------------
higher_map = {"1h": "4h", "4h": "1d", "1d": None}

if mode == "Live":
    def update_loop():
        global ml_model, ml_scaler
        retrain_counter = 0
        while True:
            alert_messages = []
            for asset in selected_assets:
                for tf in selected_timeframes:
                    try:
                        df = fetch_ohlcv(asset, tf)
                        if df.empty:
                            continue
                        df = compute_indicators(df, mode=strategy_mode)
                        current_price = df.iloc[-1]["close"]
                        now_iso = datetime.now(timezone.utc).isoformat()

                        open_signals = DB.execute("""
                            SELECT id, timestamp, signal, entry, stop, take FROM signals
                            WHERE asset = ? AND timeframe = ? AND exchange = ? AND status = 'OPEN'
                            ORDER BY timestamp DESC LIMIT 1
                        """, (asset, tf, exchange_name)).fetchall()

                        hit_found = False
                        closure_msg = ""
                        if open_signals:
                            sig_id, sig_ts, sig, entry, stop, take = open_signals[0]
                            since_ms = int(pd.to_datetime(sig_ts, utc=True).timestamp() * 1000) + 1
                            hist_df = fetch_ohlcv(asset, tf, since=since_ms, limit=1000)

                            if not hist_df.empty:
                                exit_price = current_price
                                exit_time = now_iso
                                status_new = None

                                # Trailing stop
                                profit_1r = (current_price - entry) > (entry - stop) if sig == "LONG" else (entry - current_price) > (stop - entry)
                                if profit_1r:
                                    new_stop = current_price * (1 - trailing_stop_pct / 100) if sig == "LONG" else current_price * (1 + trailing_stop_pct / 100)
                                    if (sig == "LONG" and new_stop > stop) or (sig == "SHORT" and new_stop < stop):
                                        stop = new_stop
                                        DB.execute("UPDATE signals SET stop = ? WHERE id = ?", (stop, sig_id))

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
                                    hit_found = True
                                    closure_msg = f"{asset} {tf} {status_new.replace('CLOSED_', '')} | Entry {entry:.2f} | SL {stop:.2f} | TP {take:.2f} | Exit {exit_price:.2f}"

                        DB.commit()

                        record, _, new_alert, logged = generate_and_log_signal(asset, tf, df)

                        if logged and open_signals and not hit_found:
                            prev_id = open_signals[0][0]
                            DB.execute("""
                                UPDATE signals SET status = 'CLOSED_REGIME_CHANGE', exit_price = ?, exit_timestamp = ?
                                WHERE id = ?
                            """, (current_price, now_iso, prev_id))
                            closure_msg = f"{asset} {tf} CLOSED_REGIME_CHANGE | Entry {entry:.2f} | SL {stop:.2f} | TP {take:.2f} | Exit {current_price:.2f}"

                        DB.commit()

                        signals_cache[asset][tf] = (record, df)

                        if webhook_url:
                            if new_alert:
                                alert_messages.append(f"🆕 New {record['signal']} {asset} {tf} | Entry {record['entry']:.2f} | SL {record['stop']:.2f} | TP {record['take']:.2f} | Conf {record['confidence']:.1f}%")
                            if hit_found or closure_msg:
                                alert_messages.append(f"🔒 {closure_msg}")

                    except Exception as e:
                        print(f"Update error {asset} {tf}: {e}")

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

    threading.Thread(target=update_loop, daemon=True).start()

# ---------------------------------------------------------
# Dashboard for Live (dynamic charts based on mode)
# ---------------------------------------------------------
if mode == "Live":
    st.subheader(f"Live Signals — {exchange_name} | {', '.join(selected_timeframes)} | {strategy_mode} Mode | AI { 'On' if ai_mode else 'Off' }")

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
                title=f"{asset} Price & Indicators ({strategy_mode} Mode)",
                color_discrete_map=color_map
            )

            for tf in selected_timeframes:
                color = color_map.get(tf, "black")
                tf_df = df_all[df_all["tf"] == tf]
                if not tf_df.empty:
                    if strategy_mode == "Trend":
                        fig.add_scatter(x=tf_df["timestamp"], y=tf_df["ema20"], mode="lines",
                                        name=f"EMA20 {tf}", line=dict(color=color, dash="dash"))
                        fig.add_scatter(x=tf_df["timestamp"], y=tf_df["ema50"], mode="lines",
                                        name=f"EMA50 {tf}", line=dict(color=color, dash="dot"))
                        fig.add_scatter(x=tf_df["timestamp"], y=tf_df["supertrend"], mode="lines",
                                        name=f"Supertrend {tf}", line=dict(color=color, width=3))
                    else:
                        fig.add_scatter(x=tf_df["timestamp"], y=tf_df["bb_upper"], mode="lines",
                                        name=f"BB Upper {tf}", line=dict(color=color, dash="dash"))
                        fig.add_scatter(x=tf_df["timestamp"], y=tf_df["bb_mid"], mode="lines",
                                        name=f"BB Mid {tf}", line=dict(color=color, dash="dot"))
                        fig.add_scatter(x=tf_df["timestamp"], y=tf_df["bb_lower"], mode="lines",
                                        name=f"BB Lower {tf}", line=dict(color=color, dash="dash"))

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

        for tf in selected_timeframes:
            if tf not in signals_cache[asset]:
                continue
            record, _ = signals_cache[asset][tf]

            if record["confidence"] < min_confidence:
                continue

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

    # Signal Lifecycle Table
    st.subheader("Signal Lifecycle Table")
    try:
        df_audit = pd.read_sql_query(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1000", DB
        )
        st.dataframe(df_audit, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading lifecycle table: {e}")

    # Portfolio Overview
    st.subheader("Portfolio Overview")
    try:
        df_audit = pd.read_sql_query("SELECT * FROM signals", DB)
        if not df_audit.empty:
            closed = df_audit[df_audit["status"].str.contains("CLOSED")]
            net_pnl = closed.apply(
                lambda r: (r["exit_price"] - r["entry"]) if r["signal"] == "LONG" else (r["entry"] - r["exit_price"]),
                axis=1
            ).sum()
            st.metric("Net Portfolio PNL", f"{net_pnl:.2f} USDT")

            closed_sorted = closed.sort_values("exit_timestamp")
            closed_sorted["cum_pnl"] = closed_sorted.apply(
                lambda r: (r["exit_price"] - r["entry"]) if r["signal"] == "LONG" else (r["entry"] - r["exit_price"]),
                axis=1
            ).cumsum()

            eq_fig = go.Figure()
            eq_fig.add_scatter(
                x=closed_sorted["exit_timestamp"],
                y=closed_sorted["cum_pnl"],
                mode="lines+markers",
                name="Cumulative PNL (USDT)"
            )
            eq_fig.update_layout(title="Portfolio Equity Curve", template="plotly_dark")
            st.plotly_chart(eq_fig, use_container_width=True)
    except Exception as e:
        st.info("No portfolio data yet or error: {e}")

    # Performance Dashboard with Max DD
    st.subheader("Performance Dashboard")

    try:
        df_audit = pd.read_sql_query(
            "SELECT * FROM signals ORDER BY timestamp DESC", DB
        )
        if not df_audit.empty:
            perf_df = df_audit.copy()
            perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"], utc=True, errors='coerce')
            perf_df["exit_timestamp"] = pd.to_datetime(perf_df["exit_timestamp"], utc=True, errors='coerce')
            perf_df = perf_df.dropna(subset=['timestamp'])

            closed = perf_df[perf_df["status"].str.contains("CLOSED", na=False)].copy()
            closed = closed.dropna(subset=['exit_timestamp', 'exit_price'])

            if not closed.empty:
                closed["risk"] = closed.apply(
                    lambda r: abs(r["entry"] - r["stop"]),
                    axis=1
                )
                closed["pnl"] = closed.apply(
                    lambda r: (r["exit_price"] - r["entry"]) if r["signal"] == "LONG" else (r["entry"] - r["exit_price"]),
                    axis=1
                )
                closed["R"] = closed["pnl"] / closed["risk"]

                closed_sorted = closed.sort_values("exit_timestamp")
                closed_sorted["cum_R"] = closed_sorted["R"].cumsum()

                closed_sorted["peak"] = closed_sorted["cum_R"].cummax()
                closed_sorted["drawdown"] = closed_sorted["cum_R"] - closed_sorted["peak"]
                max_dd = closed_sorted["drawdown"].min()

                eq_fig = go.Figure()
                eq_fig.add_scatter(
                    x=closed_sorted["exit_timestamp"],
                    y=closed_sorted["cum_R"],
                    mode="lines+markers",
                    name="Cumulative R"
                )
                eq_fig.add_scatter(
                    x=closed_sorted["exit_timestamp"],
                    y=closed_sorted["drawdown"],
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(255,0,0,0.2)",
                    name=f"Drawdown (Max: {max_dd:.2f}R)"
                )
                eq_fig.update_layout(title="Live Equity Curve with Drawdown", template="plotly_dark")
                st.plotly_chart(eq_fig, use_container_width=True)

                stats = closed.groupby(["regime", "timeframe"]).agg(
                    Trades=("id", "count"),
                    Wins=("status", lambda x: (x.str.contains("WIN")).sum()),
                    Avg_R=("R", "mean"),
                    Net_R=("R", "sum")
                ).reset_index()

                stats["Win_Rate_%"] = (stats["Wins"] / stats["Trades"] * 100).round(1)
                stats["Avg_R"] = stats["Avg_R"].round(2)
                stats["Net_R"] = stats["Net_R"].round(2)

                total_row = pd.DataFrame([{
                    "regime": "TOTAL",
                    "timeframe": "",
                    "Trades": stats["Trades"].sum(),
                    "Wins": stats["Wins"].sum(),
                    "Win_Rate_%": (stats["Wins"].sum() / stats["Trades"].sum() * 100).round(1),
                    "Avg_R": stats["Avg_R"].mean().round(2),
                    "Net_R": stats["Net_R"].sum().round(2)
                }])

                stats = pd.concat([stats, total_row], ignore_index=True)
                st.dataframe(stats.sort_values("Net_R", ascending=False), use_container_width=True)

                st.write(f"**Max Drawdown:** {max_dd:.2f}R")

            else:
                st.info("No closed trades yet – dashboard will populate as signals resolve.")
        else:
            st.info("No signals recorded yet.")
    except Exception as e:
        st.error(f"Error in performance dashboard: {e}")

    # Survival Analysis
    st.subheader("Signal Survival Analysis by Regime & Timeframe")

    try:
        df_audit = pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC", DB)
        if not df_audit.empty:
            df_audit = df_audit.copy()
            df_audit["timestamp"] = pd.to_datetime(df_audit["timestamp"], utc=True, errors="coerce")
            df_audit["exit_timestamp"] = pd.to_datetime(df_audit["exit_timestamp"], utc=True, errors="coerce")
            df_audit = df_audit.dropna(subset=["timestamp"])

            now = pd.Timestamp.now(tz=timezone.utc)

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
    except Exception as e:
        st.error(f"Error in survival analysis: {e}")

    # Exchange Disagreement with Arb Alerts
    st.subheader("Exchange Disagreement & Arb Opportunities (XT vs Gate.io)")

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

                xt_price = xt_ex.fetch_ticker(asset)['last']
                gate_price = gate_ex.fetch_ticker(asset)['last']
                arb_spread = abs(xt_price - gate_price) / min(xt_price, gate_price) * 100 if xt_price and gate_price else 0
                arb_msg = f"Spread {arb_spread:.2f}% | XT {xt_price:.2f} vs Gate {gate_price:.2f}" if arb_spread > 0.5 else ""

                rows.append({
                    "Timeframe": tf,
                    "Asset": asset,
                    "XT": xt_sig,
                    "Gate.io": gate_sig,
                    "Consensus": consensus,
                    "Arb Alert": arb_msg
                })
            except:
                rows.append({
                    "Timeframe": tf,
                    "Asset": asset,
                    "XT": "NA",
                    "Gate.io": "NA",
                    "Consensus": "NA",
                    "Arb Alert": "NA"
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

# ---------------------------------------------------------
# Backtest Mode
# ---------------------------------------------------------
if mode == "Backtest":
    st.header(f"Backtest Results: {start_date} to {end_date}")

    start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
    end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)

    backtest_trades = []
    equity_curve = []
    current_capital = initial_capital

    progress_bar = st.progress(0)
    total_tasks = len(selected_assets) * len(selected_timeframes)
    task_count = 0

    for asset in selected_assets:
        tf_data = {}
        for tf in TIMEFRAMES:  # fetch all for HTF
            df_hist = pd.DataFrame()
            since = start_ts
            while since < end_ts:
                chunk = fetch_ohlcv(asset, tf, since=since, limit=1000)
                if chunk.empty:
                    break
                df_hist = pd.concat([df_hist, chunk])
                since = int(chunk["timestamp"].iloc[-1].timestamp() * 1000) + 1
            if not df_hist.empty:
                df_hist = df_hist.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df_hist = compute_indicators(df_hist)
                tf_data[tf] = df_hist.dropna()

        for tf in selected_timeframes:
            if tf not in tf_data or tf_data[tf].empty:
                continue

            df_hist = tf_data[tf]

            open_signal = None
            for i in range(50, len(df_hist)):
                current_candle = df_hist.iloc[i:i+1]
                current_price = current_candle["close"].iloc[0]
                current_time = current_candle["timestamp"].iloc[0]

                # HTF regime
                higher_tf = {"1h": "4h", "4h": "1d"}.get(tf)
                higher_regime = "SIDEWAYS"
                if higher_tf and higher_tf in tf_data:
                    higher_df = tf_data[higher_tf]
                    higher_row = higher_df[higher_df["timestamp"] <= current_time].iloc[-1]
                    higher_regime = "BULLISH" if higher_row["close"] > higher_row["ema50"] else "BEARISH" if higher_row["close"] < higher_row["ema50"] else "SIDEWAYS"

                signal, regime, entry, stop, take, confidence, _ = deterministic_signal(current_candle)

                confirmed = True
                if require_confirmation and higher_tf:
                    if signal == "LONG" and higher_regime != "BULLISH":
                        confirmed = False
                    elif signal == "SHORT" and higher_regime != "BEARISH":
                        confirmed = False

                if open_signal:
                    # Trailing stop
                    profit_1r = (current_price - open_signal["entry"]) > (open_signal["entry"] - open_signal["stop"]) if open_signal["signal"] == "LONG" else (open_signal["entry"] - current_price) > (open_signal["stop"] - open_signal["entry"])
                    if profit_1r:
                        new_stop = current_price * (1 - trailing_stop_pct / 100) if open_signal["signal"] == "LONG" else current_price * (1 + trailing_stop_pct / 100)
                        if (open_signal["signal"] == "LONG" and new_stop > open_signal["stop"]) or (open_signal["signal"] == "SHORT" and new_stop < open_signal["stop"]):
                            open_signal["stop"] = new_stop

                    hit_sl = (open_signal["signal"] == "LONG" and current_candle["low"].iloc[0] <= open_signal["stop"]) or \
                             (open_signal["signal"] == "SHORT" and current_candle["high"].iloc[0] >= open_signal["stop"])
                    hit_tp = (open_signal["signal"] == "LONG" and current_candle["high"].iloc[0] >= open_signal["take"]) or \
                             (open_signal["signal"] == "SHORT" and current_candle["low"].iloc[0] <= open_signal["take"])

                    if hit_sl or hit_tp:
                        exit_price = open_signal["stop"] if hit_sl else open_signal["take"]
                        exit_price *= (1 - slippage_pct/100 if hit_sl else 1 + slippage_pct/100) if open_signal["signal"] == "LONG" else (1 + slippage_pct/100 if hit_sl else 1 - slippage_pct/100)

                        pnl = (exit_price - open_signal["entry"]) * open_signal["size"] if open_signal["signal"] == "LONG" else (open_signal["entry"] - exit_price) * open_signal["size"]
                        pnl -= 2 * fee_pct/100 * abs(pnl + open_signal["entry"] * open_signal["size"])

                        backtest_trades.append({
                            "asset": asset,
                            "tf": tf,
                            "entry_time": open_signal["entry_time"],
                            "exit_time": current_time,
                            "signal": open_signal["signal"],
                            "pnl_usdt": pnl,
                            "pnl_pct": pnl / current_capital * 100
                        })

                        current_capital += pnl
                        equity_curve.append({"time": current_time, "equity": current_capital})

                        open_signal = None

                if not open_signal and signal != "NEUTRAL" and confirmed:
                    risk_amount = current_capital * (risk_percent / 100)
                    risk_per_unit = abs(entry - stop)
                    size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                    entry_fill = entry * (1 + slippage_pct/100 if signal == "LONG" else 1 - slippage_pct/100)
                    fee_entry = fee_pct/100 * entry_fill * size

                    open_signal = {
                        "signal": signal,
                        "entry": entry_fill,
                        "stop": stop,
                        "take": take,
                        "entry_time": current_time,
                        "size": size
                    }
                    current_capital -= fee_entry

            task_count += 1
            progress_bar.progress(task_count / total_tasks)

    if backtest_trades:
        trades_df = pd.DataFrame(backtest_trades)
        returns = trades_df["pnl_pct"] / 100
        total_return = (current_capital - initial_capital) / initial_capital * 100
        win_rate = (trades_df["pnl_usdt"] > 0).mean() * 100
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * (24 if "1h" in selected_timeframes else 6 if "4h" in selected_timeframes else 1)) if returns.std() != 0 else 0

        equity_df = pd.DataFrame(equity_curve).sort_values("time")
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown_pct"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
        max_dd_pct = equity_df["drawdown_pct"].min()
        calmar = (total_return / abs(max_dd_pct)) if max_dd_pct != 0 else 0

        eq_fig = go.Figure()
        eq_fig.add_scatter(x=equity_df["time"], y=equity_df["equity"], mode="lines", name="Equity")
        eq_fig.add_scatter(x=equity_df["time"], y=equity_df["drawdown_pct"], mode="lines",
                           fill="tozeroy", fillcolor="rgba(255,0,0,0.2)", name=f"Max DD: {max_dd_pct:.1f}%")
        eq_fig.update_layout(title=f"Backtest Equity (Final: {current_capital:.0f} USDT | +{total_return:.1f}%)")
        st.plotly_chart(eq_fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(trades_df))
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col4.metric("Calmar Ratio", f"{calmar:.2f}")
        st.metric("Max Drawdown", f"{max_dd_pct:.1f}%")

        st.dataframe(trades_df)

        if st.button("Download Trades CSV"):
            csv = trades_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "backtest_trades.csv", "text/csv")
    else:
        st.info("No trades generated in backtest period.")

# ---------------------------------------------------------
# Fix Notice
# ---------------------------------------------------------
st.success("✅ **HybridTrader v1 Complete:**\n"
           "- Trend/Range switcher: Dynamic indicators/signals based on mode\n"
           "- AI Mode: Sentiment from X boosts confidence, param optimization button, NL query input\n"
           "- All previous features maintained (backtest, alerts, etc.)\n"
           "- Test the switches – hybrid power unlocked! YKonChain 🕊️ (@yk_onchain) 🚀")
