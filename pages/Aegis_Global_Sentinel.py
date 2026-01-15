import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================
# STREAMLIT CONFIG
# ============================

st.set_page_config(
    page_title="Aegis Sentinel Pro",
    layout="wide"
)

st.title("📡 Aegis Sentinel – Production Signal Engine")
st.caption("1H Entry • 4H / 1D Confirmation • ML Confidence Weighting")

# ============================
# CONFIGURATION
# ============================

EXCHANGES = {
    "Bitget": ccxt.bitget(),
    "Gate.io": ccxt.gateio(),
    "XT": ccxt.xt()
}

SYMBOLS = ["BTC/USDT", "ETH/USDT"]

TIMEFRAMES = {
    "entry": "1h",
    "confirm_4h": "4h",
    "confirm_1d": "1d"
}

CONFIDENCE_THRESHOLD = 0.65
MAX_BARS = 200
REFRESH_SECONDS = 60

# ============================
# HARDEN CCXT
# ============================

for ex in EXCHANGES.values():
    ex.enableRateLimit = True
    ex.timeout = 20000

# ============================
# DATA FETCHING
# ============================

@st.cache_data(ttl=300)
def fetch_ohlcv(_exchange, symbol, timeframe):
    """
    _exchange: CCXT exchange object (leading underscore to avoid Streamlit hashing issue)
    """
    data = _exchange.fetch_ohlcv(symbol, timeframe, limit=MAX_BARS)
    df = pd.DataFrame(
        data,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# ============================
# TECHNICAL STRUCTURE
# ============================

def compute_structure(df):
    df["ema_fast"] = df["close"].ewm(span=20).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["rsi"] = 100 - (100 / (1 + df["close"].pct_change().rolling(14).mean()))
    return df

def trend_bias(df):
    if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]:
        return 1
    elif df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]:
        return -1
    return 0

# ============================
# ML CONFIDENCE
# ============================

def ml_confidence(entry_df, confirm_4h, confirm_1d):
    """
    Deterministic logistic-style confidence
    No external ML, fully reproducible
    """
    weights = {
        "trend_alignment": 0.4,
        "momentum": 0.3,
        "volatility": 0.3
    }

    # Trend alignment
    trends = [trend_bias(entry_df), trend_bias(confirm_4h), trend_bias(confirm_1d)]
    trend_alignment = 1 if len(set(trends)) == 1 else 0

    # Momentum (RSI distance from 50)
    momentum = min(max((entry_df["rsi"].iloc[-1] - 50)/50, -1), 1)

    # Volatility
    volatility = entry_df["close"].pct_change().std()
    volatility_score = 1 - min(volatility*10, 1)

    raw = (
        weights["trend_alignment"]*trend_alignment +
        weights["momentum"]*abs(momentum) +
        weights["volatility"]*volatility_score
    )

    confidence = 1 / (1 + np.exp(-5*(raw-0.5)))
    return round(confidence, 4)

# ============================
# SIGNAL ENGINE
# ============================

def generate_signals():
    signals = []

    for ex_name, ex in EXCHANGES.items():
        for symbol in SYMBOLS:
            try:
                entry_df = compute_structure(fetch_ohlcv(ex, symbol, TIMEFRAMES["entry"]))
                confirm_4h = compute_structure(fetch_ohlcv(ex, symbol, TIMEFRAMES["confirm_4h"]))
                confirm_1d = compute_structure(fetch_ohlcv(ex, symbol, TIMEFRAMES["confirm_1d"]))

                bias = trend_bias(entry_df)
                if bias == 0:
                    continue

                confidence = ml_confidence(entry_df, confirm_4h, confirm_1d)
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                signals.append({
                    "Exchange": ex_name,
                    "Pair": symbol,
                    "Direction": "LONG" if bias == 1 else "SHORT",
                    "ML Confidence": confidence,
                    "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                })

            except Exception as e:
                st.warning(f"{ex_name} {symbol}: {str(e)}")

    return signals

# ============================
# UI LOOP
# ============================

placeholder = st.empty()

while True:
    with placeholder.container():
        st.subheader("📈 Live Signals")

        signals = generate_signals()
        if signals:
            df = pd.DataFrame(signals)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No valid signals at this time.")

        st.caption(f"Auto-refresh every {REFRESH_SECONDS} seconds")

    time.sleep(REFRESH_SECONDS)
