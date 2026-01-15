import ccxt
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timezone

# ============================
# CONFIGURATION
# ============================

EXCHANGES = {
    "bitget": ccxt.bitget(),
    "gateio": ccxt.gateio(),
    "xt": ccxt.xt()
}

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = {
    "entry": "1h",
    "confirm_1": "4h",
    "confirm_2": "1d"
}

SCHEDULER_INTERVAL_SECONDS = 60
CONFIDENCE_THRESHOLD = 0.65
MAX_BARS = 200

# ============================
# HARDENING
# ============================

for ex in EXCHANGES.values():
    ex.enableRateLimit = True
    ex.timeout = 20000

# ============================
# DATA FETCHING
# ============================

def fetch_ohlcv(exchange, symbol, timeframe):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=MAX_BARS)
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
    except Exception as e:
        print(f"[DATA ERROR] {exchange.id} {symbol} {timeframe}: {e}")
        return None

# ============================
# TECHNICAL STRUCTURE
# ============================

def compute_structure(df):
    df["ema_fast"] = df["close"].ewm(span=20).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["rsi"] = 100 - (
        100 / (1 + df["close"].pct_change().rolling(14).mean())
    )
    return df

def trend_bias(df):
    if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]:
        return 1
    elif df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]:
        return -1
    return 0

# ============================
# ML CONFIDENCE WEIGHTING
# ============================

def ml_confidence(entry_df, confirm_4h, confirm_1d):
    """
    Deterministic logistic-style confidence scoring
    NOT a trained model
    """

    weights = {
        "trend_alignment": 0.4,
        "momentum": 0.3,
        "volatility": 0.3
    }

    entry_trend = trend_bias(entry_df)
    c4_trend = trend_bias(confirm_4h)
    c1d_trend = trend_bias(confirm_1d)

    trend_alignment = 1 if entry_trend == c4_trend == c1d_trend else 0

    momentum = min(
        max((entry_df["rsi"].iloc[-1] - 50) / 50, -1),
        1
    )

    volatility = entry_df["close"].pct_change().std()
    volatility_score = 1 - min(volatility * 10, 1)

    raw_score = (
        weights["trend_alignment"] * trend_alignment +
        weights["momentum"] * abs(momentum) +
        weights["volatility"] * volatility_score
    )

    confidence = 1 / (1 + np.exp(-5 * (raw_score - 0.5)))
    return round(confidence, 4)

# ============================
# SIGNAL ENGINE
# ============================

def generate_signal(exchange, symbol):
    entry = fetch_ohlcv(exchange, symbol, TIMEFRAMES["entry"])
    c4 = fetch_ohlcv(exchange, symbol, TIMEFRAMES["confirm_1"])
    c1d = fetch_ohlcv(exchange, symbol, TIMEFRAMES["confirm_2"])

    if entry is None or c4 is None or c1d is None:
        return None

    entry = compute_structure(entry)
    c4 = compute_structure(c4)
    c1d = compute_structure(c1d)

    entry_bias = trend_bias(entry)

    if entry_bias == 0:
        return None

    confidence = ml_confidence(entry, c4, c1d)

    if confidence < CONFIDENCE_THRESHOLD:
        return None

    direction = "LONG" if entry_bias == 1 else "SHORT"

    return {
        "exchange": exchange.id,
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ============================
# AUTO-SCHEDULER
# ============================

last_signal_cache = {}

def scheduler_loop():
    while True:
        for ex in EXCHANGES.values():
            for symbol in SYMBOLS:
                key = f"{ex.id}-{symbol}"

                signal = generate_signal(ex, symbol)
                if signal:
                    last_time = last_signal_cache.get(key)
                    current_time = signal["timestamp"]

                    if last_time != current_time:
                        last_signal_cache[key] = current_time
                        print("🔥 SIGNAL:", signal)

        time.sleep(SCHEDULER_INTERVAL_SECONDS)

# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    print("🚀 Signal Engine Started (ML Confidence + Auto Scheduler)")
    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()

    while True:
        time.sleep(1)
