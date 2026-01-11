import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# -----------------------------
# 1️⃣ ATR Function
# -----------------------------
def atr(df, n=14):
    if df.empty:
        return pd.Series(dtype=float)
    tr = np.maximum(
        df["h"] - df["l"],
        np.maximum(abs(df["h"] - df["c"].shift()), abs(df["l"] - df["c"].shift()))
    )
    return tr.rolling(n).mean()

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------
def features(df):
    if df.empty:
        return df
    df = df.copy()
    df["ema9"] = df["c"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["c"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
    df["atr"] = atr(df)
    df["target"] = df["c"].shift(-1)
    return df.dropna()

# -----------------------------
# 3️⃣ Version-Safe Exchange Loader
# -----------------------------
def get_exchange_class(exchange_names):
    """
    Tries multiple class names and returns the first available ccxt.Exchange class.
    """
    for name in exchange_names:
        try:
            return getattr(ccxt, name.lower())
        except AttributeError:
            continue
    return None

EXCHANGE_LOOKUP = {
    "XT": ["xt"],
    "Bitget": ["bitget"],
    "Gate.io": ["gateio"],
    "KuCoin": ["kucoin"],
    "Huobi": ["huobi", "huobipro"]
}

EXCHANGE_CLASSES = {}
for name, possible_names in EXCHANGE_LOOKUP.items():
    ex_class = get_exchange_class(possible_names)
    if ex_class:
        EXCHANGE_CLASSES[name] = ex_class
    else:
        st.warning(f"⚠️ {name} exchange not found in your CCXT version. Skipping.")

# -----------------------------
# 4️⃣ Fetch OHLCV
# -----------------------------
def safe_fetch(symbol, exchange_name, timeframe='1h', limit=200):
    if exchange_name not in EXCHANGE_CLASSES:
        st.error(f"Exchange {exchange_name} not loaded correctly")
        return pd.DataFrame()
    try:
        ex = EXCHANGE_CLASSES[exchange_name]({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
        df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.warning(f"{symbol} {timeframe} fetch failed on {exchange_name}: {e}")
        return pd.DataFrame(columns=['ts','o','h','l','c','v','dt'])

# -----------------------------
# 5️⃣ Ensemble Signal (RF + EMA)
# -----------------------------
def ensemble_signal(df):
    if df.empty:
        return 0, 0
    df_train = df.dropna()
    feature_cols = ['c', 'v', 'ema9', 'ema21', 'atr']
    if df_train.empty or any(col not in df_train.columns for col in feature_cols):
        return 0, 0
    X = df_train[feature_cols]
    y = df_train['target']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    last_features = X.iloc[-1].values.reshape(1, -1)
    pred_price = model.predict(last_features)[0]
    ema_diff = abs(df['ema9'].iloc[-1] - df['ema21'].iloc[-1]) / df['ema21'].iloc[-1]
    confidence = min(98.5, 75 + (ema_diff * 500))
    return pred_price, confidence

# -----------------------------
# 6️⃣ Build Signal
# -----------------------------
def build_signal(df, pred_price, conf, asset, exchange, timeframe_label):
    if df.empty:
        return None
    return {
        "asset": asset,
        "exchange": exchange,
        "timeframe": timeframe_label,
        "bias": "LONG" if pred_price > df["c"].iloc[-1] else "SHORT",
        "price": df["c"].iloc[-1],
        "objective": pred_price,
        "invalidation": df["c"].iloc[-1] * 0.995,
        "confidence": conf,
        "regime": "RF+EMA Ensemble"
    }

# -----------------------------
# 7️⃣ Assets
# -----------------------------
ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT",
    "ADA/USDT","LINK/USDT","TRX/USDT","SUI/USDT","PEPE/USDT",
    "BNB/USDT","MATIC/USDT","LTC/USDT","AVAX/USDT","UNI/USDT",
    "ATOM/USDT","NEAR/USDT","FTM/USDT","ALGO/USDT","VET/USDT"
]

# -----------------------------
# 8️⃣ Streamlit Page
# -----------------------------
st.set_page_config(page_title="All-Exchange AI Signal Scanner", layout="wide")
st.title("🧠 All-Exchange AI Signal Dashboard")
st.markdown("Scans all exchanges for all assets and returns the best-qualified signals.")

# -----------------------------
# 9️⃣ Scan All Exchanges Mode
# -----------------------------
signals = []

for asset in ASSETS:
    best_signal = None
    best_conf = 0

    for exchange_name in EXCHANGE_CLASSES.keys():
        df_1h = features(safe_fetch(asset, exchange_name, "1h"))
        df_4h = features(safe_fetch(asset, exchange_name, "4h"))

        if df_1h.empty or df_4h.empty:
            continue

        pred_1h, conf_1h = ensemble_signal(df_1h)
        pred_4h, conf_4h = ensemble_signal(df_4h)
        dir_1h = "LONG" if pred_1h > df_1h["c"].iloc[-1] else "SHORT"
        dir_4h = "LONG" if pred_4h > df_4h["c"].iloc[-1] else "SHORT"

        reason = ""
        if conf_1h <= 60:
            reason = f"1H confidence too low ({conf_1h:.2f}%)"
        elif conf_4h <= 60:
            reason = f"4H confidence too low ({conf_4h:.2f}%)"
        elif dir_1h != dir_4h:
            reason = f"Direction mismatch (1H={dir_1h},4H={dir_4h})"

        if reason:
            continue

        avg_conf = (conf_1h + conf_4h)/2
        if avg_conf > best_conf:
            best_conf = avg_conf
            best_signal = build_signal(df_1h, pred_1h, avg_conf, asset, exchange_name, "1H (4H Confirmed)")

    if best_signal:
        signals.append(best_signal)
    else:
        st.info(f"{asset}: No qualified signal on any exchange.")

# -----------------------------
# 10️⃣ Display Signals & Charts
# -----------------------------
if signals:
    st.subheader("📈 Best-Qualified Signals Across Exchanges")
    st.dataframe(pd.DataFrame(signals))

    for sig in signals:
        asset = sig['asset']
        exchange_name = sig['exchange']
        df_1h = features(safe_fetch(asset, exchange_name, "1h"))
        if df_1h.empty:
            continue

        current_price = df_1h["c"].iloc[-1]
        future_dt = [df_1h["dt"].iloc[-1] + timedelta(hours=i) for i in range(5)]
        future_prices = [current_price + ((sig['objective'] - current_price)/4)*i for i in range(5)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_1h["dt"].tail(30), y=df_1h["c"].tail(30),
                                 name="Actual Price", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(x=future_dt, y=future_prices,
                                 name="Predicted Move", line=dict(color="orange", dash="dash")))
        fig.update_layout(title=f"{asset} | {exchange_name} | Bias: {sig['bias']} | Confidence: {sig['confidence']:.2f}%",
                          template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No qualified signals under current market conditions on any exchange.")
