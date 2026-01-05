import streamlit as st
import ccxt
import pandas as pd
import plotly.express as px
import ta
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="CryptoForge ML", layout="wide")
st.title("🔮 CryptoForge ML – Signals + LSTM Prediction")

symbols = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT']
timeframe_options = ['5m', '15m', '1h', '4h', '1d']

symbol = st.sidebar.selectbox("Cryptocurrency", symbols)
timeframe = st.sidebar.selectbox("Timeframe", timeframe_options)
horizon_days = st.sidebar.selectbox("Prediction Horizon (days)", [1, 7, 30])

# ----------------------------
# DATA FETCHING (XT)
# ----------------------------

@st.cache_data(ttl=300)
def fetch_data(symbol, timeframe, limit=2000):
    try:
        exchange = ccxt.xt({
            "apiKey": "e2613e3f-1a0d-4df1-b1e2-b03c09ee20c9",
            "secret": "289116468a1562cd7a553e43b44baa54e1a8be83",
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot"
            }
        })

        # Explicit market load
        exchange.load_markets()

        if symbol not in exchange.symbols:
            raise ValueError(f"Symbol {symbol} not supported on XT")

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    except Exception:
        # Explicitly fail without fallback
        return None


df = fetch_data(symbol, timeframe)

# ---- HARD FAIL IF XT NOT REACHABLE ----
if df is None or df.empty:
    st.error("❌ XT not reachable")
    st.stop()

# ----------------------------
# INDICATORS
# ----------------------------

def add_indicators(df):
    df["ma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["bb_high"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
    df["bb_low"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"]
    ).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"]
    ).adx()
    return df


df = add_indicators(df)

# ----------------------------
# SIGNAL LOGIC
# ----------------------------

def generate_signal(row):
    buy_score = sell_score = 0

    if row["close"] > row["ma_20"]:
        buy_score += 1
    else:
        sell_score += 1

    if row["rsi"] < 30:
        buy_score += 2
    if row["rsi"] > 70:
        sell_score += 2

    if row["macd"] > 0:
        buy_score += 1
    else:
        sell_score += 1

    if row["close"] < row["bb_low"]:
        buy_score += 1
    if row["close"] > row["bb_high"]:
        sell_score += 1

    if row["adx"] > 25:
        buy_score += 1

    if buy_score > sell_score + 1:
        return "STRONG BUY"
    elif buy_score > sell_score:
        return "BUY"
    elif sell_score > buy_score + 1:
        return "STRONG SELL"
    elif sell_score > buy_score:
        return "SELL"
    else:
        return "NEUTRAL"


latest = df.iloc[-1]
signal = generate_signal(latest)

# ----------------------------
# DISPLAY
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Price & Signal")
    st.metric("Price", f"${latest['close']:,.2f}")

    color = "green" if "BUY" in signal else "red" if "SELL" in signal else "gray"
    st.markdown(
        f"<h2 style='color:{color}; text-align:center;'>{signal}</h2>",
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("Price Chart")
    fig = px.line(df, y="close", title=f"{symbol} – {timeframe}")
    fig.add_scatter(y=df["ma_20"], name="MA20")
    fig.add_scatter(y=df["bb_high"], name="BB High")
    fig.add_scatter(y=df["bb_low"], name="BB Low")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# LSTM MODEL
# ----------------------------

@st.cache_resource
def train_lstm_model(_df):
    data = _df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    seq_len = 60
    X, y = [], []

    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    return model, scaler


model, scaler = train_lstm_model(df)

if st.button("Predict Future Price"):
    last_seq = scaler.transform(df["close"][-60:].values.reshape(-1, 1))
    pred_scaled = model.predict(last_seq.reshape(1, 60, 1), verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    st.success(
        f"Predicted {symbol} price in {horizon_days} days: **${pred_price:,.2f}**"
    )

    st.info("⚠️ LSTM predictions are speculative and not financial advice.")
