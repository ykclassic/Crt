import ccxt
import pandas as pd
from datetime import datetime, timezone
from config import *
from db_utils import init_db, insert_signal

# Validate exchange support in the current environment before execution
if EXCHANGE_ID not in ccxt.exchanges:
    raise EnvironmentError(
        f"Exchange '{EXCHANGE_ID}' is not supported by the currently installed CCXT version ({ccxt.__version__}). "
        f"Please run 'pip install --upgrade ccxt' in your GitHub Actions runner or update requirements.txt."
    )

init_db(DB_FILE)
exchange = getattr(ccxt, EXCHANGE_ID)()

def fetch_df(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    return df

def ema(series, period):
    return series.ewm(span=period).mean()

for symbol in ASSETS:
    try:
        df_1h = fetch_df(symbol, EXECUTION_TF)
        df_4h = fetch_df(symbol, CONFIRM_TF_1)
        df_1d = fetch_df(symbol, CONFIRM_TF_2)

        df_1h["ema20"] = ema(df_1h["close"], 20)
        df_4h["ema50"] = ema(df_4h["close"], 50)
        df_1d["ema50"] = ema(df_1d["close"], 50)

        price = df_1h["close"].iloc[-1]
        direction = None

        if price > df_1h["ema20"].iloc[-1] and \
           df_4h["close"].iloc[-1] > df_4h["ema50"].iloc[-1] and \
           df_1d["close"].iloc[-1] > df_1d["ema50"].iloc[-1]:
            direction = "LONG"

        if price < df_1h["ema20"].iloc[-1] and \
           df_4h["close"].iloc[-1] < df_4h["ema50"].iloc[-1] and \
           df_1d["close"].iloc[-1] < df_1d["ema50"].iloc[-1]:
            direction = "SHORT"

        if direction:
            sl = price * (1 - RISK_PERCENT) if direction == "LONG" else price * (1 + RISK_PERCENT)
            tp = price * (1 + REWARD_PERCENT) if direction == "LONG" else price * (1 - REWARD_PERCENT)

            insert_signal(DB_FILE, {
                "engine_id": "CORE_ENGINE",
                "symbol": symbol,
                "timeframe": EXECUTION_TF,
                "direction": direction,
                "entry": price,
                "stop_loss": sl,
                "take_profit": tp,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            })

    except Exception as e:
        print(f"Core error {symbol}: {e}")
