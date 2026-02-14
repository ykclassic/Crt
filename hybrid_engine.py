import ccxt
import pandas as pd
from datetime import datetime, timezone
from config import *
from db_utils import init_db, insert_signal

DB_NAME = "hybrid.db"
init_db(DB_NAME)

exchange = getattr(ccxt, EXCHANGE_ID)()

def fetch_df(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    df = pd.DataFrame(
        ohlcv,
        columns=["time", "open", "high", "low", "close", "volume"]
    )
    return df

for symbol in ASSETS:
    try:
        df = fetch_df(symbol, EXECUTION_TF)

        df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

        price = df["close"].iloc[-1]
        direction = None

        if df["ema8"].iloc[-1] > df["ema21"].iloc[-1]:
            direction = "LONG"
        elif df["ema8"].iloc[-1] < df["ema21"].iloc[-1]:
            direction = "SHORT"

        if direction:
            sl = price * (1 - RISK_PERCENT) if direction == "LONG" else price * (1 + RISK_PERCENT)
            tp = price * (1 + REWARD_PERCENT) if direction == "LONG" else price * (1 - REWARD_PERCENT)

            insert_signal(DB_NAME, (
                symbol,
                direction,
                price,
                sl,
                tp,
                datetime.now(timezone.utc).isoformat()  # ✅ fixed
            ))

    except Exception as e:
        print(f"Hybrid error {symbol}: {e}")
