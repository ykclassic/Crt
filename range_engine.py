import ccxt
import pandas as pd
from datetime import datetime, timezone
datetime.now(timezone.utc).isoformat()
from config import *
from db_utils import init_db, insert_signal

DB_NAME = "range.db"
init_db(DB_NAME)

exchange = getattr(ccxt, EXCHANGE_ID)()

def fetch_df(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    return df

for symbol in ASSETS:
    try:
        df = fetch_df(symbol, EXECUTION_TF)
        df["ma20"] = df["close"].rolling(20).mean()
        df["std"] = df["close"].rolling(20).std()
        df["upper"] = df["ma20"] + 2 * df["std"]
        df["lower"] = df["ma20"] - 2 * df["std"]

        price = df["close"].iloc[-1]
        direction = None

        if price <= df["lower"].iloc[-1]:
            direction = "LONG"

        if price >= df["upper"].iloc[-1]:
            direction = "SHORT"

        if direction:
            sl = price * (1 - RISK_PERCENT) if direction == "LONG" else price * (1 + RISK_PERCENT)
            tp = price * (1 + REWARD_PERCENT) if direction == "LONG" else price * (1 - REWARD_PERCENT)

            insert_signal(DB_NAME, (
                symbol, direction, price, sl, tp,
                datetime.utcnow().isoformat()
            ))

    except Exception as e:
        print(f"Range error {symbol}: {e}")
