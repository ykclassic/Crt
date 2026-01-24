import pandas as pd
import numpy as np
import ccxt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]
TIMEFRAME = "1h"
LIMIT = 2000  # ~3 months per asset
PREDICTION_HORIZON = 6  # Hours ahead for target

def fetch_and_prepare_data():
    ex = ccxt.xt({"enableRateLimit": True})
    all_dfs = []

    for asset in ASSETS:
        logging.info(f"Fetching {LIMIT} candles for {asset}...")
        try:
            ohlcv = ex.fetch_ohlcv(asset, TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['asset'] = asset
            all_dfs.append(df)
        except Exception as e:
            logging.error(f"Fetch failed for {asset}: {e}")

    if not all_dfs:
        raise ValueError("No data fetched")

    data = pd.concat(all_dfs)
    data = data.sort_values('ts')

    # Features
    delta = data['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    data['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))

    data['ema20'] = data['close'].ewm(span=20).mean()
    data['dist_ema'] = (data['close'] - data['ema20']) / data['close'] * 100

    data['vol_change'] = data['volume'].pct_change()

    tr = pd.concat([data['high'] - data['low'],
                    (data['high'] - data['close'].shift()).abs(),
                    (data['low'] - data['close'].shift()).abs()], axis=1).max(axis=1)
    data['atr'] = tr.rolling(14).mean()
    data['atr_norm'] = data['atr'] / data['close'] * 100

    # Lagged returns for momentum
    for lag in [1, 3, 6]:
        data[f'return_lag_{lag}'] = data['close'].pct_change(lag) * 100

    # Target: price up in next N hours?
    data['target'] = (data['close'].shift(-PREDICTION_HORIZON) > data['close']).astype(int)

    data = data.dropna()
    return data

def train_model():
    logging.info("Preparing multi-asset dataset...")
    df = fetch_and_prepare_data()

    features = ['rsi', 'vol_change', 'dist_ema', 'atr_norm',
                 'return_lag_1', 'return_lag_3', 'return_lag_6']
    X = df[features]
    y = df['target']

    # Chronological split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    logging.info("Training XGBoost model...")
    model.fit(X_train_scaled, y_train)

    # Evaluation
    preds = model.predict(X_test_scaled)
    logging.info("Test Set Report:\n" + classification_report(y_test, preds))

    # Cross-validation (time-series aware)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='f1')
    logging.info(f"CV F1 Scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Save model + scaler
    with open("nexus_brain.pkl", "wb") as f:
        pickle.dump((model, scaler), f)

    logging.info("🧠 Enhanced XGBoost model trained and saved (multi-asset, richer features).")

if __name__ == "__main__":
    train_model()
