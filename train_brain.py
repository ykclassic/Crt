import sqlite3
import pandas as pd
import numpy as np
import ccxt
import pickle
import logging
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier # Ensure 'pip install xgboost' is run
from datetime import datetime
from config import DB_FILE, MODEL_FILE, VOTING_METHOD

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def train_ensemble_model():
    logging.info("--- PHASE 3: ENSEMBLE TRAINING INITIALIZED ---")
    
    if not os.path.exists(DB_FILE):
        logging.error("No database found. AI cannot learn.")
        return

    conn = sqlite3.connect(DB_FILE)
    df_signals = pd.read_sql_query("SELECT asset, timeframe, signal, entry, sl, tp, ts FROM signals", conn)
    conn.close()

    if len(df_signals) < 10:
        logging.warning(f"Insufficient data ({len(df_signals)}). Need 10+ for Ensemble.")
        return

    ex = ccxt.gateio()
    training_data, labels = [], []

    for _, row in df_signals.iterrows():
        try:
            ts_ms = int(datetime.fromisoformat(row['ts']).timestamp() * 1000)
            # Context Features
            ohlcv = ex.fetch_ohlcv(row['asset'], row['timeframe'], since=ts_ms - (20 * 60 * 60 * 1000), limit=30)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            rsi = (100 - (100 / (1 + df['c'].diff().clip(lower=0).rolling(14).mean() / -df['c'].diff().clip(upper=0).rolling(14).mean()))).iloc[-1]
            dist_ema = (df['c'].iloc[-1] - df['c'].ewm(span=20).mean().iloc[-1]) / df['c'].iloc[-1]
            vol_change = df['v'].pct_change().iloc[-1]

            # Outcome Label
            outcome_ohlcv = ex.fetch_ohlcv(row['asset'], row['timeframe'], since=ts_ms, limit=50)
            won = 0
            for candle in outcome_ohlcv:
                if row['signal'] == "LONG":
                    if candle[2] >= row['tp']: won = 1; break
                    if candle[3] <= row['sl']: won = 0; break
                else:
                    if candle[3] <= row['tp']: won = 1; break
                    if candle[2] >= row['sl']: won = 0; break
            
            training_data.append([rsi, vol_change, dist_ema])
            labels.append(won)
        except: continue

    X, y = np.array(training_data), np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # CREATE THE ENSEMBLE
    clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=5)
    clf3 = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')

    ensemble = VotingClassifier(
        estimators=[('gb', clf1), ('rf', clf2), ('xgb', clf3)],
        voting=VOTING_METHOD
    )
    
    ensemble.fit(X_scaled, y)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((ensemble, scaler), f)
    
    logging.info(f"PHASE 3 SUCCESS: Ensemble trained with 3 models on {len(X)} samples.")

if __name__ == "__main__":
    train_ensemble_model()
