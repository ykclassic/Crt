import sqlite3
import pandas as pd
import numpy as np
import pickle
import logging
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from config import DB_FILE, MODEL_FILE, VOTING_METHOD

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def train_ensemble_model():
    logging.info("--- PHASE 3: ROBUST ENSEMBLE TRAINING ---")
    
    if not os.path.exists(DB_FILE):
        logging.error("Database not found.")
        return

    conn = sqlite3.connect(DB_FILE)
    # We only train on audited signals (Phase 5 requirement)
    df = pd.read_sql_query("SELECT rsi, vol_change, dist_ema, result FROM signals WHERE result IS NOT NULL", conn)
    conn.close()

    if len(df) < 5:
        logging.warning("Insufficient audited data to train. Need at least 5 concluded trades.")
        return

    # Prepare features and labels
    X = df[['rsi', 'vol_change', 'dist_ema']]
    y = df['result']

    # 1. Define Base Estimators
    clf1 = GradientBoostingClassifier(n_estimators=100)
    clf2 = RandomForestClassifier(n_estimators=100)
    clf3 = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')

    # 2. Define the Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[('gb', clf1), ('rf', clf2), ('xgb', clf3)],
        voting=VOTING_METHOD
    )

    # 3. Create a Pipeline that handles Missing Values (NaNs)
    # This fixes the "Instance by using an imputer" error
    brain_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Fill NaNs with the middle value
        ('scaler', StandardScaler()),                   # Normalize data
        ('model', ensemble)                            # The Committee
    ])

    # 4. Fit the robust model
    brain_pipeline.fit(X, y)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(brain_pipeline, f)
    
    logging.info(f"SUCCESS: Ensemble trained on {len(df)} samples with NaN protection.")

if __name__ == "__main__":
    train_ensemble_model()
