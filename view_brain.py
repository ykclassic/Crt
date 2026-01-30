import pickle
import pandas as pd
import logging

MODEL_FILE = "nexus_brain.pkl"

def inspect_brain():
    try:
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
        
        print("--- NEXUS BRAIN INSPECTION ---")
        print(f"Model Type: {type(model).__name__}")
        
        # Check Feature Importances
        # Features are: [RSI, Vol_Change, Dist_EMA]
        features = ["RSI", "Volume Change", "EMA Distance"]
        importances = model.feature_importances_
        
        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)
        
        print("\nWhat the AI has learned to prioritize:")
        print(feat_df.to_string(index=False))
        
    except FileNotFoundError:
        print("Error: nexus_brain.pkl not found. Has the training run yet?")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_brain()
