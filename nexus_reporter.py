import pickle
import os
import logging
from config import MODEL_FILE, WEBHOOK_URL
import requests

def get_ai_narrative():
    if not os.path.exists(MODEL_FILE):
        return "AI is currently in 'Observation Mode' (No brain file found)."

    try:
        with open(MODEL_FILE, "rb") as f:
            model, _ = pickle.load(f)
        
        # Features index must match your train_brain.py: [RSI, Vol_Change, Dist_EMA]
        importances = model.feature_importances_
        traits = {
            "Momentum (RSI)": importances[0],
            "Volatility (Volume)": importances[1],
            "Trend (EMA Distance)": importances[2]
        }
        
        # Sort to find the dominant trait
        top_trait = max(traits, key=traits.get)
        
        # Natural Language Templates
        if top_trait == "Momentum (RSI)":
            narrative = "The AI is currently **Aggressive**. It has learned that RSI extremes are the best predictors for wins in this market."
        elif top_trait == "Volatility (Volume)":
            narrative = "The AI is currently **Reactive**. It is prioritizing high-volume spikes over price patterns to filter out fakeouts."
        else:
            narrative = "The AI is currently **Conservative**. It is focusing on 'Mean Reversion' (EMA Distance), waiting for prices to overextend before signaling."

        return f"🧠 **AI Intelligence Update**:\n> {narrative}\n> *Top Priority: {top_trait} ({traits[top_trait]:.1%})*"

    except Exception as e:
        return f"⚠️ Could not interpret AI brain: {e}"

def send_sunday_report():
    ai_status = get_ai_narrative()
    full_report = f"📊 **NEXUS WEEKLY STRATEGIC BRIEF**\n\n{ai_status}\n\n..."
    # Existing reporting logic here
    requests.post(WEBHOOK_URL, json={"content": full_report})

if __name__ == "__main__":
    send_sunday_report()
