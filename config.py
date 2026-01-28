import os

# --- DATABASE & STORAGE ---
DB_FILE = "nexus.db"
MODEL_FILE = "nexus_brain.pkl"
PERFORMANCE_FILE = "performance.json"
DAYS_TO_KEEP = 7  # Maintenance cleanup threshold

# --- NOTIFICATIONS ---
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- ASSETS & TIMEFRAMES ---
ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
DEFAULT_TIMEFRAME = '1h'
TIMEFRAMES = ['15m', '1h', '4h']

# --- ENGINE REGISTRY ---
# Must match the STRATEGY_ID in each engine file for the auditor to work
ENGINES = {
    "core": "Nexus Core",
    "ai": "Nexus AI Predictor",
    "hybrid_v1": "Nexus Hybrid V1",
    "rangemaster": "Nexus Rangemaster"
}

# --- STRATEGY PARAMETERS ---
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 1.5
RR_RATIO = 2.0

# --- PERFORMANCE AUDIT THRESHOLDS ---
KILL_THRESHOLD = 40.0      # Disable engine if Win Rate falls below this %
RECOVERY_THRESHOLD = 50.0  # Re-enable engine if Win Rate recovers to this %
