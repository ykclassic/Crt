import os

# --- DATABASE & FILES ---
DB_FILE = "nexus.db"
MODEL_FILE = "nexus_brain.pkl"
PERFORMANCE_FILE = "performance.json"

# --- API & NOTIFICATIONS ---
# Replace with your actual Discord Webhook URL or use Environment Variables
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "YOUR_DISCORD_WEBHOOK_HERE")

# --- ASSET SETTINGS ---
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
DEFAULT_TIMEFRAME = "4h"  # Used by Core Engine for macro trend

# --- ENGINE REGISTRY ---
ENGINES = {
    "core": "Nexus Core",
    "ai": "Nexus AI (Phase 1)",
    "hybrid_v1": "Nexus Hybrid",
    "rangemaster": "Nexus Rangemaster"
}

# --- PERFORMANCE & KILL-SWITCH SETTINGS ---
# Thresholds for the Auditor to disable/enable engines
KILL_THRESHOLD = 45.0      # Disable engine if Win Rate falls below 45%
RECOVERY_THRESHOLD = 52.0  # Re-enable engine if Win Rate climbs back to 52%
MIN_TRADES_FOR_AUDIT = 5   # Minimum trades before the Kill-Switch can trigger

# --- PHASE 1: DYNAMIC RISK MANAGEMENT ---
TOTAL_CAPITAL = 1000.0        # Your total trading bankroll in USDT
RISK_PER_TRADE = 0.02         # Risk 2% of capital per trade ($20 per trade on $1000)
ATR_MULTIPLIER = 2.0          # Stop Loss = 2.0 * Average True Range (Volatility)
RR_RATIO = 2.0                # Risk:Reward Ratio (Target 2x your risk)
MIN_CONFIDENCE_FOR_SIZE_BOOST = 75.0  # AI confidence required to increase size

# --- FEATURE SETTINGS (For AI Training) ---
ATR_PERIOD = 14               # Standard period for volatility calculation
EMA_PERIOD = 20               # Trend baseline for AI distance calculation
RSI_PERIOD = 14               # Standard momentum period
