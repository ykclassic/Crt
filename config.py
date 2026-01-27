import os
from datetime import timedelta

# Shared DB
DB_FILE = "nexus.db"

# Engines
ENGINES = {
    "core": "NEXUS CORE",
    "hybrid": "NEXUS HYBRID V1",
    "range": "NEXUS RANGEMASTER",
    "ai": "NEXUS AI PREDICT"
}

# ATR settings (consistent across all engines)
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 2.0
RR_RATIO = 1.5

# Timeframes & Assets
DEFAULT_TIMEFRAME = "1h"
TIMEFRAMES = ["1h"]  # List format for the loop in run_alerts
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

# Maintenance
DAYS_TO_KEEP = 30

# Webhook
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Performance
PERFORMANCE_FILE = "performance.json"
MODEL_FILE = "nexus_brain.pkl"
JOURNAL_DB = "nexus_journal.db"

# Execution Mode
DRY_RUN = False  # Set to True to disable real alerts/orders during testing
