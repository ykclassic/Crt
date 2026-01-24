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

# Timeframes (all engines use 1h for now)
DEFAULT_TIMEFRAME = "1h"

# Maintenance
DAYS_TO_KEEP = 30

# Webhook
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Performance
PERFORMANCE_FILE = "performance.json"
MODEL_FILE = "nexus_brain.pkl"
JOURNAL_DB = "nexus_journal.db"
