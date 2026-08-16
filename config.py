import os
from pathlib import Path

# =========================
# Exchange
# =========================
EXCHANGE_ID = "xt"

# =========================
# Assets
# =========================
ASSETS = os.getenv(
    "ASSETS",
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT",
).split(",")

TRADING_PAIRS = ASSETS

# =========================
# Timeframes
# =========================
EXECUTION_TF = "1h"
CONFIRM_TF_1 = "4h"
CONFIRM_TF_2 = "1d"

TIMEFRAMES = [EXECUTION_TF, CONFIRM_TF_1, CONFIRM_TF_2]

# =========================
# Risk Model
# =========================
RISK_PERCENT = 0.02
REWARD_PERCENT = 0.05

# =========================
# Database Ownership
# =========================
# nexus_signals.db is pipeline-owned. Governance must treat it as read-only.
DB_FILE = os.getenv("DB_FILE", "nexus_signals.db")

# nexus_governance.db is governance-owned and is the only database that
# alert_monitor.py and audit_intelligence.py may modify.
GOVERNANCE_DB_FILE = os.getenv(
    "GOVERNANCE_DB_FILE",
    "nexus_governance.db",
)

# =========================
# Governance & AI Learning
# =========================
PERFORMANCE_FILE = os.getenv(
    "PERFORMANCE_FILE",
    "performance.json",
)
KILL_THRESHOLD = 45.0
RECOVERY_THRESHOLD = 55.0

# =========================
# Webhook
# =========================
WEBHOOK_URL = os.getenv("WEBHOOK_URL")


# =========================
# Path Helpers
# =========================
BASE_DIR = Path(__file__).resolve().parent
SIGNALS_DB_PATH = BASE_DIR / DB_FILE
GOVERNANCE_DB_PATH = BASE_DIR / GOVERNANCE_DB_FILE
PERFORMANCE_PATH = BASE_DIR / PERFORMANCE_FILE
