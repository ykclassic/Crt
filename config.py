import os

# =========================
# Exchange
# =========================
EXCHANGE_ID = "gateio"

# =========================
# Assets (can override via GitHub Secrets)
# Example secret:
# ASSETS=BTC/USDT,ETH/USDT
# =========================
ASSETS = os.getenv(
    "ASSETS",
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT"
).split(",")

# =========================
# Timeframes
# =========================
EXECUTION_TF = "1h"
CONFIRM_TF_1 = "4h"
CONFIRM_TF_2 = "1d"

TIMEFRAMES = [EXECUTION_TF, CONFIRM_TF_1, CONFIRM_TF_2]

# =========================
# Risk Model (Percent Based)
# =========================
RISK_PERCENT = 0.02      # 2% stop loss
REWARD_PERCENT = 0.05    # 5% take profit

# =========================
# Database
# =========================
DB_FILE = os.getenv("DB_FILE", "nexus_signals.db")

# =========================
# Webhook (Optional)
# =========================
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
