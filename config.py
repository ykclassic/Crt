import os

# Exchange
EXCHANGE_ID = "gateio"

# Assets (comma separated in repo secrets if needed)
ASSETS = os.getenv(
    "ASSETS",
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT"
).split(",")

# Timeframes
EXECUTION_TF = "1h"
CONFIRM_TF_1 = "4h"
CONFIRM_TF_2 = "1d"

# Risk parameters
RISK_PERCENT = 0.02
REWARD_PERCENT = 0.05

# Webhook
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
