import requests
import logging
from config import WEBHOOK_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | DISPATCH | %(levelname)s | %(message)s"
)

def dispatch_signal(pair, direction, tier,
                    confidence, entry,
                    stop_loss, take_profit):

    if not WEBHOOK_URL:
        logging.warning("Webhook not configured")
        return

    message = f"""
Asset: {pair}
Direction: {direction}
Tier: {tier}
Confidence: {round(confidence, 2)}
Entry: {round(entry, 4)}
Stop Loss: {round(stop_loss, 4)}
Take Profit: {round(take_profit, 4)}
"""

    try:
        requests.post(WEBHOOK_URL, json={"content": message})
        logging.info(f"{pair} dispatched")
    except Exception as e:
        logging.error(f"{pair} dispatch failed: {e}")
