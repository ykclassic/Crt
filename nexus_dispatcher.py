import requests
from config import WEBHOOK_URL
from run_consensus import consensus_results

for symbol, data, label in consensus_results:
    direction = data[1]
    entry = data[2]
    sl = data[3]
    tp = data[4]

    message = f"""
Asset: {symbol}
Direction: {direction}
Entry: {entry}
Stop Loss: {sl}
Take Profit: {tp}
Consensus: {label}
"""

    requests.post(WEBHOOK_URL, json={"content": message})
