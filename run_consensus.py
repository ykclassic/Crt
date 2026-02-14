from db_utils import get_latest_signal
from collections import defaultdict

core = get_latest_signal("core.db")
range_s = get_latest_signal("range.db")
hybrid = get_latest_signal("hybrid.db")

grouped = defaultdict(list)

for source, data in [
    ("core", core),
    ("range", range_s),
    ("hybrid", hybrid)
]:
    for row in data:
        grouped[row[0]].append((source, row))

consensus_results = []

for symbol, signals in grouped.items():
    if len(signals) < 1:
        continue

    directions = [s[1][1] for s in signals]

    agreement = max(directions.count("LONG"), directions.count("SHORT"))

    if agreement == 3:
        label = "DIAMOND CONSENSUS"
    elif agreement == 2:
        label = "GOLD CONSENSUS"
    else:
        label = "STANDARD ALERT"

    consensus_results.append((symbol, signals[0][1], label))

print(consensus_results)
