import os
import sqlite3
import json
import requests
import pickle

# --- CONFIG ---
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
FILES_TO_CHECK = {
    "DB: Core": "nexus_core.db",
    "DB: Hybrid": "hybrid_v1.db",
    "DB: Range": "rangemaster.db",
    "DB: AI": "nexus_ai.db",
    "DB: Journal": "nexus_journal.db",
    "Intelligence": "performance.json",
    "AI Brain": "nexus_brain.pkl"
}

def check_system():
    print("🔍 Nexus Sanity Check Starting...")
    results = []
    
    # 1. File Existence & DB Integrity
    for label, path in FILES_TO_CHECK.items():
        if os.path.exists(path):
            status = "✅ FOUND"
            if path.endswith(".db"):
                try:
                    conn = sqlite3.connect(path)
                    conn.execute("SELECT name FROM sqlite_master LIMIT 1")
                    conn.close()
                    status += " (INTEGRITY OK)"
                except:
                    status = "❌ CORRUPT"
        else:
            status = "⚠️ MISSING"
        results.append(f"{label}: {status}")

    # 2. Discord Connectivity
    discord_status = "❌ NOT CONFIGURED"
    if WEBHOOK_URL:
        try:
            resp = requests.post(WEBHOOK_URL, json={"content": "🛠️ **Nexus Sanity Check**: Connection Verified."})
            discord_status = "✅ VERIFIED" if resp.status_code in [200, 204] else f"❌ ERROR {resp.status_code}"
        except:
            discord_status = "❌ CONNECTION FAILED"
    
    # Summary
    print("\n--- SYSTEM REPORT ---")
    for r in results: print(r)
    print(f"Discord Webhook: {discord_status}")
    print("----------------------\n")

if __name__ == "__main__":
    check_system()
