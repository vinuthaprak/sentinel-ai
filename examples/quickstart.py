#!/usr/bin/env python3
"""
SentinelAI Quickstart — 30-second integration example.

Shows how to instrument ANY existing model in 3 lines of code.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.sdk import SentinelSDK

# 1. Create the SDK instance (once, at module level)
sentinel = SentinelSDK(
    model_name="my-churn-model",
    server_url="http://localhost:8765",
)

# 2. Wrap your existing predict function with @sentinel.monitor
@sentinel.monitor
def predict_churn(user_features: dict) -> dict:
    """Your existing model code — completely unchanged."""
    # Replace this with your real model
    score = user_features.get("days_inactive", 0) / 365
    return {
        "label": "will_churn" if score > 0.5 else "retained",
        "confidence": min(score + 0.1, 1.0),
    }

# 3. Call your model normally — SentinelAI captures everything automatically
if __name__ == "__main__":
    import random

    print("Running 50 predictions through SentinelAI...")
    for i in range(50):
        result = predict_churn({
            "days_inactive": random.randint(0, 400),
            "num_logins_30d": random.randint(0, 50),
            "support_tickets": random.randint(0, 5),
            "contract_type": random.choice(["monthly", "annual"]),
        })
        if i % 10 == 0:
            print(f"  [{i}] label={result['label']} confidence={result['confidence']:.2f}")

    sentinel.flush()
    print("\nDone! Check http://localhost:8765 to see your model's telemetry.")
    print("SDK stats:", sentinel.stats())
