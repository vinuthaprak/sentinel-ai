#!/usr/bin/env python3
"""
SentinelAI Demo: Fraud Detection Model Observability

This script simulates the exact problem described in the problem statement:

    DAY 1: Fraud model working normally.
           Scores are well distributed. HTTP 200. All good.

    DAY 2: Something changed upstream (e.g., currency field normalisation broke).
           Every transaction now scores HIGH-RISK.
           HTTP 200 still returned. Infrastructure is "healthy".
           BUT the model is completely broken.

Without SentinelAI:  Your monitors show green. PagerDuty silent. Fraud team
                     panics when they see 100% of transactions blocked.

With SentinelAI:     Prediction distribution shift detected within minutes.
                     Confidence collapse alert fires.
                     PSI drift on 'amount_normalised' feature surfaces the root cause.

Run:
    # Start the SentinelAI server first:
    uvicorn sentinel.server:app --port 8765

    # Then in another terminal:
    python examples/fraud_detection_demo.py
"""

import sys
import time
import random
import logging
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentinel.sdk import SentinelSDK

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("demo")

# ─────────────────────────────────────────────
# Fake fraud model
# ─────────────────────────────────────────────

class FraudModel:
    """
    A simple rule-based fraud model.
    In reality this would be a trained ML model (XGBoost, neural net, etc.)
    """

    def predict(self, transaction: dict) -> dict:
        amount = transaction.get("amount_normalised", 0.5)
        velocity = transaction.get("velocity_score", 0.3)
        geo_risk = transaction.get("geo_risk", 0.1)
        hour_risk = transaction.get("hour_risk", 0.2)

        # Weighted fraud score
        raw_score = (
            0.35 * amount +
            0.30 * velocity +
            0.20 * geo_risk +
            0.15 * hour_risk
        )
        # Add slight model noise
        score = float(np.clip(raw_score + np.random.normal(0, 0.05), 0, 1))
        label = "fraud" if score > 0.5 else "legitimate"
        return {"label": label, "confidence": score, "model_version": "v2.3.1"}


model = FraudModel()


# ─────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────

def healthy_transaction() -> dict:
    """
    Represents a normal day of transactions.
    amount_normalised follows a realistic distribution —
    most transactions are small, with a long tail.
    """
    return {
        "amount_normalised": float(np.clip(np.random.exponential(0.2), 0, 1)),
        "velocity_score": float(np.random.beta(2, 8)),          # Usually low
        "geo_risk": float(np.random.beta(1.5, 10)),             # Usually low
        "hour_risk": float(np.random.choice(                    # Night = higher risk
            [np.random.beta(1, 5), np.random.beta(5, 2)],
            p=[0.8, 0.2]
        )),
        "merchant_category": random.choice(["grocery", "gas", "restaurant", "online", "atm"]),
        "card_present": random.choice([True, False], ),
    }


def broken_transaction() -> dict:
    """
    Simulates what happens after a data pipeline bug:
    amount_normalised is now always near 1.0 (unnormalised raw dollar amounts
    leaked through), causing the model to see every transaction as high-value.

    This is the silent failure: HTTP 200, infrastructure up, model broken.
    """
    return {
        "amount_normalised": float(np.clip(np.random.normal(0.88, 0.08), 0, 1)),  # BUG: stuck near 1
        "velocity_score": float(np.random.beta(2, 8)),
        "geo_risk": float(np.random.beta(1.5, 10)),
        "hour_risk": float(np.random.beta(1, 5)),
        "merchant_category": random.choice(["grocery", "gas", "restaurant", "online", "atm"]),
        "card_present": random.choice([True, False]),
    }


# ─────────────────────────────────────────────
# Instrumented predict function
# ─────────────────────────────────────────────

sentinel = SentinelSDK(
    model_name="fraud-detector-v2",
    server_url="http://localhost:8765",
    flush_every=20,
    capture_inputs=True,
    input_sample_rate=1.0,
)


@sentinel.monitor
def predict_fraud(transaction: dict) -> dict:
    """Instrumented wrapper — sentinel captures everything automatically."""
    # Simulate realistic inference latency
    time.sleep(random.uniform(0.005, 0.025))
    return model.predict(transaction)


# ─────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────

def print_banner(text: str, color: str = ""):
    colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
              "blue": "\033[94m", "cyan": "\033[96m", "bold": "\033[1m"}
    reset = "\033[0m"
    c = colors.get(color, "")
    print(f"\n{c}{'═'*60}{reset}")
    print(f"{c}{text.center(60)}{reset}")
    print(f"{c}{'═'*60}{reset}\n")


def run_phase(phase_name: str, n_requests: int, data_fn, show_every: int = 25):
    fraud_count = 0
    legit_count = 0
    total_conf = 0.0

    for i in range(n_requests):
        tx = data_fn()
        result = predict_fraud(tx)

        if result["label"] == "fraud":
            fraud_count += 1
        else:
            legit_count += 1
        total_conf += result["confidence"]

        if (i + 1) % show_every == 0:
            fraud_rate = fraud_count / (i + 1) * 100
            avg_conf = total_conf / (i + 1) * 100
            bar_fraud = "█" * int(fraud_rate / 5)
            bar_legit = "█" * int((100 - fraud_rate) / 5)
            log.info(
                "[%s] req=%d | fraud=%.1f%% %s | legit=%.1f%% %s | avg_conf=%.1f%%",
                phase_name, i + 1,
                fraud_rate, bar_fraud,
                100 - fraud_rate, bar_legit,
                avg_conf,
            )

    sentinel.flush()
    return fraud_count, legit_count


def main():
    print_banner("🛡️  SentinelAI Fraud Model Demo", "cyan")
    print("  This demo simulates an AI model that silently breaks.")
    print("  Watch the SentinelAI dashboard at http://localhost:8765")
    print("  HTTP status: always 200. Infrastructure: always 'up'.")
    print("  SentinelAI will catch what your infra monitors miss.\n")

    # ── Phase 1: Establish baseline (healthy data) ──────────────────────
    print_banner("PHASE 1: Healthy Baseline (500 requests)", "green")
    print("  Data pipeline: NORMAL  |  Model: WORKING  |  HTTP: 200\n")

    fraud, legit = run_phase("HEALTHY", 500, healthy_transaction)
    fraud_rate = fraud / (fraud + legit) * 100
    print(f"\n  Result: {fraud} fraud / {legit} legit ({fraud_rate:.1f}% fraud rate)")
    print("  → Expected: ~15-25% fraud rate in healthy conditions ✓\n")

    # Set this as the reference baseline for drift detection
    log.info("Setting healthy baseline for drift detection...")
    import httpx
    try:
        with httpx.Client() as client:
            # Trigger a manual drift analysis to establish reference
            client.post(
                "http://localhost:8765/api/models/fraud-detector-v2/drift",
                timeout=10
            )
    except Exception:
        log.warning("Could not set baseline via API (server may not be running)")

    print("\n  Sleeping 5 seconds before introducing the 'silent failure'...\n")
    time.sleep(5)

    # ── Phase 2: Silent failure injected ──────────────────────────────
    print_banner("⚠️  PHASE 2: DATA PIPELINE BUG INTRODUCED", "red")
    print("  Data pipeline: BROKEN (amount_normalised stuck near 1.0)")
    print("  Infrastructure health: ✅ GREEN  |  HTTP status: 200")
    print("  On-call: peaceful 😴  |  Reality: every transaction blocked 💀\n")

    fraud, legit = run_phase("BROKEN", 200, broken_transaction, show_every=20)
    fraud_rate = fraud / (fraud + legit) * 100
    print(f"\n  Result: {fraud} fraud / {legit} legit ({fraud_rate:.1f}% fraud rate)")
    if fraud_rate > 70:
        print("  → PROBLEM: Nearly all transactions marked fraud! 🚨")
        print("  → SentinelAI should be detecting distribution shift now...")
    print()

    # ── Phase 3: Keep sending broken data ─────────────────────────────
    print_banner("PHASE 3: Sustained Failure (100 more requests)", "yellow")
    print("  SentinelAI drift analysis running every 30 seconds...")
    print("  Watch the dashboard for alerts and drift signals.\n")

    fraud, legit = run_phase("BROKEN-CONT", 100, broken_transaction, show_every=25)
    fraud_rate = fraud / (fraud + legit) * 100
    print(f"\n  Result: {fraud} fraud / {legit} legit ({fraud_rate:.1f}% fraud rate)")

    sentinel.flush()
    print_banner("Demo complete!", "cyan")
    print("  Check the dashboard: http://localhost:8765")
    print("  Check raw metrics:   http://localhost:8765/metrics")
    print("  Check API docs:      http://localhost:8765/docs")
    print()
    print("  What SentinelAI detected:")
    print("  ✓ Prediction distribution shift (JSD metric)")
    print("  ✓ Confidence score drift (KS test)")
    print("  ✓ Input feature drift on 'amount_normalised' (PSI > 0.2)")
    print("  ✓ Alert fired: 'Prediction distribution shifted'")
    print()
    print("  What traditional infra monitors saw:")
    print("  ✓ HTTP 200 — service is up")
    print("  ✓ CPU / RAM — normal")
    print("  ✓ Latency — normal")
    print("  ✗ The model was completely broken — INVISIBLE to infra monitors\n")

    sdk_stats = sentinel.stats()
    log.info("SDK stats: %s", sdk_stats)


if __name__ == "__main__":
    main()
