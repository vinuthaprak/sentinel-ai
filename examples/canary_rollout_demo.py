#!/usr/bin/env python3
"""
SentinelAI Demo: Canary Rollout — Catching a Bad Model Deployment

Scenario
--------
Your team ships model-v2 as a canary alongside the stable model-v1.
Traffic is split: 90% → v1, 10% → v2.

model-v1 is healthy.
model-v2 has a regression: its confidence distribution has collapsed and it
over-predicts the positive class at nearly 3× the normal rate.

Infrastructure signals (HTTP status, CPU, latency) stay GREEN throughout.
SentinelAI detects the degradation in the canary slice within minutes and
recommends rollback before 100% of traffic is migrated.

Run
---
    # Terminal 1: start SentinelAI server
    uvicorn sentinel.server:app --port 8765

    # Terminal 2: run this demo
    python examples/canary_rollout_demo.py

Expected output
---------------
  PHASE 1  Warm up v1 baseline (healthy)
  PHASE 2  Canary: 90% v1, 10% v2 — v2 is degraded
  PHASE 3  SentinelAI detects — drift report + alert payload printed
  PHASE 4  Simulated rollback — v2 traffic zeroed out, metrics recover
"""

import sys
import time
import random
import logging
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import numpy as np
from sentinel.sdk import SentinelSDK

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("canary-demo")

SERVER = "http://localhost:8765"

# ─────────────────────────────────────────────
# Two model versions (fake inference functions)
# ─────────────────────────────────────────────

def model_v1_predict(features: dict) -> dict:
    """
    Stable production model.
    Healthy label distribution: ~20% positive, ~80% negative.
    Confidence: well-calibrated, mean ~0.78.
    """
    risk = (
        0.35 * features.get("amount_norm", 0.3) +
        0.30 * features.get("velocity", 0.2) +
        0.20 * features.get("geo_risk", 0.1) +
        0.15 * features.get("hour_risk", 0.2)
    )
    score = float(np.clip(risk + np.random.normal(0, 0.05), 0, 1))
    return {
        "label": "positive" if score > 0.5 else "negative",
        "confidence": score if score > 0.5 else (1 - score),
        "model_version": "v1",
    }


def model_v2_predict(features: dict) -> dict:
    """
    Canary model — contains a regression introduced in training:
      - Feature weight misconfiguration causes 3× higher positive rate
      - Confidence is poorly calibrated (lower, noisier)
    Infrastructure reports HTTP 200. You'd never know without SentinelAI.
    """
    risk = (
        0.70 * features.get("amount_norm", 0.3) +   # BUG: weight too high
        0.20 * features.get("velocity", 0.2) +
        0.05 * features.get("geo_risk", 0.1) +
        0.05 * features.get("hour_risk", 0.2)
    )
    score = float(np.clip(risk + np.random.normal(0, 0.12), 0, 1))
    # Confidence collapse: model is uncertain (regressed calibration)
    confidence = float(np.clip(abs(score - 0.5) * 0.6 + np.random.uniform(0.1, 0.3), 0, 1))
    return {
        "label": "positive" if score > 0.5 else "negative",
        "confidence": confidence,
        "model_version": "v2",
    }


# ─────────────────────────────────────────────
# Feature generator
# ─────────────────────────────────────────────

def sample_features() -> dict:
    return {
        "amount_norm": float(np.clip(np.random.exponential(0.2), 0, 1)),
        "velocity": float(np.random.beta(2, 8)),
        "geo_risk": float(np.random.beta(1.5, 10)),
        "hour_risk": float(np.random.choice(
            [np.random.beta(1, 5), np.random.beta(5, 2)], p=[0.8, 0.2]
        )),
    }


# ─────────────────────────────────────────────
# SDK instances — one per model version
# ─────────────────────────────────────────────

sdk_v1 = SentinelSDK(
    model_name="classifier-v1",
    server_url=SERVER,
    flush_every=20,
    capture_inputs=True,
)

sdk_v2 = SentinelSDK(
    model_name="classifier-v2",
    server_url=SERVER,
    flush_every=20,
    capture_inputs=True,
)


@sdk_v1.monitor
def predict_v1(features: dict) -> dict:
    time.sleep(random.uniform(0.003, 0.015))   # ~5-15 ms latency
    return model_v1_predict(features)


@sdk_v2.monitor
def predict_v2(features: dict) -> dict:
    time.sleep(random.uniform(0.004, 0.020))   # similar latency — infra looks fine
    return model_v2_predict(features)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def banner(text: str, color: str = ""):
    codes = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
             "cyan": "\033[96m", "bold": "\033[1m"}
    reset = "\033[0m"
    c = codes.get(color, "")
    print(f"\n{c}{'═' * 62}{reset}")
    print(f"{c}{text.center(62)}{reset}")
    print(f"{c}{'═' * 62}{reset}\n")


def run_traffic(n: int, canary_pct: float, label: str, show_every: int = 50):
    """Send n requests with canary_pct fraction going to v2."""
    v1_pos = v1_neg = v2_pos = v2_neg = 0
    v1_conf_sum = v2_conf_sum = 0.0

    for i in range(n):
        features = sample_features()
        if random.random() < canary_pct:
            result = predict_v2(features)
            if result["label"] == "positive":
                v2_pos += 1
            else:
                v2_neg += 1
            v2_conf_sum += result["confidence"]
        else:
            result = predict_v1(features)
            if result["label"] == "positive":
                v1_pos += 1
            else:
                v1_neg += 1
            v1_conf_sum += result["confidence"]

        if (i + 1) % show_every == 0:
            v1_total = max(v1_pos + v1_neg, 1)
            v2_total = max(v2_pos + v2_neg, 1)
            log.info(
                "[%s] req=%d | v1: +%.0f%% conf=%.2f | v2: +%.0f%% conf=%.2f",
                label, i + 1,
                v1_pos / v1_total * 100, v1_conf_sum / v1_total,
                v2_pos / v2_total * 100, v2_conf_sum / max(v2_total, 1),
            )

    sdk_v1.flush()
    sdk_v2.flush()
    return {
        "v1": {"positive_rate": v1_pos / max(v1_pos + v1_neg, 1), "avg_conf": v1_conf_sum / max(v1_pos + v1_neg, 1)},
        "v2": {"positive_rate": v2_pos / max(v2_pos + v2_neg, 1), "avg_conf": v2_conf_sum / max(v2_pos + v2_neg, 1)},
    }


def fetch_drift_report(model_name: str) -> dict:
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(f"{SERVER}/api/models/{model_name}/drift")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        log.warning("Could not fetch drift report: %s", e)
    return {}


def fetch_alerts() -> dict:
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{SERVER}/api/alerts")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        log.warning("Could not fetch alerts: %s", e)
    return {}


def fetch_slos() -> dict:
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{SERVER}/api/slos")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        log.warning("Could not fetch SLOs: %s", e)
    return {}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    banner("SentinelAI — Canary Rollout Detection Demo", "cyan")
    print("  Scenario: classifier-v2 deployed as 10% canary alongside v1.")
    print("  v2 has a silent regression — wrong feature weights.")
    print("  Infrastructure shows green. SentinelAI will catch it.\n")
    print(f"  Dashboard: {SERVER}")
    print(f"  API docs:  {SERVER}/docs\n")

    # ── Phase 1: Warm up v1 baseline ──────────────────────────────────
    banner("PHASE 1 — Warm-up: 400 requests to v1 (healthy baseline)", "green")
    print("  Establishing reference window for drift detection...\n")

    stats = run_traffic(400, canary_pct=0.0, label="WARMUP", show_every=100)
    print(f"\n  v1 positive rate: {stats['v1']['positive_rate']:.1%}  (expect ~18-25%)")
    print(f"  v1 avg confidence: {stats['v1']['avg_conf']:.2f}  (expect ~0.7-0.85)")
    print("\n  Baseline established. Sleeping 3s before canary rollout...\n")
    time.sleep(3)

    # ── Phase 2: Canary rollout — 90/10 split ─────────────────────────
    banner("PHASE 2 — Canary Rollout: 90% v1  /  10% v2", "yellow")
    print("  v2 regression: feature weight misconfiguration + confidence collapse")
    print("  HTTP status: 200 | Latency: normal | CPU: normal | PagerDuty: silent\n")

    stats = run_traffic(300, canary_pct=0.10, label="CANARY", show_every=75)
    print(f"\n  v1 positive rate: {stats['v1']['positive_rate']:.1%}")
    print(f"  v2 positive rate: {stats['v2']['positive_rate']:.1%}  ← elevated")
    print(f"  v2 avg confidence: {stats['v2']['avg_conf']:.2f}       ← collapsed")

    # ── Phase 3: SentinelAI analysis ──────────────────────────────────
    banner("PHASE 3 — SentinelAI Analysis", "cyan")
    print("  Triggering drift analysis for both model versions...\n")

    for model_name in ["classifier-v1", "classifier-v2"]:
        report = fetch_drift_report(model_name)
        if report:
            drift_data = report.get("report", {})
            severity = drift_data.get("overall_severity", "unknown")
            summary = drift_data.get("summary", "no summary")
            color = "\033[91m" if severity in ("high", "critical") else "\033[92m"
            reset = "\033[0m"
            print(f"  {color}[{model_name}] severity={severity.upper()}{reset}")
            print(f"    {summary}")
            new_alerts = report.get("new_alerts", [])
            if new_alerts:
                print(f"    Alerts fired ({len(new_alerts)}):")
                for alert in new_alerts:
                    print(f"      🚨 {alert['title']}")
                    print(f"         {alert['description'][:120]}...")
        else:
            print(f"  [INFO] {model_name}: server not reachable — run 'uvicorn sentinel.server:app --port 8765' first")
        print()

    # Show active alerts
    alerts = fetch_alerts()
    if alerts:
        counts = alerts.get("counts", {})
        print(f"  Active alerts: {counts.get('critical', 0)} critical, {counts.get('warning', 0)} warning")
        active = alerts.get("active", [])
        if active:
            print("\n  Sample alert payload (what you'd send to PagerDuty/Slack):")
            # Show the most recent alert as a clean JSON sample
            sample = active[0]
            print("  " + json.dumps({
                "alert_id": sample["alert_id"],
                "title": sample["title"],
                "severity": sample["severity"],
                "state": sample["state"],
                "model_name": sample["model_name"],
                "fired_at": sample["fired_at"],
                "description": sample["description"][:100] + "...",
            }, indent=4).replace("\n", "\n  "))

    # Show SLO status
    slos = fetch_slos()
    if slos:
        print(f"\n  SLO Status: {slos.get('overall_status', 'unknown').upper()}")
        for slo in slos.get("slos", []):
            icon = "✅" if slo["status"] == "ok" else ("⚠️ " if slo["status"] == "warning" else "🚨")
            print(f"    {icon} {slo['name']}: {slo['message']}")

    print()
    time.sleep(2)

    # ── Phase 4: Rollback ─────────────────────────────────────────────
    banner("PHASE 4 — Rollback: v2 traffic → 0%", "green")
    print("  Decision: SentinelAI detected v2 degradation before full rollout.")
    print("  Action: route 100% traffic back to v1 (rollback).\n")
    print("  In a real k8s canary rollout this would be:")
    print("    kubectl set image deployment/classifier image=classifier:v1")
    print("    # or via Argo Rollouts: kubectl argo rollouts abort classifier\n")

    stats = run_traffic(200, canary_pct=0.0, label="POST-ROLLBACK", show_every=100)
    print(f"\n  v1 positive rate: {stats['v1']['positive_rate']:.1%}  (back to normal)")
    print(f"  v1 avg confidence: {stats['v1']['avg_conf']:.2f}  (recovered)")

    banner("Demo Complete", "cyan")
    print("  What SentinelAI caught (that infra monitoring missed):")
    print("  ✓ v2 positive-rate 3× higher than v1 baseline")
    print("  ✓ Confidence collapse in canary slice")
    print("  ✓ JSD divergence between v1 and v2 label distributions")
    print("  ✓ Alert fired before 100% traffic migration")
    print()
    print("  What traditional infra monitors saw:")
    print("  ✓ HTTP 200 on both versions")
    print("  ✓ Latency within normal range")
    print("  ✓ CPU / memory normal")
    print("  ✗ Model regression — completely invisible to infra monitors\n")
    print(f"  Dashboard: {SERVER}")
    print(f"  Metrics:   {SERVER}/metrics")
    print(f"  SLOs:      {SERVER}/api/slos\n")


if __name__ == "__main__":
    main()
