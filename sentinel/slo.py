"""
SentinelAI SLO (Service Level Objective) Engine

Defines and evaluates AI-specific SLOs that go beyond traditional infra SLOs.
Two categories:

  Infrastructure SLOs   — latency, availability, error rate
  Model Behavior SLOs   — prediction distribution, confidence, drift thresholds

Each SLO has a target, current value, and compliance status so you can answer:
"Is our AI service meeting its reliability contract right now?"

SLO Definitions (change via SLOConfig):
  - inference_p99_latency_ms   : 99% of inferences complete under 200 ms
  - error_rate_pct             : fewer than 1% of inference calls error
  - prediction_distribution_jsd: JSD drift below 0.1 (stable label distribution)
  - avg_confidence             : average confidence above 0.65 (model not confused)
  - uptime_pct                 : server available > 99.5% of the time (proxy: health checks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SLOStatus(str, Enum):
    OK = "ok"               # Within target
    WARNING = "warning"     # Within 20% of the limit — heading toward breach
    BREACHED = "breached"   # Target violated


@dataclass
class SLODefinition:
    name: str
    description: str
    unit: str
    target: float                # The threshold value
    direction: str               # "below" (lower is better) or "above" (higher is better)
    warning_buffer_pct: float = 0.20  # Warn when within 20% of limit

    def evaluate(self, current_value: Optional[float]) -> "SLOResult":
        if current_value is None:
            return SLOResult(
                slo=self,
                current_value=None,
                status=SLOStatus.OK,
                compliance_pct=100.0,
                message="No data yet",
            )

        if self.direction == "below":
            breached = current_value > self.target
            warn_threshold = self.target * (1 - self.warning_buffer_pct)
            warning = not breached and current_value > warn_threshold
            # Compliance: how far are we from the limit? (100% = exactly at limit)
            compliance = min(100.0, (self.target / max(current_value, 1e-9)) * 100)
        else:  # above
            breached = current_value < self.target
            warn_threshold = self.target * (1 + self.warning_buffer_pct)
            warning = not breached and current_value < warn_threshold
            compliance = min(100.0, (current_value / max(self.target, 1e-9)) * 100)

        if breached:
            status = SLOStatus.BREACHED
        elif warning:
            status = SLOStatus.WARNING
        else:
            status = SLOStatus.OK

        return SLOResult(
            slo=self,
            current_value=round(current_value, 4),
            status=status,
            compliance_pct=round(compliance, 2),
            message=self._message(current_value, status),
        )

    def _message(self, value: float, status: SLOStatus) -> str:
        direction_word = "≤" if self.direction == "below" else "≥"
        return (
            f"{self.name}: {value:.4g} {self.unit} "
            f"(target {direction_word} {self.target} {self.unit}) — {status.value.upper()}"
        )


@dataclass
class SLOResult:
    slo: SLODefinition
    current_value: Optional[float]
    status: SLOStatus
    compliance_pct: float
    message: str

    def to_dict(self) -> dict:
        return {
            "name": self.slo.name,
            "description": self.slo.description,
            "unit": self.slo.unit,
            "target": self.slo.target,
            "direction": self.slo.direction,
            "current_value": self.current_value,
            "status": self.status.value,
            "compliance_pct": self.compliance_pct,
            "message": self.message,
        }


# ─────────────────────────────────────────────
# Default SLO definitions
# ─────────────────────────────────────────────

DEFAULT_SLOS: List[SLODefinition] = [
    SLODefinition(
        name="inference_p99_latency_ms",
        description="99th-percentile inference latency must stay below 200 ms",
        unit="ms",
        target=200.0,
        direction="below",
    ),
    SLODefinition(
        name="error_rate_pct",
        description="Inference error rate must stay below 1%",
        unit="%",
        target=1.0,
        direction="below",
    ),
    SLODefinition(
        name="avg_confidence",
        description="Average prediction confidence must stay above 0.65 — below indicates OOD inputs",
        unit="(0-1)",
        target=0.65,
        direction="above",
    ),
    SLODefinition(
        name="prediction_jsd_drift",
        description="Jensen-Shannon Divergence of prediction labels must stay below 0.10",
        unit="JSD",
        target=0.10,
        direction="below",
    ),
    SLODefinition(
        name="active_critical_alerts",
        description="No more than 0 critical alerts firing at any time",
        unit="alerts",
        target=0.0,
        direction="below",
        warning_buffer_pct=0.0,  # Any critical alert is immediately a warning
    ),
]


class SLOEngine:
    """
    Evaluates all SLOs given a model summary and alert counts.
    Call evaluate() to get the current SLO compliance snapshot.
    """

    def __init__(self, slos: Optional[List[SLODefinition]] = None):
        self._slos = slos or DEFAULT_SLOS

    def evaluate(
        self,
        model_summaries: List[dict],
        alert_counts: dict,
        latest_drift_reports: Optional[Dict[str, dict]] = None,
    ) -> List[SLOResult]:
        """
        Aggregate metrics across all models and evaluate each SLO.
        Multi-model: uses the worst-case value across models for each metric.
        """
        results = []

        # Aggregate across models (worst-case approach for SRE honesty)
        p99s = [s.get("p99_latency_ms", 0) for s in model_summaries if s.get("p99_latency_ms")]
        error_rates = [s.get("error_rate", 0) * 100 for s in model_summaries]
        confidences = [s.get("avg_confidence", 1.0) for s in model_summaries if s.get("avg_confidence")]

        # Extract JSD from latest drift reports if available
        jsd_values = []
        if latest_drift_reports:
            for model_name, report in latest_drift_reports.items():
                for r in report.get("results", []):
                    if r.get("test_name") == "jensen_shannon_divergence":
                        jsd_values.append(r.get("statistic", 0.0))

        values: Dict[str, Optional[float]] = {
            "inference_p99_latency_ms": max(p99s) if p99s else None,
            "error_rate_pct": max(error_rates) if error_rates else None,
            "avg_confidence": min(confidences) if confidences else None,
            "prediction_jsd_drift": max(jsd_values) if jsd_values else None,
            "active_critical_alerts": float(alert_counts.get("critical", 0)),
        }

        for slo in self._slos:
            result = slo.evaluate(values.get(slo.name))
            results.append(result)

        return results

    def summary(self, results: List[SLOResult]) -> dict:
        total = len(results)
        breached = sum(1 for r in results if r.status == SLOStatus.BREACHED)
        warning = sum(1 for r in results if r.status == SLOStatus.WARNING)
        ok = total - breached - warning
        overall = (
            SLOStatus.BREACHED if breached > 0
            else SLOStatus.WARNING if warning > 0
            else SLOStatus.OK
        )
        return {
            "overall_status": overall.value,
            "total": total,
            "ok": ok,
            "warning": warning,
            "breached": breached,
            "slos": [r.to_dict() for r in results],
        }
