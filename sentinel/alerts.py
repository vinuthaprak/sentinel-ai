"""
SentinelAI Alert Engine

Evaluates drift reports and system metrics against configurable thresholds
and generates structured alerts. Supports pluggable notification channels.

Alert lifecycle:
  FIRING → ACKNOWLEDGED → RESOLVED

Each alert has a fingerprint (model + rule + feature) so the same condition
doesn't generate duplicate alerts while still firing.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional

from .drift import DriftReport, DriftSeverity

logger = logging.getLogger("sentinel.alerts")


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(str, Enum):
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    alert_id: str
    fingerprint: str          # dedup key
    model_name: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    title: str
    description: str
    fired_at: str
    resolved_at: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "fingerprint": self.fingerprint,
            "model_name": self.model_name,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "state": self.state.value,
            "title": self.title,
            "description": self.description,
            "fired_at": self.fired_at,
            "resolved_at": self.resolved_at,
            "labels": self.labels,
            "annotations": self.annotations,
        }


# ─────────────────────────────────────────────
# Built-in Alert Rules
# ─────────────────────────────────────────────

class AlertRules:
    """
    Declarative alert rules that evaluate DriftReports and system snapshots.
    Add your own by subclassing or registering a rule function.
    """

    @staticmethod
    def prediction_distribution_shift(report: DriftReport) -> Optional[Alert]:
        for r in report.results:
            if r.feature_name == "prediction_labels" and r.drift_detected:
                sev = AlertSeverity.CRITICAL if r.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL) else AlertSeverity.WARNING
                details = r.details
                ref_dist = details.get("reference_distribution", {})
                cur_dist = details.get("current_distribution", {})
                shift_lines = [
                    f"  {lbl}: {round(ref_dist.get(lbl, 0)*100,1)}% → {round(cur_dist.get(lbl, 0)*100,1)}%"
                    for lbl in cur_dist
                ]
                return Alert(
                    alert_id=str(uuid.uuid4()),
                    fingerprint=f"{report.model_name}:prediction_distribution_shift",
                    model_name=report.model_name,
                    rule_name="prediction_distribution_shift",
                    severity=sev,
                    state=AlertState.FIRING,
                    title=f"[{sev.value.upper()}] Prediction distribution shifted — {report.model_name}",
                    description=(
                        f"The prediction label distribution for model '{report.model_name}' "
                        f"has shifted significantly (JSD={r.statistic:.4f}, threshold={r.threshold}).\n"
                        f"Label distribution change:\n" + "\n".join(shift_lines)
                    ),
                    fired_at=report.timestamp,
                    labels={"model": report.model_name, "test": "jsd"},
                    annotations={"runbook": "https://github.com/sentinel-ai/runbooks/prediction-drift"},
                )
        return None

    @staticmethod
    def confidence_collapse(report: DriftReport) -> Optional[Alert]:
        """Fires when average confidence score drops sharply — often means the model is confused."""
        for r in report.results:
            if r.feature_name == "confidence_score" and r.drift_detected:
                ref_mean = r.details.get("ref_mean", 0)
                cur_mean = r.details.get("cur_mean", 0)
                drop = ref_mean - cur_mean
                if drop > 0.1:  # Confidence dropped by more than 10 pp
                    sev = AlertSeverity.CRITICAL if drop > 0.2 else AlertSeverity.WARNING
                    return Alert(
                        alert_id=str(uuid.uuid4()),
                        fingerprint=f"{report.model_name}:confidence_collapse",
                        model_name=report.model_name,
                        rule_name="confidence_collapse",
                        severity=sev,
                        state=AlertState.FIRING,
                        title=f"[{sev.value.upper()}] Confidence collapse detected — {report.model_name}",
                        description=(
                            f"Average confidence for model '{report.model_name}' "
                            f"dropped from {ref_mean:.2%} to {cur_mean:.2%} "
                            f"(Δ={drop:.2%}). Model may be receiving out-of-distribution inputs."
                        ),
                        fired_at=report.timestamp,
                        labels={"model": report.model_name, "test": "ks_confidence"},
                        annotations={"runbook": "https://github.com/sentinel-ai/runbooks/confidence-drop"},
                    )
        return None

    @staticmethod
    def high_psi_feature(report: DriftReport) -> Optional[Alert]:
        """Fires when any input feature has high PSI — indicates data pipeline issues."""
        worst = None
        for r in report.results:
            if r.test_name == "psi" and r.drift_detected and r.feature_name.startswith("input:"):
                if worst is None or r.statistic > worst.statistic:
                    worst = r
        if worst:
            sev = AlertSeverity.CRITICAL if worst.severity == DriftSeverity.CRITICAL else AlertSeverity.WARNING
            return Alert(
                alert_id=str(uuid.uuid4()),
                fingerprint=f"{report.model_name}:input_feature_drift:{worst.feature_name}",
                model_name=report.model_name,
                rule_name="high_psi_feature",
                severity=sev,
                state=AlertState.FIRING,
                title=f"[{sev.value.upper()}] Input feature drift — {worst.feature_name}",
                description=(
                    f"Feature '{worst.feature_name}' has PSI={worst.statistic:.4f} "
                    f"(threshold=0.2). This likely indicates a data pipeline "
                    f"change or upstream data quality issue."
                ),
                fired_at=report.timestamp,
                labels={"model": report.model_name, "feature": worst.feature_name, "test": "psi"},
                annotations={"runbook": "https://github.com/sentinel-ai/runbooks/feature-drift"},
            )
        return None

    @staticmethod
    def high_error_rate(report: DriftReport) -> Optional[Alert]:
        for r in report.results:
            if r.feature_name == "error_rate" and r.drift_detected:
                cur_rate = r.details.get("current_error_rate", 0)
                sev = AlertSeverity.CRITICAL if cur_rate > 0.15 else AlertSeverity.WARNING
                return Alert(
                    alert_id=str(uuid.uuid4()),
                    fingerprint=f"{report.model_name}:high_error_rate",
                    model_name=report.model_name,
                    rule_name="high_error_rate",
                    severity=sev,
                    state=AlertState.FIRING,
                    title=f"[{sev.value.upper()}] Error rate spike — {report.model_name}",
                    description=(
                        f"Error rate for model '{report.model_name}' is "
                        f"{cur_rate:.1%} (reference: {r.details.get('reference_error_rate', 0):.1%}). "
                        "This may indicate a regression or infrastructure issue."
                    ),
                    fired_at=report.timestamp,
                    labels={"model": report.model_name},
                    annotations={"runbook": "https://github.com/sentinel-ai/runbooks/error-spike"},
                )
        return None


# ─────────────────────────────────────────────
# Alert Manager
# ─────────────────────────────────────────────

class AlertManager:
    """
    Evaluates drift reports against all registered rules,
    deduplicates alerts by fingerprint, and stores the active alert set.
    """

    def __init__(self, max_history: int = 500):
        self._active: Dict[str, Alert] = {}    # fingerprint → Alert
        self._history: List[Alert] = []
        self._max_history = max_history
        self._rules: List[Callable] = [
            AlertRules.prediction_distribution_shift,
            AlertRules.confidence_collapse,
            AlertRules.high_psi_feature,
            AlertRules.high_error_rate,
        ]
        self._notification_hooks: List[Callable] = []

    def register_rule(self, rule_fn: Callable):
        """Add a custom alert rule. rule_fn(DriftReport) -> Optional[Alert]"""
        self._rules.append(rule_fn)

    def register_notification_hook(self, hook: Callable):
        """Add a notification callback. hook(Alert) -> None"""
        self._notification_hooks.append(hook)

    def evaluate(self, report: DriftReport) -> List[Alert]:
        """Run all rules against a drift report and return newly fired alerts."""
        new_alerts: List[Alert] = []
        for rule in self._rules:
            try:
                alert = rule(report)
                if alert:
                    if alert.fingerprint not in self._active:
                        self._active[alert.fingerprint] = alert
                        self._history.append(alert)
                        if len(self._history) > self._max_history:
                            self._history.pop(0)
                        new_alerts.append(alert)
                        logger.warning("ALERT FIRED: %s", alert.title)
                        self._notify(alert)
                    # else: already firing, suppress duplicate
            except Exception as exc:
                logger.error("Rule %s raised an exception: %s", rule.__name__, exc)

        # Auto-resolve alerts whose conditions no longer hold
        self._auto_resolve(report)
        return new_alerts

    def acknowledge(self, fingerprint: str) -> bool:
        if fingerprint in self._active:
            self._active[fingerprint].state = AlertState.ACKNOWLEDGED
            return True
        return False

    def resolve(self, fingerprint: str) -> bool:
        if fingerprint in self._active:
            alert = self._active.pop(fingerprint)
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc).isoformat()
            self._history.append(alert)
            return True
        return False

    def get_active(self) -> List[dict]:
        return [a.to_dict() for a in sorted(
            self._active.values(),
            key=lambda a: a.fired_at,
            reverse=True,
        )]

    def get_history(self, limit: int = 50) -> List[dict]:
        return [a.to_dict() for a in reversed(self._history[-limit:])]

    def get_counts(self) -> dict:
        active = list(self._active.values())
        return {
            "total_active": len(active),
            "critical": sum(1 for a in active if a.severity == AlertSeverity.CRITICAL),
            "warning": sum(1 for a in active if a.severity == AlertSeverity.WARNING),
            "info": sum(1 for a in active if a.severity == AlertSeverity.INFO),
        }

    def _auto_resolve(self, report: DriftReport):
        """Resolve alerts whose corresponding signal is no longer drifted."""
        drifted_features = {r.feature_name for r in report.results if r.drift_detected}
        to_resolve = []
        for fp, alert in self._active.items():
            if alert.model_name == report.model_name:
                # Simple heuristic: if no drift at all, resolve everything for this model
                if report.overall_severity == DriftSeverity.NONE:
                    to_resolve.append(fp)
        for fp in to_resolve:
            self.resolve(fp)

    def _notify(self, alert: Alert):
        for hook in self._notification_hooks:
            try:
                hook(alert)
            except Exception as exc:
                logger.error("Notification hook failed: %s", exc)
