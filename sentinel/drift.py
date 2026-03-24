"""
SentinelAI Drift Detection Engine

Implements multiple statistical tests to detect distribution shift in:
  - Input features (data drift)
  - Prediction labels (concept drift proxy)
  - Confidence scores (confidence drift)

Algorithms implemented:
  1. Kolmogorov-Smirnov (KS) Test       — continuous feature drift
  2. Chi-Squared Test                    — categorical feature drift
  3. Population Stability Index (PSI)    — industry-standard credit/fraud metric
  4. CUSUM (Cumulative Sum)              — sequential change detection
  5. Jensen-Shannon Divergence           — prediction distribution shift
  6. Z-score rolling anomaly             — confidence score spikes

Each detector returns a DriftResult with a severity level so the
alert engine can decide whether to page or just log.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque, Counter

import numpy as np
from scipy import stats

logger = logging.getLogger("sentinel.drift")


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    feature_name: str
    test_name: str
    statistic: float
    p_value: Optional[float]
    threshold: float
    drift_detected: bool
    severity: DriftSeverity
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "test_name": self.test_name,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "severity": self.severity.value,
            "details": self.details,
        }


@dataclass
class DriftReport:
    model_name: str
    timestamp: str
    results: List[DriftResult]
    overall_severity: DriftSeverity
    summary: str

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
            "overall_severity": self.overall_severity.value,
            "summary": self.summary,
            "drift_detected": any(r.drift_detected for r in self.results),
        }


# ─────────────────────────────────────────────
# Individual statistical tests
# ─────────────────────────────────────────────

def ks_test(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "feature",
    alpha: float = 0.05,
) -> DriftResult:
    """
    Kolmogorov-Smirnov two-sample test for continuous distributions.
    Detects ANY shape change (mean, variance, skew).
    """
    if len(reference) < 10 or len(current) < 10:
        return DriftResult(feature_name, "ks_test", 0.0, 1.0, alpha, False, DriftSeverity.NONE,
                           {"reason": "insufficient_data"})
    stat, p_value = stats.ks_2samp(reference, current)
    drift = p_value < alpha

    if not drift:
        severity = DriftSeverity.NONE
    elif stat < 0.1:
        severity = DriftSeverity.LOW
    elif stat < 0.2:
        severity = DriftSeverity.MEDIUM
    elif stat < 0.35:
        severity = DriftSeverity.HIGH
    else:
        severity = DriftSeverity.CRITICAL

    return DriftResult(
        feature_name=feature_name,
        test_name="ks_test",
        statistic=float(stat),
        p_value=float(p_value),
        threshold=alpha,
        drift_detected=drift,
        severity=severity,
        details={
            "ref_mean": float(np.mean(reference)),
            "cur_mean": float(np.mean(current)),
            "ref_std": float(np.std(reference)),
            "cur_std": float(np.std(current)),
            "ref_n": len(reference),
            "cur_n": len(current),
        },
    )


def chi2_test(
    reference: List[str],
    current: List[str],
    feature_name: str = "feature",
    alpha: float = 0.05,
) -> DriftResult:
    """
    Chi-squared test for categorical feature drift.
    Builds a frequency table over the union of categories.
    """
    if len(reference) < 10 or len(current) < 10:
        return DriftResult(feature_name, "chi2_test", 0.0, 1.0, alpha, False, DriftSeverity.NONE,
                           {"reason": "insufficient_data"})
    all_cats = sorted(set(reference) | set(current))
    ref_counts = Counter(reference)
    cur_counts = Counter(current)

    ref_freq = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
    cur_freq = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)

    # Normalise to proportions then scale to current size for chi2
    ref_norm = ref_freq / ref_freq.sum()
    expected = ref_norm * cur_freq.sum()
    # Avoid zero expected cells
    mask = expected > 0
    if mask.sum() < 2:
        return DriftResult(feature_name, "chi2_test", 0.0, 1.0, alpha, False, DriftSeverity.NONE,
                           {"reason": "degenerate_distribution"})

    stat, p_value = stats.chisquare(cur_freq[mask], f_exp=expected[mask])
    drift = p_value < alpha

    severity = _p_to_severity(p_value, alpha) if drift else DriftSeverity.NONE
    return DriftResult(
        feature_name=feature_name,
        test_name="chi2_test",
        statistic=float(stat),
        p_value=float(p_value),
        threshold=alpha,
        drift_detected=drift,
        severity=severity,
        details={
            "categories": all_cats[:20],
            "ref_distribution": {c: float(ref_counts.get(c, 0) / len(reference)) for c in all_cats[:10]},
            "cur_distribution": {c: float(cur_counts.get(c, 0) / len(current)) for c in all_cats[:10]},
        },
    )


def psi_score(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "feature",
    n_bins: int = 10,
) -> DriftResult:
    """
    Population Stability Index (PSI).
    Industry standard for monitoring credit/fraud model stability.

    PSI < 0.1  → No significant shift
    PSI < 0.2  → Moderate shift, investigate
    PSI >= 0.2 → Significant shift, retrain likely needed
    """
    if len(reference) < 20 or len(current) < 20:
        return DriftResult(feature_name, "psi", 0.0, 0.2, 0.2, False, DriftSeverity.NONE,
                           {"reason": "insufficient_data"})

    # Build bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return DriftResult(feature_name, "psi", 0.0, 0.2, 0.2, False, DriftSeverity.NONE,
                           {"reason": "constant_reference"})

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    if psi < 0.1:
        severity = DriftSeverity.NONE
        drift = False
    elif psi < 0.2:
        severity = DriftSeverity.LOW
        drift = True
    elif psi < 0.3:
        severity = DriftSeverity.MEDIUM
        drift = True
    elif psi < 0.5:
        severity = DriftSeverity.HIGH
        drift = True
    else:
        severity = DriftSeverity.CRITICAL
        drift = True

    return DriftResult(
        feature_name=feature_name,
        test_name="psi",
        statistic=psi,
        p_value=None,
        threshold=0.2,
        drift_detected=drift,
        severity=severity,
        details={
            "interpretation": (
                "stable" if psi < 0.1
                else "slight_shift" if psi < 0.2
                else "significant_shift"
            ),
        },
    )


def jensen_shannon_divergence(
    ref_labels: List[str],
    cur_labels: List[str],
    feature_name: str = "prediction_distribution",
) -> DriftResult:
    """
    Jensen-Shannon Divergence between label distributions.
    Symmetric, bounded [0, 1]. Great for prediction distribution monitoring.

    JSD > 0.1  → Notable shift
    JSD > 0.2  → Significant shift (alert)
    """
    all_labels = sorted(set(ref_labels) | set(cur_labels))
    ref_c = Counter(ref_labels)
    cur_c = Counter(cur_labels)

    ref_p = np.array([ref_c.get(l, 0) for l in all_labels], dtype=float)
    cur_p = np.array([cur_c.get(l, 0) for l in all_labels], dtype=float)

    ref_p = ref_p / ref_p.sum() if ref_p.sum() > 0 else ref_p + 1e-10
    cur_p = cur_p / cur_p.sum() if cur_p.sum() > 0 else cur_p + 1e-10

    m = 0.5 * (ref_p + cur_p)
    jsd = float(0.5 * np.sum(ref_p * np.log(ref_p / m + 1e-10)) +
                0.5 * np.sum(cur_p * np.log(cur_p / m + 1e-10)))
    jsd = max(0.0, min(1.0, jsd))  # Clamp to [0,1]

    drift = jsd > 0.1
    if jsd < 0.05:
        severity = DriftSeverity.NONE
    elif jsd < 0.1:
        severity = DriftSeverity.LOW
    elif jsd < 0.2:
        severity = DriftSeverity.MEDIUM
    elif jsd < 0.35:
        severity = DriftSeverity.HIGH
    else:
        severity = DriftSeverity.CRITICAL

    return DriftResult(
        feature_name=feature_name,
        test_name="jensen_shannon_divergence",
        statistic=jsd,
        p_value=None,
        threshold=0.1,
        drift_detected=drift,
        severity=severity,
        details={
            "reference_distribution": {l: round(float(ref_c.get(l, 0)) / len(ref_labels), 3) for l in all_labels},
            "current_distribution": {l: round(float(cur_c.get(l, 0)) / len(cur_labels), 3) for l in all_labels},
        },
    )


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) sequential change-point detector.
    Ideal for catching gradual drifts in a single time-ordered signal
    (e.g., mean confidence score per hour).
    """

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self._pos: float = 0.0
        self._neg: float = 0.0
        self._reference_mean: Optional[float] = None
        self._reference_std: Optional[float] = None
        self._history: deque = deque(maxlen=500)

    def fit(self, reference: np.ndarray):
        self._reference_mean = float(np.mean(reference))
        self._reference_std = max(float(np.std(reference)), 1e-6)
        self._pos = 0.0
        self._neg = 0.0

    def update(self, value: float) -> Tuple[bool, float, float]:
        """
        Add one observation. Returns (alarm, cusum_pos, cusum_neg).
        """
        if self._reference_mean is None:
            self._history.append(value)
            return False, 0.0, 0.0

        z = (value - self._reference_mean) / self._reference_std
        self._pos = max(0.0, self._pos + z - self.drift)
        self._neg = max(0.0, self._neg - z - self.drift)
        self._history.append(value)

        alarm = self._pos > self.threshold or self._neg > self.threshold
        return alarm, self._pos, self._neg

    def reset(self):
        self._pos = 0.0
        self._neg = 0.0


# ─────────────────────────────────────────────
# Drift Analysis Engine
# ─────────────────────────────────────────────

class DriftAnalysisEngine:
    """
    Orchestrates all drift detectors given a reference baseline
    and a window of recent predictions.
    """

    def __init__(
        self,
        model_name: str,
        reference_window: int = 500,
        current_window: int = 100,
        alpha: float = 0.05,
    ):
        self.model_name = model_name
        self.reference_window = reference_window
        self.current_window = current_window
        self.alpha = alpha
        self._cusum_confidence = CUSUMDetector(threshold=5.0, drift=0.3)
        self._baseline_set = False

    def set_baseline(self, records: List[dict]):
        """Call with historical/reference data to establish the healthy baseline."""
        confidences = [r["confidence"] for r in records if r.get("confidence") is not None]
        if confidences:
            self._cusum_confidence.fit(np.array(confidences))
            self._baseline_set = True
        logger.info("Baseline set with %d records for model %s", len(records), self.model_name)

    def analyse(
        self,
        reference_records: List[dict],
        current_records: List[dict],
        timestamp: str,
    ) -> DriftReport:
        """
        Run all drift tests comparing reference vs current records.
        Returns a full DriftReport.
        """
        results: List[DriftResult] = []

        # 1. Prediction label distribution (JSD)
        ref_labels = [r["label"] for r in reference_records if r.get("label")]
        cur_labels = [r["label"] for r in current_records if r.get("label")]
        if ref_labels and cur_labels:
            results.append(jensen_shannon_divergence(ref_labels, cur_labels, "prediction_labels"))

        # 2. Confidence score drift (KS + PSI)
        ref_conf = np.array([r["confidence"] for r in reference_records if r.get("confidence") is not None])
        cur_conf = np.array([r["confidence"] for r in current_records if r.get("confidence") is not None])
        if len(ref_conf) >= 10 and len(cur_conf) >= 10:
            results.append(ks_test(ref_conf, cur_conf, "confidence_score", self.alpha))
            results.append(psi_score(ref_conf, cur_conf, "confidence_psi"))

        # 3. Per-feature drift (numeric features in inputs)
        feature_names = self._collect_numeric_features(reference_records, current_records)
        for feat in feature_names[:15]:  # Cap at 15 features to avoid noise overload
            ref_vals = np.array([r["inputs"].get(feat) for r in reference_records
                                  if r.get("inputs") and r["inputs"].get(feat) is not None], dtype=float)
            cur_vals = np.array([r["inputs"].get(feat) for r in current_records
                                  if r.get("inputs") and r["inputs"].get(feat) is not None], dtype=float)
            if len(ref_vals) >= 10 and len(cur_vals) >= 10:
                ks = ks_test(ref_vals, cur_vals, f"input:{feat}", self.alpha)
                psi = psi_score(ref_vals, cur_vals, f"input:{feat}:psi")
                # Only surface the worse of the two
                if ks.severity.value >= psi.severity.value:
                    results.append(ks)
                else:
                    results.append(psi)

        # 4. Latency drift
        ref_lat = np.array([r["latency_ms"] for r in reference_records if r.get("latency_ms") is not None])
        cur_lat = np.array([r["latency_ms"] for r in current_records if r.get("latency_ms") is not None])
        if len(ref_lat) >= 10 and len(cur_lat) >= 10:
            results.append(ks_test(ref_lat, cur_lat, "latency_ms", self.alpha))

        # 5. Error rate anomaly
        ref_err_rate = sum(1 for r in reference_records if r.get("error")) / max(len(reference_records), 1)
        cur_err_rate = sum(1 for r in current_records if r.get("error")) / max(len(current_records), 1)
        err_delta = cur_err_rate - ref_err_rate
        if err_delta > 0.05:
            severity = DriftSeverity.HIGH if err_delta > 0.15 else DriftSeverity.MEDIUM
            results.append(DriftResult(
                feature_name="error_rate",
                test_name="error_rate_delta",
                statistic=err_delta,
                p_value=None,
                threshold=0.05,
                drift_detected=True,
                severity=severity,
                details={"reference_error_rate": round(ref_err_rate, 4), "current_error_rate": round(cur_err_rate, 4)},
            ))

        overall_severity = self._aggregate_severity(results)
        summary = self._build_summary(results, overall_severity)

        return DriftReport(
            model_name=self.model_name,
            timestamp=timestamp,
            results=results,
            overall_severity=overall_severity,
            summary=summary,
        )

    def _collect_numeric_features(self, ref_records: List[dict], cur_records: List[dict]) -> List[str]:
        feature_names: set = set()
        for records in [ref_records[:50], cur_records[:50]]:
            for r in records:
                if r.get("inputs"):
                    for k, v in r["inputs"].items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            feature_names.add(k)
        return sorted(feature_names)

    def _aggregate_severity(self, results: List[DriftResult]) -> DriftSeverity:
        if not results:
            return DriftSeverity.NONE
        order = [DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MEDIUM,
                 DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        max_sev = max(results, key=lambda r: order.index(r.severity)).severity
        return max_sev

    def _build_summary(self, results: List[DriftResult], severity: DriftSeverity) -> str:
        drifted = [r for r in results if r.drift_detected]
        if not drifted:
            return f"No drift detected across {len(results)} monitored signals."
        features = ", ".join(r.feature_name for r in drifted[:5])
        extra = f" (+{len(drifted)-5} more)" if len(drifted) > 5 else ""
        return (
            f"DRIFT DETECTED [{severity.value.upper()}] — "
            f"{len(drifted)}/{len(results)} signals drifted: {features}{extra}"
        )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _p_to_severity(p_value: float, alpha: float) -> DriftSeverity:
    ratio = alpha / max(p_value, 1e-10)
    if ratio < 2:
        return DriftSeverity.LOW
    elif ratio < 10:
        return DriftSeverity.MEDIUM
    elif ratio < 100:
        return DriftSeverity.HIGH
    else:
        return DriftSeverity.CRITICAL
