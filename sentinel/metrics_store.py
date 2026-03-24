"""
SentinelAI Metrics Store

Thread-safe in-memory time-series store with configurable retention windows.
Holds both system metrics and model prediction records for drift analysis.

Design principles:
  - Zero external dependencies (no Redis, no Postgres required to get started)
  - Rolling windows: reference (historical) + current (recent)
  - O(1) append, O(n) drain — suitable for real-time dashboards
"""

from __future__ import annotations

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger("sentinel.metrics_store")


# ─────────────────────────────────────────────
# System Snapshot
# ─────────────────────────────────────────────

@dataclass
class SystemSnapshot:
    timestamp: float
    cpu_pct: float
    mem_pct: float
    mem_used_mb: float
    request_count: int            # cumulative since start
    error_count: int              # cumulative since start
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "cpu_pct": round(self.cpu_pct, 1),
            "mem_pct": round(self.mem_pct, 1),
            "mem_used_mb": round(self.mem_used_mb, 1),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
        }


# ─────────────────────────────────────────────
# Prediction Window
# ─────────────────────────────────────────────

class PredictionWindow:
    """
    Holds two sliding windows of prediction records per model:
      - reference: the historical "good" baseline (larger window)
      - current: recent activity being compared against baseline

    Auto-promotes current → reference after enough samples accumulate
    so the system is self-calibrating.
    """

    def __init__(
        self,
        model_name: str,
        reference_size: int = 1000,
        current_size: int = 200,
    ):
        self.model_name = model_name
        self._reference: Deque[dict] = deque(maxlen=reference_size)
        self._current: Deque[dict] = deque(maxlen=current_size)
        self._lock = threading.RLock()
        self._total_received = 0
        self._baseline_locked = False   # Once set manually, don't auto-rotate

    def ingest(self, record: dict):
        with self._lock:
            self._current.append(record)
            self._total_received += 1
            # Promote current → reference periodically for self-calibration
            if not self._baseline_locked and self._total_received % 200 == 0:
                self._rotate()

    def ingest_batch(self, records: List[dict]):
        for r in records:
            self.ingest(r)

    def set_reference(self, records: List[dict]):
        """Manually lock in a reference baseline."""
        with self._lock:
            self._reference.clear()
            for r in records:
                self._reference.append(r)
            self._baseline_locked = True
            logger.info("Reference baseline locked for model %s (%d records)", self.model_name, len(records))

    def get_reference(self) -> List[dict]:
        with self._lock:
            return list(self._reference)

    def get_current(self) -> List[dict]:
        with self._lock:
            return list(self._current)

    def has_enough_data(self, min_reference: int = 30, min_current: int = 10) -> bool:
        with self._lock:
            return len(self._reference) >= min_reference and len(self._current) >= min_current

    def stats(self) -> dict:
        with self._lock:
            return {
                "model_name": self.model_name,
                "reference_size": len(self._reference),
                "current_size": len(self._current),
                "total_received": self._total_received,
                "baseline_locked": self._baseline_locked,
            }

    def _rotate(self):
        """Slide the current window into reference."""
        for r in list(self._current):
            self._reference.append(r)
        logger.debug("Auto-rotated current→reference for model %s", self.model_name)


# ─────────────────────────────────────────────
# Rolling Latency Tracker
# ─────────────────────────────────────────────

class LatencyTracker:
    """Tracks latency samples in a rolling window and computes percentiles."""

    def __init__(self, window: int = 1000):
        self._samples: Deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, latency_ms: float):
        with self._lock:
            self._samples.append(latency_ms)

    def percentiles(self) -> Tuple[float, float, float]:
        with self._lock:
            if not self._samples:
                return 0.0, 0.0, 0.0
            import numpy as np
            arr = list(self._samples)
            return (
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 95)),
                float(np.percentile(arr, 99)),
            )


# ─────────────────────────────────────────────
# Central Metrics Store
# ─────────────────────────────────────────────

class MetricsStore:
    """
    The central in-memory store for the entire SentinelAI server.

    Holds:
      - System metrics history (CPU, RAM, request rates)
      - Per-model prediction windows (reference + current)
      - Per-model latency trackers
      - Cumulative counters
    """

    def __init__(
        self,
        system_history: int = 4320,   # keep 4320 snapshots (~6 hours at 5s interval)
        reference_size: int = 1000,
        current_size: int = 200,
    ):
        self._system_history: Deque[SystemSnapshot] = deque(maxlen=system_history)
        self._models: Dict[str, PredictionWindow] = {}
        self._latency: Dict[str, LatencyTracker] = {}
        self._request_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._reference_size = reference_size
        self._current_size = current_size
        self._start_time = time.time()

    # ── System metrics ────────────────────────────────────────

    def snapshot_system(self):
        """Capture current system resource usage. Call this on a timer."""
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        with self._lock:
            # Aggregate latency across all models
            all_latencies: List[float] = []
            for tracker in self._latency.values():
                all_latencies.extend(list(tracker._samples))

            total_req = sum(self._request_counts.values())
            total_err = sum(self._error_counts.values())

            import numpy as np
            if all_latencies:
                p50 = float(np.percentile(all_latencies, 50))
                p95 = float(np.percentile(all_latencies, 95))
                p99 = float(np.percentile(all_latencies, 99))
            else:
                p50 = p95 = p99 = 0.0

            snap = SystemSnapshot(
                timestamp=time.time(),
                cpu_pct=cpu,
                mem_pct=mem.percent,
                mem_used_mb=mem.used / 1024 / 1024,
                request_count=total_req,
                error_count=total_err,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
            )
            self._system_history.append(snap)

    def get_system_history(
        self,
        last_n: int = 60,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        last_minutes: Optional[int] = None,
    ) -> List[dict]:
        with self._lock:
            snaps = list(self._system_history)
            # Time-range filtering
            if last_minutes is not None:
                cutoff = time.time() - last_minutes * 60
                snaps = [s for s in snaps if s.timestamp >= cutoff]
            elif since is not None or until is not None:
                since_ts = since.timestamp() if since else 0
                until_ts = until.timestamp() if until else float("inf")
                snaps = [s for s in snaps if since_ts <= s.timestamp <= until_ts]
            else:
                snaps = snaps[-last_n:]
            return [s.to_dict() for s in snaps]

    def get_latest_system(self) -> Optional[dict]:
        with self._lock:
            if self._system_history:
                return self._system_history[-1].to_dict()
            return None

    # ── Prediction ingestion ──────────────────────────────────

    def ingest_predictions(self, records: List[dict]):
        """Accept a batch of prediction records from the SDK."""
        by_model: Dict[str, List[dict]] = {}
        for r in records:
            by_model.setdefault(r.get("model_name", "unknown"), []).append(r)

        with self._lock:
            for model_name, model_records in by_model.items():
                if model_name not in self._models:
                    self._models[model_name] = PredictionWindow(
                        model_name, self._reference_size, self._current_size
                    )
                    self._latency[model_name] = LatencyTracker()
                    self._request_counts[model_name] = 0
                    self._error_counts[model_name] = 0

                self._models[model_name].ingest_batch(model_records)
                for r in model_records:
                    self._request_counts[model_name] += 1
                    if r.get("error"):
                        self._error_counts[model_name] += 1
                    if r.get("latency_ms"):
                        self._latency[model_name].record(r["latency_ms"])

    # ── Per-model accessors ───────────────────────────────────

    def get_model_names(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())

    def get_prediction_window(self, model_name: str) -> Optional[PredictionWindow]:
        with self._lock:
            return self._models.get(model_name)

    def set_model_baseline(self, model_name: str, records: List[dict]):
        with self._lock:
            if model_name not in self._models:
                self._models[model_name] = PredictionWindow(
                    model_name, self._reference_size, self._current_size
                )
            self._models[model_name].set_reference(records)

    def get_model_summary(self, model_name: str) -> dict:
        with self._lock:
            if model_name not in self._models:
                return {}
            window = self._models[model_name]
            tracker = self._latency.get(model_name)
            p50, p95, p99 = tracker.percentiles() if tracker else (0, 0, 0)
            req = self._request_counts.get(model_name, 0)
            err = self._error_counts.get(model_name, 0)

            current = window.get_current()
            labels = [r["label"] for r in current if r.get("label")]
            confidences = [r["confidence"] for r in current if r.get("confidence") is not None]

            label_dist: Dict[str, int] = {}
            for lbl in labels:
                label_dist[lbl] = label_dist.get(lbl, 0) + 1

            return {
                "model_name": model_name,
                "request_count": req,
                "error_count": err,
                "error_rate": round(err / max(req, 1), 4),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "current_window_size": len(current),
                "reference_window_size": len(window.get_reference()),
                "label_distribution": label_dist,
                "avg_confidence": round(sum(confidences) / max(len(confidences), 1), 4),
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }

    def get_all_model_summaries(self) -> List[dict]:
        with self._lock:
            return [self.get_model_summary(m) for m in self._models.keys()]
