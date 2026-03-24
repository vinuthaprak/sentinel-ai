"""
SentinelAI SDK — Model Instrumentation Layer

Drop-in decorator to wrap any model inference function and
automatically capture prediction distributions, confidence
scores, latency, and error rates for observability.

Usage:
    from sentinel.sdk import SentinelSDK

    sentinel = SentinelSDK(
        model_name="fraud-detector-v2",
        server_url="http://localhost:8765"
    )

    @sentinel.monitor
    def predict(transaction: dict) -> dict:
        score = model.predict(transaction)
        return {"label": "fraud" if score > 0.5 else "legit", "confidence": score}
"""

import time
import uuid
import asyncio
import logging
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timezone
from collections import deque

import httpx
import numpy as np

logger = logging.getLogger("sentinel.sdk")


class PredictionRecord:
    """Single captured prediction event."""

    def __init__(
        self,
        model_name: str,
        prediction_id: str,
        inputs: Optional[Dict],
        output: Any,
        label: Optional[str],
        confidence: Optional[float],
        latency_ms: float,
        error: Optional[str],
        timestamp: datetime,
        metadata: Optional[Dict] = None,
    ):
        self.model_name = model_name
        self.prediction_id = prediction_id
        self.inputs = inputs or {}
        self.output = output
        self.label = label
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.error = error
        self.timestamp = timestamp
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "prediction_id": self.prediction_id,
            "inputs": self.inputs,
            "output": str(self.output) if self.output is not None else None,
            "label": self.label,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 3),
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class LocalBuffer:
    """
    Thread-safe in-process buffer that flushes to the sentinel server.
    Falls back to local-only mode if the server is unreachable.
    """

    def __init__(self, max_size: int = 500, flush_every: int = 50):
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._flush_every = flush_every
        self._count_since_flush = 0

    def add(self, record: PredictionRecord):
        with self._lock:
            self._buffer.append(record.to_dict())
            self._count_since_flush += 1

    def drain(self) -> List[dict]:
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            self._count_since_flush = 0
            return items

    def should_flush(self) -> bool:
        return self._count_since_flush >= self._flush_every

    def __len__(self):
        return len(self._buffer)


class SentinelSDK:
    """
    Main entry point for model instrumentation.

    Wraps model inference functions to capture:
    - Prediction label / output class distribution
    - Confidence / probability scores
    - Input feature snapshots (for drift detection)
    - Latency per inference
    - Error rate

    Then ships everything to the SentinelAI server for analysis.
    """

    def __init__(
        self,
        model_name: str,
        server_url: str = "http://localhost:8765",
        flush_every: int = 25,
        async_flush: bool = True,
        capture_inputs: bool = True,
        input_sample_rate: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.model_name = model_name
        self.server_url = server_url.rstrip("/")
        self.capture_inputs = capture_inputs
        self.input_sample_rate = input_sample_rate
        self.tags = tags or {}
        self.async_flush = async_flush
        self._buffer = LocalBuffer(max_size=1000, flush_every=flush_every)
        self._flush_lock = threading.Lock()
        self._client = httpx.Client(timeout=3.0)
        self._total_predictions = 0
        self._total_errors = 0
        logger.info(
            "SentinelAI SDK initialised | model=%s | server=%s",
            model_name,
            server_url,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def monitor(self, func: Callable = None, *, label_key: str = "label", confidence_key: str = "confidence"):
        """
        Decorator that instruments a model prediction function.

        The wrapped function should return a dict that (optionally) includes:
            - label_key   (default "label")      : the predicted class
            - confidence_key (default "confidence"): float 0-1 probability

        Example:
            @sentinel.monitor
            def predict(features: dict) -> dict:
                return {"label": "fraud", "confidence": 0.93}

            # or with custom key names:
            @sentinel.monitor(label_key="class", confidence_key="prob")
            def predict(features: dict) -> dict:
                return {"class": "cat", "prob": 0.87}
        """
        if func is None:
            # called as @sentinel.monitor(label_key=...) with arguments
            return functools.partial(self.monitor, label_key=label_key, confidence_key=confidence_key)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._invoke_sync(func, args, kwargs, label_key, confidence_key)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._invoke_async(func, args, kwargs, label_key, confidence_key)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    def record_prediction(
        self,
        *,
        inputs: Optional[Dict] = None,
        output: Any = None,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Manual recording — use when you can't use the decorator."""
        record = PredictionRecord(
            model_name=self.model_name,
            prediction_id=str(uuid.uuid4()),
            inputs=inputs if self._should_sample() else {},
            output=output,
            label=label,
            confidence=confidence,
            latency_ms=latency_ms,
            error=error,
            timestamp=datetime.now(timezone.utc),
            metadata={**self.tags, **(metadata or {})},
        )
        self._buffer.add(record)
        self._total_predictions += 1
        if error:
            self._total_errors += 1
        if self._buffer.should_flush():
            self._maybe_flush()

    def flush(self):
        """Force-flush the local buffer to the server."""
        self._flush_to_server()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _invoke_sync(self, func, args, kwargs, label_key, confidence_key):
        start = time.perf_counter()
        error = None
        output = None
        try:
            output = func(*args, **kwargs)
            return output
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            inputs = self._extract_inputs(args, kwargs)
            label, confidence = self._extract_label_confidence(output, label_key, confidence_key)
            self.record_prediction(
                inputs=inputs,
                output=output,
                label=label,
                confidence=confidence,
                latency_ms=latency_ms,
                error=error,
            )

    async def _invoke_async(self, func, args, kwargs, label_key, confidence_key):
        start = time.perf_counter()
        error = None
        output = None
        try:
            output = await func(*args, **kwargs)
            return output
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            inputs = self._extract_inputs(args, kwargs)
            label, confidence = self._extract_label_confidence(output, label_key, confidence_key)
            self.record_prediction(
                inputs=inputs,
                output=output,
                label=label,
                confidence=confidence,
                latency_ms=latency_ms,
                error=error,
            )

    def _extract_inputs(self, args, kwargs) -> dict:
        if not self.capture_inputs or not self._should_sample():
            return {}
        combined: dict = {}
        for i, arg in enumerate(args):
            if isinstance(arg, dict):
                combined.update(arg)
            elif hasattr(arg, "__dict__"):
                combined.update(vars(arg))
        combined.update({k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))})
        # Flatten numpy types for JSON serialisation
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in combined.items()}

    def _extract_label_confidence(self, output, label_key, confidence_key):
        label = None
        confidence = None
        if isinstance(output, dict):
            label = str(output.get(label_key)) if output.get(label_key) is not None else None
            raw_conf = output.get(confidence_key)
            if raw_conf is not None:
                try:
                    confidence = float(raw_conf)
                except (TypeError, ValueError):
                    pass
        elif isinstance(output, (int, float)):
            confidence = float(output)
        elif isinstance(output, str):
            label = output
        return label, confidence

    def _should_sample(self) -> bool:
        if self.input_sample_rate >= 1.0:
            return True
        return np.random.random() < self.input_sample_rate

    def _maybe_flush(self):
        if self.async_flush:
            t = threading.Thread(target=self._flush_to_server, daemon=True)
            t.start()
        else:
            self._flush_to_server()

    def _flush_to_server(self):
        records = self._buffer.drain()
        if not records:
            return
        try:
            resp = self._client.post(
                f"{self.server_url}/ingest/predictions",
                json={"records": records},
            )
            if resp.status_code != 200:
                logger.warning("Sentinel flush returned HTTP %s", resp.status_code)
        except Exception as exc:
            logger.debug("Sentinel server unreachable, records kept locally: %s", exc)
            # Re-add records to buffer on failure
            with self._buffer._lock:
                for r in records:
                    self._buffer._buffer.appendleft(r)

    def stats(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_predictions": self._total_predictions,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(self._total_predictions, 1),
            "buffer_size": len(self._buffer),
        }

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass
