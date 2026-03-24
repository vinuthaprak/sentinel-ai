"""
SentinelAI Observability Server

FastAPI application that:
  1. Receives prediction telemetry from instrumented models (via SDK)
  2. Runs drift analysis on a background schedule
  3. Exposes REST + WebSocket APIs for the dashboard
  4. Exposes a /metrics endpoint in Prometheus text format
  5. Serves the single-file dashboard at /

Run:
    uvicorn sentinel.server:app --host 0.0.0.0 --port 8765 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import psutil
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .metrics_store import MetricsStore
from .drift import DriftAnalysisEngine
from .alerts import AlertManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("sentinel.server")

# ─────────────────────────────────────────────
# Global singletons
# ─────────────────────────────────────────────

store = MetricsStore(system_history=300, reference_size=1000, current_size=200)
alert_manager = AlertManager(max_history=500)
_drift_engines: Dict[str, DriftAnalysisEngine] = {}
_ws_connections: List[WebSocket] = []

# ─────────────────────────────────────────────
# Background tasks
# ─────────────────────────────────────────────

async def system_metrics_task():
    """Snapshots system metrics every 5 seconds."""
    while True:
        try:
            store.snapshot_system()
        except Exception as exc:
            logger.error("System snapshot failed: %s", exc)
        await asyncio.sleep(5)


async def drift_analysis_task():
    """Runs drift analysis for all known models every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        for model_name in store.get_model_names():
            try:
                window = store.get_prediction_window(model_name)
                if window and window.has_enough_data(min_reference=30, min_current=10):
                    if model_name not in _drift_engines:
                        _drift_engines[model_name] = DriftAnalysisEngine(model_name)
                    engine = _drift_engines[model_name]
                    report = engine.analyse(
                        window.get_reference(),
                        window.get_current(),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    new_alerts = alert_manager.evaluate(report)
                    if new_alerts or report.overall_severity.value != "none":
                        logger.warning("Drift analysis: %s", report.summary)
                    # Push update to dashboard via WebSocket
                    await broadcast_event("drift_update", {
                        "model_name": model_name,
                        "report": report.to_dict(),
                        "new_alerts": [a.to_dict() for a in new_alerts],
                    })
            except Exception as exc:
                logger.error("Drift analysis failed for %s: %s", model_name, exc)


async def broadcast_event(event_type: str, data: dict):
    """Push a JSON event to all connected WebSocket clients."""
    message = json.dumps({"event": event_type, "data": data, "ts": time.time()})
    disconnected = []
    for ws in _ws_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_connections.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SentinelAI server starting up...")
    store.snapshot_system()
    t1 = asyncio.create_task(system_metrics_task())
    t2 = asyncio.create_task(drift_analysis_task())
    yield
    t1.cancel()
    t2.cancel()
    logger.info("SentinelAI server shutdown.")


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="SentinelAI",
    description="AI Reliability & Observability System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class PredictionBatch(BaseModel):
    records: List[Dict[str, Any]]


class BaselineRequest(BaseModel):
    model_name: str
    records: List[Dict[str, Any]]


class AcknowledgeRequest(BaseModel):
    fingerprint: str


# ─────────────────────────────────────────────
# Dashboard (serves the HTML UI)
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>SentinelAI</h1><p>Dashboard not found. Place dashboard/index.html next to the server.</p>")


# ─────────────────────────────────────────────
# Ingestion endpoints (called by SDK)
# ─────────────────────────────────────────────

@app.post("/ingest/predictions", tags=["ingestion"])
async def ingest_predictions(batch: PredictionBatch):
    """SDK calls this to ship prediction telemetry."""
    store.ingest_predictions(batch.records)
    # Push real-time update to dashboard
    affected_models = list({r.get("model_name", "unknown") for r in batch.records})
    for model_name in affected_models:
        summary = store.get_model_summary(model_name)
        await broadcast_event("prediction_ingested", {"model_name": model_name, "summary": summary})
    return {"status": "ok", "ingested": len(batch.records)}


@app.post("/ingest/baseline", tags=["ingestion"])
async def set_baseline(req: BaselineRequest):
    """Manually set the reference baseline for a model."""
    store.set_model_baseline(req.model_name, req.records)
    return {"status": "ok", "model": req.model_name, "baseline_size": len(req.records)}


# ─────────────────────────────────────────────
# System metrics API
# ─────────────────────────────────────────────

@app.get("/api/system/current", tags=["system"])
async def get_system_current():
    snap = store.get_latest_system()
    if not snap:
        raise HTTPException(status_code=503, detail="No system data yet")
    return snap


@app.get("/api/system/history", tags=["system"])
async def get_system_history(
    last_n: int = 60,
    last_minutes: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
):
    """
    Fetch system metric snapshots with flexible time-range filtering.

    Priority order:
      1. last_minutes=30          → last 30 minutes of data
      2. since=ISO&until=ISO      → explicit timestamp range
      3. last_n=60                → last N snapshots (default)

    Examples:
      /api/system/history?last_minutes=5
      /api/system/history?last_minutes=60
      /api/system/history?since=2026-03-23T10:00:00Z&until=2026-03-23T10:30:00Z
    """
    since_dt = datetime.fromisoformat(since.replace("Z", "+00:00")) if since else None
    until_dt = datetime.fromisoformat(until.replace("Z", "+00:00")) if until else None
    return {
        "snapshots": store.get_system_history(
            last_n=min(last_n, 4320),
            last_minutes=last_minutes,
            since=since_dt,
            until=until_dt,
        ),
        "query": {
            "last_minutes": last_minutes,
            "since": since,
            "until": until,
            "last_n": last_n,
        }
    }


# ─────────────────────────────────────────────
# Model metrics API
# ─────────────────────────────────────────────

@app.get("/api/models", tags=["models"])
async def list_models():
    return {"models": store.get_all_model_summaries()}


@app.get("/api/models/{model_name}", tags=["models"])
async def get_model(model_name: str):
    summary = store.get_model_summary(model_name)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return summary


@app.get("/api/models/{model_name}/predictions", tags=["models"])
async def get_model_predictions(model_name: str, window: str = "current"):
    w = store.get_prediction_window(model_name)
    if not w:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    records = w.get_current() if window == "current" else w.get_reference()
    return {"model_name": model_name, "window": window, "count": len(records), "records": records[-100:]}


@app.post("/api/models/{model_name}/drift", tags=["models"])
async def trigger_drift_analysis(model_name: str):
    """Manually trigger drift analysis for a specific model."""
    window = store.get_prediction_window(model_name)
    if not window:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    if not window.has_enough_data(min_reference=20, min_current=5):
        return {"status": "insufficient_data", "reference_size": len(window.get_reference()),
                "current_size": len(window.get_current())}

    if model_name not in _drift_engines:
        _drift_engines[model_name] = DriftAnalysisEngine(model_name)
    engine = _drift_engines[model_name]
    report = engine.analyse(
        window.get_reference(),
        window.get_current(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    new_alerts = alert_manager.evaluate(report)
    return {
        "status": "ok",
        "report": report.to_dict(),
        "new_alerts": [a.to_dict() for a in new_alerts],
    }


# ─────────────────────────────────────────────
# Alerts API
# ─────────────────────────────────────────────

@app.get("/api/alerts", tags=["alerts"])
async def get_alerts():
    return {
        "counts": alert_manager.get_counts(),
        "active": alert_manager.get_active(),
    }


@app.get("/api/alerts/history", tags=["alerts"])
async def get_alert_history(limit: int = 50):
    return {"alerts": alert_manager.get_history(limit=limit)}


@app.post("/api/alerts/acknowledge", tags=["alerts"])
async def acknowledge_alert(req: AcknowledgeRequest):
    ok = alert_manager.acknowledge(req.fingerprint)
    if not ok:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged"}


@app.post("/api/alerts/resolve", tags=["alerts"])
async def resolve_alert(req: AcknowledgeRequest):
    ok = alert_manager.resolve(req.fingerprint)
    if not ok:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "resolved"}


# ─────────────────────────────────────────────
# Prometheus /metrics endpoint
# ─────────────────────────────────────────────

@app.get("/metrics", response_class=PlainTextResponse, tags=["observability"])
async def prometheus_metrics():
    """Prometheus-compatible text format metrics scrape endpoint."""
    lines = [
        "# HELP sentinel_requests_total Total predictions processed per model",
        "# TYPE sentinel_requests_total counter",
    ]
    for summary in store.get_all_model_summaries():
        model = summary["model_name"].replace("-", "_").replace(".", "_")
        lines.append(f'sentinel_requests_total{{model="{summary["model_name"]}"}} {summary["request_count"]}')

    lines += [
        "# HELP sentinel_error_rate Error rate per model",
        "# TYPE sentinel_error_rate gauge",
    ]
    for summary in store.get_all_model_summaries():
        lines.append(f'sentinel_error_rate{{model="{summary["model_name"]}"}} {summary["error_rate"]}')

    lines += [
        "# HELP sentinel_p95_latency_ms P95 inference latency in milliseconds",
        "# TYPE sentinel_p95_latency_ms gauge",
    ]
    for summary in store.get_all_model_summaries():
        lines.append(f'sentinel_p95_latency_ms{{model="{summary["model_name"]}"}} {summary["p95_latency_ms"]}')

    lines += [
        "# HELP sentinel_avg_confidence Average prediction confidence per model",
        "# TYPE sentinel_avg_confidence gauge",
    ]
    for summary in store.get_all_model_summaries():
        lines.append(f'sentinel_avg_confidence{{model="{summary["model_name"]}"}} {summary["avg_confidence"]}')

    snap = store.get_latest_system()
    if snap:
        lines += [
            f"# HELP sentinel_cpu_pct CPU utilisation %",
            f"# TYPE sentinel_cpu_pct gauge",
            f'sentinel_cpu_pct {snap["cpu_pct"]}',
            f"# HELP sentinel_mem_pct Memory utilisation %",
            f"# TYPE sentinel_mem_pct gauge",
            f'sentinel_mem_pct {snap["mem_pct"]}',
        ]

    lines += [
        "# HELP sentinel_active_alerts Active firing alert count",
        "# TYPE sentinel_active_alerts gauge",
    ]
    counts = alert_manager.get_counts()
    lines.append(f'sentinel_active_alerts{{severity="critical"}} {counts["critical"]}')
    lines.append(f'sentinel_active_alerts{{severity="warning"}} {counts["warning"]}')

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
# WebSocket (real-time dashboard feed)
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_connections.append(websocket)
    logger.info("WebSocket client connected (total=%d)", len(_ws_connections))
    try:
        # Send initial state on connect
        await websocket.send_text(json.dumps({
            "event": "init",
            "data": {
                "models": store.get_all_model_summaries(),
                "alerts": alert_manager.get_active(),
                "system": store.get_latest_system(),
            },
            "ts": time.time(),
        }))
        # Keep connection alive, client can send pings
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping":
                    await websocket.send_text(json.dumps({"event": "pong", "ts": time.time()}))
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({
                    "event": "heartbeat",
                    "data": {
                        "system": store.get_latest_system(),
                        "alerts": alert_manager.get_counts(),
                    },
                    "ts": time.time(),
                }))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)


# ─────────────────────────────────────────────
# Health / info
# ─────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "0.1.0", "uptime_seconds": time.time()}


@app.get("/api/summary", tags=["meta"])
async def summary():
    return {
        "models": store.get_model_names(),
        "model_count": len(store.get_model_names()),
        "alerts": alert_manager.get_counts(),
        "system": store.get_latest_system(),
    }
