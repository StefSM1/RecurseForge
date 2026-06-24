"""
harness/dashboard_server.py
============================
FastAPI backend for the RecurseForge Dashboard.

Provides:
  - WebSocket endpoint for real-time event streaming from the engine
  - REST endpoints for health checks, history, and system resources
  - Bridge between the engine's event bus and the browser

Run standalone:
    python harness/dashboard_server.py

The server starts on port 8100. The Vite frontend connects via WebSocket
directly (bypassing Vite proxy for real-time reliability).
"""

import asyncio
import logging
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("recurseforge.dashboard")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_event_log: list[dict] = []
_connected_clients: set[WebSocket] = set()

# Thread-safe queue for bridging engine events to the asyncio event loop.
# The event bus dispatcher thread puts events here; an asyncio task reads
# and broadcasts them to WebSocket clients.
_event_queue: queue.Queue[dict] = queue.Queue()

# Background bridge task reference
_bridge_task: asyncio.Task | None = None
_dashboard_started_bus = False

_chat_runs_lock = threading.Lock()
_chat_runs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Lifespan: start the queue-to-WebSocket bridge task
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the event queue bridge task once uvicorn's event loop is active."""
    global _bridge_task, _dashboard_started_bus
    _dashboard_started_bus = _ensure_event_bus_running()
    _bridge_task = asyncio.create_task(_queue_to_websocket_bridge())
    _start_event_bus_bridge()
    logger.info("[Dashboard] Lifespan: queue bridge task started")
    yield
    # Cleanup on shutdown
    if _bridge_task:
        _bridge_task.cancel()
    _bridge_task = None
    if _dashboard_started_bus:
        try:
            from harness.event_bus import get_event_bus
            get_event_bus().stop()
        except Exception as exc:
            logger.debug("[Dashboard] Event bus shutdown skipped: %s", exc)
    _dashboard_started_bus = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RecurseForge Dashboard", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Queue-to-WebSocket bridge (asyncio task running inside the event loop)
# ---------------------------------------------------------------------------

async def _queue_to_websocket_bridge():
    """
    Background asyncio task that drains the thread-safe event queue
    and broadcasts events to all connected WebSocket clients.

    This runs INSIDE the uvicorn event loop, so it can directly await
    ws.send_json() without any cross-thread scheduling.
    """
    global _connected_clients
    logger.info("[Dashboard] Queue bridge task running")
    while True:
        try:
            # Non-blocking drain of all pending events
            events_sent = 0
            while not _event_queue.empty():
                try:
                    event = _event_queue.get_nowait()
                    _event_log.append(event)
                    events_sent += 1

                    # Broadcast to all connected clients
                    dead = set()
                    for ws in list(_connected_clients):
                        try:
                            await ws.send_json(event)
                        except Exception:
                            dead.add(ws)
                    _connected_clients -= dead
                except queue.Empty:
                    break

            if events_sent > 0:
                logger.info("[Dashboard] Bridge: broadcast %d events to %d clients",
                             events_sent, len(_connected_clients))

            # Keep log bounded
            if len(_event_log) > 1000:
                _event_log[:] = _event_log[-500:]

            # Sleep briefly before next poll (50ms = responsive enough for real-time)
            await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            logger.info("[Dashboard] Queue bridge task cancelled")
            break
        except Exception as e:
            logger.error("[Dashboard] Queue bridge error: %s", e)
            await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time event streaming."""
    await websocket.accept()
    _connected_clients.add(websocket)
    logger.info("[Dashboard] WebSocket client connected (%d total)",
                len(_connected_clients))

    # Send existing event history on connect
    for event in _event_log[-50:]:
        await websocket.send_json(event)

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug("[Dashboard] Received from client: %s", data[:100])
    except WebSocketDisconnect:
        _connected_clients.discard(websocket)
        logger.info("[Dashboard] WebSocket client disconnected (%d remaining)",
                    len(_connected_clients))


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "connected_clients": len(_connected_clients),
        "events_logged": len(_event_log),
        "queue_pending": _event_queue.qsize(),
        "timestamp": time.time(),
    }


@app.get("/api/history")
async def get_history(limit: int = 100):
    """Return recent event history for the History mode."""
    return {
        "events": _event_log[-limit:],
        "total": len(_event_log),
    }


@app.get("/api/resources")
async def get_resources():
    """Return current system resource usage.

    NOTE: psutil.cpu_percent() is blocking, so we run it in a thread executor
    to avoid blocking the asyncio event loop.
    """
    resources = {
        "vram_mb": 0,
        "cpu_percent": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "gpu_percent": 0.0,
        "timestamp": time.time(),
    }

    # Run blocking psutil calls in a thread executor to not block the event loop
    try:
        import psutil
        loop = asyncio.get_running_loop()
        cpu = await loop.run_in_executor(None, psutil.cpu_percent, 0.1)
        resources["cpu_percent"] = cpu
        mem = psutil.virtual_memory()
        resources["ram_used_gb"] = round(mem.used / (1024**3), 2)
        resources["ram_total_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        resources["vram_mb"] = info.used // (1024 * 1024)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        resources["gpu_percent"] = float(util.gpu)
    except Exception:
        pass

    return resources


@app.post("/api/chat/runs")
async def start_chat_run(request_data: dict):
    """Start one dashboard-owned recursive agent run from the chat panel."""
    message = str(request_data.get("message", "")).strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    with _chat_runs_lock:
        active = [
            run for run in _chat_runs.values()
            if run["status"] in {"pending", "running", "stopping"}
        ]
        if active:
            raise HTTPException(
                status_code=409,
                detail="another chat run is already active",
            )

        run_id = "chat-{}".format(uuid4().hex)
        record = {
            "run_id": run_id,
            "prompt": message,
            "status": "pending",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "final_output": None,
            "error": None,
            "stop_requested": False,
        }
        _chat_runs[run_id] = record

    thread = threading.Thread(
        target=_run_chat_graph,
        args=(run_id, message),
        daemon=True,
        name="dashboard-chat-{}".format(run_id[:13]),
    )
    thread.start()

    return _public_chat_run(record)


@app.get("/api/chat/runs/{run_id}")
async def get_chat_run(run_id: str):
    """Return the current status and final output for a chat-launched run."""
    record = _get_chat_run_or_404(run_id)
    return _public_chat_run(record)


@app.post("/api/chat/runs/{run_id}/stop")
async def stop_chat_run(run_id: str):
    """Best-effort stop: marks the run stopped/canceling in dashboard state."""
    with _chat_runs_lock:
        record = _chat_runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")

        record["stop_requested"] = True
        if record["status"] in {"pending", "running"}:
            record["status"] = "stopping"
        elif record["status"] == "stopping":
            pass

    return _public_chat_run(record)


@app.post("/api/test/event")
async def inject_test_event(request_data: dict):
    """Inject a test event into the WebSocket stream (for development/testing)."""
    event = {
        "event_type": request_data.get("event_type", "node_spawn"),
        "payload": request_data.get("payload", {}),
        "timestamp": time.time(),
    }
    # Put directly into the queue -- the bridge task will broadcast it
    _event_queue.put(event)
    return {"status": "ok", "event": event}


@app.post("/api/exit")
async def exit_server():
    """Gracefully shut down the dashboard server and CLI process."""
    import os
    import signal

    logger.info("[Dashboard] Exit requested via /api/exit")

    for ws in list(_connected_clients):
        try:
            await ws.close(code=1000, reason="Server shutting down")
        except Exception:
            pass

    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting_down"}


# ---------------------------------------------------------------------------
# Event bus bridge (engine events -> queue -> WebSocket)
# ---------------------------------------------------------------------------

def _start_event_bus_bridge():
    """Subscribe to the engine event bus and forward events to the queue.

    The queue is drained by the _queue_to_websocket_bridge asyncio task
    running inside the uvicorn event loop.
    """
    try:
        from harness.event_bus import get_event_bus
        from engine.interfaces import EventType

        bus = get_event_bus()

        def on_event(event):
            """Callback from event bus dispatcher thread -> put into queue."""
            event_dict = event.model_dump(mode="json")
            _event_queue.put(event_dict)
            logger.debug("[Dashboard] Bridge: queued event %s", event.event_type)

        for event_type in EventType:
            bus.subscribe(event_type.value, on_event)

        logger.info("[Dashboard] Event bus bridge started (subscribed to %d event types)",
                    len(EventType))
    except ImportError as e:
        logger.warning("[Dashboard] Could not connect to event bus: %s. "
                       "Dashboard will work without live events.", e)


# ---------------------------------------------------------------------------
# Chat run helpers
# ---------------------------------------------------------------------------

def _ensure_event_bus_running() -> bool:
    """Start the event bus when dashboard_server owns the process lifecycle."""
    try:
        from harness.event_bus import get_event_bus
        bus = get_event_bus()
        was_running = getattr(bus, "_running", False)
        bus.start()
        return not was_running
    except Exception as exc:
        logger.warning("[Dashboard] Could not start event bus: %s", exc)
        return False


def _get_chat_run_or_404(run_id: str) -> dict[str, Any]:
    with _chat_runs_lock:
        record = _chat_runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return dict(record)


def _public_chat_run(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "prompt": record["prompt"],
        "status": record["status"],
        "created_at": record["created_at"],
        "started_at": record["started_at"],
        "completed_at": record["completed_at"],
        "final_output": record["final_output"],
        "error": record["error"],
        "stop_requested": record["stop_requested"],
    }


def _set_chat_run(run_id: str, **updates: Any) -> dict[str, Any] | None:
    with _chat_runs_lock:
        record = _chat_runs.get(run_id)
        if record is None:
            return None
        record.update(updates)
        return dict(record)


def _load_dashboard_config() -> dict[str, Any]:
    config_path = _PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _initial_graph_state(run_id: str, prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": "root",
        "run_id": run_id,
        "task_description": prompt,
        "status": "init",
        "children": [],
        "depth": 0,
        "results": [],
        "direct_answer": "",
        "config": config,
    }


def _format_chat_output(result: dict[str, Any]) -> str:
    direct = str(result.get("direct_answer") or "").strip()
    if direct:
        return direct

    results = result.get("results", [])
    if not results:
        return "Run completed without a textual result."

    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        task = str(item.get("task") or item.get("node_id") or "Sub-agent")
        success = bool(item.get("success"))
        result_text = str(item.get("result") or "").strip()
        status = "success" if success else "failed"
        lines.append("[{}] {} ({})".format(index, task, status))
        if result_text:
            lines.append(result_text)
        if item.get("code_executed"):
            stdout = str(item.get("stdout") or "").strip()
            stderr = str(item.get("stderr") or "").strip()
            exit_code = item.get("exit_code", "?")
            lines.append("Sandbox exit: {}".format(exit_code))
            if stdout:
                lines.append("stdout:\n{}".format(stdout))
            if stderr:
                lines.append("stderr:\n{}".format(stderr))
        lines.append("")

    return "\n".join(lines).strip()


def _run_chat_graph(run_id: str, prompt: str) -> None:
    """Invoke the existing recursive graph for a chat-submitted prompt."""
    _set_chat_run(run_id, status="running", started_at=time.time())
    try:
        from engine.graph import build_graph

        config = _load_dashboard_config()
        graph = build_graph(config)
        result = graph.invoke(_initial_graph_state(run_id, prompt, config))
        output = _format_chat_output(result)

        with _chat_runs_lock:
            record = _chat_runs.get(run_id)
            if record is None:
                return
            record["completed_at"] = time.time()
            if record.get("stop_requested"):
                record["status"] = "stopped"
                record["final_output"] = None
            else:
                record["status"] = "success" if result.get("status") == "done" else "failed"
                record["final_output"] = output
    except Exception as exc:
        logger.exception("[Dashboard] Chat run %s failed", run_id)
        with _chat_runs_lock:
            record = _chat_runs.get(run_id)
            if record is None:
                return
            record["completed_at"] = time.time()
            record["status"] = "stopped" if record.get("stop_requested") else "failed"
            record["error"] = str(exc)


# ---------------------------------------------------------------------------
# Main (standalone mode)
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("[Dashboard] Starting RecurseForge Dashboard server on port 8100")
    logger.info("[Dashboard] Frontend expected at http://localhost:5173")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8100,
        log_level="info",
    )


if __name__ == "__main__":
    main()
