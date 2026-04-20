"""
Leaves OS HTTP API — FastAPI server on 127.0.0.1:8765.

Wraps the agentd Unix socket and the IntentParser pipeline.
All inference work is offloaded to a thread pool to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.intent_parser import IntentParser
from config import AUDIT_DB_PATH, INFERENCE_SERVER_URL
from observability import configure_logging, render_prometheus_text
from cortex.briefing import BriefingGenerator
from cortex.indexer import CortexIndexer
from cortex.adapters.filesystem import FilesystemAdapter
from db.audit import (
    complete_intent,
    get_db,
    get_intent_actions,
    get_intent_transitions,
    get_recent_intents,
    log_intent_created,
)
from db.intent_store import (
    activate_intent,
    deactivate_intent,
    delete_intent,
    fire_intent,
    get_active_intents,
    get_all_intents,
    get_intent,
    store_persistent_intent,
)
from errors import LeavesError, LeavesErrorCode

log = logging.getLogger(__name__)

_SOCK_PATH = pathlib.Path.home() / ".leaves" / "agentd.sock"
_SOCKET_TIMEOUT = 180.0  # covers worst-case inference + execution
_REGISTRY_DIR = pathlib.Path(__file__).parent.parent / "agents" / "registry"
_START_INFERENCE_SCRIPT = (
    pathlib.Path(__file__).parent.parent / "scripts" / "start-inference.sh"
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


_inference_healthy: bool = False
_inference_check_task: Optional[asyncio.Task] = None


async def _inference_health_loop() -> None:
    """Background coroutine that checks inference server health every 30s."""
    global _inference_healthy
    while True:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{INFERENCE_SERVER_URL}/health")
                _inference_healthy = resp.status_code == 200
        except Exception:
            _inference_healthy = False
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_check_task
    configure_logging()
    await _ensure_agentd()
    _inference_check_task = asyncio.create_task(_inference_health_loop())
    yield
    _inference_check_task.cancel()
    try:
        await _inference_check_task
    except asyncio.CancelledError:
        pass


async def _ensure_agentd() -> None:
    if not _SOCK_PATH.exists():
        log.warning(
            "agentd socket not found at %s — start with: python3 agentd.py", _SOCK_PATH
        )
        return
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(_SOCK_PATH)), timeout=3.0
        )
        writer.write(json.dumps({"_ping": True}).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=3.0)
        writer.close()
        await writer.wait_closed()
        resp = json.loads(line)
        if resp.get("_pong"):
            log.info("agentd is up at %s", _SOCK_PATH)
        else:
            log.warning("agentd ping returned unexpected response: %s", resp)
    except Exception as e:
        log.warning("agentd not reachable at startup: %s", e)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Leaves OS API", version="0.1.0", lifespan=lifespan)

# CORS: no credentials, explicit method list, and only the dev-server
# origins we actually support. allow_credentials=True with localhost
# origins was a DNS-rebinding amplifier — see security audit F-4.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# Host-header allowlist — defense against DNS rebinding. The API binds
# to 127.0.0.1, but a rebound attacker domain pointing at our service
# would otherwise be honored. We compare the hostname portion only; any
# port the user runs on is fine. `testserver` is FastAPI's TestClient
# default and is allowed so tests don't have to override the Host header.
# See security audit F-4.
_ALLOWED_HOST_NAMES = frozenset({
    "127.0.0.1", "localhost", "::1",
    # Reserved test hostnames per RFC 6761 / FastAPI + httpx test defaults.
    # No public DNS will ever resolve these to a real host, so allowing them
    # lets in-process ASGI tests run without overriding the Host header.
    "testserver", "test",
})


def _host_allowed(host_header: str) -> bool:
    if not host_header:
        return False
    # Strip port. IPv6 hosts arrive as "[::1]:8765"; handle the bracketed form.
    h = host_header.strip().lower()
    if h.startswith("["):
        end = h.find("]")
        if end == -1:
            return False
        return h[1:end] in _ALLOWED_HOST_NAMES
    name = h.split(":", 1)[0]
    return name in _ALLOWED_HOST_NAMES


@app.middleware("http")
async def _enforce_host_header(request, call_next):
    if not _host_allowed(request.headers.get("host", "")):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=421,
            content={"detail": "Host header not allowed"},
        )
    return await call_next(request)

# Singletons — created lazily on first request
_parser: Optional[IntentParser] = None
_db = None
_indexer: Optional[CortexIndexer] = None


def _get_parser() -> IntentParser:
    global _parser
    if _parser is None:
        _parser = IntentParser()
    return _parser


def _get_db():
    global _db
    if _db is None:
        _db = get_db()
    return _db


def _get_indexer() -> CortexIndexer:
    global _indexer
    if _indexer is None:
        _lance_path = pathlib.Path(__file__).parent.parent / "rag" / ".lancedb"
        _indexer = CortexIndexer(_get_db(), _lance_path)
        _indexer.register(FilesystemAdapter())
    return _indexer


# ---------------------------------------------------------------------------
# Plan store (in-memory, thread-safe, TTL-based)
# ---------------------------------------------------------------------------

_plan_store: dict[str, dict] = {}
_plan_store_lock = threading.Lock()
_PLAN_TTL_MINUTES = 5


def _store_plan(intent_id: str, goal_spec: dict) -> None:
    with _plan_store_lock:
        _plan_store[intent_id] = {
            "goal_spec": goal_spec,
            "expires_at": datetime.utcnow() + timedelta(minutes=_PLAN_TTL_MINUTES),
        }
        # Evict expired plans while we have the lock
        now = datetime.utcnow()
        expired = [k for k, v in _plan_store.items() if v["expires_at"] < now]
        for k in expired:
            del _plan_store[k]


def _retrieve_plan(intent_id: str) -> dict | None:
    with _plan_store_lock:
        entry = _plan_store.get(intent_id)
        if entry is None:
            return None
        if entry["expires_at"] < datetime.utcnow():
            del _plan_store[intent_id]
            return None
        del _plan_store[intent_id]  # single-use
        return entry["goal_spec"]


# ---------------------------------------------------------------------------
# Socket helper
# ---------------------------------------------------------------------------


async def _send_goalspec(goal_spec: dict) -> tuple[dict, str, dict]:
    """
    Send a GoalSpec to the agentd Unix socket and return
    (results, summary, sandbox). `sandbox` reflects ground-truth Landlock
    state from the runner subprocess: {"active": bool, "reason": str,
    "authorized_resources": [str]}. On an old agentd that doesn't report
    sandbox, a safe default is substituted so the API response still
    carries honest telemetry rather than a hardcoded True.
    Raises LeavesError on connection failure or agentd error response.
    """
    _16MB = 16 * 1024 * 1024
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(_SOCK_PATH), limit=_16MB), timeout=5.0
        )
    except (FileNotFoundError, ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
        raise LeavesError(
            LeavesErrorCode.AGENT_NOT_AVAILABLE,
            detail=f"agentd not reachable: {e}",
        )

    payload = (
        json.dumps({"goal_spec": goal_spec, "from_state": "PARSING"}).encode() + b"\n"
    )
    try:
        writer.write(payload)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=_SOCKET_TIMEOUT)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    resp = json.loads(line)
    if resp.get("ok"):
        sandbox = resp.get("sandbox") or {
            "active": False,
            "reason": "agentd_did_not_report",
            "authorized_resources": goal_spec.get("authorization", {}).get("resources", []),
        }
        return resp["results"], resp["summary"], sandbox

    code_str = resp.get("code", "INTERNAL_ERROR")
    try:
        code = LeavesErrorCode[code_str]
    except KeyError:
        code = LeavesErrorCode.INTERNAL_ERROR
    raise LeavesError(code, detail=resp.get("detail") or resp.get("error"))


async def _fetch_session_context() -> str | None:
    """
    Ask agentd for the live session_context prompt block (from the
    compositor event watcher). Best-effort: returns None on any failure.
    The block is injected into the L2 system prompt so the parser can
    reference open windows, focused app, and recent process exits.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(_SOCK_PATH)), timeout=1.0
        )
        writer.write(json.dumps({"_get_session_context": True}).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=1.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        resp = json.loads(line)
        block = resp.get("session_context") or ""
        return block or None
    except Exception:
        return None


async def _notify_watcher_reload() -> None:
    """Tell agentd to reload filesystem watches. Best-effort — silently ignores failures."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(_SOCK_PATH)), timeout=3.0
        )
        writer.write(json.dumps({"_reload_watcher": True}).encode() + b"\n")
        await writer.drain()
        await asyncio.wait_for(reader.readline(), timeout=3.0)
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass  # watcher reload is best-effort


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IntentRequest(BaseModel):
    text: str
    wayland_display: str | None = None


class ExecuteRequest(BaseModel):
    intent_id: str
    confirmed: bool = True


class PersistRequest(BaseModel):
    name: str
    text: str
    trigger_type: str = "manual"
    trigger_config: Optional[dict] = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Injection detection — scans file content in execution results
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?prior\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+(a\s+)?new\s+mode", re.IGNORECASE),
    re.compile(r"new\s+system\s+prompt:", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[hidden:?\s*", re.IGNORECASE),
    re.compile(r"IGNORE\s+ALL\s+PREVIOUS", re.IGNORECASE),
    re.compile(r"read\s+~/\.ssh", re.IGNORECASE),
    re.compile(r"cat\s+~/\.ssh", re.IGNORECASE),
    re.compile(r"exfil", re.IGNORECASE),
    re.compile(r"write\s+(its\s+)?contents?\s+to\s+/tmp", re.IGNORECASE),
]


def _scan_for_injections(results: dict) -> tuple[bool, str]:
    """
    Scan execution results for prompt injection patterns.
    Returns (detected: bool, injection_content: str).
    """
    for action_id, result in results.items():
        if not isinstance(result, dict):
            continue
        content = result.get("content", "")
        if not content:
            continue
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(content)
            if match:
                # Extract context around the match
                start = max(0, match.start() - 20)
                end = min(len(content), match.end() + 80)
                snippet = content[start:end].strip()
                return True, snippet
    return False, ""


def _format_result_text(goal_spec: dict, results: dict) -> str:
    """Build a human-readable one-liner from execution results."""
    parts: list[str] = []
    for action in goal_spec.get("actions", []):
        aid = action.get("action_id", "")
        r = results.get(aid)
        if not isinstance(r, dict):
            continue
        if "error" in r:
            parts.append(r["error"])
            continue
        agent = action.get("agent", "")
        atype = action.get("type", "").upper()
        if agent == "system":
            qt = action.get("params", {}).get("query_type", "")
            if "launched" in r:
                parts.append(f"Launched {r['launched']} (PID {r.get('pid', '?')})")
            elif "terminated" in r:
                parts.append(f"Terminated {r.get('target', '?')} ({r.get('count', 0)} process{'es' if r.get('count', 1) != 1 else ''})")
            elif qt == "cpu" or "usage_percent" in r:
                lines = []
                if r.get("model"):
                    lines.append(r["model"])
                usage = r.get("usage_percent")
                freq = r.get("freq_mhz")
                phys = r.get("cores_physical") or r.get("cores")
                logi = r.get("cores_logical")
                temp = r.get("temp_celsius")
                if phys and logi and phys != logi:
                    lines.append(f"{phys} cores / {logi} threads")
                elif phys:
                    lines.append(f"{phys} cores")
                if freq:
                    lines.append(f"{freq} MHz")
                if usage is not None:
                    lines.append(f"Usage: {usage}%")
                if temp:
                    lines.append(f"Temp: {temp}°C")
                parts.append("\n".join(lines) if lines else "CPU info unavailable")
            elif qt == "memory" or ("percent" in r and "total_gb" in r and "available_gb" in r):
                parts.append(
                    f"RAM: {r.get('used_gb', '?')} GB used / "
                    f"{r.get('total_gb', '?')} GB total ({r.get('percent', '?')}%)\n"
                    f"Available: {r.get('available_gb', '?')} GB"
                )
            elif qt == "disk" or ("percent" in r and "free_gb" in r):
                parts.append(
                    f"Disk ({r.get('path', '/')}):\n"
                    f"{r.get('used_gb', '?')} GB used / "
                    f"{r.get('total_gb', '?')} GB total ({r.get('percent', '?')}%)\n"
                    f"Free: {r.get('free_gb', '?')} GB"
                )
            elif qt == "uptime" or "uptime_seconds" in r:
                h = int(r["uptime_seconds"]) // 3600
                m = (int(r["uptime_seconds"]) % 3600) // 60
                parts.append(f"Uptime: {h}h {m}m\nBoot: {r.get('boot_time_iso', '?')}")
            elif qt == "processes" or "processes" in r:
                procs = r.get("processes", [])
                lines = [f"{r.get('count', len(procs))} processes:"]
                for p in procs[:10]:
                    lines.append(
                        f"  {p.get('name', '?'):20s}  "
                        f"CPU {p.get('cpu_percent', 0):5.1f}%  "
                        f"MEM {p.get('memory_mb', 0):7.1f} MB"
                    )
                parts.append("\n".join(lines))
            else:
                # Unknown system result — show as key-value pairs
                lines = [f"  {k}: {v}" for k, v in r.items() if k != "error"]
                parts.append("\n".join(lines) if lines else str(r))
        elif agent == "file":
            if atype == "QUERY" and "files" in r:
                n = r.get("count", len(r["files"]))
                if n == 0:
                    parts.append("No files found")
                else:
                    # Show all files — the compositor's expanded card
                    # view handles scrolling for large result sets.
                    names = "\n".join(
                        f"  {f.get('name', f)}" if isinstance(f, dict) else f"  {f}"
                        for f in r["files"]
                    )
                    parts.append(f"{n} file(s):\n{names}")
            elif atype == "READ" and "content" in r:
                length = r.get("content_length", len(r.get("content", "")))
                parts.append(f"Read {length} chars")
            elif atype in ("WRITE", "DELETE", "MOVE", "COPY"):
                if r.get("created") == "directory":
                    parts.append(f"Created folder: {r.get('path', '?')}")
                else:
                    parts.append(f"{atype.capitalize()} done")
        elif agent == "web":
            if "results" in r:
                n = r.get("result_count", len(r["results"]))
                parts.append(f"{n} web result(s)")
            elif "content" in r:
                parts.append(f"Fetched {r.get('url', 'page')}")
    return " · ".join(parts) if parts else ""


def _build_result_response(
    goal_spec: dict,
    results: dict,
    summary: str,
    duration_ms: float,
    sandbox: dict,
) -> dict:
    injection_detected, injection_content = _scan_for_injections(results)
    auth = goal_spec.get("authorization", {})
    result_text = _format_result_text(goal_spec, results)

    resp = {
        "intent_id": goal_spec["intent_id"],
        "status": "done",
        "natural_text": goal_spec["natural_text"],
        "actions": [
            {
                "action_id": a.get("action_id"),
                "type": a.get("type"),
                "agent": a.get("agent"),
                "params": a.get("params", {}),
            }
            for a in goal_spec.get("actions", [])
        ],
        "authorization": auth,
        "results": results,
        "summary": summary,
        "result_text": result_text,
        "duration_ms": round(duration_ms, 1),
        "model_latency_ms": round(
            goal_spec.get("metadata", {}).get("parse_latency_ms", 0), 1
        ),
        "sandbox_active": bool(sandbox.get("active", False)),
        "sandbox_reason": sandbox.get("reason", "unknown"),
        "authorized_paths": sandbox.get("authorized_resources")
            or auth.get("resources", []),
    }

    if injection_detected:
        resp["injection_detected"] = True
        resp["injection_content"] = injection_content

    return resp


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/intent/plan")
async def plan_intent(body: IntentRequest):
    """
    Parse intent into GoalSpec without executing.
    Returns the plan for user review before execution.
    """
    parser = _get_parser()
    session_context = await _fetch_session_context()
    try:
        goal_spec = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: parser.parse(body.text, session_context=session_context),
        )
    except LeavesError as e:
        if e.code == LeavesErrorCode.NOT_IMPLEMENTED:
            return {
                "intent_id": None,
                "status": "not_implemented",
                "message": e.user_message,
            }
        raise HTTPException(status_code=500, detail=e.user_message)

    intent_id = goal_spec["intent_id"]

    # Thread compositor's WAYLAND_DISPLAY into goal_spec so launched apps
    # connect to the correct compositor session.
    if body.wayland_display:
        goal_spec.setdefault("metadata", {})["wayland_display"] = body.wayland_display

    # Force preview_required for destructive actions — the compositor's
    # Y/N overlay is the actual safety layer, not the API's confirmed flag.
    _DESTRUCTIVE = {"DELETE", "MOVE", "WRITE", "COPY"}
    has_destructive = any(
        a.get("type", "").upper() in _DESTRUCTIVE
        for a in goal_spec.get("actions", [])
    )
    if has_destructive:
        goal_spec.setdefault("authorization", {})["preview_required"] = True

    _store_plan(intent_id, goal_spec)

    return {
        "intent_id": intent_id,
        "status": "planned",
        "natural_text": goal_spec["natural_text"],
        "category": goal_spec["category"],
        "actions": [
            {
                "action_id": a["action_id"],
                "type": a["type"],
                "agent": a["agent"],
                "params": a.get("params", {}),
            }
            for a in goal_spec.get("actions", [])
        ],
        "authorization": goal_spec.get("authorization", {}),
        "model_latency_ms": round(
            goal_spec.get("metadata", {}).get("parse_latency_ms", 0), 1
        ),
        "plan_expires_in_seconds": _PLAN_TTL_MINUTES * 60,
    }


@app.post("/v1/intent/execute")
async def execute_intent(body: ExecuteRequest):
    """
    Execute a previously planned intent by intent_id.
    The plan must have been created by /v1/intent/plan within
    the last 5 minutes.
    """
    goal_spec = _retrieve_plan(body.intent_id)
    if goal_spec is None:
        raise HTTPException(
            status_code=404,
            detail="Plan not found or expired. "
            "Re-submit via /v1/intent/plan first.",
        )

    t_start = time.monotonic()
    db = _get_db()
    intent_id = goal_spec["intent_id"]
    log_intent_created(db, intent_id, goal_spec.get("natural_text", ""), goal_spec)

    sandbox: dict = {
        "active": False, "reason": "not_executed",
        "authorized_resources": goal_spec.get("authorization", {}).get("resources", []),
    }
    try:
        results, summary, sandbox = await _send_goalspec(goal_spec)
        status = "done"
        db_state = "DONE"
    except LeavesError as e:
        results = {}
        summary = e.user_message
        status = "failed"
        db_state = "FAILED"

    duration_ms = (time.monotonic() - t_start) * 1000
    complete_intent(db, intent_id, db_state, summary, duration_ms)

    if status == "failed":
        return {
            "intent_id": intent_id,
            "status": "failed",
            "natural_text": goal_spec.get("natural_text", ""),
            "summary": summary,
            "duration_ms": round(duration_ms, 1),
        }
    return _build_result_response(goal_spec, results, summary, duration_ms, sandbox)


@app.post("/v1/intent")
async def submit_intent(body: IntentRequest, auto_execute: bool = True):
    """
    Convenience endpoint: plan then execute in one call.
    Use ?auto_execute=false to plan only (same as /v1/intent/plan).
    """
    plan = await plan_intent(body)
    if plan.get("status") == "not_implemented":
        return plan
    if not auto_execute:
        return plan
    execute_body = ExecuteRequest(intent_id=plan["intent_id"], confirmed=True)
    return await execute_intent(execute_body)


@app.get("/v1/history")
async def get_history(limit: int = Query(default=20, ge=1, le=200)):
    db = _get_db()
    rows = get_recent_intents(db, limit)
    return [
        {
            "intent_id": r["intent_id"],
            "natural_text": r["natural_text"],
            "state": r["state"],
            "duration_ms": r["duration_ms"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


_NON_DESTRUCTIVE_TYPES = {"QUERY", "READ"}


@app.get("/v1/history/{intent_id}/detail")
async def get_history_detail(intent_id: str):
    """
    Full audit trace for a past intent: original goal_spec, per-action
    timing and results, state transitions, and any logged errors.
    Powers the compositor's time-machine pane.
    """
    db = _get_db()
    row = db.execute(
        "SELECT * FROM intents WHERE intent_id = ?", (intent_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Intent not found")

    actions = get_intent_actions(db, intent_id)
    transitions = get_intent_transitions(db, intent_id)
    error_rows = db.execute(
        """
        SELECT error_code, error_detail, occurred_at
        FROM errors WHERE intent_id = ? ORDER BY occurred_at ASC
        """,
        (intent_id,),
    ).fetchall()

    goal_spec: Optional[dict] = None
    if row["goal_spec_json"]:
        try:
            goal_spec = json.loads(row["goal_spec_json"])
        except json.JSONDecodeError:
            goal_spec = None

    def _parse_json(raw):
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _dur(a):
        if a["completed_at"] and a["started_at"]:
            return round((a["completed_at"] - a["started_at"]) * 1000, 1)
        return None

    return {
        "intent_id": intent_id,
        "natural_text": row["natural_text"],
        "category": row["category"],
        "state": row["state"],
        "result_message": row["result_message"],
        "duration_ms": row["duration_ms"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "goal_spec": goal_spec,
        "actions": [
            {
                "action_id": a["action_id"],
                "type": a["action_type"],
                "agent": a["agent"],
                "params": _parse_json(a["params_json"]) or {},
                "result": _parse_json(a["result_json"]),
                "error_code": a["error_code"],
                "error_detail": a["error_detail"],
                "started_at": a["started_at"],
                "completed_at": a["completed_at"],
                "duration_ms": _dur(a),
            }
            for a in actions
        ],
        "transitions": transitions,
        "errors": [dict(e) for e in error_rows],
    }


@app.post("/v1/history/{intent_id}/replay")
async def replay_history(intent_id: str):
    """
    Re-execute a stored GoalSpec under a new intent_id and report a
    structural diff against the original run.

    Safety: replay is only allowed when every planned action is
    non-destructive (QUERY or READ). For destructive intents, return
    422 — the user should re-plan via /v1/intent/plan so they see the
    preview overlay before re-deleting/re-moving data.
    """
    import uuid

    db = _get_db()
    row = db.execute(
        "SELECT * FROM intents WHERE intent_id = ?", (intent_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    if not row["goal_spec_json"]:
        raise HTTPException(
            status_code=422,
            detail="Original intent has no stored GoalSpec — nothing to replay.",
        )
    try:
        original_spec = json.loads(row["goal_spec_json"])
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Stored GoalSpec is malformed: {e}"
        )

    planned_actions = original_spec.get("actions", [])
    non_destructive = all(
        (a.get("type") or "").upper() in _NON_DESTRUCTIVE_TYPES
        for a in planned_actions
    )
    if not non_destructive or not planned_actions:
        return {
            "status": "refused",
            "reason": "destructive_replay_blocked",
            "detail": (
                "Replay is restricted to QUERY/READ intents. "
                "Re-plan via /v1/intent/plan to see the preview overlay."
            ),
            "original": {
                "intent_id": intent_id,
                "natural_text": row["natural_text"],
                "state": row["state"],
                "action_types": [a.get("type") for a in planned_actions],
            },
        }

    new_intent_id = str(uuid.uuid4())
    new_spec = json.loads(json.dumps(original_spec))  # deep copy
    new_spec["intent_id"] = new_intent_id
    new_spec.setdefault("metadata", {})["replayed_from"] = intent_id

    t_start = time.monotonic()
    log_intent_created(db, new_intent_id, new_spec.get("natural_text", ""), new_spec)

    try:
        results, summary, sandbox = await _send_goalspec(new_spec)
        new_state = "DONE"
        new_status = "done"
    except LeavesError as e:
        results, sandbox = {}, {
            "active": False, "reason": "not_executed", "authorized_resources": []
        }
        summary = e.user_message
        new_state = "FAILED"
        new_status = "failed"

    new_duration_ms = (time.monotonic() - t_start) * 1000
    complete_intent(db, new_intent_id, new_state, summary, new_duration_ms)

    return {
        "status": new_status,
        "matches_state": new_state == row["state"],
        "original": {
            "intent_id": intent_id,
            "natural_text": row["natural_text"],
            "state": row["state"],
            "result_message": row["result_message"],
            "duration_ms": row["duration_ms"],
        },
        "replay": {
            "intent_id": new_intent_id,
            "state": new_state,
            "result_message": summary,
            "duration_ms": round(new_duration_ms, 1),
            "sandbox_active": bool(sandbox.get("active", False)),
        },
    }


@app.get("/v1/agents")
async def get_agents():
    manifests = []
    for p in sorted(_REGISTRY_DIR.glob("*.json")):
        try:
            manifests.append(json.loads(p.read_text()))
        except Exception:
            pass
    return manifests


@app.get("/v1/metrics")
async def get_metrics():
    """
    Prometheus 0.0.4 text exposition. Counters and the inference-latency
    histogram are exposed; scrape with `curl http://127.0.0.1:8765/v1/metrics`.
    """
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=render_prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/v1/health")
async def get_health():
    agentd_ok = _SOCK_PATH.exists()

    issues = []
    if not agentd_ok:
        issues.append("agentd not running — start with: python3 agentd.py")
    if not _inference_healthy:
        issues.append("inference server not running — start with: bash scripts/start-inference.sh")

    return {
        "status": "ok" if (agentd_ok and _inference_healthy) else "degraded",
        "agentd": agentd_ok,
        "inference": _inference_healthy,
        "model": _read_model_from_script(),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Cortex — knowledge graph search / timeline / status
# ---------------------------------------------------------------------------


@app.get("/v1/cortex/search")
async def cortex_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(default=10, ge=1, le=100),
    source_type: Optional[str] = None,
):
    """Semantic search over the knowledge graph."""
    indexer = _get_indexer()
    source_types = [source_type] if source_type else None
    results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: indexer.search(q, top_k=top_k, source_types=source_types)
    )
    return {"query": q, "count": len(results), "results": results}


@app.get("/v1/cortex/timeline")
async def cortex_timeline(
    hours: int = Query(default=24, ge=1, le=720),
    source_type: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
):
    """What changed in the last N hours?"""
    indexer = _get_indexer()
    source_types = [source_type] if source_type else None
    results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: indexer.timeline(hours=hours, source_types=source_types, limit=limit)
    )
    return {"hours": hours, "count": len(results), "results": results}


@app.get("/v1/cortex/status")
async def cortex_status():
    """Knowledge graph indexing statistics."""
    indexer = _get_indexer()
    return await asyncio.get_event_loop().run_in_executor(None, indexer.status)


@app.post("/v1/cortex/index")
async def cortex_index(full: bool = False):
    """Trigger an indexing run. full=true for full re-index."""
    indexer = _get_indexer()
    if full:
        stats = await asyncio.get_event_loop().run_in_executor(None, indexer.run_once)
    else:
        stats = await asyncio.get_event_loop().run_in_executor(None, indexer.run_incremental)
    return stats


@app.get("/v1/cortex/briefing")
async def cortex_briefing(hours: int = Query(default=12, ge=1, le=720)):
    """Generate a briefing of recent changes."""
    indexer = _get_indexer()
    bg = BriefingGenerator(indexer._kg)
    briefing = await asyncio.get_event_loop().run_in_executor(
        None, lambda: bg.generate(hours=hours)
    )
    return briefing


# ---------------------------------------------------------------------------
# Persistent intents
# ---------------------------------------------------------------------------


@app.post("/v1/intent/persist")
async def persist_intent(body: PersistRequest):
    """Parse an intent and store it as a persistent intent with a trigger."""
    parser = _get_parser()
    db = _get_db()
    session_context = await _fetch_session_context()

    try:
        goal_spec = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: parser.parse(body.text, session_context=session_context),
        )
    except LeavesError as e:
        if e.code == LeavesErrorCode.NOT_IMPLEMENTED:
            raise HTTPException(status_code=400, detail=e.user_message)
        raise HTTPException(status_code=500, detail=e.user_message)

    # Validate trigger config for filesystem triggers
    if body.trigger_type == "filesystem":
        if not body.trigger_config or "path" not in body.trigger_config:
            raise HTTPException(
                status_code=400,
                detail="filesystem trigger requires trigger_config with 'path'",
            )

    intent_id = store_persistent_intent(
        db,
        name=body.name,
        goalspec=goal_spec,
        trigger_type=body.trigger_type,
        trigger_config=body.trigger_config,
    )

    # Notify agentd to reload filesystem watches
    await _notify_watcher_reload()

    return {"id": intent_id, "status": "stored", "name": body.name}


@app.get("/v1/intent/persistent")
async def list_persistent_intents(active_only: bool = True):
    db = _get_db()
    if active_only:
        intents = get_active_intents(db)
    else:
        intents = get_all_intents(db)
    return [
        {
            "id": i["id"],
            "name": i["name"],
            "trigger_type": i["trigger_type"],
            "trigger_config": i.get("trigger_config"),
            "active": i["active"],
            "fire_count": i["fire_count"],
            "last_fired": i.get("last_fired"),
            "created_at": i["created_at"],
        }
        for i in intents
    ]


@app.get("/v1/intent/persistent/{intent_id}")
async def get_persistent_intent(intent_id: str):
    db = _get_db()
    intent = get_intent(db, intent_id)
    if intent is None:
        raise HTTPException(status_code=404, detail="Persistent intent not found")
    return intent


@app.delete("/v1/intent/persistent/{intent_id}")
async def delete_persistent_intent(intent_id: str):
    db = _get_db()
    try:
        delete_intent(db, intent_id)
    except LeavesError as e:
        if e.code == LeavesErrorCode.INTENT_NOT_FOUND:
            raise HTTPException(status_code=404, detail=e.user_message)
        raise HTTPException(status_code=500, detail=e.user_message)
    return {"status": "deleted", "id": intent_id}


@app.post("/v1/intent/persistent/{intent_id}/fire")
async def fire_persistent_intent(intent_id: str):
    """Manually fire a persistent intent — execute its GoalSpec now."""
    db = _get_db()
    intent = get_intent(db, intent_id)
    if intent is None:
        raise HTTPException(status_code=404, detail="Persistent intent not found")

    goal_spec = intent["goalspec"]
    fire_intent(db, intent_id)

    t_start = time.monotonic()
    gs_intent_id = goal_spec["intent_id"]
    log_intent_created(db, gs_intent_id, goal_spec.get("natural_text", ""), goal_spec)

    try:
        results, summary, _sandbox = await _send_goalspec(goal_spec)
        status = "done"
        db_state = "DONE"
    except LeavesError as e:
        results = {}
        summary = e.user_message
        status = "failed"
        db_state = "FAILED"

    duration_ms = (time.monotonic() - t_start) * 1000
    complete_intent(db, gs_intent_id, db_state, summary, duration_ms)

    return {
        "intent_id": gs_intent_id,
        "persistent_id": intent_id,
        "status": status,
        "summary": summary,
        "results": results,
        "duration_ms": round(duration_ms, 1),
        "fire_count": intent["fire_count"] + 1,
    }


@app.post("/v1/intent/persistent/{intent_id}/pause")
async def pause_persistent_intent(intent_id: str):
    db = _get_db()
    try:
        deactivate_intent(db, intent_id)
    except LeavesError as e:
        if e.code == LeavesErrorCode.INTENT_NOT_FOUND:
            raise HTTPException(status_code=404, detail=e.user_message)
        if e.code == LeavesErrorCode.INTENT_ALREADY_INACTIVE:
            raise HTTPException(status_code=409, detail=e.user_message)
        raise HTTPException(status_code=500, detail=e.user_message)
    return {"status": "paused", "id": intent_id}


@app.post("/v1/intent/persistent/{intent_id}/resume")
async def resume_persistent_intent(intent_id: str):
    db = _get_db()
    try:
        activate_intent(db, intent_id)
    except LeavesError as e:
        if e.code == LeavesErrorCode.INTENT_NOT_FOUND:
            raise HTTPException(status_code=404, detail=e.user_message)
        raise HTTPException(status_code=500, detail=e.user_message)
    return {"status": "resumed", "id": intent_id}


def _read_model_from_script() -> str:
    """Extract the primary MODEL variable value from start-inference.sh."""
    try:
        for line in _START_INFERENCE_SCRIPT.read_text().splitlines():
            line = line.strip()
            # Match: MODEL="..." or MODEL=... but NOT MODEL_FALLBACK_...
            if line.startswith("MODEL=") and not line.startswith("MODEL_FALLBACK"):
                val = line[len("MODEL="):].strip().strip('"\'')
                # val may contain $(...) or variable references — just take the filename
                return pathlib.Path(val).name
    except Exception:
        pass
    return "unknown"
