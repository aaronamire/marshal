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
from config import INFERENCE_SERVER_URL
from db.audit import complete_intent, get_db, get_recent_intents, log_intent_created
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _ensure_agentd()
    yield


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons — created lazily on first request
_parser: Optional[IntentParser] = None
_db = None


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


async def _send_goalspec(goal_spec: dict) -> tuple[dict, str]:
    """
    Send a GoalSpec to the agentd Unix socket and return (results, summary).
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
        return resp["results"], resp["summary"]

    code_str = resp.get("code", "INTERNAL_ERROR")
    try:
        code = LeavesErrorCode[code_str]
    except KeyError:
        code = LeavesErrorCode.INTERNAL_ERROR
    raise LeavesError(code, detail=resp.get("detail") or resp.get("error"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IntentRequest(BaseModel):
    text: str


class ExecuteRequest(BaseModel):
    intent_id: str
    confirmed: bool = True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_result_response(
    goal_spec: dict, results: dict, summary: str, duration_ms: float
) -> dict:
    return {
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
        "authorization": goal_spec.get("authorization", {}),
        "results": results,
        "summary": summary,
        "duration_ms": round(duration_ms, 1),
        "model_latency_ms": round(
            goal_spec.get("metadata", {}).get("parse_latency_ms", 0), 1
        ),
    }


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
    try:
        goal_spec = await asyncio.get_event_loop().run_in_executor(
            None, parser.parse, body.text
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

    try:
        results, summary = await _send_goalspec(goal_spec)
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
    return _build_result_response(goal_spec, results, summary, duration_ms)


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


@app.get("/v1/agents")
async def get_agents():
    manifests = []
    for p in sorted(_REGISTRY_DIR.glob("*.json")):
        try:
            manifests.append(json.loads(p.read_text()))
        except Exception:
            pass
    return manifests


@app.get("/v1/health")
async def get_health():
    agentd_ok = _SOCK_PATH.exists()

    inference_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{INFERENCE_SERVER_URL}/health")
            inference_ok = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "agentd": agentd_ok,
        "inference": inference_ok,
        "model": _read_model_from_script(),
    }


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
