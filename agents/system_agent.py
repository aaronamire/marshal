"""
System agent — read-only system information queries.

All tools are non-destructive: they only read kernel/hardware counters.
No state is modified; no files are written.

Dispatch: action type QUERY → routed by params["query_type"]:
  "cpu"       → sys_cpu()
  "memory"    → sys_memory()
  "disk"      → sys_disk()
  "processes" → sys_processes(top_n)
  "uptime"    → sys_uptime()
"""
from __future__ import annotations

import datetime
import time
from typing import Any, Optional

import psutil

from agents.base_agent import BaseAgent
from db.audit import log_action_started, log_action_completed
from errors import LeavesError, LeavesErrorCode


class SystemAgent(BaseAgent):
    AGENT_TYPE = "system"

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def execute_action(self, action: dict) -> dict:
        action_type = action.get("type", "").upper()
        action_id = action.get("action_id", "unknown")
        params = action.get("params", {})

        if action_type != "QUERY":
            raise LeavesError(
                LeavesErrorCode.NOT_IMPLEMENTED,
                detail=f"SystemAgent only handles QUERY actions, got '{action_type}'",
            )

        query_type = params.get("query_type", "")
        top_n = params.get("top_n", 10)

        dispatch = {
            "cpu":       lambda: self.sys_cpu(),
            "memory":    lambda: self.sys_memory(),
            "disk":      lambda: self.sys_disk(params.get("path")),
            "processes": lambda: self.sys_processes(top_n),
            "uptime":    lambda: self.sys_uptime(),
        }

        handler = dispatch.get(query_type)
        if handler is None:
            # Unknown query_type — try cpu as default for vague system queries
            handler = dispatch["cpu"]

        row_id = self._audit_start(action_id, "QUERY", params)
        try:
            result = handler()
            self._audit_end(row_id, result)
            return result
        except LeavesError:
            raise
        except Exception as e:
            err = LeavesError(LeavesErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    def sys_cpu(self) -> dict:
        usage = psutil.cpu_percent(interval=0.1)
        freq = psutil.cpu_freq()
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count()

        temp: Optional[float] = None
        try:
            sensors = psutil.sensors_temperatures()
            if sensors:
                # Try common sensor names in order of preference
                for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                    entries = sensors.get(key, [])
                    if entries:
                        temp = round(entries[0].current, 1)
                        break
        except (AttributeError, Exception):
            pass  # sensors_temperatures() not available on all platforms

        return {
            "usage_percent": round(usage, 1),
            "freq_mhz": round(freq.current, 1) if freq else None,
            "cores": cores,
            "temp_celsius": temp,
        }

    def sys_memory(self) -> dict:
        vm = psutil.virtual_memory()
        gb = 1024 ** 3
        return {
            "total_gb": round(vm.total / gb, 2),
            "used_gb": round(vm.used / gb, 2),
            "available_gb": round(vm.available / gb, 2),
            "percent": round(vm.percent, 1),
        }

    def sys_disk(self, path: Optional[str] = None) -> dict:
        import pathlib
        target = pathlib.Path(path).expanduser() if path else pathlib.Path.home()
        usage = psutil.disk_usage(str(target))
        gb = 1024 ** 3
        return {
            "total_gb": round(usage.total / gb, 2),
            "used_gb": round(usage.used / gb, 2),
            "free_gb": round(usage.free / gb, 2),
            "percent": round(usage.percent, 1),
            "path": str(target),
        }

    def sys_processes(self, top_n: int = 10) -> dict:
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status"]):
            try:
                info = p.info
                procs.append({
                    "pid": info["pid"],
                    "name": info["name"] or "",
                    "cpu_percent": round(info["cpu_percent"] or 0.0, 1),
                    "memory_mb": round(
                        (info["memory_info"].rss if info["memory_info"] else 0) / (1024 ** 2), 1
                    ),
                    "status": info["status"] or "",
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda p: p["cpu_percent"], reverse=True)
        return {"processes": procs[:top_n], "count": min(top_n, len(procs))}

    def sys_uptime(self) -> dict:
        boot_ts = psutil.boot_time()
        uptime_s = time.time() - boot_ts
        boot_iso = datetime.datetime.fromtimestamp(
            boot_ts, tz=datetime.timezone.utc
        ).isoformat()
        return {
            "uptime_seconds": round(uptime_s, 1),
            "boot_time_iso": boot_iso,
        }

    # ------------------------------------------------------------------
    # Audit helpers (same pattern as FileAgent)
    # ------------------------------------------------------------------

    def _audit_start(self, action_id: str, action_type: str, params: dict) -> Optional[int]:
        try:
            return log_action_started(
                self._db,
                intent_id=self.intent_id,
                action_id=action_id,
                action_type=action_type,
                agent=self.AGENT_TYPE,
                params=params,
            )
        except Exception:
            return None

    def _audit_end(
        self,
        row_id: Optional[int],
        result: Optional[dict] = None,
        error: Optional[LeavesError] = None,
    ) -> None:
        if row_id is None:
            return
        try:
            log_action_completed(
                self._db,
                row_id=row_id,
                result=result,
                error_code=error.code.value if error else None,
                error_detail=error.detail if error else None,
            )
        except Exception:
            pass
