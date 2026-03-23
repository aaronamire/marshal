"""
System agent — system information, app launch, and app terminate.

Dispatch by action type:
  QUERY  → routed by params["query_type"]:
    "cpu"       → sys_cpu()
    "memory"    → sys_memory()
    "disk"      → sys_disk()
    "processes" → sys_processes(top_n)
    "uptime"    → sys_uptime()
  WRITE  → launch a program (params["program"])
  DELETE → terminate a program by name or PID (params["target"])
"""
from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import time
from typing import Optional

import psutil

from agents.base_agent import BaseAgent
from errors import LeavesError, LeavesErrorCode

# Programs that must never be terminated via the agent.
_PROTECTED_PROCESSES = frozenset({
    "systemd", "init", "sshd", "login", "dbus-daemon",
    "pipewire", "pulseaudio", "Xorg", "Xwayland",
    "leaves-compositor", "agentd", "python3",
})

# Max processes to kill in a single terminate action (safety cap).
_MAX_TERMINATE = 5


class SystemAgent(BaseAgent):
    AGENT_TYPE = "system"

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def execute_action(self, action: dict) -> dict:
        action_type = action.get("type", "").upper()
        action_id = action.get("action_id", "unknown")
        params = action.get("params", {})

        if action_type == "QUERY":
            return self._handle_query(action_id, params)
        elif action_type == "WRITE":
            return self._handle_launch(action_id, params)
        elif action_type == "DELETE":
            return self._handle_terminate(action_id, params)
        else:
            raise LeavesError(
                LeavesErrorCode.NOT_IMPLEMENTED,
                detail=f"SystemAgent handles QUERY/WRITE/DELETE, got '{action_type}'",
            )

    # ------------------------------------------------------------------
    # QUERY dispatch
    # ------------------------------------------------------------------

    def _handle_query(self, action_id: str, params: dict) -> dict:
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
    # WRITE — launch a program
    # ------------------------------------------------------------------

    def _handle_launch(self, action_id: str, params: dict) -> dict:
        program = params.get("program", "").strip()
        if not program:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No program specified for launch",
            )

        # Resolve the program to a real path (validates it exists in PATH)
        exe = shutil.which(program)
        if exe is None:
            raise LeavesError(
                LeavesErrorCode.FILE_NOT_FOUND,
                detail=f"Program '{program}' not found in PATH",
            )

        row_id = self._audit_start(action_id, "WRITE", params)
        try:
            # Launch detached from our process group so it outlives us.
            # stdout/stderr to devnull — GUI apps don't need a terminal.
            proc = subprocess.Popen(
                [exe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            result = {
                "launched": program,
                "pid": proc.pid,
                "path": exe,
            }
            self._audit_end(row_id, result)
            return result
        except OSError as e:
            err = LeavesError(
                LeavesErrorCode.INTERNAL_ERROR,
                detail=f"Failed to launch '{program}': {e}",
                cause=e,
            )
            self._audit_end(row_id, error=err)
            raise err

    # ------------------------------------------------------------------
    # DELETE — terminate a program by name or PID
    # ------------------------------------------------------------------

    def _handle_terminate(self, action_id: str, params: dict) -> dict:
        target = params.get("target", "").strip()
        if not target:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No target specified for terminate",
            )

        row_id = self._audit_start(action_id, "DELETE", params)
        try:
            result = self._terminate(target)
            self._audit_end(row_id, result)
            return result
        except LeavesError:
            raise
        except Exception as e:
            err = LeavesError(LeavesErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def _terminate(self, target: str) -> dict:
        """Terminate by PID (if numeric) or by process name."""
        # Try as PID first
        if target.isdigit():
            return self._terminate_pid(int(target))

        # By name — find matching processes owned by current user
        uid = os.getuid()
        matches = []
        for p in psutil.process_iter(["pid", "name", "uids"]):
            try:
                info = p.info
                if info["name"] and info["name"].lower() == target.lower():
                    # Only kill our own processes
                    if info["uids"] and info["uids"].real == uid:
                        matches.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not matches:
            raise LeavesError(
                LeavesErrorCode.FILE_NOT_FOUND,
                detail=f"No running process named '{target}' owned by current user",
            )

        # Safety: refuse to kill protected system processes
        if target.lower() in {n.lower() for n in _PROTECTED_PROCESSES}:
            raise LeavesError(
                LeavesErrorCode.PERMISSION_DENIED,
                detail=f"'{target}' is a protected system process and cannot be terminated",
            )

        # Cap the number of processes we'll kill
        if len(matches) > _MAX_TERMINATE:
            raise LeavesError(
                LeavesErrorCode.PERMISSION_DENIED,
                detail=(
                    f"Found {len(matches)} processes named '{target}' — "
                    f"refusing to kill more than {_MAX_TERMINATE} at once"
                ),
            )

        terminated = []
        for p in matches:
            terminated.append(self._kill_process(p))

        return {
            "target": target,
            "terminated": terminated,
            "count": len(terminated),
        }

    def _terminate_pid(self, pid: int) -> dict:
        """Terminate a single process by PID."""
        try:
            p = psutil.Process(pid)
        except psutil.NoSuchProcess:
            raise LeavesError(
                LeavesErrorCode.FILE_NOT_FOUND,
                detail=f"No process with PID {pid}",
            )

        # Only kill our own processes
        try:
            if p.uids().real != os.getuid():
                raise LeavesError(
                    LeavesErrorCode.PERMISSION_DENIED,
                    detail=f"PID {pid} ({p.name()}) is not owned by current user",
                )
        except psutil.AccessDenied:
            raise LeavesError(
                LeavesErrorCode.PERMISSION_DENIED,
                detail=f"Cannot access PID {pid} — not owned by current user",
            )

        name = p.name()
        if name.lower() in {n.lower() for n in _PROTECTED_PROCESSES}:
            raise LeavesError(
                LeavesErrorCode.PERMISSION_DENIED,
                detail=f"PID {pid} ({name}) is a protected system process",
            )

        info = self._kill_process(p)
        return {
            "target": str(pid),
            "terminated": [info],
            "count": 1,
        }

    @staticmethod
    def _kill_process(proc: psutil.Process) -> dict:
        """SIGTERM, wait 3s, then SIGKILL if still alive."""
        pid = proc.pid
        name = proc.name()
        try:
            proc.terminate()  # SIGTERM
            try:
                proc.wait(timeout=3)
                return {"pid": pid, "name": name, "signal": "SIGTERM"}
            except psutil.TimeoutExpired:
                proc.kill()  # SIGKILL
                proc.wait(timeout=2)
                return {"pid": pid, "name": name, "signal": "SIGKILL"}
        except psutil.NoSuchProcess:
            return {"pid": pid, "name": name, "signal": "already_exited"}

    # ------------------------------------------------------------------
    # Query tool methods
    # ------------------------------------------------------------------

    def sys_cpu(self) -> dict:
        usage = psutil.cpu_percent(interval=0.1)
        freq = psutil.cpu_freq()
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count()

        temp: Optional[float] = None
        try:
            sensors = psutil.sensors_temperatures()
            if sensors:
                for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                    entries = sensors.get(key, [])
                    if entries:
                        temp = round(entries[0].current, 1)
                        break
        except (AttributeError, Exception):
            pass

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

    # _audit_start / _audit_end inherited from BaseAgent
