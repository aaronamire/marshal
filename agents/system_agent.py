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
import pathlib
import shutil
import subprocess
import time
from typing import Optional

import psutil

from agents.base_agent import BaseAgent
from errors import MarshalError, MarshalErrorCode

# Common program aliases — maps user-friendly names to real executables.
# Checked in order; first one found in PATH wins.
_PROGRAM_ALIASES: dict[str, list[str]] = {
    "google": ["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"],
    "chrome": ["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"],
    "browser": ["firefox", "google-chrome-stable", "chromium", "epiphany"],
    "files": ["nautilus", "thunar", "dolphin", "pcmanfm", "nemo"],
    "file manager": ["nautilus", "thunar", "dolphin", "pcmanfm", "nemo"],
    "text editor": ["gedit", "kate", "mousepad", "xed", "gnome-text-editor"],
    "editor": ["gedit", "kate", "mousepad", "xed", "gnome-text-editor"],
    "terminal": ["marshal-terminal", "foot", "alacritty", "kitty", "wezterm", "gnome-terminal", "xterm"],
    "term": ["marshal-terminal", "foot", "alacritty", "kitty", "wezterm", "gnome-terminal", "xterm"],
    "calculator": ["gnome-calculator", "kcalc", "galculator", "qalculate-gtk"],
    "settings": ["gnome-control-center", "xfce4-settings-manager", "systemsettings"],
}

# Executables that are Chromium-based — need special flags to force a new
# instance and ensure Wayland rendering on the correct compositor.
_CHROMIUM_EXECUTABLES = frozenset({
    "google-chrome-stable", "google-chrome", "google-chrome-beta",
    "google-chrome-unstable", "chromium", "chromium-browser",
    "brave", "brave-browser", "vivaldi", "vivaldi-stable",
    "microsoft-edge", "microsoft-edge-stable",
})

# Firefox-based — need --new-instance to avoid connecting to an existing
# instance running on a different Wayland compositor.
_FIREFOX_EXECUTABLES = frozenset({
    "firefox", "firefox-esr", "firefox-developer-edition",
    "librewolf", "waterfox", "floorp",
})

# Base directory for per-compositor Chromium profiles.  The actual
# directory includes the WAYLAND_DISPLAY name so that Chrome instances
# on different compositors never share a SingletonLock file.
_MARSHAL_CHROME_BASE = pathlib.Path.home() / ".marshal"

# Programs that must never be terminated via the agent.
_PROTECTED_PROCESSES = frozenset({
    "systemd", "init", "sshd", "login", "dbus-daemon",
    "pipewire", "pulseaudio", "Xorg", "Xwayland",
    "marshal-compositor", "agentd", "python3",
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
        self._current_action = action  # store for socket resolution

        if action_type == "QUERY":
            return self._handle_query(action_id, params)
        elif action_type == "WRITE":
            return self._handle_launch(action_id, params)
        elif action_type == "DELETE":
            return self._handle_terminate(action_id, params)
        else:
            raise MarshalError(
                MarshalErrorCode.NOT_IMPLEMENTED,
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
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    # ------------------------------------------------------------------
    # WRITE — launch a program
    # ------------------------------------------------------------------

    def _handle_launch(self, action_id: str, params: dict) -> dict:
        program = params.get("program", "").strip()
        if not program:
            raise MarshalError(
                MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No program specified for launch",
            )

        # Try the literal name first, then check aliases.
        exe = shutil.which(program)
        if exe is None:
            candidates = _PROGRAM_ALIASES.get(program.lower(), [])
            for candidate in candidates:
                exe = shutil.which(candidate)
                if exe:
                    break
        if exe is None:
            raise MarshalError(
                MarshalErrorCode.FILE_NOT_FOUND,
                detail=f"Program '{program}' not found in PATH",
            )

        exe_basename = os.path.basename(exe)

        row_id = self._audit_start(action_id, "WRITE", params)
        try:
            env = os.environ.copy()
            if "XDG_RUNTIME_DIR" not in env:
                env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
            xrd = env["XDG_RUNTIME_DIR"]

            resolved_display: Optional[str] = None

            # ---------- Socket resolution (4 channels, most direct first) -----
            #
            # Priority order:
            #   1. Direct from goal_spec (injected into action by coordinator)
            #   2. Marker file ~/.marshal/wayland-display (written by compositor)
            #   3. os.environ — BUT only if sandboxed_runner set it from
            #      goal_spec metadata.  The *inherited* WAYLAND_DISPLAY in
            #      os.environ may point to Hyprland (the parent shell's
            #      compositor).  We detect this by comparing with the marker.
            #   4. Scan XDG_RUNTIME_DIR

            # 1. DIRECT from goal_spec metadata
            direct_display = (
                getattr(self, "_current_action", {}).get("_wayland_display")
            )
            if direct_display and os.path.exists(os.path.join(xrd, direct_display)):
                resolved_display = direct_display

            # 2. Marker file (written by compositor on startup — authoritative)
            marker_display: Optional[str] = None
            if not resolved_display:
                try:
                    marker = pathlib.Path.home() / ".marshal" / "wayland-display"
                    candidate = marker.read_text().strip()
                    if candidate and os.path.exists(os.path.join(xrd, candidate)):
                        marker_display = candidate
                        resolved_display = candidate
                except (OSError, ValueError):
                    pass

            # 3. os.environ — only trust it if it matches the marker file,
            #    OR if the marker couldn't be read.
            if not resolved_display:
                meta_display = os.environ.get("WAYLAND_DISPLAY")
                if meta_display and os.path.exists(os.path.join(xrd, meta_display)):
                    resolved_display = meta_display

            # 4. Scan XDG_RUNTIME_DIR for the newest wayland socket.
            if not resolved_display:
                try:
                    import glob as _glob
                    sockets = sorted(_glob.glob(os.path.join(xrd, "wayland-*")))
                    for s in reversed(sockets):
                        if not s.endswith(".lock"):
                            resolved_display = os.path.basename(s)
                            break
                except OSError:
                    pass

            if resolved_display:
                env["WAYLAND_DISPLAY"] = resolved_display

            # ---------- Diagnostic log ----------
            _diag_path = pathlib.Path.home() / ".marshal" / "launch-debug.log"
            try:
                import glob as _dg
                sockets = sorted(_dg.glob(os.path.join(xrd, "wayland-*")))
                with open(_diag_path, "w") as _df:
                    _df.write(f"direct_display (action):  {direct_display}\n")
                    _df.write(f"marker_display:           {marker_display}\n")
                    _df.write(f"os.environ WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY')}\n")
                    _df.write(f"RESOLVED → {resolved_display}\n")
                    _df.write(f"env[WAYLAND_DISPLAY]:     {env.get('WAYLAND_DISPLAY')}\n")
                    _df.write(f"sockets:                  {sockets}\n")
            except Exception:
                pass

            # Build the command line — browsers need special flags so they
            # don't connect to an existing instance on a different compositor.
            cmd: list[str] = [exe]

            if exe_basename in _CHROMIUM_EXECUTABLES:
                # Per-compositor data dir: include the Wayland display name
                # so each compositor gets its own Chrome instance with its
                # own SingletonLock.  A static path like "chrome-profile"
                # would let an old Chrome instance (running on Hyprland)
                # hijack the launch via the singleton mechanism.
                disp_tag = resolved_display or "default"
                chrome_data_dir = _MARSHAL_CHROME_BASE / f"chrome-{disp_tag}"
                chrome_data_dir.mkdir(parents=True, exist_ok=True)
                cmd += [
                    f"--user-data-dir={chrome_data_dir}",
                    "--ozone-platform=wayland",
                ]
            elif exe_basename in _FIREFOX_EXECUTABLES:
                cmd.append("--new-instance")
                env["MOZ_ENABLE_WAYLAND"] = "1"

            # Append final command to diagnostic log
            try:
                with open(pathlib.Path.home() / ".marshal" / "launch-debug.log", "a") as _df:
                    _df.write(f"cmd:                          {cmd}\n")
                    _df.write(f"chrome_data_dir:              {locals().get('chrome_data_dir', 'N/A')}\n")
            except Exception:
                pass

            # Launch detached from our process group so it outlives us.
            # Use PIPE instead of DEVNULL to avoid opening /dev/null,
            # which Landlock's PATH_BENEATH can't grant on char devices.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                start_new_session=True,
                env=env,
            )
            # Close our end of the pipes immediately — the child keeps
            # running with broken pipes (harmless for GUI apps).
            if proc.stdin:
                proc.stdin.close()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            result = {
                "launched": program,
                "pid": proc.pid,
                "path": exe,
            }
            self._audit_end(row_id, result)
            return result
        except OSError as e:
            err = MarshalError(
                MarshalErrorCode.INTERNAL_ERROR,
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
            raise MarshalError(
                MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No target specified for terminate",
            )

        row_id = self._audit_start(action_id, "DELETE", params)
        try:
            result = self._terminate(target)
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def _terminate(self, target: str) -> dict:
        """Terminate by PID (if numeric) or by process name."""
        # Try as PID first
        if target.isdigit():
            return self._terminate_pid(int(target))

        # By name — find matching processes owned by current user.
        # Use fuzzy matching: exact match first, then prefix/contains match.
        # e.g. "firefox" matches "firefox", "firefox-esr", "firefox-bin",
        #       ".firefox-wrapped", "Web Content" (child of firefox) etc.
        uid = os.getuid()
        exact_matches: list[psutil.Process] = []
        fuzzy_matches: list[psutil.Process] = []
        target_lower = target.lower()

        for p in psutil.process_iter(["pid", "name", "cmdline", "uids"]):
            try:
                info = p.info
                if not (info["uids"] and info["uids"].real == uid):
                    continue
                pname = (info["name"] or "").lower()
                # Exact match
                if pname == target_lower:
                    exact_matches.append(p)
                # Fuzzy: process name starts with or contains target
                elif target_lower in pname or pname.startswith(target_lower):
                    fuzzy_matches.append(p)
                # Check cmdline for the program name (e.g. /usr/lib/firefox/firefox)
                elif info.get("cmdline"):
                    cmd0 = (info["cmdline"][0] if info["cmdline"] else "").lower()
                    if target_lower in os.path.basename(cmd0):
                        fuzzy_matches.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        matches = exact_matches or fuzzy_matches
        if not matches:
            raise MarshalError(
                MarshalErrorCode.PROCESS_NOT_FOUND,
                detail=f"No running process named '{target}' owned by current user",
            )

        # Safety: refuse to kill protected system processes
        if target.lower() in {n.lower() for n in _PROTECTED_PROCESSES}:
            raise MarshalError(
                MarshalErrorCode.PERMISSION_DENIED,
                detail=f"'{target}' is a protected system process and cannot be terminated",
            )

        # Cap the number of processes we'll kill
        if len(matches) > _MAX_TERMINATE:
            raise MarshalError(
                MarshalErrorCode.PERMISSION_DENIED,
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
            raise MarshalError(
                MarshalErrorCode.PROCESS_NOT_FOUND,
                detail=f"No process with PID {pid}",
            )

        # Only kill our own processes
        try:
            if p.uids().real != os.getuid():
                raise MarshalError(
                    MarshalErrorCode.PERMISSION_DENIED,
                    detail=f"PID {pid} ({p.name()}) is not owned by current user",
                )
        except psutil.AccessDenied:
            raise MarshalError(
                MarshalErrorCode.PERMISSION_DENIED,
                detail=f"Cannot access PID {pid} — not owned by current user",
            )

        name = p.name()
        if name.lower() in {n.lower() for n in _PROTECTED_PROCESSES}:
            raise MarshalError(
                MarshalErrorCode.PERMISSION_DENIED,
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
        cores_physical = psutil.cpu_count(logical=False)
        cores_logical = psutil.cpu_count()

        # CPU model name from /proc/cpuinfo
        model: Optional[str] = None
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

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
            "model": model,
            "usage_percent": round(usage, 1),
            "freq_mhz": round(freq.current, 1) if freq else None,
            "cores_physical": cores_physical,
            "cores_logical": cores_logical,
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
