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

# Programs Marshal ships in-tree. shutil.which() won't find these unless the
# user installs them system-wide, but they live in known builddir locations.
# Lookup order: shutil.which() first, then this map.
_MARSHAL_ROOT = pathlib.Path(__file__).parent.parent
_MARSHAL_INTREE_BINARIES: dict[str, pathlib.Path] = {
    "marshal-terminal":   _MARSHAL_ROOT / "terminal"   / "builddir" / "marshal-terminal",
    "marshal-compositor": _MARSHAL_ROOT / "compositor" / "builddir" / "marshal-compositor",
}


def _resolve_executable(name: str) -> Optional[str]:
    """Return absolute path of `name` from PATH or the in-tree builddir map."""
    found = shutil.which(name)
    if found:
        return found
    intree = _MARSHAL_INTREE_BINARIES.get(name)
    if intree and intree.is_file() and os.access(intree, os.X_OK):
        return str(intree)
    return None


# Common program aliases — maps user-friendly names to real executables.
# Checked in order; first one resolvable wins.
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


def _llama_server_running() -> bool:
    """True iff at least one llama-server process is alive for this user."""
    import subprocess as sp
    try:
        out = sp.run(
            ["pgrep", "-u", str(os.getuid()), "-f", "llama-server"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return bool(out.stdout.strip())
    except (FileNotFoundError, sp.TimeoutExpired):
        return False


def _stop_inference() -> tuple[bool, str]:
    """
    Bring the inference server down. Tries systemd user units first, then
    falls back to SIGTERM on running llama-server processes. Returns
    (stopped_anything, method_used).
    """
    import shutil
    import signal
    import subprocess as sp

    if shutil.which("systemctl"):
        for unit in ("marshal-inference.service",):
            check = sp.run(
                ["systemctl", "--user", "list-unit-files", unit, "--no-legend"],
                capture_output=True, text=True, timeout=4, check=False,
            )
            if unit in check.stdout:
                rc = sp.run(
                    ["systemctl", "--user", "stop", unit],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                if rc.returncode == 0:
                    return True, f"systemd ({unit})"

    # Bare SIGTERM
    try:
        out = sp.run(
            ["pgrep", "-u", str(os.getuid()), "-f", "llama-server"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        pids = [int(p) for p in out.stdout.split() if p.isdigit()]
    except (FileNotFoundError, sp.TimeoutExpired):
        pids = []

    if not pids:
        return False, "none"

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    return True, "SIGTERM"


def _restart_inference() -> tuple[bool, str]:
    """
    Bring down the running llama-server (if any) and bring it up again with
    the current tier.json. Tries systemd --user units first, then falls back
    to a bare process replace.

    Returns (success, method_used).
    """
    import shutil
    import signal
    import subprocess as sp
    import time as _time

    # 1) Systemd user units.
    if shutil.which("systemctl"):
        for unit in ("marshal-inference.service",):
            check = sp.run(
                ["systemctl", "--user", "list-unit-files", unit, "--no-legend"],
                capture_output=True, text=True, timeout=4, check=False,
            )
            if unit in check.stdout:
                rc = sp.run(
                    ["systemctl", "--user", "restart", unit],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                if rc.returncode == 0:
                    return True, f"systemd ({unit})"

    # 2) Bare process restart. Find any running llama-server, send SIGTERM,
    #    wait briefly, then re-launch start-inference.sh in the background.
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "start-inference.sh"
    if not script.exists():
        return False, "none"

    # SIGTERM existing llama-server processes
    try:
        out = sp.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        for pid_str in out.stdout.split():
            try:
                os.kill(int(pid_str), signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        # Brief wait for graceful exit
        for _ in range(20):
            check = sp.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if not check.stdout.strip():
                break
            _time.sleep(0.1)
    except (FileNotFoundError, sp.TimeoutExpired):
        pass

    # Re-spawn detached so it survives this request
    try:
        sp.Popen(
            ["bash", str(script)],
            stdout=sp.DEVNULL, stderr=sp.DEVNULL, stdin=sp.DEVNULL,
            start_new_session=True, close_fds=True,
        )
        return True, "bare process"
    except OSError:
        return False, "spawn failed"


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
        # Inline-routed: inference status QUERY shares the toggle handler
        # so the user-disabled marker and live llama-server check are in one
        # place.
        inference_action = params.get("inference_action")
        if inference_action == "status":
            return self._handle_inference_toggle(action_id, "status")

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
        # Inline-routed sub-action: tier switch lives under SystemAgent.WRITE
        # so the existing audit + dispatch path covers it. New action_type
        # values would force schema changes across the planner / enforcer.
        switch_tier = params.get("switch_tier")
        if switch_tier:
            return self._handle_switch_tier(action_id, switch_tier)

        inference_action = params.get("inference_action")
        if inference_action:
            return self._handle_inference_toggle(action_id, inference_action)

        program = params.get("program", "").strip()
        if not program:
            raise MarshalError(
                MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No program specified for launch",
            )

        # Try the literal name first, then check aliases. _resolve_executable
        # checks PATH first, then falls back to Marshal's in-tree builddir
        # binaries so e.g. `marshal-terminal` works even when not installed.
        exe = _resolve_executable(program)
        if exe is None:
            candidates = _PROGRAM_ALIASES.get(program.lower(), [])
            for candidate in candidates:
                exe = _resolve_executable(candidate)
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
            #
            # Originally this used stdout/stderr=PIPE with an immediate
            # close on the parent side, on the theory that "GUI apps don't
            # write to stderr." That assumption was wrong: many Wayland
            # clients (including marshal-terminal) print connection
            # diagnostics at startup. Writing to a closed pipe delivers
            # SIGPIPE, which by default terminates the process — so the
            # GUI window the agent "successfully launched" would die a
            # millisecond later, before mapping any surface, and the user
            # would see "done" but no window.
            #
            # /dev/null is fine here because launch actions skip Landlock
            # (see agents/sandboxed_runner.py:_is_launch). The /dev/null
            # PATH_BENEATH-can't-grant comment that justified PIPE applies
            # only to the sandboxed code path.
            launch_log_dir = pathlib.Path.home() / ".marshal" / "logs" / "launches"
            launch_log_dir.mkdir(parents=True, exist_ok=True)
            launch_log = launch_log_dir / f"{exe_basename}.log"
            log_fd = open(launch_log, "ab", buffering=0)
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=log_fd,
                    stderr=log_fd,
                    start_new_session=True,
                    env=env,
                )
            finally:
                # The child has its own fd dup; we don't need ours.
                log_fd.close()
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
    # WRITE sub-action — switch the active model tier
    # ------------------------------------------------------------------

    def _handle_switch_tier(self, action_id: str, tier_name: str) -> dict:
        """
        Persist a forced tier choice to ~/.marshal/tier.json AND swap the
        running inference server in-place so the new model is live without
        the user touching anything else.

        Restart strategy (first that works wins):
          1. systemctl --user restart marshal-inference
          2. pkill llama-server  +  scripts/start-inference.sh &  (bare)
        """
        try:
            import hardware
        except ImportError as e:
            raise MarshalError(
                MarshalErrorCode.INTERNAL_ERROR,
                detail=f"hardware module unavailable: {e}",
                cause=e,
            )

        tier_name = (tier_name or "").strip().lower()
        if tier_name not in hardware.TIER_BY_NAME:
            raise MarshalError(
                MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                detail=(
                    f"Unknown tier '{tier_name}'. "
                    f"Choices: {', '.join(t.name for t in hardware.TIERS)}"
                ),
            )

        row_id = self._audit_start(action_id, "WRITE", {"switch_tier": tier_name})
        try:
            tier = hardware.TIER_BY_NAME[tier_name]
            profile = hardware.probe()
            decision = hardware.TierDecision(
                chosen=tier.name,
                reason=f"forced via 'switch {tier_name}'",
                profile=profile,
                bench=None,
                estimated_gen_tok_s={},
                disqualified={},
                timestamp=hardware._utc_timestamp(),
            )
            hardware.write_decision(decision, hardware.CONFIG_PATH)

            restart_status, restart_via = _restart_inference()

            summary = (
                f"Tier set to {tier.name} ({tier.param_billions:.1f}B, "
                f"{tier.model_file}). "
            )
            if restart_status:
                summary += (
                    f"Inference server restarting via {restart_via}; "
                    f"new model will be live in ~5–30 s."
                )
            else:
                summary += (
                    "Could not auto-restart inference — start it manually "
                    "with: bash scripts/start-inference.sh"
                )

            result = {
                "action": "switch_tier",
                "tier": tier.name,
                "model_file": tier.model_file,
                "param_billions": tier.param_billions,
                "restarted": restart_status,
                "restart_method": restart_via,
                "summary": summary,
            }
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(
                MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e
            )
            self._audit_end(row_id, error=err)
            raise err

    # ------------------------------------------------------------------
    # WRITE/QUERY sub-action — inference server power toggle
    # ------------------------------------------------------------------

    def _handle_inference_toggle(self, action_id: str, action: str) -> dict:
        """
        action ∈ {"on", "off", "toggle", "status"}.

        - on:   delete marker, restart inference (start fresh if not running)
        - off:  write marker, stop inference (graceful SIGTERM, fallback systemctl)
        - toggle: read marker, flip
        - status: report whether llama-server is up + whether user disabled it
        """
        marker = pathlib.Path.home() / ".marshal" / "inference-disabled"

        # Resolve "toggle" up-front so the rest of the function only deals
        # with concrete on/off/status.
        if action == "toggle":
            action = "on" if marker.exists() else "off"

        if action == "status":
            running = _llama_server_running()
            disabled = marker.exists()
            return {
                "action": "inference_status",
                "running": running,
                "disabled_by_user": disabled,
                "summary": (
                    f"Inference server: "
                    f"{'running' if running else 'stopped'}"
                    f"{' (disabled by user)' if disabled else ''}."
                ),
            }

        row_id = self._audit_start(action_id, "WRITE", {"inference_action": action})
        try:
            if action == "off":
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                stopped, method = _stop_inference()
                result = {
                    "action": "inference_off",
                    "stopped": stopped,
                    "method": method,
                    "summary": (
                        "Inference server turned off."
                        if stopped else
                        "Marker set; no running llama-server process found."
                    ),
                }
            elif action == "on":
                # Drop the user-disabled marker first, otherwise the next
                # intent would still hit the friendly "inference is off"
                # prompt even though the server is up.
                try:
                    marker.unlink()
                except FileNotFoundError:
                    pass
                started, method = _restart_inference()
                result = {
                    "action": "inference_on",
                    "started": started,
                    "method": method,
                    "summary": (
                        f"Inference server starting via {method}; "
                        f"ready in ~5–30 s."
                        if started else
                        "Could not start inference automatically — "
                        "run: bash scripts/start-inference.sh"
                    ),
                }
            else:
                raise MarshalError(
                    MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                    detail=f"Unknown inference_action {action!r}",
                )
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(
                MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e
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
        for p in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_info", "status", "username"]
        ):
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
                    "username": info.get("username") or "",
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda p: p["cpu_percent"], reverse=True)
        # `count` = how many rows we're returning (matches len(processes));
        # `total` = the full process universe so the renderer can show
        # "… N more" without lying about the slice size.
        returned = procs[:top_n]
        return {
            "processes": returned,
            "count": len(returned),
            "total": len(procs),
        }

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
