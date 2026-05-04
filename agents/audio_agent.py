"""
Audio agent — PipeWire/PulseAudio volume and device control.

Backed by `wpctl` (WirePlumber CLI) which ships with PipeWire and works
unchanged against pipewire-pulse, so the same code covers PW-native and PA
systems with no Python deps. Falls back to `pactl` if wpctl isn't on PATH.
"""
from __future__ import annotations

import re
import shutil
import subprocess

from agents.base_agent import BaseAgent
from errors import MarshalError, MarshalErrorCode


class AudioAgent(BaseAgent):
    AGENT_TYPE = "audio"

    def execute_action(self, action: dict) -> dict:
        action_type = action.get("type", "").upper()
        action_id = action.get("action_id", "unknown")
        params = action.get("params", {})

        if action_type == "QUERY":
            return self._handle_query(action_id, params)
        elif action_type == "WRITE":
            return self._handle_write(action_id, params)
        else:
            raise MarshalError(
                MarshalErrorCode.AGENT_NOT_AVAILABLE,
                detail=f"AudioAgent supports QUERY and WRITE, got {action_type}",
            )

    # ------------------------------------------------------------------
    # QUERY handlers
    # ------------------------------------------------------------------

    def _handle_query(self, action_id: str, params: dict) -> dict:
        query_type = params.get("query_type", "status")
        row_id = self._audit_start(action_id, "QUERY", params)
        try:
            vol, muted = _wpctl_get_volume()
            if query_type == "devices":
                result = {"sinks": _wpctl_list_sinks()}
            elif query_type == "volume":
                result = {"volume_percent": vol, "muted": muted}
            else:  # "status" or default
                result = {
                    "volume_percent": vol,
                    "muted": muted,
                    "sinks": _wpctl_list_sinks(),
                }
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def _get_volume(self, pulse) -> dict:
        sink = pulse.server_info().default_sink_name
        for s in pulse.sink_list():
            if s.name == sink:
                vol = round(pulse.volume_get_all_chans(s) * 100)
                return {
                    "volume_percent": vol,
                    "muted": bool(s.mute),
                    "sink_name": s.description,
                }
        return {"error": "No default sink found"}

    # ------------------------------------------------------------------
    # WRITE handlers
    # ------------------------------------------------------------------

    def _handle_write(self, action_id: str, params: dict) -> dict:
        audio_action = params.get("audio_action", "set_volume")
        row_id = self._audit_start(action_id, "WRITE", params)
        try:
            if audio_action == "set_volume":
                result = _wpctl_set_volume(params.get("level"))
            elif audio_action == "mute":
                _wpctl_set_mute("1")
                result = {"action": "mute", "muted": True}
            elif audio_action == "unmute":
                _wpctl_set_mute("0")
                result = {"action": "unmute", "muted": False}
            elif audio_action == "toggle_mute":
                _wpctl_set_mute("toggle")
                _, muted = _wpctl_get_volume()
                result = {"action": "toggle_mute", "muted": muted}
            else:
                raise MarshalError(
                    MarshalErrorCode.AGENT_NOT_AVAILABLE,
                    detail=f"Unknown audio_action: {audio_action}",
                )
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

# ----------------------------------------------------------------------
# wpctl helpers (module-level so the agent stays stateless)
# ----------------------------------------------------------------------

_WPCTL_DEFAULT_SINK = "@DEFAULT_AUDIO_SINK@"


def _which_audio_cli() -> str:
    if shutil.which("wpctl"):
        return "wpctl"
    if shutil.which("pactl"):
        return "pactl"
    raise MarshalError(
        MarshalErrorCode.AGENT_NOT_AVAILABLE,
        detail="Neither wpctl nor pactl found on PATH",
    )


def _run(argv: list[str]) -> str:
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=5, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise MarshalError(
            MarshalErrorCode.INTERNAL_ERROR,
            detail=f"{argv[0]} failed: {e}",
            cause=e,
        )
    if out.returncode != 0:
        raise MarshalError(
            MarshalErrorCode.INTERNAL_ERROR,
            detail=f"{' '.join(argv)} exited {out.returncode}: "
                   f"{out.stderr.strip() or out.stdout.strip()}",
        )
    return out.stdout


def _wpctl_get_volume() -> tuple[int, bool]:
    cli = _which_audio_cli()
    if cli == "wpctl":
        # "Volume: 0.42 [MUTED]"
        s = _run(["wpctl", "get-volume", _WPCTL_DEFAULT_SINK])
        m = re.search(r"Volume:\s*([0-9.]+)", s)
        if not m:
            raise MarshalError(MarshalErrorCode.INTERNAL_ERROR,
                               detail=f"unparseable wpctl output: {s!r}")
        pct = round(float(m.group(1)) * 100)
        muted = "MUTED" in s
        return pct, muted
    # pactl fallback
    s = _run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
    m = re.search(r"(\d+)%", s)
    pct = int(m.group(1)) if m else 0
    s2 = _run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"]).strip()
    muted = s2.endswith("yes")
    return pct, muted


def _wpctl_set_volume(level) -> dict:
    if level is None:
        raise MarshalError(
            MarshalErrorCode.INFERENCE_BAD_RESPONSE,
            detail="No volume level provided",
        )
    cli = _which_audio_cli()
    # Relative steps from compositor hotkeys / "volume up/down" intents
    if isinstance(level, str) and level.lower() in ("up", "down"):
        delta = "5%+" if level.lower() == "up" else "5%-"
        # Always unmute on adjust — matches what laptop volume keys do
        if cli == "wpctl":
            _run(["wpctl", "set-mute", _WPCTL_DEFAULT_SINK, "0"])
            _run(["wpctl", "set-volume", "-l", "1.0",
                  _WPCTL_DEFAULT_SINK, delta])
        else:
            _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
            _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", delta])
    else:
        try:
            pct = max(0, min(150, int(level)))
        except (TypeError, ValueError):
            raise MarshalError(
                MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                detail=f"Bad volume level {level!r}",
            )
        if cli == "wpctl":
            _run(["wpctl", "set-mute", _WPCTL_DEFAULT_SINK, "0"])
            _run(["wpctl", "set-volume", "-l", "1.0",
                  _WPCTL_DEFAULT_SINK, f"{pct}%"])
        else:
            _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
            _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"])
    pct_now, muted = _wpctl_get_volume()
    return {"action": "set_volume", "volume_percent": pct_now, "muted": muted}


def _wpctl_set_mute(state: str) -> None:
    """state: '0' = unmute, '1' = mute, 'toggle' = flip."""
    cli = _which_audio_cli()
    if cli == "wpctl":
        _run(["wpctl", "set-mute", _WPCTL_DEFAULT_SINK, state])
    else:
        pa_state = "toggle" if state == "toggle" else state
        _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", pa_state])


def _wpctl_list_sinks() -> list[dict]:
    """Best-effort sink listing — wpctl status output is human-readable, so
    we just return the raw block; the agent's QUERY consumers don't iterate."""
    try:
        s = _run(["wpctl", "status"])
    except MarshalError:
        return []
    # Pull the "Sinks:" section from wpctl status
    sinks: list[dict] = []
    in_sinks = False
    for line in s.splitlines():
        stripped = line.strip()
        if stripped.startswith("Sinks:"):
            in_sinks = True
            continue
        if in_sinks:
            if not stripped or stripped.endswith(":"):
                break
            m = re.match(r"\*?\s*(\d+)\.\s+(.+?)\s*\[", stripped)
            if m:
                sinks.append({"id": int(m.group(1)),
                              "description": m.group(2).strip(),
                              "default": stripped.lstrip().startswith("*")})
    return sinks
