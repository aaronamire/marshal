"""
Audio agent — PipeWire/PulseAudio volume and device control.

Dispatch by action type:
  QUERY  → routed by params["query_type"]:
    "volume"      → get current volume level + mute state
    "devices"     → list audio sinks/sources
    "status"      → current default sink + volume + mute
  WRITE  → routed by params["audio_action"]:
    "set_volume"  → set volume to N% (params["level"])
    "mute"        → mute default sink
    "unmute"      → unmute default sink
    "toggle_mute" → toggle mute on default sink
    "set_sink"    → set default sink (params["sink"])
"""
from __future__ import annotations

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
            pulse = self._get_pulsectl()
            if query_type == "volume":
                result = self._get_volume(pulse)
            elif query_type == "devices":
                result = self._list_devices(pulse)
            else:  # "status" or default
                result = self._get_status(pulse)
            pulse.close()
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

    def _list_devices(self, pulse) -> dict:
        default_sink = pulse.server_info().default_sink_name
        sinks = []
        for s in pulse.sink_list():
            sinks.append({
                "name": s.name,
                "description": s.description,
                "volume_percent": round(pulse.volume_get_all_chans(s) * 100),
                "muted": bool(s.mute),
                "default": s.name == default_sink,
            })
        sources = []
        for s in pulse.source_list():
            if ".monitor" not in s.name:
                sources.append({
                    "name": s.name,
                    "description": s.description,
                    "volume_percent": round(pulse.volume_get_all_chans(s) * 100),
                    "muted": bool(s.mute),
                })
        return {"sinks": sinks, "sources": sources}

    def _get_status(self, pulse) -> dict:
        info = pulse.server_info()
        sink_name = info.default_sink_name
        for s in pulse.sink_list():
            if s.name == sink_name:
                return {
                    "default_sink": s.description,
                    "sink_name": s.name,
                    "volume_percent": round(pulse.volume_get_all_chans(s) * 100),
                    "muted": bool(s.mute),
                    "server": info.server_name,
                }
        return {"default_sink": sink_name, "error": "Sink details unavailable"}

    # ------------------------------------------------------------------
    # WRITE handlers
    # ------------------------------------------------------------------

    def _handle_write(self, action_id: str, params: dict) -> dict:
        audio_action = params.get("audio_action", "set_volume")
        row_id = self._audit_start(action_id, "WRITE", params)
        try:
            pulse = self._get_pulsectl()
            if audio_action == "set_volume":
                result = self._set_volume(pulse, params)
            elif audio_action == "mute":
                result = self._set_mute(pulse, True)
            elif audio_action == "unmute":
                result = self._set_mute(pulse, False)
            elif audio_action == "toggle_mute":
                result = self._toggle_mute(pulse)
            elif audio_action == "set_sink":
                result = self._set_default_sink(pulse, params)
            else:
                raise MarshalError(
                    MarshalErrorCode.AGENT_NOT_AVAILABLE,
                    detail=f"Unknown audio_action: {audio_action}",
                )
            pulse.close()
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def _set_volume(self, pulse, params: dict) -> dict:
        level = params.get("level")
        if level is None:
            raise MarshalError(
                MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No volume level provided",
            )
        level = max(0, min(150, int(level)))  # clamp 0-150%
        sink = self._get_default_sink(pulse)
        pulse.volume_set_all_chans(sink, level / 100.0)
        return {
            "action": "set_volume",
            "volume_percent": level,
            "sink": sink.description,
        }

    def _set_mute(self, pulse, mute: bool) -> dict:
        sink = self._get_default_sink(pulse)
        pulse.mute(sink, mute)
        return {
            "action": "mute" if mute else "unmute",
            "muted": mute,
            "sink": sink.description,
        }

    def _toggle_mute(self, pulse) -> dict:
        sink = self._get_default_sink(pulse)
        new_mute = not bool(sink.mute)
        pulse.mute(sink, new_mute)
        return {
            "action": "toggle_mute",
            "muted": new_mute,
            "sink": sink.description,
        }

    def _set_default_sink(self, pulse, params: dict) -> dict:
        target = params.get("sink", "")
        if not target:
            raise MarshalError(
                MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No sink name provided",
            )
        # Match by name or description (case-insensitive substring)
        target_lower = target.lower()
        for s in pulse.sink_list():
            if target_lower in s.name.lower() or target_lower in s.description.lower():
                pulse.default_set(s)
                return {
                    "action": "set_sink",
                    "sink_name": s.name,
                    "sink_description": s.description,
                }
        raise MarshalError(
            MarshalErrorCode.INTERNAL_ERROR,
            detail=f"No sink matching '{target}' found",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pulsectl(self):
        try:
            import pulsectl
        except ImportError:
            raise MarshalError(
                MarshalErrorCode.AGENT_NOT_AVAILABLE,
                detail="pulsectl not installed (pip install pulsectl)",
            )
        return pulsectl.Pulse("marshal")

    def _get_default_sink(self, pulse):
        sink_name = pulse.server_info().default_sink_name
        for s in pulse.sink_list():
            if s.name == sink_name:
                return s
        raise MarshalError(
            MarshalErrorCode.INTERNAL_ERROR,
            detail="No default audio sink found",
        )
