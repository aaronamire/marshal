"""
Power agent — suspend, hibernate, brightness, and battery via logind + sysfs.

Dispatch by action type:
  QUERY  → routed by params["query_type"]:
    "battery"     → battery percentage, charging state, time remaining
    "brightness"  → current screen brightness level
    "status"      → battery + power source + brightness summary
  WRITE  → routed by params["power_action"]:
    "suspend"        → suspend via logind
    "hibernate"      → hibernate via logind
    "set_brightness" → set screen brightness (params["level"], 0-100%)
    "lock"           → lock screen via loginctl
"""
from __future__ import annotations

import os
import subprocess

from agents.base_agent import BaseAgent
from errors import MarshalError, MarshalErrorCode


# Backlight sysfs paths — first match wins
_BACKLIGHT_DIR = "/sys/class/backlight"


class PowerAgent(BaseAgent):
    AGENT_TYPE = "power"

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
                detail=f"PowerAgent supports QUERY and WRITE, got {action_type}",
            )

    # ------------------------------------------------------------------
    # QUERY handlers
    # ------------------------------------------------------------------

    def _handle_query(self, action_id: str, params: dict) -> dict:
        query_type = params.get("query_type", "status")
        row_id = self._audit_start(action_id, "QUERY", params)
        try:
            if query_type == "battery":
                result = self._get_battery()
            elif query_type == "brightness":
                result = self._get_brightness()
            else:  # "status"
                result = self._get_power_status()
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def _get_battery(self) -> dict:
        """Read battery info from sysfs (same paths as compositor status.c)."""
        for bat_name in ("BAT0", "BAT1", "macsmc-battery"):
            bat_path = f"/sys/class/power_supply/{bat_name}"
            if os.path.isdir(bat_path):
                return self._read_battery_sysfs(bat_path, bat_name)
        return {"available": False, "error": "No battery found"}

    def _read_battery_sysfs(self, path: str, name: str) -> dict:
        result = {"available": True, "device": name}

        capacity = self._read_sysfs_int(f"{path}/capacity")
        if capacity is not None:
            result["percent"] = capacity

        status = self._read_sysfs_str(f"{path}/status")
        if status:
            result["charging"] = status.lower() in ("charging", "full")
            result["status"] = status.lower()

        # Energy/charge rates for time estimation
        energy_now = self._read_sysfs_int(f"{path}/energy_now")
        energy_full = self._read_sysfs_int(f"{path}/energy_full")
        power_now = self._read_sysfs_int(f"{path}/power_now")

        if energy_now is not None and power_now and power_now > 0:
            if status and status.lower() == "discharging":
                hours_left = energy_now / power_now
                result["time_remaining_hours"] = round(hours_left, 1)
            elif status and status.lower() == "charging" and energy_full:
                hours_left = (energy_full - energy_now) / power_now
                result["time_to_full_hours"] = round(hours_left, 1)

        return result

    def _get_brightness(self) -> dict:
        """Read current backlight brightness from sysfs."""
        bl = self._find_backlight()
        if not bl:
            return {"available": False, "error": "No backlight device found"}
        current = self._read_sysfs_int(f"{bl}/brightness")
        max_val = self._read_sysfs_int(f"{bl}/max_brightness")
        if current is not None and max_val and max_val > 0:
            return {
                "available": True,
                "brightness_percent": round(current / max_val * 100),
                "brightness_raw": current,
                "max_brightness": max_val,
                "device": os.path.basename(bl),
            }
        return {"available": False, "error": "Could not read brightness values"}

    def _get_power_status(self) -> dict:
        """Combined battery + brightness status."""
        result = {}
        battery = self._get_battery()
        result["battery"] = battery

        brightness = self._get_brightness()
        result["brightness"] = brightness

        # AC power
        for ps_name in ("AC", "ADP0", "ADP1", "ACAD"):
            online_path = f"/sys/class/power_supply/{ps_name}/online"
            val = self._read_sysfs_int(online_path)
            if val is not None:
                result["ac_power"] = bool(val)
                break

        return result

    # ------------------------------------------------------------------
    # WRITE handlers
    # ------------------------------------------------------------------

    def _handle_write(self, action_id: str, params: dict) -> dict:
        power_action = params.get("power_action", "suspend")
        row_id = self._audit_start(action_id, "WRITE", params)
        try:
            if power_action == "suspend":
                result = self._suspend()
            elif power_action == "hibernate":
                result = self._hibernate()
            elif power_action == "set_brightness":
                result = self._set_brightness(params)
            elif power_action == "lock":
                result = self._lock_screen()
            else:
                raise MarshalError(
                    MarshalErrorCode.AGENT_NOT_AVAILABLE,
                    detail=f"Unknown power_action: {power_action}",
                )
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def _suspend(self) -> dict:
        """Suspend via loginctl (talks to logind over D-Bus)."""
        try:
            r = subprocess.run(
                ["loginctl", "suspend"],
                capture_output=True, text=True, timeout=10,
            )
            return {
                "action": "suspend",
                "success": r.returncode == 0,
                "error": r.stderr.strip() if r.returncode != 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {"action": "suspend", "success": False, "error": "Timed out"}
        except FileNotFoundError:
            # Fallback to systemctl
            try:
                r = subprocess.run(
                    ["systemctl", "suspend"],
                    capture_output=True, text=True, timeout=10,
                )
                return {"action": "suspend", "success": r.returncode == 0}
            except (FileNotFoundError, subprocess.TimeoutExpired):
                raise MarshalError(
                    MarshalErrorCode.AGENT_NOT_AVAILABLE,
                    detail="Neither loginctl nor systemctl available for suspend",
                )

    def _hibernate(self) -> dict:
        """Hibernate via loginctl."""
        try:
            r = subprocess.run(
                ["loginctl", "hibernate"],
                capture_output=True, text=True, timeout=10,
            )
            return {
                "action": "hibernate",
                "success": r.returncode == 0,
                "error": r.stderr.strip() if r.returncode != 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {"action": "hibernate", "success": False, "error": "Timed out"}
        except FileNotFoundError:
            try:
                r = subprocess.run(
                    ["systemctl", "hibernate"],
                    capture_output=True, text=True, timeout=10,
                )
                return {"action": "hibernate", "success": r.returncode == 0}
            except (FileNotFoundError, subprocess.TimeoutExpired):
                raise MarshalError(
                    MarshalErrorCode.AGENT_NOT_AVAILABLE,
                    detail="Neither loginctl nor systemctl available for hibernate",
                )

    def _set_brightness(self, params: dict) -> dict:
        """Set screen brightness. Tries brightnessctl first, then direct sysfs."""
        level = params.get("level")
        if level is None:
            raise MarshalError(
                MarshalErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No brightness level provided",
            )
        level = max(0, min(100, int(level)))

        # Try brightnessctl first (handles permissions via udev rules)
        try:
            r = subprocess.run(
                ["brightnessctl", "set", f"{level}%"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return {
                    "action": "set_brightness",
                    "brightness_percent": level,
                    "method": "brightnessctl",
                }
        except FileNotFoundError:
            pass

        # Fallback: direct sysfs write (needs video group or root)
        bl = self._find_backlight()
        if not bl:
            raise MarshalError(
                MarshalErrorCode.AGENT_NOT_AVAILABLE,
                detail="No backlight device found",
            )
        max_val = self._read_sysfs_int(f"{bl}/max_brightness")
        if not max_val:
            raise MarshalError(
                MarshalErrorCode.INTERNAL_ERROR,
                detail="Could not read max brightness",
            )
        raw_val = max(1, round(level / 100.0 * max_val))
        brightness_path = f"{bl}/brightness"
        try:
            with open(brightness_path, "w") as f:
                f.write(str(raw_val))
            return {
                "action": "set_brightness",
                "brightness_percent": level,
                "brightness_raw": raw_val,
                "method": "sysfs",
            }
        except PermissionError:
            raise MarshalError(
                MarshalErrorCode.PERMISSION_DENIED,
                detail=(
                    f"Cannot write to {brightness_path}. "
                    "Install brightnessctl or add user to 'video' group."
                ),
            )

    def _lock_screen(self) -> dict:
        """Lock screen via loginctl."""
        try:
            r = subprocess.run(
                ["loginctl", "lock-session"],
                capture_output=True, text=True, timeout=5,
            )
            return {"action": "lock", "success": r.returncode == 0}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"action": "lock", "success": False, "error": "loginctl not available"}

    # ------------------------------------------------------------------
    # sysfs helpers
    # ------------------------------------------------------------------

    def _find_backlight(self) -> str:
        """Find the first backlight device in sysfs."""
        try:
            for entry in sorted(os.listdir(_BACKLIGHT_DIR)):
                path = os.path.join(_BACKLIGHT_DIR, entry)
                if os.path.isdir(path):
                    return path
        except OSError:
            pass
        return ""

    def _read_sysfs_int(self, path: str):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def _read_sysfs_str(self, path: str) -> str:
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return ""
