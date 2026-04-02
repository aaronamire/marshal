"""
Network agent — WiFi management via iwd (iwctl) with NetworkManager fallback.

Dispatch by action type:
  QUERY  → routed by params["query_type"]:
    "status"    → current connection info (SSID, signal, IP)
    "scan"      → scan for available networks
    "devices"   → list network interfaces
  WRITE  → routed by params["network_action"]:
    "connect"   → connect to SSID (params["ssid"], optional params["passphrase"])
    "disconnect" → disconnect current WiFi
"""
from __future__ import annotations

import subprocess

from agents.base_agent import BaseAgent
from errors import LeavesError, LeavesErrorCode


class NetworkAgent(BaseAgent):
    AGENT_TYPE = "network"

    def execute_action(self, action: dict) -> dict:
        action_type = action.get("type", "").upper()
        action_id = action.get("action_id", "unknown")
        params = action.get("params", {})

        if action_type == "QUERY":
            return self._handle_query(action_id, params)
        elif action_type == "WRITE":
            return self._handle_write(action_id, params)
        else:
            raise LeavesError(
                LeavesErrorCode.AGENT_NOT_AVAILABLE,
                detail=f"NetworkAgent supports QUERY and WRITE, got {action_type}",
            )

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _has_iwd(self) -> bool:
        try:
            r = subprocess.run(
                ["iwctl", "version"],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _has_nmcli(self) -> bool:
        try:
            r = subprocess.run(
                ["nmcli", "--version"],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ------------------------------------------------------------------
    # QUERY handlers
    # ------------------------------------------------------------------

    def _handle_query(self, action_id: str, params: dict) -> dict:
        query_type = params.get("query_type", "status")
        row_id = self._audit_start(action_id, "QUERY", params)
        try:
            if query_type == "scan":
                result = self._scan_networks()
            elif query_type == "devices":
                result = self._list_devices()
            else:  # "status"
                result = self._get_status()
            self._audit_end(row_id, result)
            return result
        except LeavesError:
            raise
        except Exception as e:
            err = LeavesError(LeavesErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def _get_status(self) -> dict:
        """Get current connection status via iwd or nmcli."""
        if self._has_iwd():
            return self._iwd_status()
        elif self._has_nmcli():
            return self._nmcli_status()
        else:
            return self._sysfs_status()

    def _iwd_status(self) -> dict:
        """Get status via iwctl station show."""
        device = self._iwd_get_device()
        if not device:
            return {"connected": False, "error": "No WiFi device found"}
        try:
            r = subprocess.run(
                ["iwctl", "station", device, "show"],
                capture_output=True, text=True, timeout=10,
            )
            info = self._parse_iwctl_table(r.stdout)
            connected = info.get("State", "") == "connected"
            result = {
                "connected": connected,
                "device": device,
                "state": info.get("State", "unknown"),
                "backend": "iwd",
            }
            if connected:
                result["ssid"] = info.get("Connected network", "")
                result["signal"] = info.get("RSSI", "")
            return result
        except subprocess.TimeoutExpired:
            return {"connected": False, "error": "iwctl timed out"}

    def _nmcli_status(self) -> dict:
        """Get status via nmcli."""
        try:
            r = subprocess.run(
                ["nmcli", "-t", "-f",
                 "DEVICE,TYPE,STATE,CONNECTION",
                 "device", "status"],
                capture_output=True, text=True, timeout=10,
            )
            for line in r.stdout.strip().splitlines():
                parts = line.split(":")
                if len(parts) >= 4 and parts[1] == "wifi":
                    return {
                        "connected": parts[2] == "connected",
                        "device": parts[0],
                        "ssid": parts[3] if parts[2] == "connected" else "",
                        "state": parts[2],
                        "backend": "nmcli",
                    }
            return {"connected": False, "error": "No WiFi device found", "backend": "nmcli"}
        except subprocess.TimeoutExpired:
            return {"connected": False, "error": "nmcli timed out"}

    def _sysfs_status(self) -> dict:
        """Fallback: read sysfs directly (same approach as compositor status.c)."""
        import os
        net_dir = "/sys/class/net"
        try:
            for iface in os.listdir(net_dir):
                wireless_path = os.path.join(net_dir, iface, "wireless")
                if os.path.isdir(wireless_path):
                    operstate_path = os.path.join(net_dir, iface, "operstate")
                    try:
                        with open(operstate_path) as f:
                            state = f.read().strip()
                    except OSError:
                        state = "unknown"
                    connected = state == "up"
                    result = {
                        "connected": connected,
                        "device": iface,
                        "state": state,
                        "backend": "sysfs",
                    }
                    if connected:
                        try:
                            r = subprocess.run(
                                ["iwgetid", "-r", iface],
                                capture_output=True, text=True, timeout=5,
                            )
                            result["ssid"] = r.stdout.strip()
                        except (FileNotFoundError, subprocess.TimeoutExpired):
                            result["ssid"] = ""
                    return result
        except OSError:
            pass
        return {"connected": False, "error": "No WiFi interface found", "backend": "sysfs"}

    def _scan_networks(self) -> dict:
        """Scan for available WiFi networks."""
        if self._has_iwd():
            return self._iwd_scan()
        elif self._has_nmcli():
            return self._nmcli_scan()
        return {"networks": [], "error": "Neither iwd nor NetworkManager found"}

    def _iwd_scan(self) -> dict:
        device = self._iwd_get_device()
        if not device:
            return {"networks": [], "error": "No WiFi device found"}
        # Trigger scan
        subprocess.run(
            ["iwctl", "station", device, "scan"],
            capture_output=True, timeout=15,
        )
        # Get results
        try:
            r = subprocess.run(
                ["iwctl", "station", device, "get-networks"],
                capture_output=True, text=True, timeout=10,
            )
            networks = self._parse_iwctl_networks(r.stdout)
            return {"networks": networks, "device": device, "backend": "iwd"}
        except subprocess.TimeoutExpired:
            return {"networks": [], "error": "Scan timed out"}

    def _nmcli_scan(self) -> dict:
        try:
            # Trigger rescan
            subprocess.run(
                ["nmcli", "device", "wifi", "rescan"],
                capture_output=True, timeout=15,
            )
            r = subprocess.run(
                ["nmcli", "-t", "-f",
                 "SSID,SIGNAL,SECURITY,IN-USE",
                 "device", "wifi", "list"],
                capture_output=True, text=True, timeout=10,
            )
            networks = []
            seen = set()
            for line in r.stdout.strip().splitlines():
                parts = line.split(":")
                if len(parts) >= 3 and parts[0] and parts[0] not in seen:
                    seen.add(parts[0])
                    networks.append({
                        "ssid": parts[0],
                        "signal": int(parts[1]) if parts[1].isdigit() else 0,
                        "security": parts[2],
                        "connected": "*" in (parts[3] if len(parts) > 3 else ""),
                    })
            networks.sort(key=lambda n: n["signal"], reverse=True)
            return {"networks": networks, "backend": "nmcli"}
        except subprocess.TimeoutExpired:
            return {"networks": [], "error": "Scan timed out"}

    def _list_devices(self) -> dict:
        """List network interfaces."""
        import os
        devices = []
        net_dir = "/sys/class/net"
        try:
            for iface in sorted(os.listdir(net_dir)):
                iface_path = os.path.join(net_dir, iface)
                is_wireless = os.path.isdir(os.path.join(iface_path, "wireless"))
                try:
                    with open(os.path.join(iface_path, "operstate")) as f:
                        state = f.read().strip()
                except OSError:
                    state = "unknown"
                try:
                    with open(os.path.join(iface_path, "address")) as f:
                        mac = f.read().strip()
                except OSError:
                    mac = ""
                devices.append({
                    "name": iface,
                    "type": "wireless" if is_wireless else "wired",
                    "state": state,
                    "mac": mac,
                })
        except OSError:
            pass
        return {"devices": devices}

    # ------------------------------------------------------------------
    # WRITE handlers
    # ------------------------------------------------------------------

    def _handle_write(self, action_id: str, params: dict) -> dict:
        network_action = params.get("network_action", "connect")
        row_id = self._audit_start(action_id, "WRITE", params)
        try:
            if network_action == "connect":
                result = self._connect(params)
            elif network_action == "disconnect":
                result = self._disconnect()
            else:
                raise LeavesError(
                    LeavesErrorCode.AGENT_NOT_AVAILABLE,
                    detail=f"Unknown network_action: {network_action}",
                )
            self._audit_end(row_id, result)
            return result
        except LeavesError:
            raise
        except Exception as e:
            err = LeavesError(LeavesErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def _connect(self, params: dict) -> dict:
        ssid = params.get("ssid", "")
        if not ssid:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail="No SSID provided for WiFi connect",
            )
        passphrase = params.get("passphrase", "")

        if self._has_iwd():
            return self._iwd_connect(ssid, passphrase)
        elif self._has_nmcli():
            return self._nmcli_connect(ssid, passphrase)
        raise LeavesError(
            LeavesErrorCode.AGENT_NOT_AVAILABLE,
            detail="Neither iwd nor NetworkManager available",
        )

    def _iwd_connect(self, ssid: str, passphrase: str) -> dict:
        device = self._iwd_get_device()
        if not device:
            raise LeavesError(
                LeavesErrorCode.INTERNAL_ERROR,
                detail="No WiFi device found",
            )
        cmd = ["iwctl", "station", device, "connect", ssid]
        if passphrase:
            cmd.extend(["--passphrase", passphrase])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return {"action": "connect", "ssid": ssid, "success": True, "backend": "iwd"}
            return {"action": "connect", "ssid": ssid, "success": False,
                    "error": r.stderr.strip() or r.stdout.strip(), "backend": "iwd"}
        except subprocess.TimeoutExpired:
            return {"action": "connect", "ssid": ssid, "success": False,
                    "error": "Connection timed out"}

    def _nmcli_connect(self, ssid: str, passphrase: str) -> dict:
        cmd = ["nmcli", "device", "wifi", "connect", ssid]
        if passphrase:
            cmd.extend(["password", passphrase])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            success = r.returncode == 0
            return {
                "action": "connect", "ssid": ssid, "success": success,
                "message": r.stdout.strip() if success else r.stderr.strip(),
                "backend": "nmcli",
            }
        except subprocess.TimeoutExpired:
            return {"action": "connect", "ssid": ssid, "success": False,
                    "error": "Connection timed out"}

    def _disconnect(self) -> dict:
        if self._has_iwd():
            device = self._iwd_get_device()
            if device:
                r = subprocess.run(
                    ["iwctl", "station", device, "disconnect"],
                    capture_output=True, text=True, timeout=10,
                )
                return {"action": "disconnect", "success": r.returncode == 0, "backend": "iwd"}
        elif self._has_nmcli():
            r = subprocess.run(
                ["nmcli", "device", "disconnect", "wifi"],
                capture_output=True, text=True, timeout=10,
            )
            return {"action": "disconnect", "success": r.returncode == 0, "backend": "nmcli"}
        return {"action": "disconnect", "success": False, "error": "No network backend"}

    # ------------------------------------------------------------------
    # iwd helpers
    # ------------------------------------------------------------------

    def _iwd_get_device(self) -> str:
        """Get the first iwd wireless device name."""
        try:
            r = subprocess.run(
                ["iwctl", "device", "list"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.splitlines():
                stripped = line.strip()
                # Lines look like: "  wlan0   station   ..."
                parts = stripped.split()
                if len(parts) >= 2 and parts[1] == "station":
                    return parts[0]
                # Also try matching interface names directly
                if parts and parts[0].startswith("wl"):
                    return parts[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return ""

    def _parse_iwctl_table(self, output: str) -> dict:
        """Parse iwctl key-value table output."""
        info = {}
        for line in output.splitlines():
            # Lines look like "  Connected network    MySSID"
            # Trim ANSI escape codes
            import re
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
            parts = clean.split(None, 1)
            if len(parts) == 2:
                # Try to split on multiple spaces
                kv = re.split(r'\s{2,}', clean.strip(), maxsplit=1)
                if len(kv) == 2:
                    info[kv[0].strip()] = kv[1].strip()
        return info

    def _parse_iwctl_networks(self, output: str) -> list:
        """Parse iwctl get-networks output into network list."""
        import re
        networks = []
        for line in output.splitlines():
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
            if not clean or clean.startswith("---") or clean.startswith("Available"):
                continue
            # Format: "  > SSID   psk   ****"  or "    SSID   open"
            connected = clean.startswith(">")
            clean = clean.lstrip("> ")
            parts = re.split(r'\s{2,}', clean)
            if parts and parts[0]:
                network = {
                    "ssid": parts[0],
                    "security": parts[1] if len(parts) > 1 else "unknown",
                    "connected": connected,
                }
                # Signal strength shown as stars
                if len(parts) > 2:
                    stars = parts[2].count("*")
                    network["signal"] = stars * 25  # rough 0-100
                networks.append(network)
        return networks
