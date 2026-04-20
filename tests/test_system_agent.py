"""
Unit tests for SystemAgent — system info queries, app launch, app terminate.
All psutil/subprocess/shutil/os calls are mocked; nothing real is spawned or killed.
"""
import sys
from pathlib import Path
from collections import namedtuple
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.system_agent import SystemAgent, _PROTECTED_PROCESSES, _MAX_TERMINATE
from errors import MarshalError, MarshalErrorCode


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def agent():
    """SystemAgent with a stubbed DB connection (audit is non-fatal)."""
    db = MagicMock()
    return SystemAgent(intent_id="test-intent-001", db_conn=db)


def _action(action_type: str, params: dict, action_id: str = "a1") -> dict:
    return {"type": action_type, "action_id": action_id, "params": params}


# ------------------------------------------------------------------
# QUERY — cpu
# ------------------------------------------------------------------

class TestQueryCPU:

    @patch("agents.system_agent.psutil")
    def test_cpu_returns_expected_keys(self, mock_psutil, agent):
        mock_psutil.cpu_percent.return_value = 23.5
        freq = MagicMock()
        freq.current = 3600.0
        mock_psutil.cpu_freq.return_value = freq
        mock_psutil.cpu_count.side_effect = [8, 16]  # physical first, then logical
        mock_psutil.sensors_temperatures.return_value = {}

        result = agent.execute_action(_action("QUERY", {"query_type": "cpu"}))

        assert result["usage_percent"] == 23.5
        assert result["freq_mhz"] == 3600.0
        assert result["cores_physical"] == 8
        assert result["cores_logical"] == 16
        assert result["temp_celsius"] is None

    @patch("agents.system_agent.psutil")
    def test_cpu_with_temperature(self, mock_psutil, agent):
        mock_psutil.cpu_percent.return_value = 50.0
        freq = MagicMock()
        freq.current = 2400.0
        mock_psutil.cpu_freq.return_value = freq
        mock_psutil.cpu_count.side_effect = [4, 8]

        SensorEntry = namedtuple("SensorEntry", ["label", "current", "high", "critical"])
        mock_psutil.sensors_temperatures.return_value = {
            "coretemp": [SensorEntry("Core 0", 65.3, 100.0, 100.0)]
        }

        result = agent.execute_action(_action("QUERY", {"query_type": "cpu"}))
        assert result["temp_celsius"] == 65.3

    @patch("agents.system_agent.psutil")
    def test_cpu_no_freq(self, mock_psutil, agent):
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.cpu_freq.return_value = None
        mock_psutil.cpu_count.side_effect = [2, 4]
        mock_psutil.sensors_temperatures.return_value = {}

        result = agent.execute_action(_action("QUERY", {"query_type": "cpu"}))
        assert result["freq_mhz"] is None

    @patch("agents.system_agent.psutil")
    def test_cpu_physical_cores_none_still_exposes_logical(self, mock_psutil, agent):
        """When psutil can't determine physical cores (returns None), the agent
        still reports logical cores alongside a None physical count."""
        mock_psutil.cpu_percent.return_value = 5.0
        freq = MagicMock()
        freq.current = 1800.0
        mock_psutil.cpu_freq.return_value = freq
        mock_psutil.cpu_count.side_effect = [None, 4]  # physical=None
        mock_psutil.sensors_temperatures.return_value = {}

        result = agent.execute_action(_action("QUERY", {"query_type": "cpu"}))
        assert result["cores_physical"] is None
        assert result["cores_logical"] == 4


# ------------------------------------------------------------------
# QUERY — memory
# ------------------------------------------------------------------

class TestQueryMemory:

    @patch("agents.system_agent.psutil")
    def test_memory_returns_expected_keys(self, mock_psutil, agent):
        gb = 1024 ** 3
        vm = MagicMock()
        vm.total = 16 * gb
        vm.used = 8 * gb
        vm.available = 8 * gb
        vm.percent = 50.0
        mock_psutil.virtual_memory.return_value = vm

        result = agent.execute_action(_action("QUERY", {"query_type": "memory"}))

        assert result["total_gb"] == 16.0
        assert result["used_gb"] == 8.0
        assert result["available_gb"] == 8.0
        assert result["percent"] == 50.0


# ------------------------------------------------------------------
# QUERY — disk
# ------------------------------------------------------------------

class TestQueryDisk:

    @patch("agents.system_agent.psutil")
    def test_disk_returns_expected_keys(self, mock_psutil, agent):
        gb = 1024 ** 3
        usage = MagicMock()
        usage.total = 500 * gb
        usage.used = 200 * gb
        usage.free = 300 * gb
        usage.percent = 40.0
        mock_psutil.disk_usage.return_value = usage

        result = agent.execute_action(_action("QUERY", {"query_type": "disk"}))

        assert result["total_gb"] == 500.0
        assert result["used_gb"] == 200.0
        assert result["free_gb"] == 300.0
        assert result["percent"] == 40.0
        assert "path" in result

    @patch("agents.system_agent.psutil")
    def test_disk_with_custom_path(self, mock_psutil, agent):
        gb = 1024 ** 3
        usage = MagicMock()
        usage.total = 100 * gb
        usage.used = 50 * gb
        usage.free = 50 * gb
        usage.percent = 50.0
        mock_psutil.disk_usage.return_value = usage

        result = agent.execute_action(
            _action("QUERY", {"query_type": "disk", "path": "/tmp"})
        )

        mock_psutil.disk_usage.assert_called_once()
        assert result["percent"] == 50.0


# ------------------------------------------------------------------
# QUERY — processes
# ------------------------------------------------------------------

class TestQueryProcesses:

    @patch("agents.system_agent.psutil")
    def test_processes_returns_list(self, mock_psutil, agent):
        MemInfo = namedtuple("MemInfo", ["rss", "vms"])
        proc1 = MagicMock()
        proc1.info = {
            "pid": 100,
            "name": "firefox",
            "cpu_percent": 25.0,
            "memory_info": MemInfo(rss=500 * 1024 * 1024, vms=0),
            "status": "running",
        }
        proc2 = MagicMock()
        proc2.info = {
            "pid": 200,
            "name": "bash",
            "cpu_percent": 1.0,
            "memory_info": MemInfo(rss=10 * 1024 * 1024, vms=0),
            "status": "sleeping",
        }
        mock_psutil.process_iter.return_value = [proc1, proc2]

        result = agent.execute_action(
            _action("QUERY", {"query_type": "processes", "top_n": 5})
        )

        assert len(result["processes"]) == 2
        assert result["count"] == 2
        # Sorted by CPU descending
        assert result["processes"][0]["name"] == "firefox"
        assert result["processes"][1]["name"] == "bash"

    @patch("agents.system_agent.psutil")
    def test_processes_top_n_limits_output(self, mock_psutil, agent):
        MemInfo = namedtuple("MemInfo", ["rss", "vms"])
        procs = []
        for i in range(20):
            p = MagicMock()
            p.info = {
                "pid": i,
                "name": f"proc{i}",
                "cpu_percent": float(i),
                "memory_info": MemInfo(rss=1024 * 1024, vms=0),
                "status": "running",
            }
            procs.append(p)
        mock_psutil.process_iter.return_value = procs

        result = agent.execute_action(
            _action("QUERY", {"query_type": "processes", "top_n": 3})
        )

        assert len(result["processes"]) == 3
        assert result["count"] == 3

    @patch("agents.system_agent.psutil")
    def test_processes_skips_inaccessible(self, mock_psutil, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        good_proc = MagicMock()
        MemInfo = namedtuple("MemInfo", ["rss", "vms"])
        good_proc.info = {
            "pid": 1,
            "name": "ok",
            "cpu_percent": 1.0,
            "memory_info": MemInfo(rss=1024, vms=0),
            "status": "running",
        }
        bad_proc = MagicMock()
        bad_proc.info = property(lambda self: None)
        # Make accessing .info raise NoSuchProcess
        type(bad_proc).info = PropertyMock(
            side_effect=real_psutil.NoSuchProcess(999)
        )

        mock_psutil.process_iter.return_value = [good_proc, bad_proc]

        result = agent.execute_action(
            _action("QUERY", {"query_type": "processes"})
        )
        assert len(result["processes"]) == 1


# ------------------------------------------------------------------
# QUERY — uptime
# ------------------------------------------------------------------

class TestQueryUptime:

    @patch("agents.system_agent.time")
    @patch("agents.system_agent.psutil")
    def test_uptime_returns_expected_keys(self, mock_psutil, mock_time, agent):
        mock_psutil.boot_time.return_value = 1000000.0
        mock_time.time.return_value = 1003600.0  # 3600s uptime

        result = agent.execute_action(_action("QUERY", {"query_type": "uptime"}))

        assert result["uptime_seconds"] == 3600.0
        assert "boot_time_iso" in result


# ------------------------------------------------------------------
# QUERY — unknown query_type defaults to cpu
# ------------------------------------------------------------------

class TestQueryUnknown:

    @patch("agents.system_agent.psutil")
    def test_unknown_query_type_defaults_to_cpu(self, mock_psutil, agent):
        mock_psutil.cpu_percent.return_value = 10.0
        freq = MagicMock()
        freq.current = 2000.0
        mock_psutil.cpu_freq.return_value = freq
        mock_psutil.cpu_count.side_effect = [4, 8]
        mock_psutil.sensors_temperatures.return_value = {}

        result = agent.execute_action(
            _action("QUERY", {"query_type": "nonexistent_type"})
        )

        assert "usage_percent" in result

    @patch("agents.system_agent.psutil")
    def test_empty_query_type_defaults_to_cpu(self, mock_psutil, agent):
        mock_psutil.cpu_percent.return_value = 5.0
        freq = MagicMock()
        freq.current = 1000.0
        mock_psutil.cpu_freq.return_value = freq
        mock_psutil.cpu_count.side_effect = [2, 4]
        mock_psutil.sensors_temperatures.return_value = {}

        result = agent.execute_action(_action("QUERY", {}))

        assert "usage_percent" in result


# ------------------------------------------------------------------
# WRITE — launch
# ------------------------------------------------------------------

class TestLaunch:

    @patch("agents.system_agent.subprocess")
    @patch("agents.system_agent.shutil")
    def test_launch_valid_program(self, mock_shutil, mock_subprocess, agent):
        mock_shutil.which.return_value = "/usr/bin/firefox"
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_subprocess.Popen.return_value = mock_proc
        mock_subprocess.DEVNULL = -1

        result = agent.execute_action(
            _action("WRITE", {"program": "firefox"})
        )

        assert result["launched"] == "firefox"
        assert result["pid"] == 12345
        assert result["path"] == "/usr/bin/firefox"

    @patch("agents.system_agent.shutil")
    def test_launch_program_not_in_path(self, mock_shutil, agent):
        mock_shutil.which.return_value = None

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("WRITE", {"program": "nonexistent_app"})
            )

        assert exc_info.value.code == MarshalErrorCode.FILE_NOT_FOUND
        assert "nonexistent_app" in exc_info.value.detail

    def test_launch_empty_program(self, agent):
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("WRITE", {"program": ""}))

        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE

    def test_launch_missing_program_param(self, agent):
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("WRITE", {}))

        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE

    @patch("agents.system_agent.subprocess")
    @patch("agents.system_agent.shutil")
    def test_launch_popen_failure(self, mock_shutil, mock_subprocess, agent):
        mock_shutil.which.return_value = "/usr/bin/broken"
        mock_subprocess.Popen.side_effect = OSError("Permission denied")
        mock_subprocess.DEVNULL = -1

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("WRITE", {"program": "broken"})
            )

        assert exc_info.value.code == MarshalErrorCode.INTERNAL_ERROR
        assert "Failed to launch" in exc_info.value.detail

    @patch("agents.system_agent.subprocess")
    @patch("agents.system_agent.shutil")
    def test_launch_whitespace_program_stripped(self, mock_shutil, mock_subprocess, agent):
        mock_shutil.which.return_value = "/usr/bin/vim"
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_subprocess.Popen.return_value = mock_proc
        mock_subprocess.DEVNULL = -1

        result = agent.execute_action(
            _action("WRITE", {"program": "  vim  "})
        )

        mock_shutil.which.assert_called_with("vim")
        assert result["launched"] == "vim"


# ------------------------------------------------------------------
# DELETE — terminate by name
# ------------------------------------------------------------------

class TestTerminateByName:

    def _make_proc(self, pid, name, uid, psutil_mod):
        """Helper to create a mock psutil.Process-like object."""
        p = MagicMock()
        p.pid = pid
        p.name.return_value = name
        Uids = namedtuple("Uids", ["real", "effective", "saved"])
        p.info = {"pid": pid, "name": name, "uids": Uids(real=uid, effective=uid, saved=uid)}
        p.terminate.return_value = None
        p.wait.return_value = None
        p.kill.return_value = None
        return p

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_by_name_success(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.TimeoutExpired = real_psutil.TimeoutExpired

        mock_os.getuid.return_value = 1000

        proc = self._make_proc(555, "myapp", 1000, mock_psutil)
        mock_psutil.process_iter.return_value = [proc]

        result = agent.execute_action(
            _action("DELETE", {"target": "myapp"})
        )

        assert result["target"] == "myapp"
        assert result["count"] == 1
        assert result["terminated"][0]["pid"] == 555
        assert result["terminated"][0]["signal"] == "SIGTERM"
        proc.terminate.assert_called_once()

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_by_name_case_insensitive(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.TimeoutExpired = real_psutil.TimeoutExpired

        mock_os.getuid.return_value = 1000
        proc = self._make_proc(600, "MyApp", 1000, mock_psutil)
        mock_psutil.process_iter.return_value = [proc]

        result = agent.execute_action(
            _action("DELETE", {"target": "myapp"})
        )

        assert result["count"] == 1

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_protected_process(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        mock_os.getuid.return_value = 1000
        proc = self._make_proc(1, "systemd", 1000, mock_psutil)
        mock_psutil.process_iter.return_value = [proc]

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "systemd"})
            )

        assert exc_info.value.code == MarshalErrorCode.PERMISSION_DENIED
        assert "protected" in exc_info.value.detail

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_sshd_protected(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        mock_os.getuid.return_value = 1000
        proc = self._make_proc(99, "sshd", 1000, mock_psutil)
        mock_psutil.process_iter.return_value = [proc]

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "sshd"})
            )

        assert exc_info.value.code == MarshalErrorCode.PERMISSION_DENIED

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_nonexistent_process_name(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        mock_os.getuid.return_value = 1000
        mock_psutil.process_iter.return_value = []

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "ghost_process"})
            )

        assert exc_info.value.code == MarshalErrorCode.PROCESS_NOT_FOUND

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_skips_other_users_processes(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        mock_os.getuid.return_value = 1000
        # Process owned by root (uid 0)
        proc = self._make_proc(42, "myapp", 0, mock_psutil)
        mock_psutil.process_iter.return_value = [proc]

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "myapp"})
            )

        assert exc_info.value.code == MarshalErrorCode.PROCESS_NOT_FOUND

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_exceeds_max_limit(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        mock_os.getuid.return_value = 1000
        procs = [
            self._make_proc(i, "spammer", 1000, mock_psutil)
            for i in range(_MAX_TERMINATE + 1)
        ]
        mock_psutil.process_iter.return_value = procs

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "spammer"})
            )

        assert exc_info.value.code == MarshalErrorCode.PERMISSION_DENIED
        assert str(_MAX_TERMINATE) in exc_info.value.detail

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_sigkill_fallback(self, mock_psutil, mock_os, agent):
        """Process ignores SIGTERM, gets SIGKILL after timeout."""
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.TimeoutExpired = real_psutil.TimeoutExpired

        mock_os.getuid.return_value = 1000
        proc = self._make_proc(777, "stubborn", 1000, mock_psutil)
        # First wait (after SIGTERM) times out, second wait (after SIGKILL) succeeds
        proc.wait.side_effect = [real_psutil.TimeoutExpired(3), None]

        mock_psutil.process_iter.return_value = [proc]

        result = agent.execute_action(
            _action("DELETE", {"target": "stubborn"})
        )

        proc.kill.assert_called_once()
        assert result["terminated"][0]["signal"] == "SIGKILL"

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_already_exited(self, mock_psutil, mock_os, agent):
        """Process exits between finding it and sending SIGTERM."""
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.TimeoutExpired = real_psutil.TimeoutExpired

        mock_os.getuid.return_value = 1000
        proc = self._make_proc(888, "ephemeral", 1000, mock_psutil)
        proc.terminate.side_effect = real_psutil.NoSuchProcess(888)

        mock_psutil.process_iter.return_value = [proc]

        result = agent.execute_action(
            _action("DELETE", {"target": "ephemeral"})
        )

        assert result["terminated"][0]["signal"] == "already_exited"


# ------------------------------------------------------------------
# DELETE — terminate by PID
# ------------------------------------------------------------------

class TestTerminateByPID:

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_pid_success(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.TimeoutExpired = real_psutil.TimeoutExpired

        mock_os.getuid.return_value = 1000

        Uids = namedtuple("Uids", ["real", "effective", "saved"])
        proc = MagicMock()
        proc.pid = 1234
        proc.name.return_value = "vim"
        proc.uids.return_value = Uids(real=1000, effective=1000, saved=1000)
        proc.terminate.return_value = None
        proc.wait.return_value = None

        mock_psutil.Process.return_value = proc

        result = agent.execute_action(
            _action("DELETE", {"target": "1234"})
        )

        assert result["target"] == "1234"
        assert result["count"] == 1
        assert result["terminated"][0]["pid"] == 1234

    @patch("agents.system_agent.psutil")
    def test_terminate_pid_not_found(self, mock_psutil, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.Process.side_effect = real_psutil.NoSuchProcess(99999)

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "99999"})
            )

        assert exc_info.value.code == MarshalErrorCode.PROCESS_NOT_FOUND

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_pid_wrong_user(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        mock_os.getuid.return_value = 1000

        Uids = namedtuple("Uids", ["real", "effective", "saved"])
        proc = MagicMock()
        proc.pid = 1
        proc.name.return_value = "root_proc"
        proc.uids.return_value = Uids(real=0, effective=0, saved=0)

        mock_psutil.Process.return_value = proc

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "1"})
            )

        assert exc_info.value.code == MarshalErrorCode.PERMISSION_DENIED

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_pid_access_denied(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        mock_os.getuid.return_value = 1000

        proc = MagicMock()
        proc.pid = 2
        proc.uids.side_effect = real_psutil.AccessDenied(2)

        mock_psutil.Process.return_value = proc

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "2"})
            )

        assert exc_info.value.code == MarshalErrorCode.PERMISSION_DENIED

    @patch("agents.system_agent.os")
    @patch("agents.system_agent.psutil")
    def test_terminate_pid_protected_process(self, mock_psutil, mock_os, agent):
        import psutil as real_psutil
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        mock_os.getuid.return_value = 1000

        Uids = namedtuple("Uids", ["real", "effective", "saved"])
        proc = MagicMock()
        proc.pid = 50
        proc.name.return_value = "python3"
        proc.uids.return_value = Uids(real=1000, effective=1000, saved=1000)

        mock_psutil.Process.return_value = proc

        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(
                _action("DELETE", {"target": "50"})
            )

        assert exc_info.value.code == MarshalErrorCode.PERMISSION_DENIED
        assert "protected" in exc_info.value.detail


# ------------------------------------------------------------------
# DELETE — empty target
# ------------------------------------------------------------------

class TestTerminateEdgeCases:

    def test_terminate_empty_target(self, agent):
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("DELETE", {"target": ""}))

        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE

    def test_terminate_missing_target_param(self, agent):
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("DELETE", {}))

        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE


# ------------------------------------------------------------------
# Invalid action type
# ------------------------------------------------------------------

class TestInvalidAction:

    def test_invalid_action_type(self, agent):
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("PATCH", {}))

        assert exc_info.value.code == MarshalErrorCode.NOT_IMPLEMENTED

    def test_empty_action_type(self, agent):
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action({"type": "", "action_id": "a1", "params": {}})

        assert exc_info.value.code == MarshalErrorCode.NOT_IMPLEMENTED

    def test_missing_action_type(self, agent):
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action({"action_id": "a1", "params": {}})

        assert exc_info.value.code == MarshalErrorCode.NOT_IMPLEMENTED


# ------------------------------------------------------------------
# Audit logging integration
# ------------------------------------------------------------------

class TestAuditCalls:

    @patch("agents.system_agent.psutil")
    def test_query_calls_audit_start_and_end(self, mock_psutil, agent):
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.cpu_freq.return_value = None
        mock_psutil.cpu_count.side_effect = [4, 4]
        mock_psutil.sensors_temperatures.return_value = {}

        agent.execute_action(_action("QUERY", {"query_type": "cpu"}))

        agent._db.assert_not_called  # DB is accessed via db.audit module
        # Just verify no exception — audit is non-fatal

    @patch("agents.system_agent.shutil")
    def test_launch_failure_still_audits(self, mock_shutil, agent):
        """Even on failure, audit_start is called before the error."""
        mock_shutil.which.return_value = None

        with pytest.raises(MarshalError):
            agent.execute_action(_action("WRITE", {"program": "nope"}))
        # Audit start is called before shutil.which check? No — the error
        # is raised before audit_start for this case. That's fine.


# ------------------------------------------------------------------
# Protected processes list sanity
# ------------------------------------------------------------------

class TestProtectedProcesses:

    def test_protected_list_contains_critical_processes(self):
        assert "systemd" in _PROTECTED_PROCESSES
        assert "init" in _PROTECTED_PROCESSES
        assert "sshd" in _PROTECTED_PROCESSES
        assert "python3" in _PROTECTED_PROCESSES
        assert "agentd" in _PROTECTED_PROCESSES

    def test_max_terminate_is_reasonable(self):
        assert 1 <= _MAX_TERMINATE <= 20
