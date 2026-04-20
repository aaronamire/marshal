"""
Tests for install.py — the Marshal install/doctor agent.

These tests don't touch the network or mutate the system. They verify the
*shape* of the install agent: the environment probe, the topological step
ordering, the state-persistence round-trip, and the CLI dispatch. Every
step's `check()` is exercised against the live Env (fast, read-only) to
catch regressions in the check surface.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import install
from install import (
    ALL_STEPS,
    CheckResult,
    CheckStatus,
    Env,
    FailureKind,
    InstallState,
    InstallStep,
    Remediation,
    _parse_kernel,
    build_plan,
    emit_json,
    load_state,
    main,
    probe_env,
    save_state,
    steps_for,
    topological_order,
)


# ---------------------------------------------------------------------------
# Kernel parsing
# ---------------------------------------------------------------------------

class TestParseKernel:
    def test_arch_release(self):
        assert _parse_kernel("6.18.7-arch1-1") == (6, 18, 7)

    def test_plain(self):
        assert _parse_kernel("5.15.0") == (5, 15, 0)

    def test_ubuntu_dash(self):
        assert _parse_kernel("5.15.0-generic") == (5, 15, 0)

    def test_unparseable_returns_zeros(self):
        assert _parse_kernel("weird-kernel") == (0, 0, 0)


# ---------------------------------------------------------------------------
# Env probe
# ---------------------------------------------------------------------------

class TestProbeEnv:
    def test_returns_env_dataclass(self):
        env = probe_env()
        assert isinstance(env, Env)

    def test_linux_platform_detected(self):
        env = probe_env()
        assert env.os_name == "linux"

    def test_kernel_tuple_has_three_ints(self):
        env = probe_env()
        assert len(env.kernel) == 3
        assert all(isinstance(v, int) for v in env.kernel)

    def test_ram_is_positive(self):
        env = probe_env()
        assert env.ram_total_gb > 0
        assert env.ram_available_gb > 0

    def test_to_dict_is_json_serialisable(self):
        env = probe_env()
        json.dumps(env.to_dict())

    def test_supports_landlock_threshold(self):
        old = Env(
            os_name="linux", distro="arch", distro_version="", kernel=(5, 12, 0),
            arch="x86_64", python_version=(3, 12, 0), python_exec="/x",
            ram_total_gb=8.0, ram_available_gb=4.0, disk_free_gb=100.0,
            is_laptop=True, has_sudo=False, has_network=True, has_git=True,
            has_cmake=True, has_huggingface_cli=False, has_pacman=True,
            has_apt=False, has_dnf=False, has_systemctl=True,
        )
        assert not old.supports_landlock
        new = Env(**{**old.__dict__, "kernel": (5, 13, 0)})
        assert new.supports_landlock

    def test_package_manager_priority(self):
        base = dict(
            os_name="linux", distro="", distro_version="", kernel=(6, 0, 0),
            arch="x86_64", python_version=(3, 12, 0), python_exec="/x",
            ram_total_gb=8.0, ram_available_gb=4.0, disk_free_gb=100.0,
            is_laptop=True, has_sudo=False, has_network=True, has_git=True,
            has_cmake=True, has_huggingface_cli=False,
            has_pacman=True, has_apt=True, has_dnf=True, has_systemctl=True,
        )
        # pacman wins over apt and dnf
        assert Env(**base).package_manager == "pacman"
        assert Env(**{**base, "has_pacman": False}).package_manager == "apt"
        assert Env(**{**base, "has_pacman": False, "has_apt": False}).package_manager == "dnf"
        assert Env(
            **{**base, "has_pacman": False, "has_apt": False, "has_dnf": False}
        ).package_manager is None


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

class _Step(InstallStep):
    id = ""
    def check(self, env):  # pragma: no cover
        return CheckResult(CheckStatus.SATISFIED, "")
    def execute(self, env, *, on_output):  # pragma: no cover
        return install.ExecuteResult(True, "")


def _mkstep(sid: str, deps: tuple[str, ...] = (), optional: bool = False):
    cls = type(f"Step_{sid}", (_Step,), {
        "id": sid,
        "title": sid,
        "depends_on": deps,
        "optional": optional,
    })
    return cls


class TestTopologicalOrder:
    def test_linear_chain(self):
        a = _mkstep("a")
        b = _mkstep("b", ("a",))
        c = _mkstep("c", ("b",))
        order = topological_order((c, b, a))
        assert [s.id for s in order] == ["a", "b", "c"]

    def test_diamond(self):
        a = _mkstep("a")
        b = _mkstep("b", ("a",))
        c = _mkstep("c", ("a",))
        d = _mkstep("d", ("b", "c"))
        order = topological_order((a, b, c, d))
        ids = [s.id for s in order]
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_missing_dep_raises(self):
        a = _mkstep("a", ("nope",))
        with pytest.raises(ValueError, match="unknown"):
            topological_order((a,))

    def test_cycle_raises(self):
        a = _mkstep("a", ("b",))
        b = _mkstep("b", ("a",))
        with pytest.raises(ValueError, match="cycle"):
            topological_order((a, b))

    def test_stable_order_by_declaration(self):
        a = _mkstep("a")
        b = _mkstep("b")
        c = _mkstep("c")
        order = topological_order((a, b, c))
        assert [s.id for s in order] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# steps_for — transitive expansion
# ---------------------------------------------------------------------------

class TestStepsFor:
    def test_none_returns_all_steps(self):
        steps = steps_for(None)
        assert len(steps) == len(ALL_STEPS)
        assert {s.id for s in steps} == {s.id for s in ALL_STEPS}

    def test_single_step_expands_to_deps(self):
        # model depends on hardware-probe, venv; venv depends on python; …
        steps = steps_for({"model"})
        ids = {s.id for s in steps}
        assert "model" in ids
        assert "hardware-probe" in ids
        assert "venv" in ids
        assert "python" in ids

    def test_order_preserves_topological(self):
        steps = steps_for({"model"})
        ids = [s.id for s in steps]
        assert ids.index("python") < ids.index("venv")
        assert ids.index("venv") < ids.index("hardware-probe")
        assert ids.index("hardware-probe") < ids.index("model")

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="unknown step id"):
            steps_for({"totally-made-up"})


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_load_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "install-state.json")
        s = InstallState(
            completed={"os": "2026-04-20T10:00:00", "python": "2026-04-20T10:00:01"},
            fingerprint="git:abc123",
            env_snapshot={"ram_total_gb": 8.0},
        )
        save_state(s)
        loaded = load_state()
        # fingerprint gets stamped to current, but completed survives if fingerprint matches
        # To assert round-trip regardless, also pin fingerprint:
        monkeypatch.setattr(install, "_fingerprint", lambda: "git:abc123")
        loaded = load_state()
        assert loaded.completed == s.completed
        assert loaded.fingerprint == "git:abc123"

    def test_missing_state_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "missing.json")
        s = load_state()
        assert s.completed == {}
        assert s.fingerprint != ""

    def test_corrupt_state_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        path = tmp_path / "state.json"
        path.write_text("{not json")
        monkeypatch.setattr(install, "STATE_PATH", path)
        s = load_state()
        assert s.completed == {}

    def test_fingerprint_change_invalidates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(install, "_fingerprint", lambda: "v1")
        save_state(InstallState(completed={"os": "t"}, fingerprint="v1"))
        monkeypatch.setattr(install, "_fingerprint", lambda: "v2")
        s = load_state()
        assert s.completed == {}
        assert s.fingerprint == "v2"

    def test_save_is_atomic(self, tmp_path, monkeypatch):
        """Save via tmp + rename so partial writes never land on STATE_PATH."""
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        save_state(InstallState(completed={"a": "t"}))
        # Tmp file must be cleaned up by the atomic rename
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Step catalog — every step is well-formed and checkable
# ---------------------------------------------------------------------------

class TestStepCatalog:
    def test_all_steps_have_id_and_title(self):
        for cls in ALL_STEPS:
            assert cls.id, f"{cls.__name__} missing id"
            assert cls.title, f"{cls.__name__} missing title"

    def test_all_step_ids_unique(self):
        ids = [cls.id for cls in ALL_STEPS]
        assert len(ids) == len(set(ids)), f"duplicate step ids: {ids}"

    def test_all_steps_checkable_against_live_env(self):
        """Every step.check() must return a CheckResult without raising."""
        env = probe_env()
        for cls in ALL_STEPS:
            result = cls().check(env)
            assert isinstance(result, CheckResult), f"{cls.__name__} returned {type(result)}"
            assert result.status in CheckStatus

    def test_unsatisfied_non_optional_steps_have_remediation(self):
        """Users are never stuck without a next action on the core path.
        Optional steps may return UNSATISFIED without remediation (e.g. sanity
        runs on demand), so the invariant applies only to required steps."""
        env = probe_env()
        for cls in ALL_STEPS:
            if cls.optional:
                continue
            result = cls().check(env)
            if result.status in (CheckStatus.UNSATISFIED, CheckStatus.BLOCKED):
                assert result.remediation is not None, (
                    f"{cls.__name__} status={result.status.value} has no remediation"
                )
                assert result.remediation.summary


# ---------------------------------------------------------------------------
# Plan building + JSON emit
# ---------------------------------------------------------------------------

class TestBuildPlan:
    def test_build_plan_returns_pairs(self):
        env = probe_env()
        plan = build_plan(env, None)
        assert len(plan) == len(ALL_STEPS)
        for (cls, result) in plan:
            assert issubclass(cls, InstallStep)
            assert isinstance(result, CheckResult)

    def test_emit_json_is_valid_json(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        env = probe_env()
        plan = build_plan(env, {"os"})
        emit_json(env, plan)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "env" in data
        assert "steps" in data
        assert len(data["steps"]) >= 1
        assert data["steps"][0]["id"] == "os"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_default_is_plan(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        rc = main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "install plan" in out.lower()

    def test_plan_apply_doctor_are_mutex(self, capsys):
        with pytest.raises(SystemExit):
            main(["--plan", "--apply"])

    def test_doctor_returns_zero_when_all_satisfied(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        # Fake a plan where every step is SATISFIED
        env = probe_env()
        fake = [(cls, CheckResult(CheckStatus.SATISFIED, "ok")) for cls in ALL_STEPS]
        with mock.patch.object(install, "build_plan", return_value=fake):
            rc = main(["--doctor"])
        assert rc == 0

    def test_doctor_returns_one_when_unsatisfied(self, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        from install import StepOS
        fake = [(StepOS, CheckResult(CheckStatus.UNSATISFIED, "bad",
                                     Remediation("fix it")))]
        with mock.patch.object(install, "build_plan", return_value=fake):
            rc = main(["--doctor"])
        assert rc == 1

    def test_doctor_returns_two_when_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        from install import StepOS
        fake = [(StepOS, CheckResult(CheckStatus.BLOCKED, "nope",
                                     Remediation("use docker")))]
        with mock.patch.object(install, "build_plan", return_value=fake):
            rc = main(["--doctor"])
        assert rc == 2

    def test_unknown_step_id_returns_two(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        rc = main(["--plan", "--step", "does-not-exist"])
        assert rc == 2

    def test_json_output_parses(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        rc = main(["--json", "--step", "os"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["env"]["os_name"] == "linux"

    def test_force_clears_saved_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(install, "MARSHAL_HOME", tmp_path)
        monkeypatch.setattr(install, "STATE_PATH", tmp_path / "s.json")
        monkeypatch.setattr(install, "_fingerprint", lambda: "v1")
        save_state(InstallState(completed={"os": "t"}, fingerprint="v1"))
        rc = main(["--plan", "--force", "--step", "os"])
        assert rc == 0
        reloaded = load_state()
        assert reloaded.completed == {}


# ---------------------------------------------------------------------------
# Result taxonomy — immutability and shape
# ---------------------------------------------------------------------------

class TestResultTaxonomy:
    def test_remediation_is_frozen(self):
        r = Remediation("s")
        with pytest.raises(Exception):
            r.summary = "other"  # type: ignore[misc]

    def test_check_result_is_frozen(self):
        r = CheckResult(CheckStatus.SATISFIED, "ok")
        with pytest.raises(Exception):
            r.detail = "other"  # type: ignore[misc]

    def test_failure_kind_values_are_strings(self):
        for k in FailureKind:
            assert isinstance(k.value, str)
