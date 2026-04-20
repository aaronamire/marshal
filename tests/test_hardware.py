"""
Tests for hardware.py tier selection.

Static probe is thin wrapping of psutil/platform — exercised by running the
CLI. The interesting logic is select_tier's disqualification math and the
bandwidth-bound extrapolation, which is what we exercise here with synthetic
HardwareProfiles.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import hardware
from hardware import (
    BenchmarkResult,
    HardwareProfile,
    ModelDownloadError,
    TIER_BY_NAME,
    active_tier_name,
    download_model_for_tier,
    ensure_tier_model,
    read_decision,
    resolve_model_path,
    select_tier,
    write_decision,
)


def _profile(
    *,
    total: float,
    available: float,
    laptop: bool = False,
) -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=total,
        available_ram_gb=available,
        cpu_model="synthetic",
        cpu_physical_cores=4,
        cpu_logical_cores=8,
        cpu_has_avx2=True,
        cpu_has_avx512=False,
        platform="linux",
        arch="x86_64",
        is_laptop=laptop,
    )


def _bench(gen_tok_s: float, ref_b: float = 3.0) -> BenchmarkResult:
    return BenchmarkResult(
        reference_model="synthetic-qwen",
        reference_param_billions=ref_b,
        gen_tok_s_median=gen_tok_s,
        gen_tok_s_min=gen_tok_s * 0.95,
        gen_tok_s_p95=gen_tok_s,
        prompt_eval_tok_s_median=gen_tok_s * 4,
        samples=5,
        thermal_dropoff_pct=5.0,
    )


# ---------------------------------------------------------------------------
# RAM gating (no benchmark)
# ---------------------------------------------------------------------------

def test_ancient_laptop_falls_to_tiny():
    """4GB total RAM: nothing above 'tiny' should qualify."""
    d = select_tier(_profile(total=4.0, available=2.5, laptop=True))
    assert d.chosen == "tiny"
    assert "standard" in d.disqualified
    assert "pro" in d.disqualified
    assert "max" in d.disqualified


def test_typical_8gb_laptop_picks_standard():
    """User's machine: 8GB total, ~5GB free. Should pick 'standard' (3B)."""
    d = select_tier(_profile(total=8.0, available=5.0, laptop=True))
    assert d.chosen == "standard"
    assert "pro" in d.disqualified
    assert "max" in d.disqualified


def test_16gb_laptop_reaches_pro_on_ram_alone():
    """16GB RAM + headroom fits the 'pro' (7B) footprint."""
    d = select_tier(_profile(total=16.0, available=10.0, laptop=True))
    assert d.chosen == "pro"


def test_32gb_workstation_reaches_max():
    d = select_tier(_profile(total=32.0, available=20.0, laptop=False))
    assert d.chosen == "max"


def test_low_available_ram_blocks_upgrade_even_with_high_total():
    """16GB total but only 3GB free (Chrome + IDE open) → stuck at standard."""
    d = select_tier(_profile(total=16.0, available=3.0, laptop=True))
    assert d.chosen == "tiny"
    assert "standard" in d.disqualified  # 3GB < 3.5GB available floor
    assert "pro" in d.disqualified


# ---------------------------------------------------------------------------
# Benchmark-driven selection
# ---------------------------------------------------------------------------

def test_fast_bench_on_high_ram_picks_pro():
    """
    3B measured at 20 tok/s → 7B extrapolated to ~8.6 tok/s. That clears the
    'pro' 8 tok/s interactive floor. With 16GB RAM we should land on 'pro'.
    """
    profile = _profile(total=16.0, available=10.0, laptop=False)
    d = select_tier(profile, _bench(gen_tok_s=20.0))
    assert d.chosen == "pro"
    assert d.estimated_gen_tok_s["pro"] > 8.0


def test_slow_bench_forces_fallback_even_with_ram():
    """
    3B at 4 tok/s on a 16GB machine → 7B extrapolates to ~1.7 tok/s, fails the
    8 tok/s floor. Should land on 'standard' instead of 'pro' despite
    sufficient RAM.
    """
    profile = _profile(total=16.0, available=10.0, laptop=False)
    d = select_tier(profile, _bench(gen_tok_s=4.0))
    assert d.chosen == "tiny"  # 4 tok/s also fails 'standard' 6 tok/s floor
    assert "too slow" in d.disqualified["standard"].lower() or \
           "interactive floor" in d.disqualified["standard"]


def test_laptop_haircut_applied():
    """
    Identical benchmark on laptop vs desktop: laptop's extrapolation is 15%
    lower, which can flip a borderline tier decision.
    """
    bench = _bench(gen_tok_s=15.0)
    # Desktop version
    d_desk = select_tier(_profile(total=16.0, available=10.0, laptop=False), bench)
    # Laptop version
    d_lap = select_tier(_profile(total=16.0, available=10.0, laptop=True), bench)
    assert d_desk.estimated_gen_tok_s["pro"] > d_lap.estimated_gen_tok_s["pro"]
    # Specifically, the haircut should be ~15%
    ratio = d_lap.estimated_gen_tok_s["pro"] / d_desk.estimated_gen_tok_s["pro"]
    assert 0.83 < ratio < 0.87


def test_extrapolation_math():
    """tok/s scales inverse to param count at fixed quant."""
    b = _bench(gen_tok_s=30.0, ref_b=3.0)
    assert b.extrapolate_gen_tok_s(TIER_BY_NAME["standard"]) == pytest.approx(30.0)
    assert b.extrapolate_gen_tok_s(TIER_BY_NAME["pro"]) == pytest.approx(30.0 * 3 / 7, rel=0.01)
    assert b.extrapolate_gen_tok_s(TIER_BY_NAME["tiny"]) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_write_then_read_roundtrip(tmp_path: Path):
    profile = _profile(total=8.0, available=5.0, laptop=True)
    d = select_tier(profile)
    target = tmp_path / "tier.json"
    write_decision(d, target)
    assert target.exists()

    loaded = read_decision(target)
    assert loaded is not None
    assert loaded["chosen"] == d.chosen
    assert loaded["profile"]["total_ram_gb"] == 8.0


def test_write_is_atomic(tmp_path: Path):
    """The tmp-then-rename pattern should not leave half-written files."""
    target = tmp_path / "tier.json"
    d = select_tier(_profile(total=8.0, available=5.0))
    write_decision(d, target)
    # No leftover tmp file
    assert not (tmp_path / "tier.json.tmp").exists()
    # File is valid JSON
    json.loads(target.read_text())


def test_read_missing_returns_none(tmp_path: Path):
    assert read_decision(tmp_path / "absent.json") is None


# ---------------------------------------------------------------------------
# CLI sanity (doesn't actually call the server)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------

def test_resolver_returns_preferred_when_present(tmp_path: Path):
    """'pro' tier saved, 7B file on disk → resolver returns 7B."""
    models = tmp_path / "models"
    models.mkdir()
    (models / TIER_BY_NAME["pro"].model_file).touch()
    (models / TIER_BY_NAME["standard"].model_file).touch()
    result = resolve_model_path(models, preferred_tier="pro")
    assert result is not None
    assert result.name == TIER_BY_NAME["pro"].model_file


def test_resolver_walks_down_when_preferred_missing(tmp_path: Path):
    """'pro' requested but only 3B and 1.5B on disk → resolver returns 3B."""
    models = tmp_path / "models"
    models.mkdir()
    (models / TIER_BY_NAME["standard"].model_file).touch()
    (models / TIER_BY_NAME["tiny"].model_file).touch()
    result = resolve_model_path(models, preferred_tier="pro")
    assert result is not None
    assert result.name == TIER_BY_NAME["standard"].model_file


def test_resolver_returns_none_when_nothing_present(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    assert resolve_model_path(models, preferred_tier="standard") is None


def test_resolver_returns_none_when_models_dir_missing(tmp_path: Path):
    assert resolve_model_path(tmp_path / "absent", preferred_tier="standard") is None


def test_resolver_reads_saved_tier_when_no_preference(tmp_path: Path, monkeypatch):
    """With no explicit preference, resolver should read the saved decision."""
    config = tmp_path / "tier.json"
    monkeypatch.setattr("hardware.CONFIG_PATH", config)
    d = select_tier(_profile(total=16.0, available=10.0, laptop=False))
    write_decision(d, config)
    models = tmp_path / "models"
    models.mkdir()
    (models / TIER_BY_NAME[d.chosen].model_file).touch()
    result = resolve_model_path(models, config_path=config)
    assert result is not None
    assert result.name == TIER_BY_NAME[d.chosen].model_file


def test_resolver_falls_back_to_legacy_fine_tuned(tmp_path: Path):
    """If only the Phase 2 legacy artifact exists, use it."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "goalspec_qwen25_3b_q4km.gguf").touch()
    result = resolve_model_path(models, preferred_tier="pro")
    assert result is not None
    assert result.name == "goalspec_qwen25_3b_q4km.gguf"


def test_active_tier_name_unset(tmp_path: Path):
    assert active_tier_name(tmp_path / "nope.json") is None


def test_active_tier_name_returns_chosen(tmp_path: Path):
    d = select_tier(_profile(total=8.0, available=5.0))
    target = tmp_path / "tier.json"
    write_decision(d, target)
    assert active_tier_name(target) == d.chosen


def test_cli_resolve_model_exits_nonzero_when_nothing(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    rc = __import__("hardware").main(["--resolve-model", "--models-dir", str(models)])
    assert rc == 1


# ---------------------------------------------------------------------------
# Catalog integrity — every tier must be downloadable
# ---------------------------------------------------------------------------

def test_every_tier_has_download_metadata():
    for name, tier in TIER_BY_NAME.items():
        assert tier.hf_repo and "/" in tier.hf_repo, f"{name} missing hf_repo"
        assert tier.hf_filename.endswith(".gguf"), f"{name} hf_filename must be GGUF"


# ---------------------------------------------------------------------------
# Download function — mocked subprocess, no real network
# ---------------------------------------------------------------------------

def test_download_short_circuits_when_file_exists(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    tier = TIER_BY_NAME["tiny"]
    (models / tier.model_file).write_bytes(b"x" * 1024)  # pretend it's there

    calls = []
    def _never_called(*a, **kw):
        calls.append((a, kw))
        raise AssertionError("subprocess should not be invoked when file exists")
    monkeypatch.setattr("subprocess.run", _never_called)

    result = download_model_for_tier(tier, models)
    assert result.exists()
    assert not calls


def test_download_raises_when_cli_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("hardware._huggingface_cli_available", lambda: False)
    tier = TIER_BY_NAME["tiny"]
    with pytest.raises(ModelDownloadError, match="huggingface-cli not found"):
        download_model_for_tier(tier, tmp_path / "models")


def test_download_raises_when_subprocess_fails(tmp_path: Path, monkeypatch):
    import subprocess
    monkeypatch.setattr("hardware._huggingface_cli_available", lambda: True)
    class FakeProc:
        returncode = 1
        stdout = ""
        stderr = "401 Unauthorized\naccess denied"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeProc())

    tier = TIER_BY_NAME["tiny"]
    with pytest.raises(ModelDownloadError, match="401"):
        download_model_for_tier(tier, tmp_path / "models")


def test_download_success_creates_file(tmp_path: Path, monkeypatch):
    """Simulate a successful hf download by having the fake subprocess
    materialize the target file."""
    import subprocess
    monkeypatch.setattr("hardware._huggingface_cli_available", lambda: True)
    tier = TIER_BY_NAME["tiny"]
    models = tmp_path / "models"

    class FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **kw):
        # Pull the --local-dir argument and drop a stub file there
        local_dir = Path(cmd[cmd.index("--local-dir") + 1])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / tier.hf_filename).write_bytes(b"stub-gguf")
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = download_model_for_tier(tier, models)
    assert result == models / tier.model_file
    assert result.read_bytes() == b"stub-gguf"


def test_ensure_tier_model_no_consent_returns_none(tmp_path: Path):
    """If the file is missing and no consent_fn is provided, we never download."""
    tier = TIER_BY_NAME["tiny"]
    assert ensure_tier_model(tier, tmp_path / "models") is None


def test_ensure_tier_model_declined_consent(tmp_path: Path):
    tier = TIER_BY_NAME["tiny"]
    consent_calls = []
    def say_no(t):
        consent_calls.append(t.name)
        return False
    result = ensure_tier_model(tier, tmp_path / "models", consent_fn=say_no)
    assert result is None
    assert consent_calls == ["tiny"]


def test_ensure_tier_model_returns_existing(tmp_path: Path):
    tier = TIER_BY_NAME["tiny"]
    models = tmp_path / "models"
    models.mkdir()
    target = models / tier.model_file
    target.write_bytes(b"stub")

    def never_called(t):
        raise AssertionError("consent_fn should not fire when file exists")

    result = ensure_tier_model(tier, models, consent_fn=never_called)
    assert result == target


def test_cli_force_tier_bypasses_autodetect(tmp_path: Path, monkeypatch):
    """--tier pro should override auto-detection and persist."""
    monkeypatch.setattr(hardware, "CONFIG_PATH", tmp_path / "tier.json")
    rc = hardware.main(["--tier", "pro", "--no-bench"])
    assert rc == 0
    saved = read_decision(tmp_path / "tier.json")
    assert saved is not None
    assert saved["chosen"] == "pro"
    assert "forced" in saved["reason"]
