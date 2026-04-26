"""
Marshal hardware detection and model-tier selection.

The problem: the right local model for Marshal depends on the user's machine,
but specs lie. LLM inference on CPU is memory-bandwidth-bound, not
compute-bound — two "16GB RAM, modern CPU" laptops can differ 2.5x in tok/s.
And `total_ram` is not what matters at runtime; `available_ram` with Chrome +
IDE + Slack running is.

This module does three things:

1. Static probe (cheap, always runs): RAM, CPU identity, AVX flags on x86,
   laptop-vs-desktop inference via battery presence. Catches the obvious
   disqualifiers (not enough RAM to hold the model + KV cache + headroom).

2. Optional calibration (when the inference server is reachable): run a short
   generation benchmark against the live model, measure tok/s. Extrapolate to
   other tiers via a bandwidth-bound approximation (tok/s ~ 1/model_size at
   fixed quant). This is far more reliable than guessing from CPU flags.

3. Selection: pick the largest tier whose estimated generation tok/s clears
   the interactive budget AND whose memory footprint fits with headroom.
   Every disqualification is recorded with a reason so the user can see the
   math behind the choice.

GPU detection is explicitly out of scope for v1 — the CUDA/ROCm/Vulkan/Metal
fan-out is a rabbit hole that doesn't pay off for the 3B/7B CPU path most
users will land on. `HardwareProfile.gpu_detected` is a reserved hook.

Invoke directly (`python hardware.py` or `python -m hardware`) to probe,
benchmark if a server is up, print the decision, and persist it.
"""
from __future__ import annotations

import json
import platform
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import psutil


# ---------------------------------------------------------------------------
# Tier catalog
# ---------------------------------------------------------------------------
# Numbers reflect Qwen-2.5 architecture at Q4_K_M with 4K context.
#
# KV cache derivation (per token, fp16):
#   2 * num_layers * num_kv_heads * head_dim * 2 bytes
#     Qwen2.5-1.5B : 28 layers, 2 KV heads, 128 head_dim → ~29KB/tok → ~120MB @ 4K
#     Qwen2.5-3B   : 36 layers, 2 KV heads, 128 head_dim → ~37KB/tok → ~150MB @ 4K
#     Qwen2.5-7B   : 28 layers, 4 KV heads, 128 head_dim → ~56KB/tok → ~230MB @ 4K
#     Qwen2.5-14B  : 48 layers, 8 KV heads, 128 head_dim → ~192KB/tok → ~780MB @ 4K
#
# We round up and add overhead to be safe on llama.cpp bookkeeping.

@dataclass(frozen=True)
class Tier:
    name: str
    model_file: str
    param_billions: float
    model_size_gb: float           # Q4_K_M GGUF file size on disk
    kv_cache_gb_at_4k: float       # per model at 4K ctx
    min_total_ram_gb: float        # hard floor — machine cannot run this, ever
    min_available_ram_gb: float    # soft check at selection time
    min_gen_tok_s: float           # below this, tier is "too slow to be interactive"
    description: str
    # Source metadata for auto-download. Kept in the Tier itself (rather than a
    # sidecar table) so the catalog is the single source of truth — adding a
    # new tier means filling in one dataclass, not editing two places.
    hf_repo: str
    hf_filename: str

    @property
    def footprint_gb(self) -> float:
        """Model + KV cache + llama.cpp overhead (approx 0.5GB)."""
        return self.model_size_gb + self.kv_cache_gb_at_4k + 0.5


TIERS: tuple[Tier, ...] = (
    Tier(
        name="tiny",
        model_file="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        param_billions=1.5,
        model_size_gb=1.1,
        kv_cache_gb_at_4k=0.15,
        min_total_ram_gb=3.0,
        min_available_ram_gb=2.0,
        min_gen_tok_s=5.0,
        description="For old / low-RAM machines. Limited reasoning, fast.",
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        hf_filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ),
    Tier(
        name="standard",
        model_file="goalspec_qwen25_3b_q4km.gguf",
        param_billions=3.0,
        model_size_gb=2.0,
        kv_cache_gb_at_4k=0.2,
        min_total_ram_gb=6.0,
        min_available_ram_gb=3.5,
        min_gen_tok_s=6.0,
        description="Default for most laptops. Good balance of speed and quality.",
        hf_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        hf_filename="qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
    Tier(
        name="pro",
        model_file="qwen2.5-7b-instruct-q4_k_m.gguf",
        param_billions=7.0,
        model_size_gb=4.4,
        kv_cache_gb_at_4k=0.3,
        min_total_ram_gb=10.0,
        min_available_ram_gb=5.5,
        min_gen_tok_s=8.0,
        description="For modern machines with headroom. Noticeably better reasoning.",
        hf_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        hf_filename="qwen2.5-7b-instruct-q4_k_m.gguf",
    ),
    Tier(
        name="max",
        model_file="qwen2.5-14b-instruct-q4_k_m.gguf",
        param_billions=14.0,
        model_size_gb=8.2,
        kv_cache_gb_at_4k=0.8,
        min_total_ram_gb=20.0,
        min_available_ram_gb=10.0,
        min_gen_tok_s=10.0,
        description="Workstation-class. Requires a real GPU or 32GB+ of fast RAM.",
        hf_repo="Qwen/Qwen2.5-14B-Instruct-GGUF",
        hf_filename="qwen2.5-14b-instruct-q4_k_m.gguf",
    ),
)

TIER_BY_NAME: dict[str, Tier] = {t.name: t for t in TIERS}


# ---------------------------------------------------------------------------
# Static probe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareProfile:
    total_ram_gb: float
    available_ram_gb: float
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    cpu_has_avx2: bool
    cpu_has_avx512: bool
    platform: str        # "linux" | "darwin" | "windows"
    arch: str            # "x86_64" | "aarch64" | "arm64"
    is_laptop: bool
    gpu_detected: bool = False  # reserved — not populated in v1


def _read_cpu_model() -> str:
    """Best-effort CPU model string across platforms."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=2,
            )
            return out.strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _read_cpu_flags() -> set[str]:
    """Return the set of CPU flags on x86 Linux; empty on non-x86/non-Linux."""
    if platform.system() != "Linux":
        return set()
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        return set()
    return set()


def _detect_laptop() -> bool:
    """
    True if battery detected. Laptops throttle sustained load, which matters
    for tier selection — burst tok/s overstates what the user actually sees.
    psutil.sensors_battery() returns None on desktops.
    """
    try:
        return psutil.sensors_battery() is not None
    except Exception:
        return False


def probe() -> HardwareProfile:
    vm = psutil.virtual_memory()
    flags = _read_cpu_flags()
    return HardwareProfile(
        total_ram_gb=round(vm.total / 1024**3, 2),
        available_ram_gb=round(vm.available / 1024**3, 2),
        cpu_model=_read_cpu_model(),
        cpu_physical_cores=psutil.cpu_count(logical=False) or 0,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 0,
        cpu_has_avx2="avx2" in flags,
        cpu_has_avx512="avx512f" in flags,
        platform=platform.system().lower(),
        arch=platform.machine().lower(),
        is_laptop=_detect_laptop(),
    )


# ---------------------------------------------------------------------------
# Calibration benchmark (optional)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    """
    Measured inference performance against a specific model via llama-server.

    `reference_param_billions` anchors the bandwidth-bound extrapolation to
    other tiers: tok/s(X) ≈ gen_tok_s_median * (reference / X_params).
    """
    reference_model: str
    reference_param_billions: float
    gen_tok_s_median: float
    gen_tok_s_min: float
    gen_tok_s_p95: float
    prompt_eval_tok_s_median: float
    samples: int
    thermal_dropoff_pct: float  # (median - min) / median * 100; laptop throttle signal

    def extrapolate_gen_tok_s(self, target: Tier) -> float:
        """
        Bandwidth-bound approximation: tok/s is inversely proportional to
        bytes-moved-per-token, which is dominated by model size at fixed quant.
        We scale by parameter count (proxy for bytes-per-token) rather than
        file size to keep the math clean across quant variations.
        """
        if target.param_billions <= 0:
            return 0.0
        return self.gen_tok_s_median * (self.reference_param_billions / target.param_billions)


def _guess_reference_params(model_name: str) -> float:
    """Heuristic: extract Nb/NB from the model filename (e.g., '3b', '7B')."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    return float(m.group(1)) if m else 3.0  # sensible default


def calibrate(samples: int = 5, tokens_per_sample: int = 40) -> Optional[BenchmarkResult]:
    """
    Run a short generation benchmark against whatever model the llama-server
    is currently serving. Returns None if the server is unreachable — the
    caller should fall back to static-only selection.

    Keeps prompts identical across samples so KV cache reuse is consistent;
    this measures generation tok/s cleanly without prompt-processing noise.
    """
    try:
        from inference.client import InferenceClient, InferenceRequest
    except ImportError:
        return None

    client = InferenceClient()
    if not client.is_available():
        return None

    # Minimal ChatML prompt — small, predictable, unlikely to trigger schema
    # issues across model families.
    prompt = (
        "<|im_start|>system\nYou are a counter. Count the numbers.\n<|im_end|>\n"
        "<|im_start|>user\nList numbers 1 through 50, comma-separated.\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    gen_rates: list[float] = []
    prompt_rates: list[float] = []
    model_name = "unknown"

    for _ in range(samples):
        req = InferenceRequest(
            prompt=prompt,
            temperature=0.1,
            max_tokens=tokens_per_sample,
            stop_tokens=["<|im_end|>"],
        )
        try:
            resp = client.complete(req)
        except Exception:
            return None  # server flaked; bail out cleanly

        if resp.model != "unknown":
            model_name = resp.model
        seconds = resp.latency_ms / 1000.0
        if resp.tokens_predicted > 0 and seconds > 0:
            gen_rates.append(resp.tokens_predicted / seconds)
        # Prompt eval rate is noisy after cache warm; only trust the first sample
        if resp.prompt_tokens > 0 and seconds > 0 and not prompt_rates:
            prompt_rates.append(resp.prompt_tokens / seconds)

    if not gen_rates:
        return None

    gen_rates.sort()
    median = statistics.median(gen_rates)
    minimum = gen_rates[0]
    p95_idx = max(0, int(len(gen_rates) * 0.95) - 1)
    p95 = gen_rates[p95_idx] if len(gen_rates) > 1 else median
    dropoff = ((median - minimum) / median * 100.0) if median > 0 else 0.0

    return BenchmarkResult(
        reference_model=model_name,
        reference_param_billions=_guess_reference_params(model_name),
        gen_tok_s_median=round(median, 2),
        gen_tok_s_min=round(minimum, 2),
        gen_tok_s_p95=round(p95, 2),
        prompt_eval_tok_s_median=round(statistics.median(prompt_rates), 2) if prompt_rates else 0.0,
        samples=len(gen_rates),
        thermal_dropoff_pct=round(dropoff, 1),
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierDecision:
    chosen: str                          # Tier.name
    reason: str                          # one-line human-readable summary
    profile: HardwareProfile
    bench: Optional[BenchmarkResult]
    estimated_gen_tok_s: dict[str, float]  # per tier name
    disqualified: dict[str, str]           # tier name -> reason rejected
    timestamp: str

    def as_dict(self) -> dict:
        d = asdict(self)
        # Dataclasses-in-dataclasses serialize fine, but make sure it's JSON-safe
        return d


def select_tier(
    profile: HardwareProfile,
    bench: Optional[BenchmarkResult] = None,
) -> TierDecision:
    """
    Walk tiers from largest to smallest, pick the first that qualifies.

    Hard gates (always applied):
      - total_ram_gb >= tier.min_total_ram_gb
      - available_ram_gb >= tier.min_available_ram_gb
      - footprint fits with 1.5GB margin below available_ram

    Soft gate (applied only when bench is present):
      - estimated_gen_tok_s >= tier.min_gen_tok_s
      - on laptops, apply a 15% haircut to extrapolated rates (sustained
        performance is below burst; the bench runs briefly and may not hit
        thermal steady state)
    """
    estimated: dict[str, float] = {}
    disqualified: dict[str, str] = {}
    haircut = 0.85 if profile.is_laptop else 1.0

    # Iterate largest → smallest so the first qualifier wins the highest tier
    for tier in reversed(TIERS):
        reasons: list[str] = []

        if profile.total_ram_gb < tier.min_total_ram_gb:
            reasons.append(
                f"need {tier.min_total_ram_gb:.1f}GB total RAM, have {profile.total_ram_gb:.1f}GB"
            )
        if profile.available_ram_gb < tier.min_available_ram_gb:
            reasons.append(
                f"need {tier.min_available_ram_gb:.1f}GB available RAM now, "
                f"have {profile.available_ram_gb:.1f}GB free"
            )
        if profile.available_ram_gb < tier.footprint_gb + 1.5:
            reasons.append(
                f"model footprint {tier.footprint_gb:.1f}GB + 1.5GB margin exceeds "
                f"available {profile.available_ram_gb:.1f}GB"
            )

        est_rate = 0.0
        if bench is not None:
            est_rate = bench.extrapolate_gen_tok_s(tier) * haircut
            estimated[tier.name] = round(est_rate, 2)
            if est_rate < tier.min_gen_tok_s:
                reasons.append(
                    f"estimated {est_rate:.1f} tok/s < {tier.min_gen_tok_s:.1f} tok/s interactive floor"
                )
        else:
            estimated[tier.name] = 0.0

        if reasons:
            disqualified[tier.name] = "; ".join(reasons)
            continue

        # First tier from the top that qualifies.
        reason = _build_selection_reason(tier, profile, bench, est_rate)
        return TierDecision(
            chosen=tier.name,
            reason=reason,
            profile=profile,
            bench=bench,
            estimated_gen_tok_s=estimated,
            disqualified=disqualified,
            timestamp=_utc_timestamp(),
        )

    # Nothing qualified — fall back to tiny and let the user know why.
    fallback = TIERS[0]
    return TierDecision(
        chosen=fallback.name,
        reason=(
            f"No tier fully qualifies on this machine; falling back to '{fallback.name}'. "
            f"Override with: python hardware.py --tier <name>"
        ),
        profile=profile,
        bench=bench,
        estimated_gen_tok_s=estimated,
        disqualified=disqualified,
        timestamp=_utc_timestamp(),
    )


def _build_selection_reason(
    tier: Tier,
    profile: HardwareProfile,
    bench: Optional[BenchmarkResult],
    est_rate: float,
) -> str:
    parts = [
        f"'{tier.name}' ({tier.param_billions:.1f}B) fits "
        f"{profile.total_ram_gb:.0f}GB RAM"
    ]
    if bench is not None:
        parts.append(
            f"extrapolated {est_rate:.1f} tok/s from {bench.reference_param_billions:.1f}B "
            f"bench at {bench.gen_tok_s_median:.1f} tok/s"
        )
        if bench.thermal_dropoff_pct > 20.0:
            parts.append(f"thermal dropoff {bench.thermal_dropoff_pct:.0f}% (laptop throttle)")
    if profile.is_laptop:
        parts.append("laptop haircut -15% applied")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".marshal" / "tier.json"
DEFAULT_MODELS_DIR = Path(__file__).parent / "models"


def write_decision(decision: TierDecision, path: Path = CONFIG_PATH) -> Path:
    """Persist the decision so main.py / agentd can read it without re-probing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(decision.as_dict(), indent=2, default=str))
    tmp.replace(path)
    return path


def read_decision(path: Path = CONFIG_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def active_tier_name(path: Path = CONFIG_PATH) -> Optional[str]:
    """Short accessor: return the saved chosen tier name, or None if unset."""
    d = read_decision(path)
    return d["chosen"] if d and "chosen" in d else None


def resolve_model_path(
    models_dir: Path = DEFAULT_MODELS_DIR,
    preferred_tier: Optional[str] = None,
    *,
    config_path: Path = CONFIG_PATH,
) -> Optional[Path]:
    """
    Map a tier preference to a concrete GGUF path, with graceful walk-down.

    Resolution order:
      1. If `preferred_tier` is given, start there; else read the saved
         decision; else start at 'standard' as a neutral default.
      2. Check `models_dir/<tier.model_file>`. If it exists, return it.
      3. Walk *down* the tier ladder (pro → standard → tiny), returning the
         first model file that actually exists on disk.
      4. Return None if nothing matches — caller decides how to handle it.

    The walk-down means the saved tier is a *preference*, not a hard
    requirement: if the user deletes the 7B file, they drop to 3B rather
    than getting a missing-file error.
    """
    if not models_dir.exists():
        return None

    start_name = preferred_tier or active_tier_name(config_path) or "standard"
    if start_name not in TIER_BY_NAME:
        start_name = "standard"

    # Build the search order: start tier, then every smaller tier by size.
    ordered = sorted(TIERS, key=lambda t: t.param_billions, reverse=True)
    start_idx = next(
        (i for i, t in enumerate(ordered) if t.name == start_name),
        len(ordered) - 1,
    )
    walk = ordered[start_idx:]  # start tier, then smaller ones in order

    for tier in walk:
        candidate = models_dir / tier.model_file
        if candidate.exists():
            return candidate

    # Last resort: the historical fine-tuned Phase 2 artifact, still widely
    # used in this repo (MEMORY notes it as the current deployed model).
    legacy = models_dir / "goalspec_qwen25_3b_q4km.gguf"
    if legacy.exists():
        return legacy

    return None


def _utc_timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

class ModelDownloadError(RuntimeError):
    """Raised when a model download fails for any reason."""


def _huggingface_cli_available() -> bool:
    import shutil
    return shutil.which("huggingface-cli") is not None


def download_model_for_tier(
    tier: Tier,
    models_dir: Path = DEFAULT_MODELS_DIR,
    *,
    force: bool = False,
    on_progress: Optional[callable] = None,  # type: ignore[type-arg]
) -> Path:
    """
    Download the GGUF for a tier from HuggingFace into `models_dir`.

    Uses the same `huggingface-cli download` path that scripts/download-model.sh
    uses — single source of truth for where models come from. Returns the
    absolute path of the downloaded file.

    Raises ModelDownloadError if:
      - huggingface-cli is not installed
      - the subprocess returns non-zero
      - the expected file isn't present after the download claims success

    `on_progress(str)` is called with one-line status updates so the REPL can
    surface progress without this module depending on rich / the console.
    """
    import subprocess

    def _notify(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    target = models_dir / tier.model_file
    if target.exists() and not force:
        _notify(f"{tier.model_file} already present ({target.stat().st_size / 1024**3:.2f}GB)")
        return target

    if not _huggingface_cli_available():
        raise ModelDownloadError(
            "huggingface-cli not found. Install it with: pip install --upgrade huggingface_hub"
        )

    models_dir.mkdir(parents=True, exist_ok=True)

    _notify(f"downloading {tier.hf_filename} from {tier.hf_repo} (~{tier.model_size_gb:.1f}GB)...")
    try:
        proc = subprocess.run(
            [
                "huggingface-cli", "download",
                tier.hf_repo,
                tier.hf_filename,
                "--local-dir", str(models_dir),
                "--local-dir-use-symlinks", "False",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise ModelDownloadError(f"huggingface-cli invocation failed: {e}") from e

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise ModelDownloadError(
            f"huggingface-cli exited {proc.returncode}: {' / '.join(stderr_tail)}"
        )

    # huggingface-cli occasionally drops files into repo-flavored subdirs.
    # Normalize to the canonical location in models_dir.
    if not target.exists():
        for found in models_dir.rglob(tier.hf_filename):
            if found != target:
                found.replace(target)
                break

    if not target.exists():
        raise ModelDownloadError(
            f"download completed but {target.name} is not in {models_dir}"
        )

    _notify(f"saved {target.name} ({target.stat().st_size / 1024**3:.2f}GB)")
    return target


def ensure_tier_model(
    tier: Tier,
    models_dir: Path = DEFAULT_MODELS_DIR,
    *,
    consent_fn: Optional[callable] = None,  # type: ignore[type-arg]
    on_progress: Optional[callable] = None,  # type: ignore[type-arg]
) -> Optional[Path]:
    """
    If the tier's GGUF is on disk, return its path. Otherwise ask the caller
    via `consent_fn(tier) -> bool` whether to download it. Returns None if
    consent was withheld or the caller provided no consent_fn (auto-download
    never happens silently — models are 1-8GB, user should always be asked).
    """
    target = models_dir / tier.model_file
    if target.exists():
        return target
    if consent_fn is None:
        return None
    if not consent_fn(tier):
        return None
    return download_model_for_tier(tier, models_dir, on_progress=on_progress)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_decision(decision: TierDecision) -> None:
    tier = TIER_BY_NAME[decision.chosen]
    print(f"\n=== Marshal tier selection ===\n")
    print(f"Chosen tier : {decision.chosen}  ({tier.param_billions:.1f}B params)")
    print(f"Model file  : {tier.model_file}")
    print(f"Reason      : {decision.reason}")
    print()
    p = decision.profile
    print(f"Hardware    : {p.cpu_model}")
    print(f"              {p.cpu_physical_cores}c/{p.cpu_logical_cores}t, "
          f"{'AVX-512' if p.cpu_has_avx512 else 'AVX2' if p.cpu_has_avx2 else 'baseline'}, "
          f"{p.platform}/{p.arch}, {'laptop' if p.is_laptop else 'desktop'}")
    print(f"RAM         : {p.available_ram_gb:.1f}GB free / {p.total_ram_gb:.1f}GB total")
    if decision.bench is not None:
        b = decision.bench
        print(f"Benchmark   : {b.gen_tok_s_median:.1f} tok/s median, "
              f"{b.gen_tok_s_min:.1f} min ({b.samples} samples) on {b.reference_model}")
        if b.thermal_dropoff_pct > 20.0:
            print(f"              thermal dropoff {b.thermal_dropoff_pct:.0f}% — "
                  f"sustained performance will be lower than burst")
    else:
        print("Benchmark   : not run (inference server unreachable)")
    if decision.estimated_gen_tok_s and any(v > 0 for v in decision.estimated_gen_tok_s.values()):
        print("\nPer-tier estimate (gen tok/s):")
        for name in ("tiny", "standard", "pro", "max"):
            rate = decision.estimated_gen_tok_s.get(name, 0.0)
            if rate > 0:
                marker = "←" if name == decision.chosen else " "
                print(f"  {marker} {name:<9} ≈ {rate:5.1f} tok/s")
    if decision.disqualified:
        print("\nDisqualified tiers:")
        for name, reason in decision.disqualified.items():
            print(f"  - {name:<9} : {reason}")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="marshal-hardware",
        description="Detect hardware and pick the right local model tier.",
    )
    p.add_argument("--no-bench", action="store_true",
                   help="Skip the calibration benchmark (static probe only)")
    p.add_argument("--tier", choices=[t.name for t in TIERS],
                   help="Force a specific tier instead of auto-detecting")
    p.add_argument("--show", action="store_true",
                   help="Print the last saved decision and exit")
    p.add_argument("--json", action="store_true",
                   help="Emit the decision as JSON to stdout")
    p.add_argument("--resolve-model", action="store_true",
                   help="Print the absolute path of the model GGUF to load, "
                        "based on the saved tier (walks down if missing). "
                        "Exits non-zero if no model is available.")
    p.add_argument("--print-tier", action="store_true",
                   help="Print just the saved tier name (or 'standard' if unset) and exit")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                   help="Directory to search for GGUF files (for --resolve-model)")
    args = p.parse_args(argv)

    if args.resolve_model:
        resolved = resolve_model_path(args.models_dir)
        if resolved is None:
            print("no model available", file=sys.stderr)
            return 1
        print(resolved)
        return 0

    if args.print_tier:
        print(active_tier_name() or "standard")
        return 0

    if args.show:
        saved = read_decision(CONFIG_PATH)
        if saved is None:
            print(f"No saved decision at {CONFIG_PATH}. Run without --show to detect.", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(saved, indent=2))
        else:
            print(f"Saved decision ({saved.get('timestamp', '?')}): tier='{saved['chosen']}'")
            print(f"  reason: {saved['reason']}")
        return 0

    profile = probe()
    bench = None if args.no_bench else calibrate()

    if args.tier is not None:
        forced = TIER_BY_NAME[args.tier]
        decision = TierDecision(
            chosen=forced.name,
            reason=f"forced by --tier {args.tier} (auto-detect skipped)",
            profile=profile,
            bench=bench,
            estimated_gen_tok_s={},
            disqualified={},
            timestamp=_utc_timestamp(),
        )
    else:
        decision = select_tier(profile, bench)

    path = write_decision(decision, CONFIG_PATH)
    if args.json:
        print(json.dumps(decision.as_dict(), indent=2, default=str))
    else:
        _print_decision(decision)
        print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
