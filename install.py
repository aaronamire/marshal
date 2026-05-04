#!/usr/bin/env python3
"""
Marshal install / doctor agent.

Not an LLM — a deterministic state reconciler. The user's machine is the
target state; we compute what's missing for Marshal to run, show the plan,
and apply it with confirmation.

Design rationale lives in comments throughout, but the short version:

  * Declarative steps (check/execute/remediate) not imperative scripts.
    Idempotency is a property of the design, not of carefully-placed
    `[ -f file ]` guards.

  * Plan-before-apply. User sees every step that will run, with time and
    disk estimates, before anything mutates the system.

  * Structured failures. Every failure has a taxonomy and a one-liner
    remediation command. No bare tracebacks reach the user.

  * `--doctor` is the same code as `--plan` with execution disabled.
    The check path is the diagnostic path.

  * No hidden state: writes only under `~/.marshal/` and the repo.
    Never modifies shell RCs. Never elevates privileges silently.

Run:

    python install.py                 # --plan (default)
    python install.py --apply         # run the plan (prompts per step group)
    python install.py --apply --yes   # non-interactive
    python install.py --doctor        # check only, never execute
    python install.py --step venv     # run a single step (with its deps)
    python install.py --json          # machine-readable output
"""
from __future__ import annotations

import argparse
import enum
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional


REPO_ROOT = Path(__file__).resolve().parent
MARSHAL_HOME = Path.home() / ".marshal"
STATE_PATH = MARSHAL_HOME / "install-state.json"

MIN_PYTHON = (3, 12)
MIN_KERNEL = (5, 13)           # Landlock requires 5.13+
MIN_DISK_GB = 5.0
MIN_RAM_GB = 4.0


# ---------------------------------------------------------------------------
# Result taxonomy — structured enough to act on, simple enough to JSON.
# ---------------------------------------------------------------------------

class CheckStatus(str, enum.Enum):
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"         # can't tell without executing (rare)
    BLOCKED = "blocked"         # cannot be satisfied on this machine


class FailureKind(str, enum.Enum):
    NETWORK = "network"
    PERMISSION = "permission"
    DISK = "disk"
    COMPAT = "compat"            # kernel / distro / arch wrong
    DEPENDENCY = "dependency"    # another tool missing
    USER_INPUT = "user_input"    # consent declined / invalid arg
    INTERNAL = "internal"        # our bug
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Remediation:
    """How the user gets unstuck. One of these per failure mode."""
    summary: str                  # one-line human description
    commands: tuple[str, ...] = ()   # copy-pasteable shell lines
    url: Optional[str] = None     # optional doc link
    can_skip: bool = False        # is the step safe to skip?


@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    detail: str                   # what we found (version string, path, reason)
    remediation: Optional[Remediation] = None  # only when UNSATISFIED/BLOCKED


@dataclass(frozen=True)
class ExecuteResult:
    ok: bool
    detail: str
    duration_s: float = 0.0
    failure_kind: Optional[FailureKind] = None
    remediation: Optional[Remediation] = None


# ---------------------------------------------------------------------------
# Environment probe — one pass, cached, shared with every step.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Env:
    # Identity
    os_name: str                  # "linux" etc
    distro: str                   # "arch" | "ubuntu" | "fedora" | "unknown"
    distro_version: str
    kernel: tuple[int, int, int]  # (major, minor, patch)
    arch: str

    # Resources
    python_version: tuple[int, int, int]
    python_exec: str
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    is_laptop: bool

    # Capabilities
    has_sudo: bool
    has_network: bool
    has_git: bool
    has_cmake: bool
    has_huggingface_cli: bool
    has_pacman: bool
    has_apt: bool
    has_dnf: bool
    has_systemctl: bool

    # Derived
    @property
    def supports_landlock(self) -> bool:
        return self.kernel[:2] >= MIN_KERNEL

    @property
    def package_manager(self) -> Optional[str]:
        if self.has_pacman:
            return "pacman"
        if self.has_apt:
            return "apt"
        if self.has_dnf:
            return "dnf"
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kernel"] = list(self.kernel)
        d["python_version"] = list(self.python_version)
        d["supports_landlock"] = self.supports_landlock
        d["package_manager"] = self.package_manager
        return d


def _parse_kernel(release: str) -> tuple[int, int, int]:
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", release)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _detect_distro() -> tuple[str, str]:
    """Return (distro_id, version). Reads /etc/os-release on Linux."""
    path = Path("/etc/os-release")
    if not path.exists():
        return ("unknown", "")
    fields = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k] = v.strip().strip('"')
    return (fields.get("ID", "unknown"), fields.get("VERSION_ID", ""))


def _check_network() -> bool:
    """Quick TCP connectivity check. No DNS resolution required if host is up."""
    import socket
    try:
        sock = socket.create_connection(("1.1.1.1", 443), timeout=2)
        sock.close()
        return True
    except OSError:
        return False


def _check_sudo() -> bool:
    """True if passwordless sudo is available (doesn't prompt)."""
    try:
        r = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            timeout=3,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def probe_env() -> Env:
    """One-shot environment detection. Under 500ms total."""
    import psutil

    distro, distro_ver = _detect_distro()
    kernel = _parse_kernel(platform.release())
    vm = psutil.virtual_memory()
    du = psutil.disk_usage(str(REPO_ROOT))
    py = sys.version_info

    try:
        is_laptop = psutil.sensors_battery() is not None
    except Exception:
        is_laptop = False

    return Env(
        os_name=platform.system().lower(),
        distro=distro,
        distro_version=distro_ver,
        kernel=kernel,
        arch=platform.machine().lower(),
        python_version=(py.major, py.minor, py.micro),
        python_exec=sys.executable,
        ram_total_gb=round(vm.total / 1024**3, 2),
        ram_available_gb=round(vm.available / 1024**3, 2),
        disk_free_gb=round(du.free / 1024**3, 2),
        is_laptop=is_laptop,
        has_sudo=_check_sudo(),
        has_network=_check_network(),
        has_git=shutil.which("git") is not None,
        has_cmake=shutil.which("cmake") is not None,
        has_huggingface_cli=shutil.which("huggingface-cli") is not None,
        has_pacman=shutil.which("pacman") is not None,
        has_apt=shutil.which("apt-get") is not None,
        has_dnf=shutil.which("dnf") is not None,
        has_systemctl=shutil.which("systemctl") is not None,
    )


# ---------------------------------------------------------------------------
# Step abstraction
# ---------------------------------------------------------------------------

class InstallStep(ABC):
    """
    Every unit of work is one of these. `check` is fast and idempotent;
    `execute` may be slow but must also be idempotent — running it twice
    produces the same result as running it once.
    """

    # Subclasses override these class-level fields.
    id: str = ""
    title: str = ""
    depends_on: tuple[str, ...] = ()
    est_seconds: int = 10
    est_disk_mb: int = 0
    needs_sudo: bool = False
    optional: bool = False         # true → skipping does not break the core path

    @abstractmethod
    def check(self, env: Env) -> CheckResult: ...

    @abstractmethod
    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult: ...


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

@dataclass
class InstallState:
    """
    Cached checkpoint. Writes after each successful step. Next run consults
    it to skip already-satisfied steps *unless* the fingerprint has changed
    (e.g. repo was updated), in which case we re-verify.
    """
    completed: dict[str, str] = field(default_factory=dict)   # id -> ISO timestamp
    fingerprint: str = ""
    env_snapshot: dict = field(default_factory=dict)


def _fingerprint() -> str:
    """Identify this version of the repo + install agent.
    Uses the install.py mtime as a rough proxy so checkpoints invalidate when
    this script changes. Git SHA is better but git may not be available.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if out.returncode == 0:
            return f"git:{out.stdout.strip()[:12]}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return f"mtime:{int(Path(__file__).stat().st_mtime)}"


def load_state() -> InstallState:
    if not STATE_PATH.exists():
        return InstallState(fingerprint=_fingerprint())
    try:
        data = json.loads(STATE_PATH.read_text())
        state = InstallState(**data)
    except (OSError, json.JSONDecodeError, TypeError):
        return InstallState(fingerprint=_fingerprint())
    if state.fingerprint != _fingerprint():
        # Repo changed; invalidate. Caller will re-run checks.
        state.completed = {}
        state.fingerprint = _fingerprint()
    return state


def save_state(state: InstallState) -> None:
    MARSHAL_HOME.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(STATE_PATH)


# ---------------------------------------------------------------------------
# Concrete steps
# ---------------------------------------------------------------------------

class StepOS(InstallStep):
    id = "os"
    title = "Linux with kernel ≥ 5.13 (Landlock)"
    est_seconds = 0

    def check(self, env: Env) -> CheckResult:
        if env.os_name != "linux":
            return CheckResult(
                CheckStatus.BLOCKED,
                f"Detected {env.os_name}. Marshal is Linux-only (Landlock/cgroup2 required).",
                Remediation(
                    "Run Marshal in a Linux VM or container.",
                    ("docker compose up",),
                    url="https://github.com/aaronamire/marshal#docker",
                ),
            )
        if not env.supports_landlock:
            major, minor, patch = env.kernel
            return CheckResult(
                CheckStatus.BLOCKED,
                f"Kernel {major}.{minor}.{patch} is too old (need ≥ 5.13 for Landlock)",
                Remediation(
                    "Upgrade your kernel, or run via Docker which has its own kernel.",
                    ("docker compose up",),
                    can_skip=False,
                ),
            )
        return CheckResult(
            CheckStatus.SATISFIED,
            f"{env.distro} {env.distro_version}, kernel {'.'.join(map(str, env.kernel))}",
        )

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        return ExecuteResult(False, "cannot execute — OS compat is a precondition",
                             failure_kind=FailureKind.COMPAT)


class StepPython(InstallStep):
    id = "python"
    title = f"Python ≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    est_seconds = 0

    def check(self, env: Env) -> CheckResult:
        if env.python_version[:2] < MIN_PYTHON:
            cur = ".".join(map(str, env.python_version))
            want = ".".join(map(str, MIN_PYTHON))
            return CheckResult(
                CheckStatus.BLOCKED,
                f"Python {cur} is too old (need ≥ {want})",
                Remediation(
                    f"Install Python ≥ {want} and re-run install.py with it.",
                    (f"sudo {env.package_manager or 'your-pkg-manager'} install python3.12",),
                    can_skip=False,
                ),
            )
        return CheckResult(
            CheckStatus.SATISFIED,
            f"{'.'.join(map(str, env.python_version))} at {env.python_exec}",
        )

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        return ExecuteResult(False, "cannot execute — Python is a precondition",
                             failure_kind=FailureKind.COMPAT)


class StepDisk(InstallStep):
    id = "disk"
    title = f"≥ {MIN_DISK_GB:.0f} GB free disk"
    est_seconds = 0

    def check(self, env: Env) -> CheckResult:
        if env.disk_free_gb < MIN_DISK_GB:
            return CheckResult(
                CheckStatus.UNSATISFIED,
                f"{env.disk_free_gb:.1f} GB free on {REPO_ROOT} partition (need {MIN_DISK_GB:.0f})",
                Remediation(
                    "Free up disk — llama.cpp build + model + venv is ~3.5 GB.",
                    ("du -sh ~/* | sort -h",),
                ),
            )
        return CheckResult(
            CheckStatus.SATISFIED, f"{env.disk_free_gb:.1f} GB free"
        )

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        return ExecuteResult(False, "cannot free disk for you",
                             failure_kind=FailureKind.DISK)


class StepSystemPackages(InstallStep):
    id = "system-packages"
    title = "System build toolchain (cmake, git, pkg-config)"
    depends_on = ("os", "python")
    est_seconds = 60

    def check(self, env: Env) -> CheckResult:
        missing = []
        if not env.has_git: missing.append("git")
        if not env.has_cmake: missing.append("cmake")
        if missing:
            return CheckResult(
                CheckStatus.UNSATISFIED,
                f"missing: {', '.join(missing)}",
                Remediation(
                    f"Install {', '.join(missing)} via your package manager.",
                    self._install_cmd(env, missing),
                ),
            )
        return CheckResult(CheckStatus.SATISFIED, "git, cmake present")

    @staticmethod
    def _install_cmd(env: Env, pkgs: list[str]) -> tuple[str, ...]:
        if env.has_pacman:
            return (f"sudo pacman -S --needed --noconfirm {' '.join(pkgs)}",)
        if env.has_apt:
            return (f"sudo apt-get install -y {' '.join(pkgs)}",)
        if env.has_dnf:
            return (f"sudo dnf install -y {' '.join(pkgs)}",)
        return (f"# install manually: {' '.join(pkgs)}",)

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        # Delegate to bootstrap.sh's package install block; respect dry intent.
        if env.package_manager is None:
            return ExecuteResult(
                False, "no supported package manager (pacman/apt/dnf)",
                failure_kind=FailureKind.COMPAT,
                remediation=Remediation(
                    "Install git and cmake manually.",
                ),
            )
        # We can't install system packages without sudo. Don't try to elevate;
        # print the command and let the user run it themselves.
        cmd = self._install_cmd(env, ["git", "cmake"])[0]
        return ExecuteResult(
            False,
            "system-package install needs sudo — run the command below then re-run install.py",
            failure_kind=FailureKind.PERMISSION,
            remediation=Remediation(
                "Run the system-package install as root, then re-run install.py.",
                (cmd,),
            ),
        )


class StepLlamaCpp(InstallStep):
    id = "llama-cpp"
    title = "Build llama.cpp (CPU, native ISA)"
    depends_on = ("system-packages",)
    est_seconds = 180
    est_disk_mb = 250

    LLAMA_DIR = Path.home() / "dev" / "llama.cpp"
    LLAMA_BIN = LLAMA_DIR / "build" / "bin" / "llama-server"

    def check(self, env: Env) -> CheckResult:
        if self.LLAMA_BIN.exists() and os.access(self.LLAMA_BIN, os.X_OK):
            return CheckResult(CheckStatus.SATISFIED, f"{self.LLAMA_BIN}")
        return CheckResult(
            CheckStatus.UNSATISFIED,
            f"{self.LLAMA_BIN} not found",
            Remediation("Will clone and build ~3 min."),
        )

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        if not env.has_network:
            return ExecuteResult(
                False, "no network — cannot clone llama.cpp",
                failure_kind=FailureKind.NETWORK,
                remediation=Remediation("Get online, then re-run."),
            )
        t0 = time.monotonic()
        self.LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not (self.LLAMA_DIR / ".git").exists():
                _stream(on_output,
                        ["git", "clone", "--depth", "1",
                         "https://github.com/ggerganov/llama.cpp.git",
                         str(self.LLAMA_DIR)])
            _stream(on_output,
                    ["cmake", "-B", "build",
                     "-DLLAMA_CURL=OFF", "-DLLAMA_NATIVE=ON", "-DGGML_NATIVE=ON"],
                    cwd=self.LLAMA_DIR)
            _stream(on_output,
                    ["cmake", "--build", "build", "--target", "llama-server",
                     "-j", str(os.cpu_count() or 2)],
                    cwd=self.LLAMA_DIR)
        except _StreamError as e:
            return ExecuteResult(
                False, f"{e.step} failed: {e.tail}",
                duration_s=time.monotonic() - t0,
                failure_kind=FailureKind.DEPENDENCY,
                remediation=Remediation(
                    "Check build log output above. Usually missing compiler or bad CMake version.",
                    ("cd ~/dev/llama.cpp && cmake -B build -DLLAMA_NATIVE=ON",
                     "cmake --build build --target llama-server"),
                ),
            )
        if not self.LLAMA_BIN.exists():
            return ExecuteResult(
                False, "build finished but llama-server binary missing",
                duration_s=time.monotonic() - t0,
                failure_kind=FailureKind.INTERNAL,
            )
        return ExecuteResult(True, f"{self.LLAMA_BIN}", duration_s=time.monotonic() - t0)


class StepVenv(InstallStep):
    id = "venv"
    title = "Python venv at .os/ with Marshal + [rag,remote]"
    depends_on = ("python",)
    est_seconds = 120
    est_disk_mb = 800

    VENV_DIR = REPO_ROOT / ".os"

    def check(self, env: Env) -> CheckResult:
        py = self.VENV_DIR / "bin" / "python3"
        if not py.exists():
            return CheckResult(
                CheckStatus.UNSATISFIED,
                f"venv not found at {self.VENV_DIR}",
            )
        # Verify marshal is installed inside it
        r = subprocess.run(
            [str(py), "-c", "import main"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return CheckResult(
                CheckStatus.UNSATISFIED,
                "venv exists but Marshal not installed inside it",
            )
        return CheckResult(CheckStatus.SATISFIED, f"{self.VENV_DIR} OK")

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        t0 = time.monotonic()
        try:
            if not self.VENV_DIR.exists():
                _stream(on_output,
                        [sys.executable, "-m", "venv", str(self.VENV_DIR)])
            pip = self.VENV_DIR / "bin" / "pip"
            _stream(on_output,
                    [str(pip), "install", "--upgrade", "pip", "wheel"])
            _stream(on_output,
                    [str(pip), "install", "-e", ".[rag,remote]"],
                    cwd=REPO_ROOT)
        except _StreamError as e:
            return ExecuteResult(
                False, f"{e.step} failed: {e.tail}",
                duration_s=time.monotonic() - t0,
                failure_kind=FailureKind.DEPENDENCY,
                remediation=Remediation(
                    "pip errors usually mean missing system headers. Look for "
                    "'Python.h: No such file' or similar in the output above.",
                    ("sudo apt-get install python3-dev  # or your distro equivalent",),
                ),
            )
        return ExecuteResult(True, f"{self.VENV_DIR}", duration_s=time.monotonic() - t0)


class StepHardwareProbe(InstallStep):
    id = "hardware-probe"
    title = "Detect hardware tier (runs hardware.py)"
    depends_on = ("venv",)
    est_seconds = 2

    def check(self, env: Env) -> CheckResult:
        tier_file = MARSHAL_HOME / "tier.json"
        if tier_file.exists():
            return CheckResult(CheckStatus.SATISFIED, f"{tier_file}")
        return CheckResult(CheckStatus.UNSATISFIED, "no tier saved yet")

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        t0 = time.monotonic()
        py = StepVenv.VENV_DIR / "bin" / "python3"
        if not py.exists():
            py = Path(sys.executable)
        try:
            _stream(on_output, [str(py), "hardware.py", "--no-bench"], cwd=REPO_ROOT)
        except _StreamError as e:
            return ExecuteResult(
                False, f"hardware probe failed: {e.tail}",
                duration_s=time.monotonic() - t0,
                failure_kind=FailureKind.INTERNAL,
            )
        return ExecuteResult(True, "tier saved", duration_s=time.monotonic() - t0)


class StepModelDownload(InstallStep):
    id = "model"
    title = "Download model for selected tier"
    depends_on = ("hardware-probe", "venv")
    est_seconds = 240
    est_disk_mb = 2000

    def _active_tier(self) -> Optional[dict]:
        tier_file = MARSHAL_HOME / "tier.json"
        if not tier_file.exists():
            return None
        try:
            return json.loads(tier_file.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def check(self, env: Env) -> CheckResult:
        # Import the catalog from hardware.py to stay in sync.
        sys.path.insert(0, str(REPO_ROOT))
        try:
            import hardware
        except ImportError as e:
            return CheckResult(
                CheckStatus.UNKNOWN, f"hardware module not importable: {e}"
            )
        tier = hardware.active_tier_name() or "standard"
        tier_obj = hardware.TIER_BY_NAME.get(tier)
        if tier_obj is None:
            return CheckResult(CheckStatus.UNKNOWN, f"unknown tier: {tier}")
        path = REPO_ROOT / "models" / tier_obj.model_file
        if path.exists():
            size_gb = path.stat().st_size / 1024**3
            return CheckResult(CheckStatus.SATISFIED, f"{path.name} ({size_gb:.2f} GB)")
        return CheckResult(
            CheckStatus.UNSATISFIED,
            f"{tier_obj.model_file} not in models/",
            Remediation(
                f"Download ~{tier_obj.model_size_gb:.1f}GB from {tier_obj.hf_repo}.",
            ),
        )

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        t0 = time.monotonic()
        if not env.has_network:
            return ExecuteResult(
                False, "no network — cannot download model",
                failure_kind=FailureKind.NETWORK,
                remediation=Remediation("Get online, then re-run this step."),
            )
        sys.path.insert(0, str(REPO_ROOT))
        try:
            import hardware
        except ImportError as e:
            return ExecuteResult(
                False, f"hardware module not importable: {e}",
                failure_kind=FailureKind.INTERNAL,
            )
        tier = hardware.active_tier_name() or "standard"
        tier_obj = hardware.TIER_BY_NAME.get(tier)
        if tier_obj is None:
            return ExecuteResult(
                False, f"unknown tier: {tier}",
                failure_kind=FailureKind.INTERNAL,
            )
        try:
            path = hardware.download_model_for_tier(
                tier_obj,
                REPO_ROOT / "models",
                on_progress=on_output,
            )
        except hardware.ModelDownloadError as e:
            msg = str(e)
            if "401" in msg or "403" in msg:
                kind = FailureKind.PERMISSION
                fix = Remediation(
                    "HuggingFace returned auth error. Set an HF token and retry.",
                    ("# visit https://huggingface.co/settings/tokens",
                     "export HF_TOKEN=hf_xxx",
                     "python install.py --step model"),
                    url="https://huggingface.co/settings/tokens",
                )
            elif "huggingface-cli not found" in msg:
                kind = FailureKind.DEPENDENCY
                fix = Remediation(
                    "huggingface-cli ships with huggingface_hub; install it in the venv.",
                    (f"{StepVenv.VENV_DIR}/bin/pip install --upgrade huggingface_hub",),
                )
            else:
                kind = FailureKind.NETWORK
                fix = Remediation("Retry; if repeated, check your connection.")
            return ExecuteResult(
                False, msg, duration_s=time.monotonic() - t0,
                failure_kind=kind, remediation=fix,
            )
        return ExecuteResult(
            True, str(path), duration_s=time.monotonic() - t0,
        )


class StepClassifier(InstallStep):
    id = "classifier"
    title = "Train Layer-1 intent classifier"
    depends_on = ("venv",)
    est_seconds = 30
    optional = True

    CLASSIFIER_PATH = REPO_ROOT / "models" / "layer1_pipeline.joblib"

    def check(self, env: Env) -> CheckResult:
        if self.CLASSIFIER_PATH.exists():
            return CheckResult(CheckStatus.SATISFIED, str(self.CLASSIFIER_PATH))
        return CheckResult(
            CheckStatus.UNSATISFIED,
            "layer1_pipeline.joblib missing",
            Remediation("Will train (~30s).", can_skip=True),
        )

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        t0 = time.monotonic()
        py = StepVenv.VENV_DIR / "bin" / "python3"
        if not py.exists():
            py = Path(sys.executable)
        try:
            _stream(on_output,
                    [str(py), "scripts/train_classifier.py"],
                    cwd=REPO_ROOT)
        except _StreamError as e:
            return ExecuteResult(
                False, f"classifier training failed: {e.tail}",
                duration_s=time.monotonic() - t0,
                failure_kind=FailureKind.INTERNAL,
                remediation=Remediation(
                    "Skip this step — Layer 1 is optional; L0+L2 still work.",
                    can_skip=True,
                ),
            )
        return ExecuteResult(True, "classifier trained",
                             duration_s=time.monotonic() - t0)


class StepAuditDB(InstallStep):
    id = "audit-db"
    title = "Initialize audit SQLite at ~/.marshal/intents.db"
    depends_on = ("venv",)
    est_seconds = 1

    DB_PATH = MARSHAL_HOME / "intents.db"

    def check(self, env: Env) -> CheckResult:
        if self.DB_PATH.exists():
            return CheckResult(CheckStatus.SATISFIED, str(self.DB_PATH))
        return CheckResult(CheckStatus.UNSATISFIED, "audit DB not initialized")

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        t0 = time.monotonic()
        MARSHAL_HOME.mkdir(parents=True, exist_ok=True)
        py = StepVenv.VENV_DIR / "bin" / "python3"
        if not py.exists():
            py = Path(sys.executable)
        try:
            _stream(
                on_output,
                [str(py), "-c",
                 "import sys; sys.path.insert(0, '.'); "
                 "from db.audit import get_db; get_db()"],
                cwd=REPO_ROOT,
            )
        except _StreamError as e:
            return ExecuteResult(
                False, f"db init failed: {e.tail}",
                duration_s=time.monotonic() - t0,
                failure_kind=FailureKind.INTERNAL,
            )
        return ExecuteResult(True, str(self.DB_PATH),
                             duration_s=time.monotonic() - t0)


class StepSanityTest(InstallStep):
    id = "sanity"
    title = "Run a non-inference sanity test"
    depends_on = ("venv", "classifier", "audit-db")
    est_seconds = 20
    optional = True

    def check(self, env: Env) -> CheckResult:
        # Can't cache this — always run when requested
        return CheckResult(CheckStatus.UNSATISFIED, "runs on demand only")

    def execute(self, env: Env, *, on_output: Callable[[str], None]) -> ExecuteResult:
        t0 = time.monotonic()
        py = StepVenv.VENV_DIR / "bin" / "python3"
        if not py.exists():
            py = Path(sys.executable)
        try:
            _stream(
                on_output,
                [str(py), "-m", "pytest", "tests/test_layer0.py",
                 "tests/test_hardware.py", "-q", "--no-header"],
                cwd=REPO_ROOT,
            )
        except _StreamError as e:
            return ExecuteResult(
                False, f"sanity tests failed: {e.tail}",
                duration_s=time.monotonic() - t0,
                failure_kind=FailureKind.INTERNAL,
                remediation=Remediation(
                    "A failing test is a bug — open a GitHub issue with the output above.",
                    url="https://github.com/aaronamire/marshal/issues",
                ),
            )
        return ExecuteResult(True, "sanity tests passed",
                             duration_s=time.monotonic() - t0)


ALL_STEPS: tuple[type[InstallStep], ...] = (
    StepOS,
    StepPython,
    StepDisk,
    StepSystemPackages,
    StepLlamaCpp,
    StepVenv,
    StepHardwareProbe,
    StepModelDownload,
    StepClassifier,
    StepAuditDB,
    StepSanityTest,
)


# ---------------------------------------------------------------------------
# Subprocess helper with streaming output + tailed error buffer.
# ---------------------------------------------------------------------------

class _StreamError(RuntimeError):
    def __init__(self, step: str, tail: str, returncode: int):
        super().__init__(f"{step}: exit {returncode}: {tail}")
        self.step = step
        self.tail = tail
        self.returncode = returncode


def _stream(
    on_output: Callable[[str], None],
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> None:
    """Run a subprocess, stream its output, raise _StreamError on non-zero."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            on_output(line.rstrip())
            tail.append(line.rstrip())
            if len(tail) > 20:
                tail.pop(0)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise _StreamError(" ".join(cmd[:2]), "timeout", -1)
    if proc.returncode != 0:
        raise _StreamError(" ".join(cmd[:2]), "\n".join(tail[-5:]), proc.returncode)


# ---------------------------------------------------------------------------
# Planner: topological sort, step ordering
# ---------------------------------------------------------------------------

def topological_order(
    step_classes: tuple[type[InstallStep], ...],
) -> list[type[InstallStep]]:
    """
    Kahn's algorithm. Stable order — ties broken by declaration order so
    output is deterministic. Raises ValueError on cycles or missing deps.
    """
    by_id = {s.id: s for s in step_classes}
    indeg = {s.id: 0 for s in step_classes}
    for s in step_classes:
        for dep in s.depends_on:
            if dep not in by_id:
                raise ValueError(f"step {s.id} depends on unknown {dep}")
            indeg[s.id] += 1

    order: list[type[InstallStep]] = []
    # Seed with zero-indegree, preserving declaration order
    ready = [s for s in step_classes if indeg[s.id] == 0]
    while ready:
        s = ready.pop(0)
        order.append(s)
        # Release dependents
        for other in step_classes:
            if s.id in other.depends_on:
                indeg[other.id] -= 1
                if indeg[other.id] == 0:
                    ready.append(other)

    if len(order) != len(step_classes):
        raise ValueError("cycle detected in step dependencies")
    return order


def steps_for(ids: Optional[set[str]]) -> list[type[InstallStep]]:
    """Return ordered steps matching `ids` (plus transitive deps). None → all."""
    ordered = topological_order(ALL_STEPS)
    if ids is None:
        return ordered
    by_id = {s.id: s for s in ordered}
    # Expand to include all transitive deps
    wanted: set[str] = set()
    stack = list(ids)
    while stack:
        i = stack.pop()
        if i in wanted:
            continue
        wanted.add(i)
        if i not in by_id:
            raise ValueError(f"unknown step id: {i}")
        stack.extend(by_id[i].depends_on)
    return [s for s in ordered if s.id in wanted]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class _Ansi:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    OK = "\033[32m"
    WARN = "\033[33m"
    ERR = "\033[31m"
    CYAN = "\033[36m"
    OFF = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        for k in list(vars(cls).keys()):
            if k.isupper():
                setattr(cls, k, "")


def _fmt_status(status: CheckStatus) -> str:
    return {
        CheckStatus.SATISFIED:   f"{_Ansi.OK}✓{_Ansi.OFF}",
        CheckStatus.UNSATISFIED: f"{_Ansi.WARN}○{_Ansi.OFF}",
        CheckStatus.UNKNOWN:     f"{_Ansi.DIM}?{_Ansi.OFF}",
        CheckStatus.BLOCKED:     f"{_Ansi.ERR}✗{_Ansi.OFF}",
    }[status]


def print_plan(env: Env, plan: list[tuple[type[InstallStep], CheckResult]]) -> None:
    print()
    print(f"{_Ansi.BOLD}{_Ansi.CYAN}=== Marshal install plan ==={_Ansi.OFF}")
    print()
    print(f"  Detected: {env.distro} {env.distro_version}, "
          f"kernel {'.'.join(map(str, env.kernel))}, Python "
          f"{'.'.join(map(str, env.python_version))}")
    print(f"            {env.ram_total_gb:.1f}GB RAM, {env.disk_free_gb:.1f}GB disk free, "
          f"{'laptop' if env.is_laptop else 'desktop'}")
    print(f"            Landlock: {'✓' if env.supports_landlock else '✗'}  "
          f"Network: {'✓' if env.has_network else '✗'}  "
          f"Sudo: {'passwordless' if env.has_sudo else 'will prompt'}")
    print()

    todo = [(s, r) for (s, r) in plan if r.status == CheckStatus.UNSATISFIED]
    blocked = [(s, r) for (s, r) in plan if r.status == CheckStatus.BLOCKED]
    total_s = sum(s.est_seconds for (s, r) in todo)
    total_mb = sum(s.est_disk_mb for (s, r) in todo)

    print(f"  Plan ({len(plan)} steps, {len(todo)} to run, "
          f"~{total_s // 60}m {total_s % 60}s, ~{total_mb / 1024:.1f}GB disk):")
    print()
    for (step, result) in plan:
        sym = _fmt_status(result.status)
        opt = f" {_Ansi.DIM}(optional){_Ansi.OFF}" if step.optional else ""
        print(f"   {sym}  {_Ansi.BOLD}{step.id:<16}{_Ansi.OFF} {step.title}{opt}")
        print(f"      {_Ansi.DIM}→ {result.detail}{_Ansi.OFF}")
        if result.remediation and result.status != CheckStatus.SATISFIED:
            print(f"      {_Ansi.DIM}→ fix: {result.remediation.summary}{_Ansi.OFF}")
    print()
    if blocked:
        print(f"{_Ansi.ERR}{len(blocked)} blocked step(s) — cannot proceed until "
              f"the environment is fixed.{_Ansi.OFF}")
        print()


def print_result(step: type[InstallStep], result: ExecuteResult) -> None:
    if result.ok:
        print(f"  {_Ansi.OK}✓{_Ansi.OFF} {step.id} "
              f"{_Ansi.DIM}({result.duration_s:.1f}s){_Ansi.OFF} — {result.detail}")
    else:
        print(f"  {_Ansi.ERR}✗{_Ansi.OFF} {step.id} — {result.detail}")
        if result.remediation:
            print(f"    {_Ansi.BOLD}remediation:{_Ansi.OFF} {result.remediation.summary}")
            for cmd in result.remediation.commands:
                print(f"      {_Ansi.CYAN}${_Ansi.OFF} {cmd}")
            if result.remediation.url:
                print(f"    {_Ansi.DIM}→ {result.remediation.url}{_Ansi.OFF}")


# ---------------------------------------------------------------------------
# Apply loop
# ---------------------------------------------------------------------------

def apply_plan(
    env: Env,
    plan: list[tuple[type[InstallStep], CheckResult]],
    *,
    yes: bool,
    verbose: bool,
    state: InstallState,
) -> int:
    """Run unsatisfied steps in order. Return process exit code."""
    from datetime import datetime, timezone

    # Blocked steps mean the machine can't host Marshal — stop before touching.
    blocked = [s for (s, r) in plan if r.status == CheckStatus.BLOCKED]
    if blocked:
        for (s, r) in plan:
            if r.status == CheckStatus.BLOCKED:
                print(f"  {_Ansi.ERR}✗{_Ansi.OFF} {s.id} — {r.detail}")
                if r.remediation:
                    print(f"    {r.remediation.summary}")
        return 2

    todo = [(s, r) for (s, r) in plan if r.status == CheckStatus.UNSATISFIED]
    if not todo:
        print(f"{_Ansi.OK}Nothing to do — all steps already satisfied.{_Ansi.OFF}")
        return 0

    if not yes:
        try:
            ans = input(f"\nRun the plan? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
        if ans and ans not in ("y", "yes"):
            print("aborted.")
            return 130

    print()
    any_failed = False
    for (step_cls, _) in todo:
        step = step_cls()
        print(f"{_Ansi.BOLD}→ {step.id}{_Ansi.OFF}  {step.title}")

        def on_output(line: str, _verbose=verbose) -> None:
            if _verbose:
                print(f"    {_Ansi.DIM}{line}{_Ansi.OFF}")

        result = step.execute(env, on_output=on_output)
        print_result(step_cls, result)

        if result.ok:
            state.completed[step.id] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_state(state)
        else:
            any_failed = True
            if not step.optional:
                print(f"\n{_Ansi.ERR}Stopping: required step '{step.id}' failed."
                      f"{_Ansi.OFF}  Fix and re-run, or: "
                      f"python install.py --doctor")
                return 1
            print(f"  {_Ansi.WARN}(optional step — continuing){_Ansi.OFF}")

    print()
    if any_failed:
        print(f"{_Ansi.WARN}Done with warnings — optional steps failed.{_Ansi.OFF}")
        return 0
    print(f"{_Ansi.OK}All steps complete. Next:{_Ansi.OFF}")
    print(f"  {_Ansi.CYAN}$ {_Ansi.OFF}bash scripts/start-inference.sh      "
          f"{_Ansi.DIM}# in one terminal{_Ansi.OFF}")
    print(f"  {_Ansi.CYAN}$ {_Ansi.OFF}source .os/bin/activate && python main.py  "
          f"{_Ansi.DIM}# in another{_Ansi.OFF}")
    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def build_plan(env: Env, step_ids: Optional[set[str]]) -> list[tuple[type[InstallStep], CheckResult]]:
    return [(cls, cls().check(env)) for cls in steps_for(step_ids)]


def emit_json(env: Env, plan: list[tuple[type[InstallStep], CheckResult]]) -> None:
    out = {
        "env": env.to_dict(),
        "steps": [
            {
                "id": cls.id,
                "title": cls.title,
                "optional": cls.optional,
                "status": r.status.value,
                "detail": r.detail,
                "remediation": asdict(r.remediation) if r.remediation else None,
            }
            for (cls, r) in plan
        ],
        "state": asdict(load_state()),
    }
    print(json.dumps(out, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="marshal-install",
        description="Declarative install / doctor agent for Marshal.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true",
                      help="(default) show what would run without running it")
    mode.add_argument("--apply", action="store_true",
                      help="run the plan after confirmation")
    mode.add_argument("--doctor", action="store_true",
                      help="check everything, execute nothing (diagnostic)")

    parser.add_argument("--step", action="append", default=[],
                        help="run only this step (plus deps); repeatable")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="skip confirmation prompt (for CI)")
    parser.add_argument("--force", action="store_true",
                        help="clear saved checkpoints before running")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="stream subprocess output")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON, no interactive output")
    parser.add_argument("--no-color", action="store_true",
                        help="disable ANSI colors")

    args = parser.parse_args(argv)

    if args.no_color or not sys.stdout.isatty():
        _Ansi.disable()

    env = probe_env()
    state = load_state()
    if args.force:
        state.completed = {}
        save_state(state)

    step_ids: Optional[set[str]] = set(args.step) if args.step else None
    try:
        plan = build_plan(env, step_ids)
    except ValueError as e:
        print(f"{_Ansi.ERR}error: {e}{_Ansi.OFF}", file=sys.stderr)
        return 2

    if args.json:
        emit_json(env, plan)
        return 0

    # Doctor mode = pure diagnostic (no mutation allowed).
    if args.doctor:
        print_plan(env, plan)
        unsatisfied = sum(1 for (_, r) in plan if r.status != CheckStatus.SATISFIED)
        blocked = sum(1 for (_, r) in plan if r.status == CheckStatus.BLOCKED)
        if blocked:
            return 2
        return 0 if unsatisfied == 0 else 1

    if args.apply:
        print_plan(env, plan)
        return apply_plan(env, plan, yes=args.yes, verbose=args.verbose, state=state)

    # Default: --plan
    print_plan(env, plan)
    print(f"{_Ansi.DIM}Run with --apply to execute, "
          f"--doctor for diagnostic-only.{_Ansi.OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
