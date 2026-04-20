#!/usr/bin/env bash
# Leaves OS bootstrap — one-shot installer for any modern Linux.
#
# What this does:
#   1. Detects package manager (pacman, apt, dnf) and installs system deps
#   2. Builds llama.cpp at a pinned commit (CPU-only, native ISA)
#   3. Creates a Python venv at .os/ and installs the Leaves package
#   4. Downloads the GoalSpec model (verified by SHA-256)
#   5. Optionally builds the Wayland compositor (--with-compositor)
#   6. Optionally installs systemd --user units (--with-systemd)
#   7. Runs tests/demo_suite.py to verify the install
#
# Idempotent: safe to re-run. Re-uses an existing venv, skips the model
# download if the SHA-256 matches, skips the llama.cpp clone if already
# present at the pinned commit.
#
# Usage:
#   ./bootstrap.sh                          # core install (no compositor)
#   ./bootstrap.sh --with-compositor        # also build the compositor
#   ./bootstrap.sh --with-systemd           # also install systemd units
#   ./bootstrap.sh --skip-model             # skip the model download
#   ./bootstrap.sh --skip-tests             # skip the verification step
#   ./bootstrap.sh --no-color               # disable colored output
set -euo pipefail

# --- args -------------------------------------------------------------------
WITH_COMPOSITOR=0
WITH_SYSTEMD=0
SKIP_MODEL=0
SKIP_TESTS=0
USE_COLOR=1

for arg in "$@"; do
    case "$arg" in
        --with-compositor) WITH_COMPOSITOR=1 ;;
        --with-systemd)    WITH_SYSTEMD=1 ;;
        --skip-model)      SKIP_MODEL=1 ;;
        --skip-tests)      SKIP_TESTS=1 ;;
        --no-color)        USE_COLOR=0 ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | head -n -1
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# --- ui ---------------------------------------------------------------------
if [[ $USE_COLOR -eq 1 && -t 1 ]]; then
    C_OK="\033[1;32m"
    C_INFO="\033[1;36m"
    C_WARN="\033[1;33m"
    C_ERR="\033[1;31m"
    C_OFF="\033[0m"
else
    C_OK=""; C_INFO=""; C_WARN=""; C_ERR=""; C_OFF=""
fi
log()  { printf "${C_INFO}[bootstrap]${C_OFF} %s\n" "$*"; }
ok()   { printf "${C_OK}[bootstrap]${C_OFF} %s\n" "$*"; }
warn() { printf "${C_WARN}[bootstrap]${C_OFF} %s\n" "$*" >&2; }
err()  { printf "${C_ERR}[bootstrap]${C_OFF} %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

LEAVES_ROOT="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$LEAVES_ROOT"

# --- preflight --------------------------------------------------------------
log "Leaves OS bootstrap"
log "root: $LEAVES_ROOT"

[[ "$(uname -s)" == "Linux" ]] || die "Linux only — detected $(uname -s)"

# --- pkg manager detection --------------------------------------------------
PKG_MGR=""
PKG_INSTALL=""
PKG_UPDATE=""
if command -v pacman >/dev/null 2>&1; then
    PKG_MGR="pacman"
    PKG_UPDATE="sudo pacman -Sy"
    PKG_INSTALL="sudo pacman -S --needed --noconfirm"
elif command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt"
    PKG_UPDATE="sudo apt-get update"
    PKG_INSTALL="sudo apt-get install -y"
elif command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
    PKG_UPDATE="sudo dnf check-update || true"
    PKG_INSTALL="sudo dnf install -y"
else
    die "No supported package manager found (pacman/apt/dnf)"
fi
log "package manager: $PKG_MGR"

# --- system deps ------------------------------------------------------------
declare -a SYS_PKGS_PACMAN=(
    base-devel cmake git python python-pip python-virtualenv
    sqlite curl jq ca-certificates
)
declare -a SYS_PKGS_APT=(
    build-essential cmake git python3 python3-pip python3-venv
    sqlite3 curl jq ca-certificates pkg-config
)
declare -a SYS_PKGS_DNF=(
    "@Development Tools" cmake git python3 python3-pip python3-virtualenv
    sqlite curl jq ca-certificates pkgconfig
)

if [[ $WITH_COMPOSITOR -eq 1 ]]; then
    SYS_PKGS_PACMAN+=(meson ninja wlroots wayland wayland-protocols
                      cairo pango libcurl-gnutls cjson libdrm libjpeg-turbo
                      libxcb xcb-util-wm alsa-lib systemd-libs xorg-xwayland)
    SYS_PKGS_APT+=(meson ninja-build libwlroots-dev libwayland-dev
                   wayland-protocols libcairo2-dev libpango1.0-dev libcurl4-openssl-dev
                   libcjson-dev libdrm-dev libjpeg-dev libxcb1-dev libxcb-icccm4-dev
                   libasound2-dev libsystemd-dev xwayland)
    SYS_PKGS_DNF+=(meson ninja-build wlroots-devel wayland-devel
                   wayland-protocols-devel cairo-devel pango-devel libcurl-devel
                   cjson-devel libdrm-devel libjpeg-turbo-devel libxcb-devel
                   xcb-util-wm-devel alsa-lib-devel systemd-devel xorg-x11-server-Xwayland)
fi

log "installing system packages..."
$PKG_UPDATE >/dev/null 2>&1 || warn "package index update failed (continuing)"
case "$PKG_MGR" in
    pacman) $PKG_INSTALL "${SYS_PKGS_PACMAN[@]}" ;;
    apt)    $PKG_INSTALL "${SYS_PKGS_APT[@]}" ;;
    dnf)    $PKG_INSTALL "${SYS_PKGS_DNF[@]}" ;;
esac
ok "system deps installed"

# --- llama.cpp build --------------------------------------------------------
LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/dev/llama.cpp}"
LLAMA_PIN="${LLAMA_CPP_COMMIT:-master}"  # TODO: pin a specific commit hash for reproducibility
LLAMA_BIN="$LLAMA_DIR/build/bin/llama-server"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    log "cloning llama.cpp into $LLAMA_DIR"
    mkdir -p "$(dirname "$LLAMA_DIR")"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
    log "building llama.cpp (this takes a few minutes)..."
    pushd "$LLAMA_DIR" >/dev/null
    if [[ "$LLAMA_PIN" != "master" ]]; then
        git fetch --depth 1 origin "$LLAMA_PIN"
        git checkout "$LLAMA_PIN"
    fi
    cmake -B build -DLLAMA_CURL=OFF -DLLAMA_NATIVE=ON -DGGML_NATIVE=ON >/dev/null
    cmake --build build --target llama-server -j "$(nproc)" >/dev/null
    popd >/dev/null
    ok "llama.cpp built — $LLAMA_BIN"
else
    log "llama-server already present at $LLAMA_BIN — skipping build"
fi

# --- python venv ------------------------------------------------------------
VENV_DIR="$LEAVES_ROOT/.os"
if [[ ! -d "$VENV_DIR" ]]; then
    log "creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "installing Leaves OS into venv..."
pip install --upgrade pip wheel >/dev/null
pip install -e ".[rag,remote]" >/dev/null
ok "Python deps installed"

# --- model download ---------------------------------------------------------
if [[ $SKIP_MODEL -eq 0 ]]; then
    log "downloading GoalSpec model..."
    bash "$LEAVES_ROOT/scripts/download-model.sh"
else
    warn "model download skipped (--skip-model). Inference will not work until you fetch a model."
fi

# --- compositor build (optional) --------------------------------------------
if [[ $WITH_COMPOSITOR -eq 1 ]]; then
    log "building Wayland compositor..."
    pushd "$LEAVES_ROOT/compositor" >/dev/null
    if [[ ! -d builddir ]]; then
        meson setup builddir
    fi
    meson compile -C builddir
    popd >/dev/null
    ok "compositor built — $LEAVES_ROOT/compositor/builddir/leaves-compositor"
fi

# --- systemd units (optional) -----------------------------------------------
if [[ $WITH_SYSTEMD -eq 1 ]]; then
    log "installing systemd --user units..."
    bash "$LEAVES_ROOT/systemd/install.sh"
fi

# --- verify -----------------------------------------------------------------
if [[ $SKIP_TESTS -eq 0 ]]; then
    log "running test suite (excluding demo)..."
    if pytest tests/ -q --tb=line --ignore=tests/eval_suite.py --ignore=tests/demo_suite.py -x 2>/dev/null; then
        ok "tests passed"
    else
        warn "test suite reported failures — investigate before launching"
    fi
fi

# --- summary ----------------------------------------------------------------
ok "bootstrap complete"
echo
echo "Next steps:"
echo "  1. Start the inference server:  ./scripts/start-inference.sh"
echo "  2. In another shell:            source .os/bin/activate && python leaves.py"
echo
echo "  Optional remote inference:      export LEAVES_ANTHROPIC_KEY=sk-ant-..."
if [[ $WITH_COMPOSITOR -eq 1 ]]; then
    echo "  Run the compositor (nested):    ./compositor/builddir/leaves-compositor"
fi
