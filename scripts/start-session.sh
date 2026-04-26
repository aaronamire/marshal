#!/bin/bash
# Marshal session startup — orchestrates the full boot chain.
# Used by greetd/login managers or can be run manually.
#
# Works in two modes:
#   1. systemd mode: if marshal-*.service units are installed
#      (./systemd/install.sh was run), services start via systemctl --user.
#   2. standalone mode: if the units are missing or the user systemd bus
#      isn't available (e.g. plain TTY login), daemons are launched
#      directly as background processes and torn down on exit.
set -e

MARSHAL_ROOT="$(dirname "$(realpath "$0")")/.."
MARSHAL_ROOT="$(cd "$MARSHAL_ROOT" && pwd)"
PYTHON="$MARSHAL_ROOT/.os/bin/python3"
UVICORN="$MARSHAL_ROOT/.os/bin/uvicorn"

# Ensure XDG runtime dir exists
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# Working/log dirs for standalone mode
mkdir -p "$HOME/.marshal" "$HOME/.marshal/logs"
PIDS=()

cleanup() {
    # Kill any direct-launch children on exit. systemd-managed units are
    # left running — the user can stop them with systemctl.
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Reset standalone-mode leftovers from a previous run.
#
# When the previous start-session.sh was killed without exiting cleanly
# (e.g. user pressed power, compositor crashed) the agentd/api/llama-server
# processes it spawned are still bound to their sockets and ports. The new
# run then either fails to bind (EADDRINUSE) or — worse — keeps talking to
# the *old* code while the user thinks they're testing the new code.
#
# Only kill processes we recognize. Never touch systemd-managed services.
# ---------------------------------------------------------------------------
reset_standalone_leftovers() {
    local killed=0
    # API server (uvicorn api.server:app on :8765)
    local api_pid
    api_pid=$(/usr/bin/lsof -ti:8765 2>/dev/null || true)
    if [ -n "$api_pid" ]; then
        kill "$api_pid" 2>/dev/null && killed=1
    fi
    # Llama-server (port 8080)
    local llama_pid
    llama_pid=$(/usr/bin/lsof -ti:8080 2>/dev/null || true)
    if [ -n "$llama_pid" ]; then
        kill "$llama_pid" 2>/dev/null && killed=1
    fi
    # Agentd (Unix socket)
    if [ -S "$HOME/.marshal/agentd.sock" ]; then
        # Find python process holding the socket
        local agentd_pid
        agentd_pid=$(/usr/bin/lsof -t "$HOME/.marshal/agentd.sock" 2>/dev/null | head -n1)
        if [ -n "$agentd_pid" ]; then
            kill "$agentd_pid" 2>/dev/null && killed=1
        fi
        rm -f "$HOME/.marshal/agentd.sock"
    fi
    if [ $killed -eq 1 ]; then
        echo "[marshal] killed leftover standalone processes from prior run"
        sleep 1  # give the kernel a moment to release ports/sockets
    fi
}
reset_standalone_leftovers

# ---------------------------------------------------------------------------
# Detect whether systemd --user is usable AND has the marshal units loaded.
# If either is false, we fall back to direct background launches.
# ---------------------------------------------------------------------------
have_systemd_unit() {
    local unit="$1"
    [ -n "${XDG_RUNTIME_DIR:-}" ] || return 1
    [ -S "$XDG_RUNTIME_DIR/systemd/private" ] || [ -S "$XDG_RUNTIME_DIR/bus" ] || return 1
    systemctl --user cat "$unit" >/dev/null 2>&1
}

start_or_launch() {
    # $1 = unit name, $2 = direct-launch command (eval'd if used)
    local unit="$1"
    local direct="$2"
    if have_systemd_unit "$unit"; then
        echo "[marshal] starting $unit via systemd"
        systemctl --user start "$unit"
    else
        echo "[marshal] $unit not installed — launching directly"
        # shellcheck disable=SC2086
        eval "$direct" >>"$HOME/.marshal/logs/${unit%.service}.log" 2>&1 &
        PIDS+=("$!")
    fi
}

wait_for_url() {
    local url="$1" budget="$2" label="$3"
    echo "[marshal] waiting for $label..."
    if ! timeout "$budget" bash -c "until /usr/bin/curl -sf '$url' >/dev/null 2>&1; do sleep 1; done"; then
        echo "[marshal] ERROR: $label did not become healthy at $url within ${budget}s"
        echo "[marshal] check $HOME/.marshal/logs/ (standalone) or journalctl --user (systemd)"
        exit 1
    fi
    echo "[marshal] $label ready"
}

# ---------------------------------------------------------------------------
# 1. Inference server (llama.cpp)
# ---------------------------------------------------------------------------
start_or_launch \
    "marshal-inference.service" \
    "exec '$MARSHAL_ROOT/scripts/start-inference.sh'"

wait_for_url "http://127.0.0.1:8080/health" 120 "inference server"

# ---------------------------------------------------------------------------
# 1b. First-touch probe of llama.cpp
#
# Just confirms the server can take a request. The REAL KV-cache prime
# happens inside agentd after this script starts it: agentd awaits
# _warm_inference_kv_cache() with the real intent-parser system prompt,
# which is what cache_prompt prefix-matches against subsequent user
# requests. A fake prompt sent here would not match that prefix.
#
# n_predict=1 keeps the probe cheap. The point is to make sure /health
# wasn't lying.
# ---------------------------------------------------------------------------
echo "[marshal] probing inference /completion..."
PROBE_BODY='{"prompt":"hi","n_predict":1,"cache_prompt":false,"temperature":0.0}'
if /usr/bin/curl -sf --max-time 60 \
        -H "Content-Type: application/json" \
        -d "$PROBE_BODY" \
        http://127.0.0.1:8080/completion >/dev/null 2>&1; then
    echo "[marshal] llama.cpp responsive"
else
    echo "[marshal] WARNING: /completion probe failed — first user intent may stall"
fi

# ---------------------------------------------------------------------------
# 2. Agent daemon (agentd — watcher + indexer start here)
# ---------------------------------------------------------------------------
start_or_launch \
    "marshal-agentd.service" \
    "cd '$MARSHAL_ROOT' && exec '$PYTHON' agentd.py"

echo "[marshal] waiting for agentd socket..."
AGENTD_SOCK="$HOME/.marshal/agentd.sock"
if ! timeout 30 bash -c 'until [ -S "'"$AGENTD_SOCK"'" ]; do sleep 0.1; done'; then
    echo "[marshal] ERROR: agentd socket $AGENTD_SOCK never appeared"
    exit 1
fi
echo "[marshal] agentd ready"

# ---------------------------------------------------------------------------
# 3. API server (FastAPI on :8765)
# ---------------------------------------------------------------------------
start_or_launch \
    "marshal-api.service" \
    "cd '$MARSHAL_ROOT' && exec '$UVICORN' api.server:app --host 127.0.0.1 --port 8765"

wait_for_url "http://127.0.0.1:8765/v1/health" 30 "API server"

# ---------------------------------------------------------------------------
# 4. Wait for initial indexing (briefing needs data)
#    Best-effort: wait up to 60s for at least some items indexed.
#    Compositor launches either way — briefing shows "indexing..." if empty.
# ---------------------------------------------------------------------------
echo "[marshal] waiting for initial index..."
INDEXED=0
for i in $(seq 1 60); do
    STATUS=$(/usr/bin/curl -sf http://127.0.0.1:8765/v1/cortex/status 2>/dev/null || echo '{}')
    TOTAL=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(s.get('count',0) for s in d.get('sources',{}).values()))" 2>/dev/null || echo "0")
    if [ "$TOTAL" -gt 0 ] 2>/dev/null; then
        echo "[marshal] index ready ($TOTAL items)"
        INDEXED=1
        break
    fi
    sleep 1
done
if [ "$INDEXED" -eq 0 ]; then
    echo "[marshal] index still building — compositor will show loading state"
fi

# ---------------------------------------------------------------------------
# 5. Compositor (Wayland client) — always direct-launched and foregrounded
# ---------------------------------------------------------------------------
COMPOSITOR_BIN="$MARSHAL_ROOT/compositor/builddir/marshal-compositor"
if [ ! -x "$COMPOSITOR_BIN" ]; then
    echo "[marshal] ERROR: compositor not built — run ./bootstrap.sh --with-compositor"
    exit 1
fi

echo "[marshal] launching compositor..."
"$COMPOSITOR_BIN" &
COMPOSITOR_PID=$!
PIDS+=("$COMPOSITOR_PID")

# Wait for compositor to publish its WAYLAND_DISPLAY socket name, then export
# it into the D-Bus activation environment so portals/notifyd launched by
# D-Bus activation (rather than direct exec) inherit it. Without this,
# xdg-desktop-portal backends and any D-Bus-activated Wayland client fail
# to connect to the compositor.
WAYLAND_DISPLAY_FILE="$HOME/.marshal/wayland-display"
for i in $(seq 1 100); do
    if [ -s "$WAYLAND_DISPLAY_FILE" ]; then
        WAYLAND_DISPLAY="$(head -n1 "$WAYLAND_DISPLAY_FILE" | tr -d '[:space:]')"
        export WAYLAND_DISPLAY
        export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-wayland}"
        export XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-marshal}"
        export XDG_SESSION_DESKTOP="${XDG_SESSION_DESKTOP:-marshal}"
        if command -v dbus-update-activation-environment >/dev/null 2>&1; then
            dbus-update-activation-environment --systemd \
                WAYLAND_DISPLAY \
                XDG_CURRENT_DESKTOP \
                XDG_SESSION_TYPE \
                XDG_SESSION_DESKTOP \
                || echo "[marshal] dbus-update-activation-environment failed (non-fatal)"
        else
            echo "[marshal] dbus-update-activation-environment not installed — D-Bus activated services may not see WAYLAND_DISPLAY"
        fi
        break
    fi
    sleep 0.1
done

wait "$COMPOSITOR_PID"
