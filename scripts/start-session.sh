#!/bin/bash
# Marshal session startup — orchestrates the full boot chain.
# Used by greetd/login managers or can be run manually.
set -e

MARSHAL_ROOT="$(dirname "$(realpath "$0")")/.."
PYTHON="$MARSHAL_ROOT/.os/bin/python3"
UVICORN="$MARSHAL_ROOT/.os/bin/uvicorn"

# Ensure XDG runtime dir exists
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# ---------------------------------------------------------------------------
# 1. Inference server (llama.cpp)
# ---------------------------------------------------------------------------
echo "[marshal] starting inference server..."
systemctl --user start marshal-inference.service || true

echo "[marshal] waiting for inference server..."
timeout 120 bash -c 'until /usr/bin/curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; do sleep 1; done'
echo "[marshal] inference server ready"

# ---------------------------------------------------------------------------
# 2. Agent daemon (agentd — watcher + indexer start here)
# ---------------------------------------------------------------------------
echo "[marshal] starting agentd..."
systemctl --user start marshal-agentd.service || true

echo "[marshal] waiting for agentd socket..."
AGENTD_SOCK="$HOME/.marshal/agentd.sock"
timeout 30 bash -c 'until [ -S "'"$AGENTD_SOCK"'" ]; do sleep 0.1; done'
echo "[marshal] agentd ready"

# ---------------------------------------------------------------------------
# 3. API server (FastAPI on :8765)
# ---------------------------------------------------------------------------
echo "[marshal] starting API server..."
systemctl --user start marshal-api.service || true

echo "[marshal] waiting for API server..."
timeout 30 bash -c 'until /usr/bin/curl -sf http://127.0.0.1:8765/v1/health >/dev/null 2>&1; do sleep 0.5; done'
echo "[marshal] API server ready"

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
# 5. Compositor (Wayland client)
# ---------------------------------------------------------------------------
echo "[marshal] launching compositor..."
"$MARSHAL_ROOT/compositor/builddir/marshal-compositor" &
COMPOSITOR_PID=$!

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
