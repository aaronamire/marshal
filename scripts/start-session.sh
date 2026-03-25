#!/bin/bash
# Leaves OS session startup — orchestrates the full boot chain.
# Used by greetd/login managers or can be run manually.
set -e

LEAVES_ROOT="$(dirname "$(realpath "$0")")/.."
PYTHON="$LEAVES_ROOT/.os/bin/python3"
UVICORN="$LEAVES_ROOT/.os/bin/uvicorn"

# Ensure XDG runtime dir exists
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# ---------------------------------------------------------------------------
# 1. Inference server (llama.cpp)
# ---------------------------------------------------------------------------
echo "[leaves] starting inference server..."
systemctl --user start leaves-inference.service || true

echo "[leaves] waiting for inference server..."
timeout 120 bash -c 'until /usr/bin/curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; do sleep 1; done'
echo "[leaves] inference server ready"

# ---------------------------------------------------------------------------
# 2. Agent daemon (agentd — watcher + indexer start here)
# ---------------------------------------------------------------------------
echo "[leaves] starting agentd..."
systemctl --user start leaves-agentd.service || true

echo "[leaves] waiting for agentd socket..."
timeout 30 bash -c 'until [ -S /home/xan/.leaves/agentd.sock ]; do sleep 0.1; done'
echo "[leaves] agentd ready"

# ---------------------------------------------------------------------------
# 3. API server (FastAPI on :8765)
# ---------------------------------------------------------------------------
echo "[leaves] starting API server..."
systemctl --user start leaves-api.service || true

echo "[leaves] waiting for API server..."
timeout 30 bash -c 'until /usr/bin/curl -sf http://127.0.0.1:8765/v1/health >/dev/null 2>&1; do sleep 0.5; done'
echo "[leaves] API server ready"

# ---------------------------------------------------------------------------
# 4. Wait for initial indexing (briefing needs data)
#    Best-effort: wait up to 60s for at least some items indexed.
#    Compositor launches either way — briefing shows "indexing..." if empty.
# ---------------------------------------------------------------------------
echo "[leaves] waiting for initial index..."
INDEXED=0
for i in $(seq 1 60); do
    STATUS=$(/usr/bin/curl -sf http://127.0.0.1:8765/v1/cortex/status 2>/dev/null || echo '{}')
    TOTAL=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(s.get('count',0) for s in d.get('sources',{}).values()))" 2>/dev/null || echo "0")
    if [ "$TOTAL" -gt 0 ] 2>/dev/null; then
        echo "[leaves] index ready ($TOTAL items)"
        INDEXED=1
        break
    fi
    sleep 1
done
if [ "$INDEXED" -eq 0 ]; then
    echo "[leaves] index still building — compositor will show loading state"
fi

# ---------------------------------------------------------------------------
# 5. Compositor (Wayland client)
# ---------------------------------------------------------------------------
echo "[leaves] launching compositor..."
exec "$LEAVES_ROOT/compositor/build/leaves-compositor"
