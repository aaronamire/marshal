#!/bin/bash
set -e

LEAVES_ROOT="$(dirname "$(realpath "$0")")/.."
PYTHON="$LEAVES_ROOT/.os/bin/python3"
UVICORN="$LEAVES_ROOT/.os/bin/uvicorn"

echo "[leaves] starting inference server..."
systemctl --user start leaves-inference.service || true

echo "[leaves] waiting for inference server..."
timeout 120 bash -c 'until /usr/bin/curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; do sleep 1; done'

echo "[leaves] starting agentd..."
systemctl --user start leaves-agentd.service || true

echo "[leaves] waiting for agentd socket..."
timeout 30 bash -c 'until [ -S /home/xan/.leaves/agentd.sock ]; do sleep 0.1; done'

echo "[leaves] starting API server..."
systemctl --user start leaves-api.service || true
sleep 2

echo "[leaves] launching compositor..."
exec "$LEAVES_ROOT/compositor/build/leaves-compositor"
