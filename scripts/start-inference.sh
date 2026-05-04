#!/bin/bash
# Start the llama.cpp inference server for Marshal.
# CPU-only — no GPU flags by default.
#
# llama.cpp install location (matches bootstrap.sh): override with
#   LLAMA_CPP_DIR=/path/to/llama.cpp ./scripts/start-inference.sh

LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/dev/llama.cpp}"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Tier-aware model resolution. hardware.py reads ~/.marshal/tier.json and
# returns the absolute path of the GGUF that best matches the saved tier,
# walking down the ladder if the preferred model isn't installed. If the
# resolver can't produce a path (no models on disk), we fall through to the
# historical fallback chain below so a fresh clone without hardware.py
# configured still boots.
MODEL=""
if [ -x "$(command -v python3)" ]; then
    MODEL_CANDIDATE="$(cd "$REPO_ROOT" && python3 hardware.py --resolve-model 2>/dev/null)"
    if [ -n "$MODEL_CANDIDATE" ] && [ -f "$MODEL_CANDIDATE" ]; then
        MODEL="$MODEL_CANDIDATE"
        ACTIVE_TIER="$(cd "$REPO_ROOT" && python3 hardware.py --print-tier 2>/dev/null)"
        echo "Tier: ${ACTIVE_TIER:-unknown} → $(basename "$MODEL")"
    fi
fi

# Fallbacks — kept so the script still works without a tier.json.
#   Default: fine-tuned GoalSpec model Q4_K_M (ChatML, ~1.8GB)
#   Fallback: Qwen2.5-3B-Instruct Q4_K_M (ChatML, ~1.88GB)
if [ -z "$MODEL" ]; then
    MODEL="$REPO_ROOT/models/goalspec_qwen25_3b_q4km.gguf"
fi
MODEL_FALLBACK="$REPO_ROOT/models/qwen2.5-3b-instruct-q4_k_m.gguf"
PORT=8080
HOST="127.0.0.1"

# Thread count for llama.cpp.
#
# llama.cpp runs best at the number of *physical* cores. On SMT/HT systems,
# scheduling threads onto logical siblings causes pipeline contention and
# regressed throughput. /proc/cpuinfo gives a reliable physical-core count
# across all x86 vendors; we fall back to nproc/2 (typical 2-way SMT) and
# finally a hardcoded 2 if neither is available.
#
# Override with MARSHAL_LLAMA_THREADS=N to pin a specific value (useful for
# benchmarking or when running alongside other CPU-heavy workloads).
if [ -n "${MARSHAL_LLAMA_THREADS:-}" ]; then
    THREADS="$MARSHAL_LLAMA_THREADS"
elif command -v lscpu >/dev/null 2>&1; then
    PHYS_CORES=$(lscpu -p=core 2>/dev/null | grep -v '^#' | sort -u | wc -l)
    THREADS="${PHYS_CORES:-2}"
elif [ -r /proc/cpuinfo ]; then
    PHYS_CORES=$(awk -F: '/^core id/ {print $2}' /proc/cpuinfo | sort -u | wc -l)
    THREADS="${PHYS_CORES:-2}"
else
    THREADS=$(($(nproc 2>/dev/null || echo 4) / 2))
    [ "$THREADS" -lt 1 ] && THREADS=2
fi
echo "[inference] threads=$THREADS"

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Run ./bootstrap.sh, or build llama.cpp manually:"
    echo "  git clone https://github.com/ggerganov/llama.cpp.git \"$LLAMA_DIR\""
    echo "  cd \"$LLAMA_DIR\" && cmake -B build -DLLAMA_CURL=OFF -DLLAMA_NATIVE=ON && cmake --build build -j\$(nproc)"
    echo "  (set LLAMA_CPP_DIR to install elsewhere)"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    if [ -f "$MODEL_FALLBACK" ]; then
        echo "WARNING: Fine-tuned model not found at $MODEL"
        echo "         Falling back to base model: $MODEL_FALLBACK"
        MODEL="$MODEL_FALLBACK"
    else
        echo "ERROR: No model found at $MODEL or $MODEL_FALLBACK."
        echo "Run ./scripts/download-model.sh, or fetch manually:"
        echo "  source .os/bin/activate"
        echo "  python3 -c \"from huggingface_hub import hf_hub_download; \\"
        echo "    hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', \\"
        echo "    filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='models/')\""
        exit 1
    fi
fi

# If invoked manually while the systemd service is already running, restart it.
# Skip this check when we ARE the systemd service (INVOCATION_ID is set by systemd).
if [ -z "$INVOCATION_ID" ] && systemctl --user is-active --quiet marshal-inference.service 2>/dev/null; then
    echo "systemd service already active — restarting..."
    systemctl --user restart marshal-inference.service
    echo "Done. Use: journalctl --user -u marshal-inference -f"
    exit 0
fi

# Kill any existing server on this port (no systemd)
EXISTING=$(lsof -t -i:$PORT 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "Killing existing process on port $PORT (PID $EXISTING)..."
    kill "$EXISTING" 2>/dev/null
    sleep 1
fi

echo "Starting llama-server on $HOST:$PORT with $THREADS threads..."
echo "Model: $MODEL"
echo ""

# Ensure KV cache save directory exists (for --slot-save-path)
KV_CACHE_DIR="$HOME/.marshal/kv-cache"
mkdir -p "$KV_CACHE_DIR"

exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --port "$PORT" \
    --host "$HOST" \
    --threads "$THREADS" \
    --ctx-size 4096 \
    --mlock \
    --no-mmap \
    --log-disable \
    --spec-type ngram-simple \
    --draft-max 8 \
    --slot-save-path "$KV_CACHE_DIR"
