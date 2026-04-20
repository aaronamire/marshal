#!/bin/bash
# Start the llama.cpp inference server for Marshal
# CPU-only — no GPU flags. i5-7200U / Intel HD 620.

LLAMA_SERVER="$HOME/dev/llama.cpp/build/bin/llama-server"
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

# Legacy fallbacks — kept so the script still works without a tier.json.
# Phase 2: Fine-tuned GoalSpec model Q4_K_M (ChatML format, ~1.8GB)
# Phase 1: Qwen2.5-3B-Instruct Q4_K_M (ChatML format, ~1.88GB) — fallback
# Phase 0: Llama-3.2-1B-Instruct Q4_K_M (Llama3 format, 771MB) — last resort
if [ -z "$MODEL" ]; then
    MODEL="$REPO_ROOT/models/goalspec_qwen25_3b_q4km.gguf"
fi
MODEL_FALLBACK_P1="$REPO_ROOT/models/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_FALLBACK_P0="$HOME/marshal-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
PORT=8080
HOST="127.0.0.1"
THREADS=2  # Physical cores only — DO NOT use 4 (logical) on Kaby Lake HT

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Build llama.cpp first:"
    echo "  cd ~/dev/llama.cpp && cmake -B build -DLLAMA_CURL=OFF -DLLAMA_NATIVE=ON && cmake --build build -j\$(nproc)"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    if [ -f "$MODEL_FALLBACK_P1" ]; then
        echo "WARNING: Fine-tuned model not found at $MODEL"
        echo "         Falling back to Phase 1 base model: $MODEL_FALLBACK_P1"
        MODEL="$MODEL_FALLBACK_P1"
    elif [ -f "$MODEL_FALLBACK_P0" ]; then
        echo "WARNING: Qwen2.5-3B not found. Falling back to Phase 0 model."
        echo "         Set MODEL_FAMILY=llama3 in config.py for correct prompt format."
        MODEL="$MODEL_FALLBACK_P0"
    else
        echo "ERROR: No model found."
        echo "  Fine-tuned (Phase 2): $MODEL"
        echo "  Qwen2.5-3B (Phase 1): $MODEL_FALLBACK_P1"
        echo "  Llama-3.2-1B (Phase 0): $MODEL_FALLBACK_P0"
        echo ""
        echo "Download fine-tuned model from Google Drive or run Colab notebook."
        echo "Or download Qwen2.5-3B base:"
        echo "  source .os/bin/activate && python3 -c \""
        echo "  from huggingface_hub import hf_hub_download"
        echo "  hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',"
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
