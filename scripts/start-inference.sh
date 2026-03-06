#!/bin/bash
# Start the llama.cpp inference server for Leaves OS
# CPU-only — no GPU flags. i5-7200U / Intel HD 620.

LLAMA_SERVER="$HOME/dev/llama.cpp/build/bin/llama-server"
MODEL="$HOME/leaves-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
PORT=8080
HOST="127.0.0.1"
THREADS=4

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Build llama.cpp first:"
    echo "  cd ~/dev/llama.cpp && cmake -B build -DLLAMA_CURL=OFF -DLLAMA_NATIVE=ON && cmake --build build -j\$(nproc)"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Download with:"
    echo "  source .os/bin/activate && python3 -c \""
    echo "  from huggingface_hub import hf_hub_download; import os"
    echo "  hf_hub_download(repo_id='bartowski/Llama-3.2-1B-Instruct-GGUF',"
    echo "    filename='Llama-3.2-1B-Instruct-Q4_K_M.gguf',"
    echo "    local_dir=os.path.expanduser('~/leaves-models'))\""
    exit 1
fi

# Kill any existing server on this port
EXISTING=$(lsof -t -i:$PORT 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "Killing existing process on port $PORT (PID $EXISTING)..."
    kill "$EXISTING" 2>/dev/null
    sleep 1
fi

echo "Starting llama-server on $HOST:$PORT with $THREADS threads..."
echo "Model: $MODEL"
echo ""

exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --port "$PORT" \
    --host "$HOST" \
    --threads "$THREADS" \
    --ctx-size 2048 \
    --n-predict 512 \
    --log-disable
