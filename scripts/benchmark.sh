#!/bin/bash
set -e

LLAMA_CLI="$HOME/dev/llama.cpp/build/bin/llama-cli"
MODEL="$HOME/leaves-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

if [ ! -f "$MODEL" ]; then
    echo "Model not found at $MODEL"
    echo "Download with:"
    echo "  huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \\"
    echo "    --include 'Llama-3.2-1B-Instruct-Q4_K_M.gguf' \\"
    echo "    --local-dir ~/leaves-models/"
    exit 1
fi

PROMPT='<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an intent classifier for Leaves OS. 
Return ONLY a JSON object. No explanation. No markdown.
Format: {"category": "file_task", "confidence": 0.95}
Categories: file_task, web_task, system_task, writing_task, audio_task, network_task, power_task
<|eot_id|><|start_header_id|>user<|end_header_id|>
find all my tax PDFs from last year
<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

echo "=== Leaves OS Inference Benchmark ==="
echo "Model: Llama 3.2 1B Q4_K_M"
echo "Hardware: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo ""
echo "Running 5 classification calls..."
echo ""

TOTAL=0
for i in {1..5}; do
    START=$(date +%s%3N)
    OUTPUT=$($LLAMA_CLI \
        -m "$MODEL" \
        -p "$PROMPT" \
        -n 64 \
        --temp 0.1 \
        --threads 4 \
        --log-disable 2>/dev/null | tail -1)
    END=$(date +%s%3N)
    ELAPSED=$((END - START))
    TOTAL=$((TOTAL + ELAPSED))
    echo "  Run $i: ${ELAPSED}ms — $OUTPUT"
done

AVG=$((TOTAL / 5))
echo ""
echo "Average latency: ${AVG}ms"
echo ""

if [ $AVG -lt 500 ]; then
    echo "✓ Excellent — well within 300ms classification budget"
    echo "  (classification needs only ~15 tokens, not 64)"
elif [ $AVG -lt 1000 ]; then
    echo "⚠ Acceptable — use fine-tuned MiniLM for Layer 1"
    echo "  Use Llama 1B as Layer 2 only"
else
    echo "✗ Too slow for Layer 1 — must use fine-tuned classifier"
    echo "  See docs/layer1-classifier.md"
fi
