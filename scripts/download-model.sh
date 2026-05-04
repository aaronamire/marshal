#!/usr/bin/env bash
# Download and verify the Marshal GoalSpec model.
#
# Pulls the fine-tuned Qwen-2.5-3B GoalSpec model from HuggingFace and
# verifies its SHA-256 against the manifest in models/MANIFEST.sha256.
# Falls back to the Phase 1 base model if the fine-tune is unavailable.
#
# Usage: ./scripts/download-model.sh [--phase1] [--force]
set -euo pipefail

MARSHAL_ROOT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
MODELS_DIR="$MARSHAL_ROOT/models"
MANIFEST="$MODELS_DIR/MANIFEST.sha256"

# --- model registry ----------------------------------------------------------
# name|hf_repo|hf_filename|local_filename
readonly MODEL_PHASE2="goalspec|yudweb2/marshal-goalspec-3b|goalspec_qwen25_3b_q4km.gguf|goalspec_qwen25_3b_q4km.gguf"
readonly MODEL_PHASE2_7B="goalspec_7b|yudweb2/marshal-goalspec-7b|goalspec_qwen25_7b_q4km.gguf|goalspec_qwen25_7b_q4km.gguf"
readonly MODEL_PHASE1="qwen25_3b|Qwen/Qwen2.5-3B-Instruct-GGUF|qwen2.5-3b-instruct-q4_k_m.gguf|qwen2.5-3b-instruct-q4_k_m.gguf"

usage() {
    echo "Usage: $0 [--phase1|--7b] [--force]"
    echo "  --phase1    download the Phase 1 base model instead of the fine-tune"
    echo "  --7b        download the 7B fine-tune (default is the 3B fine-tune)"
    echo "  --force     redownload even if the file already exists"
    exit 1
}

PHASE1=0
SEVENB=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --phase1) PHASE1=1 ;;
        --7b)     SEVENB=1 ;;
        --force)  FORCE=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown arg: $arg"; usage ;;
    esac
done

if [[ $PHASE1 -eq 1 ]]; then
    SPEC="$MODEL_PHASE1"
elif [[ $SEVENB -eq 1 ]]; then
    SPEC="$MODEL_PHASE2_7B"
else
    SPEC="$MODEL_PHASE2"
fi

IFS='|' read -r NAME HF_REPO HF_FILE LOCAL_FILE <<< "$SPEC"
LOCAL_PATH="$MODELS_DIR/$LOCAL_FILE"

mkdir -p "$MODELS_DIR"

# --- check existing file ----------------------------------------------------
if [[ -f "$LOCAL_PATH" ]] && [[ $FORCE -eq 0 ]]; then
    echo "[download-model] $LOCAL_FILE already exists ($(du -h "$LOCAL_PATH" | cut -f1))"
    if [[ -f "$MANIFEST" ]] && grep -q "$LOCAL_FILE" "$MANIFEST"; then
        echo "[download-model] verifying SHA-256 against MANIFEST..."
        EXPECTED=$(grep "$LOCAL_FILE" "$MANIFEST" | awk '{print $1}')
        ACTUAL=$(sha256sum "$LOCAL_PATH" | awk '{print $1}')
        if [[ "$EXPECTED" == "$ACTUAL" ]]; then
            echo "[download-model] OK — checksum matches"
            exit 0
        else
            echo "[download-model] WARNING: SHA-256 mismatch for $LOCAL_FILE"
            echo "[download-model]   expected: $EXPECTED"
            echo "[download-model]   actual:   $ACTUAL"
            echo "[download-model] re-run with --force to redownload"
            exit 2
        fi
    else
        echo "[download-model] no manifest entry for $LOCAL_FILE — skipping checksum"
        exit 0
    fi
fi

# --- download ---------------------------------------------------------------
# Prefer the new `hf` CLI (huggingface_hub >=0.26) and fall back to the
# legacy `huggingface-cli` binary. Auth is optional — these repos are public.
HF_BIN=""
if command -v hf >/dev/null 2>&1; then
    HF_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_BIN="huggingface-cli"
else
    echo "[download-model] ERROR: neither 'hf' nor 'huggingface-cli' found"
    echo "  install with: pip install --upgrade huggingface_hub"
    echo "  or:           curl -LsSf https://hf.co/cli/install.sh | bash"
    exit 3
fi

echo "[download-model] fetching $HF_FILE from $HF_REPO (via $HF_BIN)..."
if [[ "$HF_BIN" == "hf" ]]; then
    hf download "$HF_REPO" "$HF_FILE" --local-dir "$MODELS_DIR"
else
    huggingface-cli download \
        "$HF_REPO" \
        "$HF_FILE" \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks False
fi

# huggingface-cli sometimes drops files in repo-flavored subdirs; normalize.
if [[ ! -f "$LOCAL_PATH" ]]; then
    found=$(find "$MODELS_DIR" -name "$HF_FILE" -type f 2>/dev/null | head -1)
    if [[ -n "$found" && "$found" != "$LOCAL_PATH" ]]; then
        mv "$found" "$LOCAL_PATH"
    fi
fi

if [[ ! -f "$LOCAL_PATH" ]]; then
    echo "[download-model] ERROR: download appeared to succeed but $LOCAL_PATH is missing"
    exit 4
fi

# --- verify -----------------------------------------------------------------
if [[ -f "$MANIFEST" ]] && grep -q "$LOCAL_FILE" "$MANIFEST"; then
    echo "[download-model] verifying SHA-256..."
    EXPECTED=$(grep "$LOCAL_FILE" "$MANIFEST" | awk '{print $1}')
    ACTUAL=$(sha256sum "$LOCAL_PATH" | awk '{print $1}')
    if [[ "$EXPECTED" != "$ACTUAL" ]]; then
        echo "[download-model] FAIL — checksum mismatch"
        echo "  expected: $EXPECTED"
        echo "  actual:   $ACTUAL"
        echo "  removing tampered/corrupt download"
        rm -f "$LOCAL_PATH"
        exit 5
    fi
    echo "[download-model] OK — $LOCAL_FILE verified"
else
    echo "[download-model] no manifest entry — recording current SHA-256:"
    sha256sum "$LOCAL_PATH" | tee -a "$MANIFEST"
    echo "[download-model] commit MANIFEST.sha256 to lock this checksum"
fi

echo "[download-model] done. model at $LOCAL_PATH"
