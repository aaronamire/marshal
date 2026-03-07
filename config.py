"""
Centralized configuration constants for Leaves OS.
All numeric thresholds and paths live here — never hardcode in business logic.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Inference server
# ---------------------------------------------------------------------------
INFERENCE_SERVER_HOST = "127.0.0.1"
INFERENCE_SERVER_PORT = 8080
INFERENCE_SERVER_URL = f"http://{INFERENCE_SERVER_HOST}:{INFERENCE_SERVER_PORT}"

# Timeouts (seconds)
# i5-7200U with --mlock --no-mmap --threads 2 --ctx-size 2048 generates ~20-30 tok/s.
# 1024 tokens at 20 tok/s = ~51s. 60s read timeout is safe; was 300s due to swap.
TIMEOUT_CONNECT_SECONDS = 5
TIMEOUT_READ_SECONDS = 60    # 1 minute — covers 1024 tokens at 20 tok/s with headroom
TIMEOUT_HARD_SECONDS = 360

# Generation parameters
INFERENCE_TEMPERATURE = 0.1
INFERENCE_MAX_TOKENS = 1024
INFERENCE_STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>"]

# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------
MIN_CONFIDENCE_THRESHOLD = 0.60
MAX_INTENT_LENGTH = 1000   # characters

# ---------------------------------------------------------------------------
# Agent / tool retry policy
# ---------------------------------------------------------------------------
MAX_TOOL_RETRIES = 3
TOOL_FAILURE_ESCALATION_THRESHOLD = 3  # same tool+args fail count before livelock abort

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent

AUDIT_DB_PATH = Path.home() / ".leaves" / "intents.db"
SCHEMA_PATH = _PROJECT_ROOT / "agents" / "schema" / "goal_spec.json"
INTENT_PARSER_PROMPT_PATH = _PROJECT_ROOT / "agents" / "prompts" / "intent_parser.txt"
FILE_AGENT_REGISTRY_PATH = _PROJECT_ROOT / "agents" / "registry" / "file-agent.json"
CLASSIFIER_PIPELINE_PATH = _PROJECT_ROOT / "models" / "layer1_pipeline.joblib"

# ---------------------------------------------------------------------------
# UI / display
# ---------------------------------------------------------------------------
LEAVES_PRIMARY_COLOR = "#7FDBFF"   # cyan-ish — used in banner and headers
LEAVES_SUCCESS_COLOR = "#2ECC40"
LEAVES_ERROR_COLOR = "#FF4136"
LEAVES_WARNING_COLOR = "#FFDC00"
LEAVES_DIM_COLOR = "#AAAAAA"

APP_NAME = "Leaves OS"
APP_VERSION = "0.2.0-phase1"

# ---------------------------------------------------------------------------
# Authorized path roots (relative to home — expanded at runtime)
# ---------------------------------------------------------------------------
# Phase 0: only home directory. Phase 1+ will add XDG dirs.
AUTHORIZED_PATH_ROOTS: list[str] = [
    "~",
]
