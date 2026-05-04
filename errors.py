"""
Typed error taxonomy for Marshal.
All user-visible errors must use MarshalError with a MarshalErrorCode.
Never raise bare Exception at a user-visible boundary.
"""
from __future__ import annotations

import enum
from typing import Optional


class MarshalErrorCode(enum.Enum):
    # Input validation
    EMPTY_INTENT = "EMPTY_INTENT"
    INTENT_TOO_LONG = "INTENT_TOO_LONG"
    INVALID_INTENT_FORMAT = "INVALID_INTENT_FORMAT"

    # Inference / parsing
    INFERENCE_UNAVAILABLE = "INFERENCE_UNAVAILABLE"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INFERENCE_BAD_RESPONSE = "INFERENCE_BAD_RESPONSE"
    JSON_PARSE_FAILED = "JSON_PARSE_FAILED"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"

    # Authorization / path safety
    AUTHORIZATION_VIOLATION = "AUTHORIZATION_VIOLATION"
    PATH_NOT_AUTHORIZED = "PATH_NOT_AUTHORIZED"
    PATH_DOES_NOT_EXIST = "PATH_DOES_NOT_EXIST"
    PATH_TRAVERSAL_DETECTED = "PATH_TRAVERSAL_DETECTED"

    # File operations
    FILE_READ_ERROR = "FILE_READ_ERROR"
    FILE_WRITE_ERROR = "FILE_WRITE_ERROR"
    FILE_DELETE_ERROR = "FILE_DELETE_ERROR"
    FILE_MOVE_ERROR = "FILE_MOVE_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"

    # State machine
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    INTENT_ALREADY_TERMINAL = "INTENT_ALREADY_TERMINAL"

    # Tool failure / livelock
    TOOL_FAILURE_ESCALATED = "TOOL_FAILURE_ESCALATED"
    MAX_RETRIES_EXCEEDED = "MAX_RETRIES_EXCEEDED"

    # Agent availability
    AGENT_NOT_AVAILABLE = "AGENT_NOT_AVAILABLE"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"

    # Database
    DB_ERROR = "DB_ERROR"

    # Authorization dialog
    USER_CANCELLED = "USER_CANCELLED"

    # Layer 2.5 routing (Phase 1)
    SEMANTIC_VALIDATION_FAILED = "SEMANTIC_VALIDATION_FAILED"  # hard semantic error, escalate
    LOW_CONFIDENCE_ESCALATION = "LOW_CONFIDENCE_ESCALATION"   # confidence below L2.5 threshold

    # Persistent intents
    INTENT_NOT_FOUND = "INTENT_NOT_FOUND"
    INTENT_ALREADY_INACTIVE = "INTENT_ALREADY_INACTIVE"
    WATCHER_PATH_INVALID = "WATCHER_PATH_INVALID"

    # Process management
    PROCESS_NOT_FOUND = "PROCESS_NOT_FOUND"

    # User toggled the inference server off via `inference off`
    INFERENCE_DISABLED = "INFERENCE_DISABLED"

    # Generic
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


# Human-readable messages shown to users (never show stack traces)
USER_MESSAGES: dict[MarshalErrorCode, str] = {
    MarshalErrorCode.EMPTY_INTENT: "Please enter an intent. Type what you want to do.",
    MarshalErrorCode.INTENT_TOO_LONG: "Your intent is too long. Please keep it under 1000 characters.",
    MarshalErrorCode.INVALID_INTENT_FORMAT: "Could not understand that intent format.",
    MarshalErrorCode.INFERENCE_UNAVAILABLE: (
        "The inference server is not running. "
        "Start it with: bash scripts/start-inference.sh"
    ),
    MarshalErrorCode.INFERENCE_TIMEOUT: "The inference server timed out. Try again or restart it.",
    MarshalErrorCode.INFERENCE_BAD_RESPONSE: "The model returned an unexpected response. Try rephrasing.",
    MarshalErrorCode.JSON_PARSE_FAILED: "Could not parse model output as JSON. Try rephrasing your intent.",
    MarshalErrorCode.SCHEMA_VALIDATION_FAILED: "Model output did not match the expected schema. Try rephrasing.",
    MarshalErrorCode.LOW_CONFIDENCE: (
        "I'm not confident enough about what you want. "
        "Try being more specific (e.g., 'find all PDFs in ~/Downloads')."
    ),
    MarshalErrorCode.AUTHORIZATION_VIOLATION: (
        "Action blocked: it violates the planned authorization contract. "
        "The agent attempted something outside the approved scope."
    ),
    MarshalErrorCode.PATH_NOT_AUTHORIZED: (
        "That path is outside your home directory. "
        "Marshal only operates within your home directory."
    ),
    MarshalErrorCode.PATH_DOES_NOT_EXIST: "That path does not exist.",
    MarshalErrorCode.PATH_TRAVERSAL_DETECTED: "Potential path traversal detected. Operation blocked.",
    MarshalErrorCode.FILE_READ_ERROR: "Could not read the file.",
    MarshalErrorCode.FILE_WRITE_ERROR: "Could not write the file.",
    MarshalErrorCode.FILE_DELETE_ERROR: "Could not delete the file.",
    MarshalErrorCode.FILE_MOVE_ERROR: "Could not move the file.",
    MarshalErrorCode.FILE_NOT_FOUND: "File not found.",
    MarshalErrorCode.PERMISSION_DENIED: "Permission denied.",
    MarshalErrorCode.INVALID_STATE_TRANSITION: "Invalid state transition in intent lifecycle.",
    MarshalErrorCode.INTENT_ALREADY_TERMINAL: "This intent has already completed or been cancelled.",
    MarshalErrorCode.TOOL_FAILURE_ESCALATED: (
        "A tool has failed too many times with the same arguments. "
        "The operation has been stopped to prevent a loop."
    ),
    MarshalErrorCode.MAX_RETRIES_EXCEEDED: "Maximum retry attempts exceeded.",
    MarshalErrorCode.AGENT_NOT_AVAILABLE: "That agent type is not available in this phase.",
    MarshalErrorCode.INFERENCE_DISABLED: (
        "The inference server is turned off. Want to do it? "
        "Type 'inference on' to start it."
    ),
    MarshalErrorCode.DEPENDENCY_FAILED: "A required action dependency failed. Skipping.",
    MarshalErrorCode.DB_ERROR: "A database error occurred. Check logs.",
    MarshalErrorCode.USER_CANCELLED: "Operation cancelled.",
    MarshalErrorCode.SEMANTIC_VALIDATION_FAILED: (
        "The intent has a structural conflict (e.g., mutation before discovery). "
        "Try being more specific about the operation order."
    ),
    MarshalErrorCode.LOW_CONFIDENCE_ESCALATION: (
        "I'm not confident enough about this intent — routing to a more capable model. "
        "Try being more specific while the escalation is in progress."
    ),
    MarshalErrorCode.INTENT_NOT_FOUND: "That persistent intent was not found.",
    MarshalErrorCode.INTENT_ALREADY_INACTIVE: "That persistent intent is already inactive.",
    MarshalErrorCode.WATCHER_PATH_INVALID: (
        "The watcher path is invalid or does not exist. "
        "Provide an absolute path or a path under your home directory."
    ),
    MarshalErrorCode.PROCESS_NOT_FOUND: "No running process found with that name.",
    MarshalErrorCode.INTERNAL_ERROR: "An internal error occurred. This is a bug.",
    MarshalErrorCode.NOT_IMPLEMENTED: (
        "That type of task isn't implemented yet. "
        "Supported: file operations, system info, web search, app launch/close, "
        "audio control, network management, power management. "
        "Email and writing tasks are coming in a future release."
    ),
}

# Sanity check at import time — all codes must have user messages
_missing = [c for c in MarshalErrorCode if c not in USER_MESSAGES]
if _missing:
    raise RuntimeError(f"Missing USER_MESSAGES for: {_missing}")


class MarshalError(Exception):
    """
    All user-visible errors in Marshal.
    Always include a MarshalErrorCode. Never show stack traces to users.
    """

    def __init__(
        self,
        code: MarshalErrorCode,
        detail: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ):
        self.code = code
        self.detail = detail
        self.cause = cause
        self.user_message = USER_MESSAGES[code]
        super().__init__(f"[{code.value}] {detail or self.user_message}")

    def __repr__(self) -> str:
        return f"MarshalError(code={self.code.value!r}, detail={self.detail!r})"
