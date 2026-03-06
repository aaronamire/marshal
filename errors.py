"""
Typed error taxonomy for Leaves OS.
All user-visible errors must use LeavesError with a LeavesErrorCode.
Never raise bare Exception at a user-visible boundary.
"""
from __future__ import annotations

import enum
from typing import Optional


class LeavesErrorCode(enum.Enum):
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

    # Generic
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


# Human-readable messages shown to users (never show stack traces)
USER_MESSAGES: dict[LeavesErrorCode, str] = {
    LeavesErrorCode.EMPTY_INTENT: "Please enter an intent. Type what you want to do.",
    LeavesErrorCode.INTENT_TOO_LONG: "Your intent is too long. Please keep it under 1000 characters.",
    LeavesErrorCode.INVALID_INTENT_FORMAT: "Could not understand that intent format.",
    LeavesErrorCode.INFERENCE_UNAVAILABLE: (
        "The inference server is not running. "
        "Start it with: bash scripts/start-inference.sh"
    ),
    LeavesErrorCode.INFERENCE_TIMEOUT: "The inference server timed out. Try again or restart it.",
    LeavesErrorCode.INFERENCE_BAD_RESPONSE: "The model returned an unexpected response. Try rephrasing.",
    LeavesErrorCode.JSON_PARSE_FAILED: "Could not parse model output as JSON. Try rephrasing your intent.",
    LeavesErrorCode.SCHEMA_VALIDATION_FAILED: "Model output did not match the expected schema. Try rephrasing.",
    LeavesErrorCode.LOW_CONFIDENCE: (
        "I'm not confident enough about what you want. "
        "Try being more specific (e.g., 'find all PDFs in ~/Downloads')."
    ),
    LeavesErrorCode.PATH_NOT_AUTHORIZED: (
        "That path is outside your home directory. "
        "Leaves OS only operates within your home directory."
    ),
    LeavesErrorCode.PATH_DOES_NOT_EXIST: "That path does not exist.",
    LeavesErrorCode.PATH_TRAVERSAL_DETECTED: "Potential path traversal detected. Operation blocked.",
    LeavesErrorCode.FILE_READ_ERROR: "Could not read the file.",
    LeavesErrorCode.FILE_WRITE_ERROR: "Could not write the file.",
    LeavesErrorCode.FILE_DELETE_ERROR: "Could not delete the file.",
    LeavesErrorCode.FILE_MOVE_ERROR: "Could not move the file.",
    LeavesErrorCode.FILE_NOT_FOUND: "File not found.",
    LeavesErrorCode.PERMISSION_DENIED: "Permission denied.",
    LeavesErrorCode.INVALID_STATE_TRANSITION: "Invalid state transition in intent lifecycle.",
    LeavesErrorCode.INTENT_ALREADY_TERMINAL: "This intent has already completed or been cancelled.",
    LeavesErrorCode.TOOL_FAILURE_ESCALATED: (
        "A tool has failed too many times with the same arguments. "
        "The operation has been stopped to prevent a loop."
    ),
    LeavesErrorCode.MAX_RETRIES_EXCEEDED: "Maximum retry attempts exceeded.",
    LeavesErrorCode.AGENT_NOT_AVAILABLE: "That agent type is not available in this phase.",
    LeavesErrorCode.DEPENDENCY_FAILED: "A required action dependency failed. Skipping.",
    LeavesErrorCode.DB_ERROR: "A database error occurred. Check logs.",
    LeavesErrorCode.USER_CANCELLED: "Operation cancelled.",
    LeavesErrorCode.INTERNAL_ERROR: "An internal error occurred. This is a bug.",
    LeavesErrorCode.NOT_IMPLEMENTED: (
        "That type of task isn't implemented yet. "
        "Phase 1 supports file operations only. "
        "Email, web, system, and writing tasks are coming in Phase 2."
    ),
}

# Sanity check at import time — all codes must have user messages
_missing = [c for c in LeavesErrorCode if c not in USER_MESSAGES]
if _missing:
    raise RuntimeError(f"Missing USER_MESSAGES for: {_missing}")


class LeavesError(Exception):
    """
    All user-visible errors in Leaves OS.
    Always include a LeavesErrorCode. Never show stack traces to users.
    """

    def __init__(
        self,
        code: LeavesErrorCode,
        detail: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ):
        self.code = code
        self.detail = detail
        self.cause = cause
        self.user_message = USER_MESSAGES[code]
        super().__init__(f"[{code.value}] {detail or self.user_message}")

    def __repr__(self) -> str:
        return f"LeavesError(code={self.code.value!r}, detail={self.detail!r})"
