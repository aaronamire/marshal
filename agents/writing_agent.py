"""
WritingAgent — generates text content via inference.

Supports COMPOSE and WRITE action types. Optionally writes
the generated content to a file if params["path"] is present.
"""
from __future__ import annotations

import os
import pathlib

from agents.base_agent import BaseAgent
from config import MODEL_FAMILY
from errors import LeavesError, LeavesErrorCode
from inference.client import InferenceClient, InferenceRequest


class WritingAgent(BaseAgent):

    AGENT_TYPE = "writing"

    def execute_action(self, action: dict) -> dict:
        action_type = action.get("type", "").upper()
        if action_type in ("COMPOSE", "WRITE"):
            return self._compose(action, action.get("params", {}))
        raise LeavesError(
            LeavesErrorCode.AGENT_NOT_AVAILABLE,
            detail=f"WritingAgent: unsupported action type '{action_type}'",
        )

    def _compose(self, action: dict, params: dict) -> dict:
        action_id = action.get("action_id", "unknown")
        topic = params.get("topic", "")
        fmt = params.get("format", "text")
        path = params.get("path")

        if not topic:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail="WritingAgent: params.topic is required",
            )

        row_id = self._audit_start(action_id, "COMPOSE", params)

        try:
            client = InferenceClient()
            user_block = (
                f"Write the following in {fmt} format. "
                f"Output only the content, no preamble:\n{topic}"
            )

            if MODEL_FAMILY == "chatml":
                prompt = (
                    f"<|im_start|>system\nYou are a helpful writing assistant."
                    f"<|im_end|>\n"
                    f"<|im_start|>user\n{user_block}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                stop = ["<|im_end|>", "<|endoftext|>"]
            else:
                prompt = (
                    f"<|begin_of_text|>"
                    f"<|start_header_id|>system<|end_header_id|>\n"
                    f"You are a helpful writing assistant.\n<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n"
                    f"{user_block}\n<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n"
                )
                stop = ["<|eot_id|>"]

            req = InferenceRequest(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1024,
                stop_tokens=stop,
            )
            resp = client.complete(req)
            content = resp.content.strip()

            result: dict = {
                "content": content,
                "format": fmt,
                "topic": topic,
            }

            if path:
                expanded = os.path.expanduser(path)
                pathlib.Path(expanded).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(expanded).write_text(content)
                result["path"] = expanded
                result["bytes_written"] = len(content.encode())

            self._audit_end(row_id, result=result)
            return result

        except LeavesError:
            raise
        except Exception as e:
            err = LeavesError(
                LeavesErrorCode.INTERNAL_ERROR,
                detail=f"WritingAgent: {e}",
                cause=e,
            )
            self._audit_end(row_id, error=err)
            raise err
