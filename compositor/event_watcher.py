"""
CompositorEventWatcher — subscribes to the compositor event socket and
maintains a live SessionContext that the intent parser can inject into
L2 prompts.

The compositor writes newline-delimited JSON to:
    $XDG_RUNTIME_DIR/leaves-compositor-events.sock

Events:
    window_opened   {app_id, title, workspace, pid?, ts_ms}
    window_closed   {app_id, workspace, pid?, ts_ms}
    window_focused  {app_id, workspace, ts_ms}
    child_exited    {app_id, pid, exit_code, ts_ms}
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

COMPOSITOR_EVENTS_SOCK = pathlib.Path(
    os.environ.get("XDG_RUNTIME_DIR", "/tmp")
) / "leaves-compositor-events.sock"


@dataclass
class SessionContext:
    open_windows: list[dict] = field(default_factory=list)
    focused_window: dict | None = None
    recent_exits: list[dict] = field(default_factory=list)
    active_workspace: int = 0

    def to_prompt_block(self) -> str:
        """Serialize to a block for injection into the L2 system prompt."""
        lines = ["## Live session state"]
        if self.open_windows:
            parts = []
            for w in self.open_windows:
                label = w.get("app_id", "?")
                if (self.focused_window and
                        w.get("app_id") == self.focused_window.get("app_id")):
                    label += " (focused)"
                parts.append(label)
            lines.append(f"Open windows: {', '.join(parts)}")
        else:
            lines.append("Open windows: none")

        if self.recent_exits:
            last = self.recent_exits[-1]
            age = int(time.time() - last.get("ts", time.time()))
            code = last.get("exit_code", 0)
            if code != 0:
                lines.append(
                    f"Recent: {last.get('app_id', '?')} exited with "
                    f"code {code} ({age}s ago)"
                )

        lines.append(f"Workspace: {self.active_workspace + 1}/4")
        return "\n".join(lines)


class CompositorEventWatcher:
    def __init__(self):
        self.context = SessionContext()
        self._running = False
        self._task: asyncio.Task | None = None
        self._exit_callbacks: list[
            Callable[[str, int, pathlib.Path], Coroutine]
        ] = []

    def on_nonzero_exit(
        self,
        callback: Callable[[str, int, pathlib.Path], Coroutine],
    ):
        """Register async callback(app_id, exit_code, scrollback_path)."""
        self._exit_callbacks.append(callback)

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._watch())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _watch(self):
        while self._running:
            try:
                reader, writer = await asyncio.open_unix_connection(
                    str(COMPOSITOR_EVENTS_SOCK))
                try:
                    async for line in reader:
                        try:
                            event = json.loads(line.decode().strip())
                            self._handle(event)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
                finally:
                    writer.close()
                    await writer.wait_closed()
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                await asyncio.sleep(2)

    def _handle(self, event: dict):
        t = event.get("type")
        app_id = event.get("app_id", "")

        if t == "window_opened":
            entry = {
                "app_id": app_id,
                "title": event.get("title", ""),
                "workspace": event.get("workspace", 0),
                "ts": time.time(),
            }
            self.context.open_windows = [
                w for w in self.context.open_windows
                if w["app_id"] != app_id
            ]
            self.context.open_windows.append(entry)
            self.context.active_workspace = event.get("workspace", 0)

        elif t == "window_closed":
            self.context.open_windows = [
                w for w in self.context.open_windows
                if w["app_id"] != app_id
            ]
            if (self.context.focused_window and
                    self.context.focused_window.get("app_id") == app_id):
                self.context.focused_window = None

        elif t == "window_focused":
            self.context.focused_window = {"app_id": app_id}
            self.context.active_workspace = event.get("workspace", 0)

        elif t == "child_exited":
            exit_code = event.get("exit_code", 0)
            record = {
                "app_id": app_id,
                "pid": event.get("pid"),
                "exit_code": exit_code,
                "ts": time.time(),
            }
            self.context.recent_exits.append(record)
            self.context.recent_exits = self.context.recent_exits[-10:]

            if exit_code != 0 and self._exit_callbacks:
                scrollback = (pathlib.Path.home() / ".leaves" /
                              "terminal-scrollback.txt")
                for cb in self._exit_callbacks:
                    asyncio.create_task(cb(app_id, exit_code, scrollback))
