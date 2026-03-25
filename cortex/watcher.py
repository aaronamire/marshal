"""
Filesystem watcher — monitors paths referenced by persistent intents.

Uses watchdog (inotify wrapper). One Observer thread watches all paths.
When an event matches a trigger, the persistent intent fires through
the normal agent execution pipeline (including Landlock sandbox).

Memory: watchdog is lightweight — one Observer + inotify watches.
inotify kernel limit: /proc/sys/fs/inotify/max_user_watches (default 65536).
"""
from __future__ import annotations

import fnmatch
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config import AUDIT_DB_PATH
from db.intent_store import fire_intent, get_active_intents

log = logging.getLogger("cortex.watcher")

# Map watchdog event types to our trigger event names
_EVENT_TYPE_MAP = {
    "created": "created",
    "modified": "modified",
    "moved": "moved",
    "deleted": "deleted",
}

# Minimum seconds between fires of the same intent (debounce)
_DEBOUNCE_SECONDS = 5.0


class _IntentEventHandler(FileSystemEventHandler):
    """Watchdog handler that matches events against persistent intent triggers."""

    def __init__(
        self,
        intents: list[dict[str, Any]],
        db_path: Path,
        on_fire: Callable[[dict], None],
    ):
        super().__init__()
        self._intents = intents
        self._db_path = db_path
        self._on_fire = on_fire
        self._last_fired: dict[str, float] = {}  # intent_id -> monotonic timestamp
        self._lock = threading.Lock()
        self._thread_db = None  # lazily opened in observer thread

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        event_type = _EVENT_TYPE_MAP.get(event.event_type)
        if event_type is None:
            return

        for intent in self._intents:
            if self._matches(event, event_type, intent):
                self._try_fire(intent)

    def _matches(
        self, event: FileSystemEvent, event_type: str, intent: dict
    ) -> bool:
        """Check if a watchdog event matches an intent's trigger config."""
        config = intent.get("trigger_config") or {}
        watch_path = config.get("path", "")

        # Check the event is under the watched path
        event_path = str(event.src_path)
        if not event_path.startswith(watch_path):
            return False

        # Check event type
        allowed_events = config.get("events", ["created"])
        if event_type not in allowed_events:
            return False

        # Check filename pattern if specified
        pattern = config.get("pattern")
        if pattern:
            filename = Path(event.src_path).name
            if not fnmatch.fnmatch(filename, pattern):
                return False

        return True

    def _get_thread_db(self):
        """Get or create a SQLite connection for the observer thread."""
        if self._thread_db is None:
            import sqlite3
            self._thread_db = sqlite3.connect(str(self._db_path))
            self._thread_db.row_factory = sqlite3.Row
            self._thread_db.execute("PRAGMA journal_mode=WAL")
            self._thread_db.execute("PRAGMA foreign_keys=ON")
        return self._thread_db

    def _try_fire(self, intent: dict) -> None:
        """Fire an intent if debounce period has passed."""
        intent_id = intent["id"]
        now = time.monotonic()

        with self._lock:
            last = self._last_fired.get(intent_id, 0.0)
            if now - last < _DEBOUNCE_SECONDS:
                log.debug("Debounced intent %s (%.1fs since last)", intent_id[:8], now - last)
                return
            self._last_fired[intent_id] = now

        log.info("Firing intent %s (%s)", intent_id[:8], intent["name"])

        try:
            db = self._get_thread_db()
            fire_intent(db, intent_id)
            goalspec = intent.get("goalspec")
            if goalspec:
                self._on_fire(goalspec)
        except Exception:
            log.exception("Failed to fire intent %s", intent_id[:8])


class IntentWatcher:
    """
    Watches filesystem paths for persistent intents and fires them on match.

    Usage:
        watcher = IntentWatcher(db, coordinator_fn)
        watcher.start()   # starts background thread
        watcher.reload()  # call when persistent intents change
        watcher.stop()    # teardown
    """

    def __init__(self, db, on_fire: Callable[[dict], None], db_path: Optional[Path] = None):
        """
        Args:
            db: SQLite connection for reading intents (main thread).
            on_fire: Called with the GoalSpec dict when an intent fires.
                     Typically AgentCoordinator.execute or similar.
            db_path: Path to the SQLite DB file. Used by the observer thread
                     to open its own connection. Defaults to AUDIT_DB_PATH.
        """
        self._db = db
        self._on_fire = on_fire
        self._db_path = db_path or AUDIT_DB_PATH
        self._observer: Optional[Observer] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Load filesystem-triggered intents and start watching."""
        with self._lock:
            self._start_observer()
        log.info("IntentWatcher started")

    def stop(self) -> None:
        """Stop all watches and the observer thread."""
        with self._lock:
            self._stop_observer()
        log.info("IntentWatcher stopped")

    def reload(self) -> None:
        """Reload watches from the database (call after adding/removing intents)."""
        with self._lock:
            self._stop_observer()
            self._start_observer()
        log.info("IntentWatcher reloaded")

    def _start_observer(self) -> None:
        """Internal: read intents, set up watches, start observer."""
        intents = get_active_intents(self._db, trigger_type="filesystem")
        if not intents:
            log.info("No filesystem-triggered intents — watcher idle")
            return

        handler = _IntentEventHandler(intents, self._db_path, self._on_fire)
        self._observer = Observer()

        # Collect unique paths to watch
        watched_paths: set[str] = set()
        for intent in intents:
            config = intent.get("trigger_config") or {}
            path = config.get("path")
            if path and Path(path).is_dir():
                watched_paths.add(path)

        for path in watched_paths:
            try:
                self._observer.schedule(handler, path, recursive=False)
                log.info("Watching: %s", path)
            except Exception:
                log.exception("Failed to watch %s", path)

        if watched_paths:
            self._observer.daemon = True
            self._observer.start()

    def _stop_observer(self) -> None:
        """Internal: stop and join the observer if running."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
