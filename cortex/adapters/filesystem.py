"""
Filesystem adapter — indexes text files under ~/.

Scans: .txt, .md, .py, .js, .ts, .rs, .c, .cpp, .h, .java, .go,
       .sh, .bash, .zsh, .json, .yaml, .yml, .toml, .ini, .cfg,
       .csv, .html, .css, .xml, .sql, .tex, .org, .rst, .log, .conf
Skips: .git/, node_modules/, __pycache__/, .cache/, .mozilla/, .thunderbird/,
       any dotdir except .config/ and .ssh/

Content: read as UTF-8 (lossy), truncated to 32KB.
Files > 1MB are skipped entirely.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from cortex.adapters.base import BaseAdapter, KnowledgeItem

INDEXABLE_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".rs", ".c", ".cpp", ".h",
    ".java", ".go", ".sh", ".bash", ".zsh", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".csv", ".html", ".css", ".xml", ".sql",
    ".tex", ".org", ".rst", ".log", ".conf",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".cache", ".local",
    ".mozilla", ".thunderbird", ".npm", ".cargo", ".rustup",
    "venv", ".venv", ".os", ".lancedb",
}

# Dotdirs we DO index
KEEP_DOTDIRS = {".config", ".ssh"}

MAX_FILE_SIZE = 1_000_000   # 1MB — skip larger files
MAX_CONTENT_LEN = 32 * 1024  # 32KB — truncate content for embedding


class FilesystemAdapter(BaseAdapter):

    def __init__(self, root: Optional[Path] = None):
        self._root = root or Path.home()

    def source_type(self) -> str:
        return "file"

    def is_available(self) -> bool:
        return self._root.exists()

    def scan(self, since: Optional[str] = None) -> Iterator[KnowledgeItem]:
        since_dt = datetime.fromisoformat(since) if since else None

        for root, dirs, files in os.walk(self._root):
            root_path = Path(root)

            # Prune skip dirs
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS
                and (not d.startswith(".") or d in KEEP_DOTDIRS)
            ]

            for fname in files:
                fpath = root_path / fname
                suffix = fpath.suffix.lower()
                if suffix not in INDEXABLE_EXTENSIONS:
                    continue

                try:
                    stat = fpath.stat()
                except (OSError, PermissionError):
                    continue

                if stat.st_size > MAX_FILE_SIZE:
                    continue
                if stat.st_size == 0:
                    continue

                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                if since_dt and mtime < since_dt:
                    continue

                try:
                    content = fpath.read_text(errors="replace")[:MAX_CONTENT_LEN]
                except (OSError, PermissionError):
                    continue

                content_hash = self.hash_content(content)

                yield KnowledgeItem(
                    source_type="file",
                    source_id=str(fpath),
                    source_path=str(fpath),
                    title=fpath.name,
                    content=content,
                    content_hash=content_hash,
                    metadata={
                        "size": stat.st_size,
                        "mtime": mtime.isoformat(),
                        "extension": suffix,
                        "parent_dir": str(fpath.parent),
                    },
                    timestamp=mtime.isoformat(),
                )
