"""
Briefing generator — "what changed while you were away?"

Queries the knowledge graph for recent changes, groups them by
source type and parent directory, and returns a structured briefing.

Template-based (no LLM dependency) for speed and reliability.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from cortex.knowledge_graph import KnowledgeGraph

log = logging.getLogger("cortex.briefing")

# Collapse directories with fewer than this many items into "other"
_MIN_GROUP_SIZE = 2
# Max items to pull from temporal query
_MAX_ITEMS = 200
# Max directory groups to show per source type
_MAX_DIR_GROUPS = 8


class BriefingGenerator:
    """
    Generates a structured briefing from recent knowledge graph changes.

    Usage:
        bg = BriefingGenerator(kg)
        briefing = bg.generate(hours=12)
    """

    def __init__(self, kg: KnowledgeGraph):
        self._kg = kg

    def generate(self, hours: int = 12) -> dict[str, Any]:
        """
        Generate a briefing covering the last `hours` hours.

        Returns:
            {
                "period_hours": 12,
                "since": "2026-03-25T00:00:00+00:00",
                "generated_at": "2026-03-25T12:00:00+00:00",
                "total_changes": 42,
                "sections": [
                    {
                        "source_type": "file",
                        "count": 42,
                        "groups": [
                            {"directory": "~/dev/leaves-os", "count": 15, "items": [...]},
                            ...
                        ]
                    }
                ],
                "headline": "42 files changed across 3 directories",
                "empty": False,
            }
        """
        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=hours)).isoformat()

        items = self._kg.temporal_query(since=since, limit=_MAX_ITEMS)

        if not items:
            return self._empty_briefing(hours, since, now)

        sections = self._group_items(items)
        total = sum(s["count"] for s in sections)
        headline = self._make_headline(sections, total, hours)

        return {
            "period_hours": hours,
            "since": since,
            "generated_at": now.isoformat(),
            "total_changes": total,
            "sections": sections,
            "headline": headline,
            "empty": False,
        }

    def status(self) -> dict[str, Any]:
        """Quick check: is the knowledge graph populated?"""
        stats = self._kg.stats()
        total = sum(s["count"] for s in stats.values())
        return {
            "indexed": total > 0,
            "total_items": total,
            "sources": stats,
        }

    def _empty_briefing(
        self, hours: int, since: str, now: datetime
    ) -> dict[str, Any]:
        stats = self._kg.stats()
        total_indexed = sum(s["count"] for s in stats.values())

        if total_indexed == 0:
            headline = "No files indexed yet. Run indexing to get started."
        else:
            headline = f"No changes in the last {hours} hours. All quiet."

        return {
            "period_hours": hours,
            "since": since,
            "generated_at": now.isoformat(),
            "total_changes": 0,
            "sections": [],
            "headline": headline,
            "empty": True,
        }

    def _group_items(self, items: list[dict]) -> list[dict[str, Any]]:
        """Group items by source_type, then cluster by parent directory."""
        by_type: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            by_type[item.get("source_type", "unknown")].append(item)

        sections = []
        for source_type, type_items in sorted(by_type.items()):
            groups = self._cluster_by_directory(type_items)
            sections.append({
                "source_type": source_type,
                "count": len(type_items),
                "groups": groups,
            })

        return sections

    def _cluster_by_directory(self, items: list[dict]) -> list[dict[str, Any]]:
        """Cluster items by parent directory, collapsing small groups."""
        by_dir: dict[str, list[dict]] = defaultdict(list)

        for item in items:
            path = item.get("source_path", "")
            parent = str(Path(path).parent) if path else "unknown"
            # Collapse home prefix for readability
            parent = _collapse_home(parent)
            by_dir[parent].append(item)

        groups = []
        other_items: list[dict] = []

        for directory, dir_items in sorted(by_dir.items(), key=lambda x: -len(x[1])):
            if len(dir_items) < _MIN_GROUP_SIZE:
                other_items.extend(dir_items)
            elif len(groups) < _MAX_DIR_GROUPS:
                groups.append({
                    "directory": directory,
                    "count": len(dir_items),
                    "items": [
                        _item_summary(i) for i in dir_items[:5]
                    ],
                })
            else:
                other_items.extend(dir_items)

        if other_items:
            groups.append({
                "directory": "other",
                "count": len(other_items),
                "items": [
                    _item_summary(i) for i in other_items[:5]
                ],
            })

        return groups

    def _make_headline(
        self, sections: list[dict], total: int, hours: int
    ) -> str:
        """Generate a one-line summary."""
        if len(sections) == 1:
            s = sections[0]
            n_dirs = len(s["groups"])
            dir_word = "directory" if n_dirs == 1 else "directories"
            return (
                f"{total} {s['source_type']}(s) changed "
                f"across {n_dirs} {dir_word} "
                f"in the last {_fmt_hours(hours)}"
            )

        parts = []
        for s in sections:
            parts.append(f"{s['count']} {s['source_type']}(s)")
        joined = ", ".join(parts)
        return f"{total} changes ({joined}) in the last {_fmt_hours(hours)}"


def _collapse_home(path: str) -> str:
    """Replace /home/<user> with ~."""
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def _item_summary(item: dict) -> dict[str, Any]:
    """Extract a compact summary from a knowledge_items row."""
    metadata = {}
    if item.get("metadata_json"):
        try:
            metadata = json.loads(item["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "title": item.get("title", ""),
        "path": _collapse_home(item.get("source_path", "")),
        "timestamp": item.get("timestamp", ""),
        "extension": metadata.get("extension", ""),
    }


def _fmt_hours(hours: int) -> str:
    """Human-readable time period."""
    if hours < 1:
        return f"{hours * 60:.0f} minutes"
    if hours == 1:
        return "hour"
    if hours < 24:
        return f"{hours} hours"
    days = hours // 24
    if days == 1:
        return "day"
    return f"{days} days"
