#!/usr/bin/env python3
"""
Synthetic GoalSpec training-data generator (v2).

Goal: ~4500 high-quality (user_text, goal_spec) pairs for fine-tuning the
intent parser, with strategic per-category distribution that closes the
known gaps (audio/network/power had 0 pairs; system/web were thin).

Distribution target:
   1000 file_task           700 system_task        700 web_task
    400 audio_task          400 network_task       400 power_task
    300 writing_task        200 multi-action (DAG) 200 NOT_IMPL refusals
    200 ambiguous/edge

Every pair is validated against agents/schema/goal_spec.json before write,
and every (user_text.lower()) key is deduped across the whole corpus.

Output: data/goalspec_training_v2.jsonl (preserves the existing corpus
under goalspec_training_pairs.jsonl for diff/comparison).

Format mirrors the existing corpus: assistant content is the GoalSpec
JSON minus `intent_id` and `metadata` (these are injected/added at
runtime; the model is fine-tuned without them).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter
from typing import Iterable

import jsonschema

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SCHEMA_PATH = _PROJECT_ROOT / "agents" / "schema" / "goal_spec.json"
_OUT_PATH = _PROJECT_ROOT / "data" / "goalspec_training_v2.jsonl"

with open(_SCHEMA_PATH) as f:
    _SCHEMA = json.load(f)

# Synthetic intent_id used only for schema validation. The training pairs
# themselves never contain intent_id (matches existing corpus format).
_VALIDATION_UUID = "00000000-0000-4000-8000-000000000000"


# ---------------------------------------------------------------------------
# Path / phrase pools
# ---------------------------------------------------------------------------

_HOME_DIRS = [
    "~", "~/Documents", "~/Downloads", "~/Desktop", "~/Pictures",
    "~/Music", "~/Videos", "~/Projects", "~/dev", "~/notes",
    "~/work", "~/tmp", "~/.config", "~/.local/share", "~/.cache",
    "~/Documents/work", "~/Documents/personal", "~/Downloads/archive",
    "~/Pictures/screenshots", "~/Music/playlists", "~/dev/leaves-os",
    "~/dev/scratch", "~/projects/website", "~/projects/api-server",
]

_ABS_DIRS = [
    "/tmp", "/var/log", "/etc", "/usr/share", "/opt",
    "/var/cache", "/var/tmp", "/srv", "/mnt/data",
]

_FILE_NAMES = [
    "notes.md", "readme.txt", "todo.md", "config.yaml", "settings.json",
    "draft.txt", "report.pdf", "meeting.md", "diary.txt", "scratch.txt",
    "summary.md", "outline.md", "journal.md", "ideas.txt",
    "credentials.env", "backup.tar.gz", "data.csv", "results.json",
    "log.txt", "debug.log", "access.log", "error.log",
    "main.py", "app.py", "server.py", "client.py", "utils.py",
    "test_main.py", "Makefile", "Dockerfile", "package.json",
    "requirements.txt", "pyproject.toml", "Cargo.toml",
    "index.html", "style.css", "script.js", "README.md",
    "old.txt", "new.txt", "tmp.txt", "junk.txt", "trash.txt",
    "photo.jpg", "screenshot.png", "diagram.svg", "icon.ico",
    "video.mp4", "audio.mp3", "track.flac", "podcast.ogg",
    "presentation.pdf", "thesis.pdf", "paper.pdf", "ebook.epub",
]

_PATTERNS = [
    "*.py", "*.txt", "*.md", "*.log", "*.pdf", "*.jpg", "*.png",
    "*.json", "*.yaml", "*.toml", "*.csv", "*.html", "*.css", "*.js",
    "*.tar.gz", "*.zip", "*.mp3", "*.mp4", "*.tmp", "*.bak",
    "*~", ".*", "**/*.py", "**/*.log", "*.conf",
]

_PATTERN_NAMES = {
    "*.py": "Python files",
    "*.txt": "text files",
    "*.md": "markdown files",
    "*.log": "log files",
    "*.pdf": "PDFs",
    "*.jpg": "JPEGs",
    "*.png": "PNG images",
    "*.json": "JSON files",
    "*.yaml": "YAML files",
    "*.toml": "TOML files",
    "*.csv": "CSV files",
    "*.html": "HTML files",
    "*.css": "CSS files",
    "*.js": "JavaScript files",
    "*.tar.gz": "tarballs",
    "*.zip": "ZIP archives",
    "*.mp3": "MP3 files",
    "*.mp4": "videos",
    "*.tmp": "temp files",
    "*.bak": "backup files",
    "*~": "editor backups",
    ".*": "hidden files",
    "**/*.py": "all Python files",
    "**/*.log": "all log files",
    "*.conf": "config files",
}

_DIR_NAMES = {
    "~": "home directory",
    "~/Documents": "Documents folder",
    "~/Downloads": "Downloads folder",
    "~/Desktop": "Desktop",
    "~/Pictures": "Pictures folder",
    "~/Music": "Music folder",
    "~/Videos": "Videos folder",
    "~/Projects": "Projects folder",
    "~/dev": "dev folder",
    "~/notes": "notes folder",
    "~/work": "work folder",
    "~/tmp": "temp folder",
    "/tmp": "/tmp",
    "/var/log": "system log directory",
    "/etc": "/etc",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(spec: dict) -> dict:
    """Return spec with synthetic intent_id for schema validation."""
    return {**spec, "intent_id": _VALIDATION_UUID}


def _validate(spec: dict) -> None:
    """Schema-validate; raises on failure."""
    jsonschema.validate(_wrap(spec), _SCHEMA)


def _spec(natural_text: str, category: str, actions: list[dict],
          resources: list[str], preview: bool, reversible: bool) -> dict:
    """Build a GoalSpec with the canonical fields the existing corpus uses."""
    # Existing corpus convention: every action has depends_on (possibly []).
    for a in actions:
        a.setdefault("depends_on", [])
    return {
        "natural_text": natural_text,
        "category": category,
        "actions": actions,
        "authorization": {
            "resources": resources,
            "preview_required": preview,
            "reversible": reversible,
        },
    }


def _pair(user: str, spec: dict) -> dict:
    """Build the JSONL line for the SFT training format."""
    return {
        "messages": [
            {"role": "user", "content": user},
            # Compact JSON to match the existing corpus byte-for-byte.
            {"role": "assistant",
             "content": json.dumps(spec, separators=(",", ":"))},
        ]
    }


def _rand_path(rng: random.Random) -> str:
    """A random ~ or absolute directory path."""
    return rng.choice(_HOME_DIRS + _ABS_DIRS)


def _rand_home_dir(rng: random.Random) -> str:
    return rng.choice(_HOME_DIRS)


def _rand_file_path(rng: random.Random) -> str:
    """A random ~/dir/file.ext path."""
    base = rng.choice(_HOME_DIRS)
    name = rng.choice(_FILE_NAMES)
    if base == "~":
        return f"~/{name}"
    return f"{base}/{name}"


# ---------------------------------------------------------------------------
# file_task generators
# ---------------------------------------------------------------------------

_READ_VERBS = [
    "read {p}",
    "show me the contents of {p}",
    "show me {p}",
    "open {p}",
    "view {p}",
    "cat {p}",
    "print {p}",
    "display {p}",
    "what's in {p}",
    "what is in {p}",
    "let me see {p}",
    "can you read {p}",
    "could you open {p}",
    "please show me {p}",
    "I want to see what's in {p}",
    "load up {p}",
    "pull up {p}",
    "what does {p} say",
    "show the contents of {p}",
    "read me {p}",
    "render {p}",
    "preview {p}",
]


def _gen_file_read(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for _ in range(450):
        verb = rng.choice(_READ_VERBS)
        path = _rand_file_path(rng)
        text = verb.format(p=path)
        yield text, _spec(
            natural_text=text,
            category="file_task",
            actions=[{
                "action_id": "act-1", "type": "READ", "agent": "file",
                "params": {"path": path}, "destructive": False,
            }],
            resources=[path], preview=False, reversible=True,
        )


_QUERY_VERBS_DIR = [
    "list {d}",
    "ls {d}",
    "show me what's in {d}",
    "what's in {d}",
    "show contents of {d}",
    "list the files in {d}",
    "list files in {d}",
    "what files are in {d}",
    "show me {d}",
    "browse {d}",
    "look in {d}",
    "scan {d}",
]

_QUERY_VERBS_PATTERN_DIR = [
    "find {pat} in {d}",
    "list all {patname} in {d}",
    "find all {patname} in {d}",
    "show me {patname} in {d}",
    "show all {patname} in {d}",
    "search for {pat} in {d}",
    "find files matching {pat} in {d}",
    "what {patname} are in {d}",
    "look for {patname} in {d}",
]


def _gen_file_query(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for _ in range(220):
        d = _rand_path(rng)
        verb = rng.choice(_QUERY_VERBS_DIR)
        text = verb.format(d=d)
        yield text, _spec(
            natural_text=text,
            category="file_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "file",
                "params": {"path": d, "recursive": False},
                "destructive": False,
            }],
            resources=[d], preview=False, reversible=True,
        )

    for _ in range(220):
        pat = rng.choice([p for p in _PATTERNS if p in _PATTERN_NAMES])
        patname = _PATTERN_NAMES[pat]
        d = rng.choice(_HOME_DIRS + _ABS_DIRS[:3])
        recursive = pat.startswith("**/") or rng.random() < 0.45
        verb = rng.choice(_QUERY_VERBS_PATTERN_DIR)
        text = verb.format(pat=pat, patname=patname, d=d)
        yield text, _spec(
            natural_text=text,
            category="file_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "file",
                "params": {"path": d, "pattern": pat, "recursive": recursive},
                "destructive": False,
            }],
            resources=[d], preview=False, reversible=True,
        )


_DELETE_VERBS = [
    "delete {p}", "remove {p}", "rm {p}", "erase {p}",
    "trash {p}", "get rid of {p}", "wipe {p}",
    "delete the file at {p}", "remove the file {p}",
    "please delete {p}", "I want to remove {p}",
    "throw away {p}", "kill {p}",
]


def _gen_file_delete(rng: random.Random) -> Iterable[tuple[str, dict]]:
    # Single-file delete (single DELETE action — matches existing corpus
    # for explicit-path deletes; the QUERY+DELETE compound is in
    # gen_multi_action).
    for _ in range(110):
        path = _rand_file_path(rng)
        verb = rng.choice(_DELETE_VERBS)
        text = verb.format(p=path)
        yield text, _spec(
            natural_text=text,
            category="file_task",
            actions=[{
                "action_id": "act-1", "type": "DELETE", "agent": "file",
                "params": {"path": path}, "destructive": True,
            }],
            resources=[path], preview=True, reversible=False,
        )


_MOVE_VERBS = [
    "move {a} to {b}", "mv {a} {b}",
    "rename {a} to {b}",
    "shift {a} into {b}",
    "relocate {a} to {b}",
    "transfer {a} to {b}",
    "send {a} to {b}",
]


def _gen_file_move(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for _ in range(90):
        a = _rand_file_path(rng)
        b_dir = _rand_home_dir(rng)
        # Sometimes destination is a directory, sometimes a renamed file.
        if rng.random() < 0.4:
            b = b_dir
        else:
            stem = pathlib.PurePosixPath(a).stem
            ext = pathlib.PurePosixPath(a).suffix
            b = f"{b_dir}/{stem}-renamed{ext}"
        verb = rng.choice(_MOVE_VERBS)
        text = verb.format(a=a, b=b)
        yield text, _spec(
            natural_text=text,
            category="file_task",
            actions=[{
                "action_id": "act-1", "type": "MOVE", "agent": "file",
                "params": {"source": a, "destination": b},
                "destructive": True,
            }],
            resources=[a, b], preview=True, reversible=False,
        )


_COPY_VERBS = [
    "copy {a} to {b}", "cp {a} {b}",
    "duplicate {a} to {b}",
    "make a copy of {a} at {b}",
    "back up {a} to {b}",
    "clone {a} to {b}",
]


def _gen_file_copy(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for _ in range(90):
        a = _rand_file_path(rng)
        b_dir = _rand_home_dir(rng)
        stem = pathlib.PurePosixPath(a).stem
        ext = pathlib.PurePosixPath(a).suffix
        b = f"{b_dir}/{stem}-copy{ext}" if rng.random() < 0.6 else b_dir
        verb = rng.choice(_COPY_VERBS)
        text = verb.format(a=a, b=b)
        yield text, _spec(
            natural_text=text,
            category="file_task",
            actions=[{
                "action_id": "act-1", "type": "COPY", "agent": "file",
                "params": {"source": a, "destination": b},
                "destructive": True,
            }],
            resources=[a, b], preview=True, reversible=False,
        )


_WRITE_VERBS_EMPTY = [
    "create file {p}", "create a file at {p}",
    "touch {p}", "make a new file {p}",
    "create an empty file at {p}", "new file {p}",
]


def _gen_file_write(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for _ in range(80):
        path = _rand_file_path(rng)
        verb = rng.choice(_WRITE_VERBS_EMPTY)
        text = verb.format(p=path)
        # Empty-file create is non-destructive (no overwrite of existing data)
        # — matches the L0 rule.
        yield text, _spec(
            natural_text=text,
            category="file_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "file",
                "params": {"path": path, "content": ""},
                "destructive": False,
            }],
            resources=[path], preview=False, reversible=True,
        )


def gen_file_task(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    items.extend(_gen_file_read(rng))
    items.extend(_gen_file_query(rng))
    items.extend(_gen_file_delete(rng))
    items.extend(_gen_file_move(rng))
    items.extend(_gen_file_copy(rng))
    items.extend(_gen_file_write(rng))
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# system_task generators
# ---------------------------------------------------------------------------

_SYS_QUERIES = {
    "cpu": [
        "show cpu usage", "what's the cpu load", "how's the cpu doing",
        "cpu usage", "current cpu load", "show me cpu stats",
        "how much cpu am I using", "is the cpu busy",
        "check cpu utilization", "show processor load",
        "what's my cpu at", "cpu status",
    ],
    "memory": [
        "how much memory is being used", "show memory usage",
        "memory status", "ram usage", "how much ram is free",
        "check memory", "show me memory stats", "what's my ram at",
        "is memory full", "how much free memory do I have",
        "show ram", "memory consumption",
    ],
    "disk": [
        "how much disk space is left", "show disk usage", "disk status",
        "how much storage do I have", "check disk space",
        "show me free disk space", "is the disk full",
        "what's disk utilization", "disk free", "df",
        "show storage", "how full is my drive",
    ],
    "processes": [
        "show running processes", "list processes",
        "what's running right now", "show me all processes",
        "what processes are active", "ps", "list all running tasks",
        "show me what's running", "process list", "running tasks",
        "what programs are running",
    ],
    "uptime": [
        "how long has this been running", "show uptime", "uptime",
        "how long has the system been up", "when did this boot",
        "system uptime", "how long since last reboot",
        "boot time", "since when has it been on",
    ],
}


def _gen_system_query(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for qt, phrases in _SYS_QUERIES.items():
        for phrase in phrases:
            yield phrase, _spec(
                natural_text=phrase,
                category="system_task",
                actions=[{
                    "action_id": "act-1", "type": "QUERY", "agent": "system",
                    "params": {"query_type": qt}, "destructive": False,
                }],
                resources=[], preview=False, reversible=True,
            )
        # Add variations with hedges/qualifiers
        for _ in range(35):
            base = rng.choice(phrases)
            wrap = rng.choice([
                "could you {b}", "can you {b}", "please {b}",
                "I'd like to {b}", "tell me, {b}", "{b}, please",
                "quickly {b}", "just {b}",
            ])
            text = wrap.format(b=base)
            yield text, _spec(
                natural_text=text,
                category="system_task",
                actions=[{
                    "action_id": "act-1", "type": "QUERY", "agent": "system",
                    "params": {"query_type": qt}, "destructive": False,
                }],
                resources=[], preview=False, reversible=True,
            )


_APPS = [
    "firefox", "chromium", "google-chrome", "brave", "vivaldi",
    "code", "vim", "nvim", "emacs", "nano", "gedit",
    "gimp", "inkscape", "krita", "blender",
    "vlc", "mpv", "spotify", "rhythmbox",
    "thunderbird", "evolution",
    "libreoffice", "gnumeric", "abiword",
    "kitty", "alacritty", "wezterm", "konsole", "gnome-terminal",
    "obs", "audacity", "kdenlive",
    "discord", "slack", "telegram-desktop", "signal-desktop",
    "calculator", "gnome-calculator", "gnome-files", "nautilus",
    "thunar", "dolphin",
    "steam", "lutris",
    "zoom", "teams",
    "htop", "btop", "neofetch",
]

_LAUNCH_VERBS = [
    "open {a}", "launch {a}", "start {a}", "run {a}",
    "fire up {a}", "boot up {a}", "spin up {a}",
    "please open {a}", "can you launch {a}",
    "could you start {a}", "I need {a} open", "start up {a}",
]


def _gen_system_launch(rng: random.Random) -> Iterable[tuple[str, dict]]:
    seen = set()
    for app in _APPS:
        for verb in _LAUNCH_VERBS:
            text = verb.format(a=app)
            if text in seen:
                continue
            seen.add(text)
            yield text, _spec(
                natural_text=text,
                category="system_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "system",
                    "params": {"program": app}, "destructive": False,
                }],
                resources=[], preview=False, reversible=True,
            )


_TERMINATE_VERBS = [
    "close {a}", "kill {a}", "quit {a}", "stop {a}", "exit {a}",
    "terminate {a}", "shut down {a}", "end {a}",
    "force quit {a}", "please close {a}", "can you kill {a}",
    "shut {a} down", "make {a} stop",
]


def _gen_system_terminate(rng: random.Random) -> Iterable[tuple[str, dict]]:
    seen = set()
    for app in _APPS:
        for verb in _TERMINATE_VERBS:
            text = verb.format(a=app)
            if text in seen:
                continue
            seen.add(text)
            yield text, _spec(
                natural_text=text,
                category="system_task",
                actions=[{
                    "action_id": "act-1", "type": "DELETE", "agent": "system",
                    "params": {"target": app}, "destructive": True,
                }],
                resources=[], preview=True, reversible=False,
            )


def gen_system_task(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    items.extend(_gen_system_query(rng))
    items.extend(_gen_system_launch(rng))
    items.extend(_gen_system_terminate(rng))
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# web_task generators
# ---------------------------------------------------------------------------

_SEARCH_TOPICS = [
    "rust async runtime comparison", "python type hints best practices",
    "how to fine-tune llama models", "kubernetes ingress vs gateway",
    "the weather in San Francisco", "espresso brewing techniques",
    "neovim lsp setup", "tailwind dark mode tutorial",
    "best mechanical keyboards 2026", "vim regex tutorial",
    "git rebase vs merge", "docker compose v2 changes",
    "react server components", "wayland vs x11",
    "linux kernel 6.18 release notes", "claude api pricing",
    "gpt-4 vs claude opus benchmarks", "self hosted password manager",
    "homelab nas build", "raspberry pi 5 review",
    "rust borrow checker explained", "postgres index types",
    "sqlite full text search", "lancedb vs qdrant",
    "speculative decoding explained", "lora fine tuning tutorial",
    "qwen 2.5 model card", "llama.cpp grammar guide",
    "best espresso machine under 1000", "ai-native operating systems",
    "writing a wayland compositor", "cgroups v2 tutorial",
    "landlock lsm guide", "seccomp filters explained",
    "io_uring performance", "ebpf for monitoring",
    "fastapi vs flask 2026", "modern python project layout",
    "uv vs poetry", "ruff vs black",
    "pytest fixtures patterns", "mypy strict mode tips",
    "hyprland config examples", "river compositor setup",
    "btrfs vs zfs", "nixos beginner guide",
    "arch linux install 2026", "alpine linux on desktop",
    "self-instruct paper summary", "constitutional ai paper",
    "model card best practices", "llm eval frameworks",
    "vllm vs sglang", "ollama vs lm studio",
]

_SEARCH_VERBS = [
    "google {q}", "search the web for {q}", "search for {q}",
    "look up {q}", "web search {q}", "search {q}",
    "find me info on {q}", "tell me about {q}",
    "what is {q}", "explain {q}",
    "show me search results for {q}", "I want to learn about {q}",
    "can you google {q}", "please search for {q}",
    "look this up: {q}", "what does the web say about {q}",
]


def _gen_web_search(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for topic in _SEARCH_TOPICS:
        for verb in _SEARCH_VERBS:
            text = verb.format(q=topic)
            yield text, _spec(
                natural_text=text,
                category="web_task",
                actions=[{
                    "action_id": "act-1", "type": "QUERY", "agent": "web",
                    "params": {"query_type": "search", "query": topic},
                    "destructive": False,
                }],
                resources=[], preview=False, reversible=True,
            )


_FETCH_URLS = [
    "https://example.com", "https://github.com/anthropics",
    "https://news.ycombinator.com", "https://lobste.rs",
    "https://arxiv.org/abs/2305.18290", "https://en.wikipedia.org/wiki/Wayland",
    "https://docs.python.org/3/library/asyncio.html",
    "https://kernel.org/doc/html/latest/admin-guide/cgroup-v2.html",
    "https://landlock.io", "https://wayland.freedesktop.org",
    "https://huggingface.co/Qwen/Qwen2.5-3B",
    "https://github.com/ggerganov/llama.cpp",
    "https://docs.anthropic.com/claude/reference",
    "https://raw.githubusercontent.com/anthropics/claude-code/main/README.md",
    "https://api.github.com/users/torvalds",
    "https://www.rust-lang.org/learn",
    "https://go.dev/doc/effective_go",
    "https://www.postgresql.org/docs/current",
    "https://sqlite.org/lang_select.html",
    "https://docs.docker.com/compose",
    "https://kubernetes.io/docs/concepts/services-networking/ingress",
    "https://nixos.org/manual/nixos/stable",
    "https://wiki.archlinux.org/title/PKGBUILD",
    "https://aur.archlinux.org/packages",
    "https://www.gnu.org/software/bash/manual",
]

_FETCH_VERBS = [
    "fetch {u}", "load {u}", "scrape {u}", "download {u}",
    "open {u}", "get the page at {u}", "show me {u}",
    "what's at {u}", "pull {u}", "grab {u}",
    "scrape the page at {u}", "show me the contents of {u}",
    "load up {u}", "fetch the url {u}",
    "get {u}", "read {u}",
]


def _gen_web_fetch(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for url in _FETCH_URLS:
        for verb in _FETCH_VERBS:
            text = verb.format(u=url)
            yield text, _spec(
                natural_text=text,
                category="web_task",
                actions=[{
                    "action_id": "act-1", "type": "QUERY", "agent": "web",
                    "params": {"query_type": "fetch", "url": url},
                    "destructive": False,
                }],
                resources=[], preview=False, reversible=True,
            )


def gen_web_task(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    items.extend(_gen_web_search(rng))
    items.extend(_gen_web_fetch(rng))
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# audio_task generators
# ---------------------------------------------------------------------------

def _gen_audio_query(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for phrase in [
        "what's my volume", "current volume", "show volume", "volume level",
        "how loud is it", "is it muted", "am I muted", "show me the volume",
        "what's my current volume", "tell me the volume",
        "volume status", "volume info", "check volume", "report volume",
        "show audio volume", "current audio level", "what's the audio level",
        "how loud", "give me the volume", "audio volume",
        "speaker volume", "system volume",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="audio_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "audio",
                "params": {"query_type": "volume"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )

    for phrase in [
        "list audio devices", "show audio devices", "what speakers do I have",
        "list my speakers", "show me audio sinks", "list audio sources",
        "show available speakers", "what audio outputs are available",
        "show audio outputs", "list output devices",
        "what audio devices are connected", "enumerate audio devices",
        "show me my audio hardware", "list sinks and sources",
        "show audio hardware", "what microphones do I have",
        "list mics", "show me input devices",
        "enumerate audio inputs", "what audio devices are available",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="audio_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "audio",
                "params": {"query_type": "devices"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )

    for phrase in [
        "audio status", "show audio status", "what's the audio doing",
        "current audio state", "audio summary", "audio info",
        "show me audio info", "what's playing", "audio overview",
        "report audio state", "give me audio status",
        "tell me about audio", "audio dashboard", "audio readout",
        "summarize audio", "audio system status",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="audio_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "audio",
                "params": {"query_type": "status"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )


_VOL_VERBS = [
    "set volume to {n}%", "set the volume to {n}",
    "volume {n}", "make volume {n}%", "change volume to {n}",
    "turn volume to {n}%", "set audio to {n}%",
    "put the volume at {n}%", "volume to {n} percent",
]


def _gen_audio_write(rng: random.Random) -> Iterable[tuple[str, dict]]:
    # set_volume across many levels
    for n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
              70, 75, 80, 85, 90, 95, 100, 110, 120]:
        for verb in _VOL_VERBS:
            text = verb.format(n=n)
            yield text, _spec(
                natural_text=text, category="audio_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "audio",
                    "params": {"audio_action": "set_volume", "level": n},
                    "destructive": True,
                }],
                resources=[], preview=False, reversible=True,
            )

    for phrase in [
        "mute", "mute audio", "mute sound", "mute the speakers",
        "silence", "be quiet", "turn off sound", "shush",
        "mute everything", "please mute",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="audio_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "audio",
                "params": {"audio_action": "mute"}, "destructive": True,
            }],
            resources=[], preview=False, reversible=True,
        )

    for phrase in [
        "unmute", "unmute audio", "turn sound on", "unmute the speakers",
        "stop muting", "audio on", "speakers on", "make sound work again",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="audio_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "audio",
                "params": {"audio_action": "unmute"}, "destructive": True,
            }],
            resources=[], preview=False, reversible=True,
        )

    for phrase in [
        "toggle mute", "switch mute", "flip mute",
        "toggle audio mute", "mute toggle",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="audio_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "audio",
                "params": {"audio_action": "toggle_mute"}, "destructive": True,
            }],
            resources=[], preview=False, reversible=True,
        )

    for sink in ["headphones", "speakers", "hdmi", "bluetooth",
                 "usb audio", "monitor speakers", "external dac",
                 "airpods", "sony wh-1000", "studio monitors",
                 "internal speakers", "displayport audio", "usb headset",
                 "logitech speakers"]:
        for verb in [
            "switch to {s}", "use {s}", "set output to {s}",
            "route audio to {s}", "play through {s}",
            "send audio to {s}", "change to {s}",
            "set default to {s}", "make {s} the default sink",
            "audio to {s}", "output to {s}",
        ]:
            text = verb.format(s=sink)
            yield text, _spec(
                natural_text=text, category="audio_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "audio",
                    "params": {"audio_action": "set_sink", "sink": sink},
                    "destructive": True,
                }],
                resources=[], preview=False, reversible=True,
            )


def gen_audio_task(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    items.extend(_gen_audio_query(rng))
    items.extend(_gen_audio_write(rng))
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# network_task generators
# ---------------------------------------------------------------------------

def _gen_network_query(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for phrase in [
        "am I online", "wifi status", "show network status",
        "are we connected", "what's my wifi status", "network info",
        "show me my network", "current network", "what's my ssid",
        "what wifi am I on", "show connection", "am I connected to wifi",
        "is the internet up", "check connection",
        "network state", "show internet status", "do I have internet",
        "is wifi connected", "what network am I on",
        "show ip address", "report network status",
        "give me network info", "tell me my wifi",
        "current ssid", "active wifi network",
        "show me what wifi I'm on",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="network_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "network",
                "params": {"query_type": "status"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )

    for phrase in [
        "scan for wifi", "show available networks",
        "list wifi networks", "what wifi is available",
        "scan wifi", "search for networks", "show nearby wifi",
        "what networks can I see", "find available wifi",
        "list available wifi", "scan for networks", "show wifi list",
        "rescan wifi", "look for wifi networks",
        "discover networks", "wifi scan", "show me nearby networks",
        "what's broadcasting", "show ssids in range",
        "list nearby ssids", "find wifi", "scan the airwaves",
        "what aps are around",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="network_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "network",
                "params": {"query_type": "scan"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )

    for phrase in [
        "list network devices", "show network interfaces",
        "what network devices do I have", "show network adapters",
        "list interfaces", "show me my network cards",
        "what nics are installed", "ip link",
        "show interfaces", "enumerate interfaces",
        "list net interfaces", "what wifi adapters do I have",
        "show network hardware", "list ethernet devices",
        "what network cards are available", "report interfaces",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="network_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "network",
                "params": {"query_type": "devices"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )


_SSIDS = [
    "HomeWiFi", "OfficeNet", "GuestNetwork", "CafeFreeWifi",
    "NETGEAR42", "Linksys-5G", "TP-Link_Home", "ATT-Fiber-1234",
    "xfinitywifi", "eduroam", "Pixel_Hotspot", "iPhone-Hotspot",
    "MyNet", "Lab-5GHz", "ConferenceWiFi", "AirportFreeWifi",
    "StarbucksWiFi", "MarriottGuest", "HiltonHonors", "AmtrakConnect",
    "DeltaWifi", "UnitedWiFi", "NeighborNet", "Apt5G",
    "FiOS_Home", "Spectrum_Mesh", "GoogleFiber-AB12", "ComcastXfinity",
]


def _gen_network_write(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for ssid in _SSIDS:
        for verb in [
            "connect to {s}", "join {s}", "connect wifi {s}",
            "switch to {s}", "log into {s}", "use wifi {s}",
            "please connect to {s}", "get on {s}",
        ]:
            text = verb.format(s=ssid)
            yield text, _spec(
                natural_text=text, category="network_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "network",
                    "params": {"network_action": "connect", "ssid": ssid},
                    "destructive": True,
                }],
                resources=[], preview=True, reversible=True,
            )

    # With password
    for ssid in _SSIDS[:8]:
        for pw in ["hunter2", "secretpass", "wifi-pass-123",
                   "MyP@ssw0rd", "guest2024"]:
            for verb in [
                "connect to {s} with password {p}",
                "join {s} password {p}",
                "log into {s} using {p}",
            ]:
                text = verb.format(s=ssid, p=pw)
                yield text, _spec(
                    natural_text=text, category="network_task",
                    actions=[{
                        "action_id": "act-1", "type": "WRITE",
                        "agent": "network",
                        "params": {"network_action": "connect",
                                   "ssid": ssid, "passphrase": pw},
                        "destructive": True,
                    }],
                    resources=[], preview=True, reversible=True,
                )

    for phrase in [
        "disconnect wifi", "disconnect from wifi", "drop the wifi connection",
        "disconnect", "go offline", "turn off wifi", "leave the network",
        "kill the wifi", "shut off wifi",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="network_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "network",
                "params": {"network_action": "disconnect"},
                "destructive": True,
            }],
            resources=[], preview=True, reversible=True,
        )


def gen_network_task(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    items.extend(_gen_network_query(rng))
    items.extend(_gen_network_write(rng))
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# power_task generators
# ---------------------------------------------------------------------------

def _gen_power_query(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for phrase in [
        "battery", "battery status", "battery level",
        "how much battery do I have", "what's my battery at",
        "show battery", "is the battery charging", "am I charging",
        "battery percentage", "how charged am I", "battery info",
        "tell me battery level", "is my laptop plugged in",
        "battery health", "show me battery info",
        "check battery", "battery state", "how's my battery",
        "is the laptop on ac power", "am I on battery",
        "remaining battery", "battery time left",
        "how long until battery dies", "time on battery",
        "show me how much juice is left", "battery readout",
        "am I plugged in", "is power connected",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="power_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "power",
                "params": {"query_type": "battery"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )

    for phrase in [
        "screen brightness", "what's my brightness",
        "show brightness level", "brightness", "current brightness",
        "how bright is the screen", "brightness setting",
        "show me the brightness", "what brightness am I at",
        "display brightness", "monitor brightness", "tell me the brightness",
        "screen brightness level", "how dim is it", "check brightness",
        "current screen brightness", "what's the screen brightness",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="power_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "power",
                "params": {"query_type": "brightness"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )

    for phrase in [
        "power status", "show power info", "power summary",
        "battery and brightness", "show me power state",
        "current power state", "power info", "system power state",
        "give me a power readout", "everything power-related",
        "show me power details", "report power status",
        "power overview", "power dashboard",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="power_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "power",
                "params": {"query_type": "status"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )


def _gen_power_write(rng: random.Random) -> Iterable[tuple[str, dict]]:
    for phrase in [
        "suspend", "suspend the system", "go to sleep", "sleep",
        "put the laptop to sleep", "suspend my computer",
        "make it sleep", "send to sleep", "suspend now",
        "sleep mode", "enter sleep", "put computer to sleep",
        "suspend the laptop", "go into sleep", "system sleep",
        "suspend it", "sleep the machine", "put it to sleep",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="power_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "power",
                "params": {"power_action": "suspend"}, "destructive": True,
            }],
            resources=[], preview=True, reversible=True,
        )

    for phrase in [
        "hibernate", "hibernate the system", "hibernate now",
        "put my laptop in hibernation", "go into hibernation",
        "deep sleep", "save state and shut down",
        "hibernate the laptop", "enter hibernation", "system hibernate",
        "hibernate it", "deep sleep mode", "save and hibernate",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="power_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "power",
                "params": {"power_action": "hibernate"}, "destructive": True,
            }],
            resources=[], preview=True, reversible=True,
        )

    for phrase in [
        "lock screen", "lock my screen", "lock", "lock the session",
        "lock it", "screen lock", "session lock", "lock the computer",
        "lock the laptop", "lock the desktop", "lock workstation",
        "secure the screen", "go to lock screen", "lock now",
    ]:
        yield phrase, _spec(
            natural_text=phrase, category="power_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "power",
                "params": {"power_action": "lock"}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )

    for n in [1, 5, 10, 12, 15, 18, 20, 25, 30, 33, 35, 40, 45, 50,
              55, 60, 65, 70, 75, 80, 85, 88, 90, 95, 100]:
        for verb in [
            "set brightness to {n}%", "brightness {n}",
            "set screen brightness to {n}", "make screen {n}% bright",
            "dim screen to {n}%", "set display to {n}% brightness",
            "brighten to {n}", "screen brightness {n} percent",
            "change brightness to {n}", "adjust brightness to {n}%",
            "put brightness at {n}", "screen to {n}% brightness",
            "display brightness {n}", "monitor brightness to {n}%",
        ]:
            text = verb.format(n=n)
            yield text, _spec(
                natural_text=text, category="power_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "power",
                    "params": {"power_action": "set_brightness", "level": n},
                    "destructive": True,
                }],
                resources=[], preview=False, reversible=True,
            )


def gen_power_task(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    items.extend(_gen_power_query(rng))
    items.extend(_gen_power_write(rng))
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# writing_task generators
# ---------------------------------------------------------------------------

_WRITING_TOPICS = [
    "the team standup notes", "Q1 retrospective summary",
    "release announcement for v0.4", "introduction to GoalSpec",
    "weekly status update", "design doc for the warm runner pool",
    "incident postmortem template", "onboarding doc for new engineers",
    "README for the compositor crate", "blog post about Layer 0 regex",
    "speakers notes for the demo", "project pitch one-pager",
    "outline for the talk on AI-native OS", "feature spec for L3 fallback",
    "changelog entry for the security audit", "architecture overview",
    "RFC for the agent authority protocol", "test plan for Phase 3",
    "user guide draft", "FAQ for the launch",
    "press release for Show HN", "launch checklist",
    "summary of the security audit findings",
    "memo about the merge freeze", "internal dogfooding report",
    "draft of the contribution guide", "new hire welcome email",
    "performance benchmark writeup", "CI pipeline overview",
    "bug triage notes", "sprint goals for next week",
]

_WRITING_FORMATS = [
    "memo", "report", "email", "blog_post", "doc", "summary",
    "outline", "spec", "letter", "note",
]

_WRITING_VERBS = [
    "write {t}", "draft {t}", "compose {t}", "create {t}",
    "put together {t}", "prepare {t}", "draft up {t}",
    "can you write {t}", "please draft {t}", "I need {t}",
    "help me write {t}", "let's write {t}",
    "write a draft of {t}", "compose a {f} on {t}",
    "create a {f} about {t}", "draft a {f} for {t}",
]


def gen_writing_task(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    for topic in _WRITING_TOPICS:
        for verb in _WRITING_VERBS:
            fmt = rng.choice(_WRITING_FORMATS)
            text = verb.format(t=topic, f=fmt)
            items.append((text, _spec(
                natural_text=text, category="writing_task",
                actions=[{
                    "action_id": "act-1", "type": "COMPOSE", "agent": "writing",
                    "params": {"topic": topic, "format": fmt},
                    "destructive": False,
                }],
                resources=[], preview=False, reversible=True,
            )))
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# multi-action (DAG with depends_on) generators
# ---------------------------------------------------------------------------

def gen_multi_action(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []

    # Pattern A: find + delete (the canonical destructive compound).
    for _ in range(140):
        pat = rng.choice([p for p in _PATTERNS if p in _PATTERN_NAMES])
        patname = _PATTERN_NAMES[pat]
        d = rng.choice(_HOME_DIRS)
        recursive = pat.startswith("**/") or rng.random() < 0.5
        verb = rng.choice([
            "delete all {pn} in {d}", "remove all {pn} in {d}",
            "clean up {pn} in {d}", "get rid of {pn} in {d}",
            "delete {p} in {d}", "remove {p} files from {d}",
            "wipe {pn} from {d}", "purge {pn} in {d}",
        ])
        text = verb.format(pn=patname, p=pat, d=d)
        items.append((text, _spec(
            natural_text=text, category="file_task",
            actions=[
                {"action_id": "act-1", "type": "QUERY", "agent": "file",
                 "params": {"path": d, "pattern": pat, "recursive": recursive},
                 "destructive": False},
                {"action_id": "act-2", "type": "DELETE", "agent": "file",
                 "params": {"path": d, "pattern": pat, "recursive": recursive},
                 "destructive": True, "depends_on": ["act-1"]},
            ],
            resources=[d], preview=True, reversible=False,
        )))

    # Pattern B: find + move (archive flow).
    for _ in range(80):
        pat = rng.choice([p for p in _PATTERNS if p in _PATTERN_NAMES])
        patname = _PATTERN_NAMES[pat]
        src = rng.choice(_HOME_DIRS)
        dst = rng.choice(_HOME_DIRS)
        if dst == src:
            dst = src + "/archive"
        verb = rng.choice([
            "move all {pn} from {s} to {d}",
            "archive {pn} from {s} into {d}",
            "shift {pn} in {s} over to {d}",
            "relocate all {pn} from {s} to {d}",
            "move {pn} from {s} into {d}",
        ])
        text = verb.format(pn=patname, s=src, d=dst)
        items.append((text, _spec(
            natural_text=text, category="file_task",
            actions=[
                {"action_id": "act-1", "type": "QUERY", "agent": "file",
                 "params": {"path": src, "pattern": pat, "recursive": False},
                 "destructive": False},
                {"action_id": "act-2", "type": "MOVE", "agent": "file",
                 "params": {"source": src, "destination": dst,
                            "pattern": pat},
                 "destructive": True, "depends_on": ["act-1"]},
            ],
            resources=[src, dst], preview=True, reversible=False,
        )))

    # Pattern C: find + copy (backup flow).
    for _ in range(60):
        pat = rng.choice([p for p in _PATTERNS if p in _PATTERN_NAMES])
        patname = _PATTERN_NAMES[pat]
        src = rng.choice(_HOME_DIRS)
        dst = rng.choice(_HOME_DIRS) + "/backup"
        verb = rng.choice([
            "back up {pn} from {s} to {d}",
            "copy all {pn} in {s} to {d}",
            "duplicate {pn} from {s} into {d}",
            "make a backup of {pn} in {s} at {d}",
        ])
        text = verb.format(pn=patname, s=src, d=dst)
        items.append((text, _spec(
            natural_text=text, category="file_task",
            actions=[
                {"action_id": "act-1", "type": "QUERY", "agent": "file",
                 "params": {"path": src, "pattern": pat, "recursive": False},
                 "destructive": False},
                {"action_id": "act-2", "type": "COPY", "agent": "file",
                 "params": {"source": src, "destination": dst,
                            "pattern": pat},
                 "destructive": True, "depends_on": ["act-1"]},
            ],
            resources=[src, dst], preview=True, reversible=False,
        )))

    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# NOT_IMPL refusals (email)
# ---------------------------------------------------------------------------

_EMAIL_RECIPIENTS = ["john", "Sarah", "the team", "my boss", "Alex",
                     "the client", "support", "hr", "marketing",
                     "everyone", "Priya", "Marcus"]
_EMAIL_TOPICS = ["the meeting", "tomorrow's sync", "the launch",
                 "our progress", "the proposal", "next steps",
                 "Friday's deadline", "the budget", "Q2 plans",
                 "the bug report", "feedback on the demo"]


def gen_not_impl(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    """Email intents — not yet implemented. Match the existing corpus shape:
    category=email_task, single COMPOSE action on agent=email."""
    items: list[tuple[str, dict]] = []
    for _ in range(n * 3):
        recipient = rng.choice(_EMAIL_RECIPIENTS)
        topic = rng.choice(_EMAIL_TOPICS)
        verb = rng.choice([
            "send an email to {r} about {t}",
            "email {r} about {t}",
            "compose an email to {r} regarding {t}",
            "draft an email to {r} about {t}",
            "write {r} about {t}",
            "shoot {r} an email about {t}",
            "ping {r} on email about {t}",
            "send {r} a quick email re: {t}",
        ])
        text = verb.format(r=recipient, t=topic)
        items.append((text, _spec(
            natural_text=text, category="email_task",
            actions=[{
                "action_id": "act-1", "type": "COMPOSE", "agent": "email",
                "params": {"topic": topic, "format": "email"},
                "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )))
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# Ambiguous / edge-case (long, polite, terse, qualifier-heavy)
# ---------------------------------------------------------------------------

def gen_ambiguous(rng: random.Random, n: int) -> list[tuple[str, dict]]:
    """Edge cases: long-winded phrasings, multiple qualifiers, unusual
    framings — all still resolving to a clear single-action GoalSpec.
    Forces the model to handle real-world phrasing diversity."""
    items: list[tuple[str, dict]] = []

    # Long-winded file reads
    for _ in range(80):
        path = _rand_file_path(rng)
        text = rng.choice([
            f"hey, when you get a chance, could you read {path} for me?",
            f"if you don't mind, please show me what's in {path}",
            f"I was wondering if you could open {path} and tell me what it says",
            f"go ahead and display {path} please",
            f"would it be possible to view {path}",
            f"can you do me a favor and read {path}",
            f"take a quick look at {path} and let me know what's in it",
        ])
        items.append((text, _spec(
            natural_text=text, category="file_task",
            actions=[{
                "action_id": "act-1", "type": "READ", "agent": "file",
                "params": {"path": path}, "destructive": False,
            }],
            resources=[path], preview=False, reversible=True,
        )))

    # Long-winded system queries
    for _ in range(60):
        qt = rng.choice(["cpu", "memory", "disk"])
        topic = {"cpu": "cpu usage", "memory": "memory usage",
                 "disk": "disk space"}[qt]
        text = rng.choice([
            f"hey can you check the {topic} for me real quick",
            f"would you mind telling me the current {topic}",
            f"I'd like to know what the {topic} is right now",
            f"give me a readout of the {topic}",
            f"what's the deal with {topic} on this machine",
            f"can you take a look at the {topic} please",
        ])
        items.append((text, _spec(
            natural_text=text, category="system_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "system",
                "params": {"query_type": qt}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )))

    # Polite app launches
    for _ in range(50):
        app = rng.choice(_APPS)
        text = rng.choice([
            f"hey, can you open {app} for me?",
            f"would you mind launching {app}",
            f"please go ahead and start {app}",
            f"I need {app} running, can you start it",
            f"could you fire up {app} please",
        ])
        items.append((text, _spec(
            natural_text=text, category="system_task",
            actions=[{
                "action_id": "act-1", "type": "WRITE", "agent": "system",
                "params": {"program": app}, "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )))

    # Verbose web searches
    for _ in range(60):
        topic = rng.choice(_SEARCH_TOPICS)
        text = rng.choice([
            f"could you go and search the web for {topic}, thanks",
            f"I'd love to read up on {topic}, can you find some info",
            f"hey can you google {topic} for me real quick",
            f"please look up everything about {topic}",
            f"can you do a web search for {topic} and show me the results",
        ])
        items.append((text, _spec(
            natural_text=text, category="web_task",
            actions=[{
                "action_id": "act-1", "type": "QUERY", "agent": "web",
                "params": {"query_type": "search", "query": topic},
                "destructive": False,
            }],
            resources=[], preview=False, reversible=True,
        )))

    # Polite power/audio
    for _ in range(80):
        choice = rng.choice(["suspend", "lock", "mute", "unmute"])
        if choice == "suspend":
            text = rng.choice([
                "could you put the system to sleep please",
                "I'm done — go ahead and suspend",
                "please suspend the laptop, I'm stepping away",
            ])
            items.append((text, _spec(
                natural_text=text, category="power_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "power",
                    "params": {"power_action": "suspend"},
                    "destructive": True,
                }],
                resources=[], preview=True, reversible=True,
            )))
        elif choice == "lock":
            text = rng.choice([
                "lock my screen, I'll be right back",
                "go ahead and lock the session please",
                "can you lock the screen real quick",
            ])
            items.append((text, _spec(
                natural_text=text, category="power_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "power",
                    "params": {"power_action": "lock"}, "destructive": False,
                }],
                resources=[], preview=False, reversible=True,
            )))
        elif choice == "mute":
            text = rng.choice([
                "please mute the audio, my call is starting",
                "can you mute everything for a sec",
                "go ahead and mute the speakers please",
            ])
            items.append((text, _spec(
                natural_text=text, category="audio_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "audio",
                    "params": {"audio_action": "mute"}, "destructive": True,
                }],
                resources=[], preview=False, reversible=True,
            )))
        else:  # unmute
            text = rng.choice([
                "alright, unmute the audio please",
                "we're back — turn the sound on again",
                "can you unmute the speakers now",
            ])
            items.append((text, _spec(
                natural_text=text, category="audio_task",
                actions=[{
                    "action_id": "act-1", "type": "WRITE", "agent": "audio",
                    "params": {"audio_action": "unmute"}, "destructive": True,
                }],
                resources=[], preview=False, reversible=True,
            )))

    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_TARGETS = {
    "file_task":      ("gen_file_task",     1000),
    "system_task":    ("gen_system_task",   700),
    "web_task":       ("gen_web_task",      700),
    "audio_task":     ("gen_audio_task",    400),
    "network_task":   ("gen_network_task",  400),
    "power_task":     ("gen_power_task",    400),
    "writing_task":   ("gen_writing_task",  300),
    "multi_action":   ("gen_multi_action",  200),
    "not_impl":       ("gen_not_impl",      200),
    "ambiguous":      ("gen_ambiguous",     200),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260419,
                    help="random seed (default: 20260419)")
    ap.add_argument("--out", type=pathlib.Path, default=_OUT_PATH,
                    help=f"output file (default: {_OUT_PATH})")
    ap.add_argument("--dry-run", action="store_true",
                    help="generate and validate but do not write")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    seen: set[str] = set()
    out_pairs: list[tuple[str, dict, str]] = []  # (text, spec, bucket)
    rejects: Counter[str] = Counter()

    for bucket, (fn_name, target) in _TARGETS.items():
        fn = globals()[fn_name]
        candidates = fn(rng, target * 3)  # over-generate for dedup headroom
        kept = 0
        for text, spec in candidates:
            if kept >= target:
                break
            key = text.strip().lower()
            if key in seen:
                rejects["duplicate"] += 1
                continue
            try:
                _validate(spec)
            except jsonschema.ValidationError as e:
                rejects[f"schema:{e.message[:40]}"] += 1
                continue
            seen.add(key)
            out_pairs.append((text, spec, bucket))
            kept += 1

        achieved = sum(1 for _, _, b in out_pairs if b == bucket)
        status = "OK " if achieved >= target else "SHORT"
        print(f"  {status} {bucket:14s} target={target:4d} got={achieved:4d}")

    print(f"\nTotal kept: {len(out_pairs)}")
    print(f"Rejected:   {sum(rejects.values())}")
    for reason, count in rejects.most_common(8):
        print(f"  {count:5d}  {reason}")

    if args.dry_run:
        print("\n(dry-run; not writing)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for text, spec, _ in out_pairs:
            f.write(json.dumps(_pair(text, spec)) + "\n")
    print(f"\nWrote {len(out_pairs)} pairs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
