"""
Unit tests for Layer 0 regex pattern matcher.
No inference server, no sklearn — pure regex logic.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.layer0 import match, Layer0Result


class TestLayer0Read:

    def test_read_tilde_path(self):
        r = match("read ~/documents/notes.txt")
        assert r.matched
        assert r.action_type == "READ"
        assert r.params["path"] == "~/documents/notes.txt"

    def test_cat_absolute_path(self):
        r = match("cat /etc/hosts")
        assert r.matched
        assert r.action_type == "READ"
        assert r.params["path"] == "/etc/hosts"

    def test_show_contents_of(self):
        r = match("show contents of ~/readme.md")
        assert r.matched
        assert r.action_type == "READ"

    def test_print_path(self):
        r = match("print /tmp/log.txt")
        assert r.matched
        assert r.action_type == "READ"

    def test_display_path(self):
        r = match("display ~/file.py")
        assert r.matched
        assert r.action_type == "READ"

    def test_read_nondestructive(self):
        r = match("read ~/file.txt")
        assert not r.params.get("destructive")


class TestLayer0Move:

    def test_move_tilde_paths(self):
        r = match("move ~/old.txt to ~/new.txt")
        assert r.matched
        assert r.action_type == "MOVE"
        assert r.params["source"] == "~/old.txt"
        assert r.params["destination"] == "~/new.txt"

    def test_mv_absolute(self):
        r = match("mv /tmp/foo /home/user/foo")
        assert r.matched
        assert r.action_type == "MOVE"

    def test_rename(self):
        r = match("rename ~/draft.txt to ~/final.txt")
        assert r.matched
        assert r.action_type == "MOVE"
        assert r.params["source"] == "~/draft.txt"
        assert r.params["destination"] == "~/final.txt"

    def test_move_destructive(self):
        r = match("move ~/a to ~/b")
        assert r.params.get("destructive") is True


class TestLayer0Copy:

    def test_copy_tilde_paths(self):
        r = match("copy ~/src.txt to ~/dst.txt")
        assert r.matched
        assert r.action_type == "COPY"
        assert r.params["source"] == "~/src.txt"
        assert r.params["destination"] == "~/dst.txt"

    def test_cp_command(self):
        r = match("cp /tmp/a /tmp/b")
        assert r.matched
        assert r.action_type == "COPY"

    def test_copy_nondestructive(self):
        r = match("copy ~/a to ~/b")
        assert not r.params.get("destructive")


class TestLayer0Delete:

    def test_delete_path(self):
        r = match("delete ~/tmp/old_file.txt")
        assert r.matched
        assert r.action_type == "DELETE"
        assert r.params["path"] == "~/tmp/old_file.txt"

    def test_remove_path(self):
        r = match("remove /tmp/junk")
        assert r.matched
        assert r.action_type == "DELETE"

    def test_rm_path(self):
        r = match("rm ~/trash/file.txt")
        assert r.matched
        assert r.action_type == "DELETE"

    def test_delete_destructive(self):
        r = match("delete ~/file.txt")
        assert r.params.get("destructive") is True


class TestLayer0Query:

    def test_ls_tilde_dir(self):
        r = match("ls ~/projects")
        assert r.matched
        assert r.action_type == "QUERY"
        assert r.params["path"] == "~/projects"

    def test_list_dir(self):
        r = match("list ~/documents")
        assert r.matched
        assert r.action_type == "QUERY"

    def test_list_files_in(self):
        r = match("list files in ~/dev")
        assert r.matched
        assert r.action_type == "QUERY"

    def test_find_glob_pattern(self):
        r = match("find *.py in ~/projects")
        assert r.matched
        assert r.action_type == "QUERY"
        assert r.params["pattern"] == "*.py"
        assert r.params["path"] == "~/projects"

    def test_find_tilde_glob(self):
        r = match("find ~/logs/*.log")
        assert r.matched
        assert r.action_type == "QUERY"

    def test_query_nondestructive(self):
        r = match("ls ~/dir")
        assert not r.params.get("destructive")


class TestLayer0NoMatch:
    """Ambiguous or multi-step commands must NOT match Layer 0."""

    def test_vague_find_large_files(self):
        # No explicit path — vague intent
        r = match("find large files")
        assert not r.matched

    def test_move_after_find(self):
        # Multi-step intent
        r = match("find all Python files and move them to ~/archive")
        assert not r.matched

    def test_bare_read(self):
        # No path
        r = match("read the file")
        assert not r.matched

    def test_delete_tmp_files_vague(self):
        # No explicit path
        r = match("delete all tmp files")
        assert not r.matched

    def test_email_intent(self):
        # Email inputs now match as NOT_IMPLEMENTED fast-path (matched=True, is_implemented=False)
        r = match("send an email to john about the meeting")
        assert r.matched
        assert not r.is_implemented

    def test_rename_bare_words(self):
        # Both paths must be explicit (start with ~, /, etc.)
        r = match("rename draft to final")
        assert not r.matched


class TestLayer0Latency:

    def test_match_latency_under_1ms(self):
        import time
        t0 = time.monotonic()
        for _ in range(100):
            match("read ~/documents/notes.txt")
        elapsed_ms = (time.monotonic() - t0) * 1000
        avg_ms = elapsed_ms / 100
        assert avg_ms < 1.0, f"Average latency {avg_ms:.3f}ms >= 1ms threshold"

    def test_no_match_latency_under_1ms(self):
        import time
        t0 = time.monotonic()
        for _ in range(100):
            match("find all large files and delete them")
        elapsed_ms = (time.monotonic() - t0) * 1000
        avg_ms = elapsed_ms / 100
        assert avg_ms < 1.0, f"Average latency {avg_ms:.3f}ms >= 1ms threshold"


class TestLayer0QuotedPaths:

    def test_double_quoted_path(self):
        r = match('read "/home/user/my file.txt"')
        assert r.matched
        assert r.action_type == "READ"
        assert r.params["path"] == "/home/user/my file.txt"

    def test_single_quoted_path(self):
        r = match("cat '/home/user/my notes.txt'")
        assert r.matched
        assert r.params["path"] == "/home/user/my notes.txt"
