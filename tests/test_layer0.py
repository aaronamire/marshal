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
        
    def test_write_to_path_not_impl(self):
        r = match("write the output to ~/report.txt")
        # Should NOT be caught by NOT_IMPLEMENTED — this is a file operation
        assert not r.matched or r.is_implemented

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


# ---------------------------------------------------------------------------
# Web search / fetch — bypass L1+L2, big latency win on the L2 fallback path
# ---------------------------------------------------------------------------


class TestLayer0WebSearch:

    def test_google_query(self):
        r = match("google rust async runtime comparison")
        assert r.matched
        assert r.agent == "web"
        assert r.action_type == "QUERY"
        assert r.params["query_type"] == "search"
        assert r.params["query"] == "rust async runtime comparison"

    def test_look_up(self):
        r = match("look up the weather in San Francisco")
        assert r.matched
        assert r.agent == "web"
        assert r.params["query"] == "the weather in San Francisco"

    def test_search_the_web_for(self):
        r = match("search the web for python tutorials")
        assert r.matched
        assert r.agent == "web"
        assert r.params["query"] == "python tutorials"

    def test_search_for(self):
        r = match("search for latest kernel news")
        assert r.matched
        assert r.params["query_type"] == "search"

    def test_bare_search(self):
        r = match("search anthropic claude")
        assert r.matched
        assert r.agent == "web"

    def test_path_in_query_falls_through(self):
        # Path-like tokens in the query → L2 (likely meant a file find)
        r = match("look up ~/file.txt")
        assert not r.matched

    def test_search_with_absolute_path_falls_through(self):
        r = match("search for /etc/hosts")
        assert not r.matched


class TestLayer0WebFetch:

    def test_fetch_https(self):
        r = match("fetch https://example.com")
        assert r.matched
        assert r.agent == "web"
        assert r.params["query_type"] == "fetch"
        assert r.params["url"].startswith("https://")

    def test_load_bare_domain(self):
        r = match("load github.com/foo/bar")
        assert r.matched
        assert r.params["url"] == "github.com/foo/bar"

    def test_scrape_page_at(self):
        r = match("scrape the page at https://x.org")
        assert r.matched
        assert r.params["url"] == "https://x.org"

    def test_download_url(self):
        r = match(
            "download https://raw.githubusercontent.com/x/y/main/README"
        )
        assert r.matched
        assert r.params["query_type"] == "fetch"

    def test_open_https_routes_to_fetch_not_app_launch(self):
        r = match("open https://github.com")
        assert r.matched
        assert r.agent == "web"
        assert r.params["query_type"] == "fetch"

    def test_open_bare_domain_does_not_match_fetch(self):
        # No scheme + "open" verb → falls through (avoids stealing from
        # APP_LAUNCH for ambiguous tokens like "open foo.com").
        # APP_LAUNCH itself rejects "foo.com" because the period in a
        # word boundary check fails to bind to a single program name in
        # most apps; this test just asserts we don't classify as fetch.
        r = match("open github.com")
        if r.matched:
            assert r.params.get("query_type") != "fetch"


class TestLayer0AppLaunchVsWeb:
    """Web rules must not steal app-launch matches for bare program names."""

    def test_open_firefox_still_app_launch(self):
        r = match("open firefox")
        assert r.matched
        assert r.agent == "system"
        assert r.action_type == "WRITE"
        assert r.params["program"] == "firefox"

    def test_launch_gimp_still_app_launch(self):
        r = match("launch gimp")
        assert r.matched
        assert r.agent == "system"
        assert r.action_type == "WRITE"


# ---------------------------------------------------------------------------
# New file READ phrasings: "what's in", "view", "open <path>"
# ---------------------------------------------------------------------------


class TestLayer0FileReadAlternates:

    def test_whats_in(self):
        r = match("what's in ~/Documents/notes.md")
        assert r.matched
        assert r.action_type == "READ"
        assert r.params["path"] == "~/Documents/notes.md"

    def test_what_is_in(self):
        r = match("what is in ~/file.txt")
        assert r.matched
        assert r.action_type == "READ"

    def test_view_path(self):
        r = match("view ~/file.py")
        assert r.matched
        assert r.action_type == "READ"

    def test_open_path_routes_to_file_read(self):
        # "open ~/file.py" → READ. App-launch's _APP_NAME doesn't permit
        # the leading ~, so this can only hit the file rule.
        r = match("open ~/dev/marshal/config.py")
        assert r.matched
        assert r.action_type == "READ"
        assert r.agent == "file"


# ---------------------------------------------------------------------------
# New WRITE (empty file) phrasings: touch, create file
# ---------------------------------------------------------------------------


class TestLayer0CreateFile:

    def test_touch(self):
        r = match("touch ~/foo.txt")
        assert r.matched
        assert r.action_type == "WRITE"
        assert r.params["path"] == "~/foo.txt"
        assert r.params["content"] == ""

    def test_create_file_at(self):
        r = match("create a file at ~/notes/y.txt")
        assert r.matched
        assert r.action_type == "WRITE"
        assert r.params["path"] == "~/notes/y.txt"

    def test_create_file_short(self):
        r = match("create file ~/x.md")
        assert r.matched
        assert r.action_type == "WRITE"

    def test_create_file_nondestructive(self):
        # New empty files don't overwrite anything → not destructive.
        r = match("touch ~/empty.txt")
        assert not r.params.get("destructive")
