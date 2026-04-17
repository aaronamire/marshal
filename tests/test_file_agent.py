"""
Unit tests for FileAgent — file operations within authorized path roots.
All operations use a temp directory as the authorized root.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.file_agent import FileAgent
from errors import LeavesError, LeavesErrorCode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = MagicMock()
    conn.execute = MagicMock(return_value=MagicMock(lastrowid=1))
    return conn


@pytest.fixture
def tree(tmp_path):
    """Create a temp directory tree for testing."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "readme.txt").write_text("hello world")
    (tmp_path / "docs" / "notes.md").write_text("# Notes\nSome notes here")
    (tmp_path / "docs" / "deep").mkdir()
    (tmp_path / "docs" / "deep" / "nested.txt").write_text("nested content")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "photo.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
    (tmp_path / "empty_dir").mkdir()
    return tmp_path


@pytest.fixture
def agent(db_conn, tree):
    with patch("agents.base_agent.log_action_started", return_value=1), \
         patch("agents.base_agent.log_action_completed"), \
         patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
        a = FileAgent(intent_id="test-file-001", db_conn=db_conn)
    return a


def _action(action_type, **params):
    return {"action_id": "act-1", "type": action_type, "params": params}


# ---------------------------------------------------------------------------
# QUERY (fs_list)
# ---------------------------------------------------------------------------

class TestQuery:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_list_directory(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-q1", db_conn)
        result = agent.execute_action(_action("QUERY", path=str(tree / "docs")))
        assert result["count"] >= 3
        names = [f["name"] for f in result["files"]]
        assert "readme.txt" in names
        assert "notes.md" in names

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_list_with_pattern(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-q2", db_conn)
        result = agent.execute_action(
            _action("QUERY", path=str(tree / "docs"), pattern="*.txt"))
        names = [f["name"] for f in result["files"]]
        assert "readme.txt" in names
        assert "notes.md" not in names

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_list_recursive(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-q3", db_conn)
        result = agent.execute_action(
            _action("QUERY", path=str(tree / "docs"), pattern="*.txt", recursive=True))
        names = [f["name"] for f in result["files"]]
        assert "nested.txt" in names
        assert "readme.txt" in names

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_list_empty_dir(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-q4", db_conn)
        result = agent.execute_action(
            _action("QUERY", path=str(tree / "empty_dir")))
        assert result["count"] == 0

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_list_nonexistent_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-q5", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(
                _action("QUERY", path=str(tree / "nonexistent")))
        assert exc.value.code == LeavesErrorCode.PATH_DOES_NOT_EXIST


# ---------------------------------------------------------------------------
# READ (fs_read_text)
# ---------------------------------------------------------------------------

class TestRead:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_read_text_file(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-r1", db_conn)
        result = agent.execute_action(
            _action("READ", path=str(tree / "docs" / "readme.txt")))
        assert result["content"] == "hello world"
        assert not result["truncated"]

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_read_nonexistent_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-r2", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(
                _action("READ", path=str(tree / "no_such_file.txt")))
        assert exc.value.code == LeavesErrorCode.FILE_NOT_FOUND

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_read_directory_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-r3", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(
                _action("READ", path=str(tree / "docs")))
        assert exc.value.code == LeavesErrorCode.FILE_READ_ERROR

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_read_truncates_large_file(self, _s, _e, db_conn, tree):
        big_file = tree / "big.txt"
        big_file.write_text("A" * 100_000)
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-r4", db_conn)
        result = agent.execute_action(_action("READ", path=str(big_file)))
        assert result["truncated"]
        assert result["size_bytes"] <= 65_536


# ---------------------------------------------------------------------------
# WRITE (fs_write)
# ---------------------------------------------------------------------------

class TestWrite:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_write_new_file(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-w1", db_conn)
        target = str(tree / "output.txt")
        result = agent.execute_action(
            _action("WRITE", path=target, content="new content"))
        assert result["bytes_written"] == len("new content".encode())
        assert Path(target).read_text() == "new content"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_write_existing_no_overwrite_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-w2", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(
                _action("WRITE", path=str(tree / "docs" / "readme.txt"),
                        content="overwrite"))
        assert exc.value.code == LeavesErrorCode.FILE_WRITE_ERROR

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_write_existing_with_overwrite(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-w3", db_conn)
        target = str(tree / "docs" / "readme.txt")
        agent.execute_action(
            _action("WRITE", path=target, content="replaced", overwrite=True))
        assert Path(target).read_text() == "replaced"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_write_creates_parent_dirs(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-w4", db_conn)
        target = str(tree / "new" / "sub" / "file.txt")
        agent.execute_action(_action("WRITE", path=target, content="deep"))
        assert Path(target).read_text() == "deep"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_write_directory(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-w5", db_conn)
        target = str(tree / "new_dir")
        result = agent.execute_action(
            _action("WRITE", path=target, is_directory=True))
        assert result["created"] == "directory"
        assert Path(target).is_dir()


# ---------------------------------------------------------------------------
# DELETE (fs_delete)
# ---------------------------------------------------------------------------

class TestDelete:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_delete_single_file(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-d1", db_conn)
        target = tree / "docs" / "readme.txt"
        assert target.exists()
        agent.execute_action(_action("DELETE", path=str(target)))
        assert not target.exists()

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_delete_nonexistent_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-d2", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(
                _action("DELETE", path=str(tree / "nonexistent.txt")))
        assert exc.value.code == LeavesErrorCode.FILE_NOT_FOUND

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_delete_directory_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-d3", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(_action("DELETE", path=str(tree / "docs")))
        assert exc.value.code == LeavesErrorCode.FILE_DELETE_ERROR

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_delete_with_pattern(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-d4", db_conn)
        result = agent.execute_action(
            _action("DELETE", path=str(tree / "docs"), pattern="*.txt"))
        assert result["deleted_count"] >= 1
        assert not (tree / "docs" / "readme.txt").exists()
        assert (tree / "docs" / "notes.md").exists()


# ---------------------------------------------------------------------------
# MOVE (fs_move)
# ---------------------------------------------------------------------------

class TestMove:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_move_single_file(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-m1", db_conn)
        src = str(tree / "docs" / "readme.txt")
        dst = str(tree / "moved.txt")
        result = agent.execute_action(
            _action("MOVE", source=src, destination=dst))
        assert result["moved"]
        assert not Path(src).exists()
        assert Path(dst).read_text() == "hello world"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_move_with_pattern(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-m2", db_conn)
        dst_dir = str(tree / "archive")
        result = agent.execute_action(
            _action("MOVE", source=str(tree / "docs"), destination=dst_dir,
                    pattern="*.txt"))
        assert result["moved_count"] >= 1
        assert (Path(dst_dir) / "readme.txt").exists()

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_move_nonexistent_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-m3", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(
                _action("MOVE", source=str(tree / "nope.txt"),
                        destination=str(tree / "dst.txt")))
        assert exc.value.code == LeavesErrorCode.FILE_NOT_FOUND


# ---------------------------------------------------------------------------
# COPY (fs_copy)
# ---------------------------------------------------------------------------

class TestCopy:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_copy_single_file(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-c1", db_conn)
        src = str(tree / "docs" / "readme.txt")
        dst = str(tree / "copy.txt")
        result = agent.execute_action(
            _action("COPY", source=src, destination=dst))
        assert result["copied"]
        assert Path(src).exists()
        assert Path(dst).read_text() == "hello world"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_copy_nonexistent_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-c2", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(
                _action("COPY", source=str(tree / "nope.txt"),
                        destination=str(tree / "dst.txt")))
        assert exc.value.code == LeavesErrorCode.FILE_NOT_FOUND


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

class TestAuthorization:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_path_outside_root_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-auth1", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(_action("READ", path="/etc/passwd"))
        assert exc.value.code == LeavesErrorCode.PATH_NOT_AUTHORIZED

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_dotdot_traversal_blocked(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-auth2", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(
                _action("READ", path=str(tree / "docs" / ".." / ".." / "etc" / "passwd")))
        assert exc.value.code == LeavesErrorCode.PATH_NOT_AUTHORIZED

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_unsupported_action_type_raises(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-auth3", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(_action("EXECUTE", path="/bin/bash"))
        assert exc.value.code == LeavesErrorCode.NOT_IMPLEMENTED


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestMeta:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_agent_type_is_file(self, _s, _e, db_conn, tree):
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-meta1", db_conn)
        assert agent.AGENT_TYPE == "file"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_inherits_base_agent(self, _s, _e, db_conn, tree):
        from agents.base_agent import BaseAgent
        with patch("agents.file_agent.AUTHORIZED_PATH_ROOTS", [str(tree)]):
            agent = FileAgent("test-meta2", db_conn)
        assert isinstance(agent, BaseAgent)
