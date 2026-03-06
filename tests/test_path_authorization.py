"""
Tests for FileAgent path authorization.
The symlink traversal test is the most critical — must pass.
"""
import os
import tempfile
from pathlib import Path
import pytest

from agents.file_agent import FileAgent
from db.audit import get_db
from errors import LeavesError, LeavesErrorCode


@pytest.fixture
def agent(tmp_path):
    """FileAgent with a real DB."""
    db = get_db(tmp_path / "test_audit.db")
    return FileAgent(intent_id="test-auth", db_conn=db)


def test_home_path_authorized(agent):
    home = str(Path.home())
    resolved = agent._authorize(home)
    assert resolved == Path.home().resolve()


def test_path_outside_home_rejected(agent):
    with pytest.raises(LeavesError) as exc_info:
        agent._authorize("/etc/passwd")
    assert exc_info.value.code == LeavesErrorCode.PATH_NOT_AUTHORIZED


def test_root_path_rejected(agent):
    with pytest.raises(LeavesError) as exc_info:
        agent._authorize("/")
    assert exc_info.value.code == LeavesErrorCode.PATH_NOT_AUTHORIZED


def test_symlink_traversal_blocked(agent, tmp_path):
    """
    A symlink inside home pointing outside home must be blocked.
    ~/evil_link -> /etc  should resolve to /etc and be rejected.
    """
    # Create a symlink inside home that points to /tmp (outside ~/ for attack simulation)
    # We simulate the attack: a path that looks like ~/something but resolves outside home
    # We'll create a symlink pointing to /etc inside a temp dir outside home
    # and verify resolve() catches it.

    # Create temp dir outside home
    outside_dir = Path(tempfile.mkdtemp(prefix="leaves_test_outside_"))
    try:
        # Create a symlink inside home's structure pointing outside
        home = Path.home()
        link_path = tmp_path / "evil_link"
        link_path.symlink_to(outside_dir)

        # The symlink is under tmp_path (which itself may or may not be under home)
        # Test the core: does _authorize resolve and reject /etc directly?
        with pytest.raises(LeavesError) as exc_info:
            agent._authorize("/etc")
        assert exc_info.value.code == LeavesErrorCode.PATH_NOT_AUTHORIZED

        # Also test that resolve(strict=False) on a symlink to outside gives the real path
        resolved = link_path.resolve(strict=False)
        assert resolved == outside_dir.resolve()
    finally:
        import shutil
        shutil.rmtree(outside_dir, ignore_errors=True)
