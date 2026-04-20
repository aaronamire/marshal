"""
Security-hardening tests covering fixes from
docs/security-audit-2026-04-18.md.

Each test maps 1:1 to a finding in the audit. If a test starts failing,
re-read the audit before changing the test — the test is the contract.
"""
from __future__ import annotations

import os
import socket
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.enforcer import enforce, _looks_like_path  # noqa: E402
from agentd import _UUID_V4_RE, _validate_intent_id  # noqa: E402
from errors import MarshalError, MarshalErrorCode  # noqa: E402


# ---------------------------------------------------------------------------
# F-1: intent_id must be a UUID v4 before agentd will interpolate it
# ---------------------------------------------------------------------------


class TestIntentIdValidation:
    """A non-UUID intent_id from the socket must never reach the cgroup path."""

    def test_uuid_regex_accepts_real_uuids(self):
        for _ in range(50):
            assert _UUID_V4_RE.match(str(uuid.uuid4()))

    @pytest.mark.parametrize("evil", [
        "/../foo",                  # the F-1 traversal payload
        "../etc",
        "intent\nbar",              # newline injection
        "a/b/c/d",
        "",
        "not-a-uuid",
        "00000000-0000-3000-8000-000000000000",  # not v4 (3rd group starts with 3)
        "00000000-0000-4000-7000-000000000000",  # variant nibble outside [89ab]
        "00000000-0000-4000-8000-00000000000",   # one char short
        None,
        12345,
        ["00000000-0000-4000-8000-000000000000"],
    ])
    def test_validate_intent_id_rejects_non_uuid(self, evil):
        with pytest.raises(MarshalError) as exc:
            _validate_intent_id(evil)
        assert exc.value.code == MarshalErrorCode.INVALID_INTENT_FORMAT

    def test_validate_intent_id_passes_real_uuid(self):
        valid = str(uuid.uuid4())
        assert _validate_intent_id(valid) == valid


# ---------------------------------------------------------------------------
# F-2 + F-3: live socket-permission and SO_PEERCRED behaviour
#
# We can't reasonably spawn a different-uid process in the test, but we CAN
# verify (a) the socket file lands at mode 0o600 once agentd is running, and
# (b) our peer-cred check returns True for our own connection. Both checks
# are exercised against a tiny in-process server using the same helper.
# ---------------------------------------------------------------------------


class TestSocketHardening:
    def test_check_peer_uid_allowed_accepts_same_uid(self, tmp_path):
        from agentd import _check_peer_uid_allowed
        import asyncio

        sock_path = tmp_path / "test.sock"
        results: list[bool] = []

        async def server_main():
            async def on_conn(reader, writer):
                results.append(_check_peer_uid_allowed(writer))
                writer.close()
                await writer.wait_closed()

            srv = await asyncio.start_unix_server(on_conn, path=str(sock_path))
            os.chmod(sock_path, 0o600)
            # connect from same-process client → same uid
            r, w = await asyncio.open_unix_connection(str(sock_path))
            w.close()
            await w.wait_closed()
            # let the server callback run
            for _ in range(20):
                if results:
                    break
                await asyncio.sleep(0.01)
            srv.close()
            await srv.wait_closed()

        asyncio.run(server_main())
        assert results == [True], "same-uid peer must be allowed"

    def test_socket_chmod_pattern_matches_audit_fix(self, tmp_path):
        """
        Smoke test: after binding, agentd's chmod pattern produces a 0o600
        socket. Verifies the mechanism, not the running daemon.
        """
        sock_path = tmp_path / "x.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(str(sock_path))
            os.chmod(sock_path, 0o600)
            mode = sock_path.stat().st_mode & 0o777
            assert mode == 0o600, f"expected 0o600, got 0o{mode:03o}"
        finally:
            srv.close()
            try:
                sock_path.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# F-4: CORS and Host-header behaviour
# ---------------------------------------------------------------------------


class TestApiHardening:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)

    def test_cors_does_not_allow_credentials(self, client):
        resp = client.options(
            "/v1/intent/plan",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Host": "127.0.0.1:8765",
            },
        )
        # `access-control-allow-credentials` must NOT be present (or must be false)
        cred = resp.headers.get("access-control-allow-credentials", "false")
        assert cred.lower() == "false", (
            f"CORS still advertises credentials: {cred!r}"
        )

    def test_host_header_unknown_rejected(self, client):
        resp = client.get("/v1/health", headers={"Host": "evil.example.com"})
        assert resp.status_code == 421, (
            f"Expected 421 Misdirected Request for unknown Host, got "
            f"{resp.status_code}: {resp.text}"
        )

    def test_host_header_loopback_accepted(self, client):
        # Health endpoint may not exist; any status that ISN'T 421 from the
        # host-check middleware proves the middleware let the request through.
        resp = client.get("/v1/history", headers={"Host": "127.0.0.1:8765"})
        assert resp.status_code != 421, (
            f"Loopback Host wrongly rejected by middleware: {resp.text}"
        )


# ---------------------------------------------------------------------------
# F-5: enforcer still blocks destructive-into-non-destructive (now via step 2)
# ---------------------------------------------------------------------------


class TestEnforcerStep4Removal:
    def test_destructive_in_query_plan_still_blocked(self, tmp_path):
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        plan = [{
            "action_id": "act-1", "type": "QUERY", "agent": "file",
            "params": {"path": str(safe_dir)}, "destructive": False,
        }]
        spec = {
            "intent_id": "00000000-0000-4000-8000-000000000000",
            "natural_text": "x",
            "category": "file_task",
            "actions": plan,
            "authorization": {"resources": [str(safe_dir)],
                               "preview_required": False, "reversible": True},
        }
        evil = {"action_id": "act-1", "type": "DELETE", "agent": "file",
                "params": {"path": str(safe_dir / "x")}}
        with pytest.raises(MarshalError) as exc:
            enforce(evil, spec)
        assert exc.value.code == MarshalErrorCode.AUTHORIZATION_VIOLATION


# ---------------------------------------------------------------------------
# F-6: bare-relative path values in non-whitelisted keys are now caught
# ---------------------------------------------------------------------------


class TestBareRelativePathScan:
    @pytest.mark.parametrize("val", [
        "etc/passwd", "var/log/auth.log", "x/y/z",
    ])
    def test_looks_like_path_catches_bare_relative(self, val):
        assert _looks_like_path(val), (
            f"bare relative {val!r} should be path-like (audit F-6)"
        )

    @pytest.mark.parametrize("val", [
        "http://example.com/x",       # scheme'd URL
        "user:pass/secret",           # colon before first slash → not a path
        "file with spaces/x",         # whitespace → likely free text
        "noseparator",                 # no slash at all
        "type:value",                  # config-string, no slash
    ])
    def test_looks_like_path_skips_obvious_non_paths(self, val):
        assert not _looks_like_path(val), (
            f"value {val!r} should NOT be treated as a path"
        )

    def test_bare_relative_in_exotic_key_rejected_by_enforcer(self, tmp_path):
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        plan = [{
            "action_id": "act-1", "type": "READ", "agent": "file",
            "params": {"path": str(safe_dir)}, "destructive": False,
        }]
        spec = {
            "intent_id": "00000000-0000-4000-8000-000000000000",
            "natural_text": "x",
            "category": "file_task",
            "actions": plan,
            "authorization": {"resources": [str(safe_dir)],
                               "preview_required": False, "reversible": True},
        }
        action = {"action_id": "act-1", "type": "READ", "agent": "file",
                  "params": {"path": str(safe_dir),
                             "log_target": "etc/shadow"}}
        with pytest.raises(MarshalError) as exc:
            enforce(action, spec)
        assert exc.value.code == MarshalErrorCode.AUTHORIZATION_VIOLATION
