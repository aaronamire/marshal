"""
File agent — executes file system operations within authorized path roots.

Security requirements (non-negotiable):
1. _authorize() uses .expanduser().resolve(strict=False) before comparison
   to defeat symlink traversal attacks.
2. Containment check uses Path.relative_to() — NOT str.startswith().
3. All tool calls go through _audit() — zero unlogged operations.
4. execute_action() is the single dispatch entry point.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from agents.base_agent import BaseAgent
from config import AUTHORIZED_PATH_ROOTS
from errors import MarshalError, MarshalErrorCode

MAX_READ_BYTES = 65_536  # 64 KiB — don't slurp huge files into memory
MAX_LIST_RESULTS = 500   # cap on fs_list results
MAX_BULK_OPS = 200       # cap on bulk move/copy/delete operations


class FileAgent(BaseAgent):
    AGENT_TYPE = "file"

    def __init__(self, intent_id: str, db_conn):
        super().__init__(intent_id, db_conn)
        self._authorized_roots: list[Path] = [
            Path(r).expanduser().resolve() for r in AUTHORIZED_PATH_ROOTS
        ]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def execute_action(self, action: dict) -> dict:
        """
        Dispatch a GoalSpec action to the appropriate tool method.
        This is the single entry point for all file operations.
        """
        action_type = action.get("type", "").upper()
        action_id = action.get("action_id", "unknown")
        params = action.get("params", {})

        dispatch = {
            "QUERY":  lambda: self.fs_list(action_id, **params),
            "READ":   lambda: self.fs_read_text(action_id, **params),
            "WRITE":  lambda: self.fs_write(action_id, **params),
            "DELETE": lambda: self.fs_delete(action_id, **params),
            "MOVE":   lambda: self.fs_move(action_id, **params),
            "COPY":   lambda: self.fs_copy(action_id, **params),
        }

        handler = dispatch.get(action_type)
        if handler is None:
            raise MarshalError(
                MarshalErrorCode.NOT_IMPLEMENTED,
                detail=f"FileAgent does not handle action type '{action_type}'",
            )
        return handler()

    # ------------------------------------------------------------------
    # Tool methods — all go through _audit()
    # ------------------------------------------------------------------

    def fs_list(
        self,
        action_id: str,
        path: str,
        pattern: str = "*",
        recursive: bool = False,
        **_: Any,
    ) -> dict:
        resolved = self._authorize(path)
        row_id = self._audit_start(
            action_id, "QUERY",
            {"path": path, "pattern": pattern, "recursive": recursive},
        )
        try:
            results = self._do_list(resolved, pattern, recursive)
            result = {"count": len(results), "files": results, "path": str(resolved)}
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.FILE_READ_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def fs_read_meta(
        self,
        action_id: str,
        path: str,
        **_: Any,
    ) -> dict:
        resolved = self._authorize(path)
        row_id = self._audit_start(action_id, "READ", {"path": path})
        try:
            if not resolved.exists():
                raise MarshalError(MarshalErrorCode.PATH_DOES_NOT_EXIST, detail=str(resolved))
            stat = resolved.stat()
            result = {
                "name": resolved.name,
                "path": str(resolved),
                "is_dir": resolved.is_dir(),
                "is_file": resolved.is_file(),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "mode": oct(stat.st_mode),
            }
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except PermissionError as e:
            err = MarshalError(MarshalErrorCode.PERMISSION_DENIED, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err
        except Exception as e:
            err = MarshalError(MarshalErrorCode.FILE_READ_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def fs_read_text(
        self,
        action_id: str,
        path: str,
        max_bytes: int = MAX_READ_BYTES,
        **_: Any,
    ) -> dict:
        resolved = self._authorize(path)
        row_id = self._audit_start(action_id, "READ", {"path": path})
        try:
            if not resolved.exists():
                raise MarshalError(MarshalErrorCode.FILE_NOT_FOUND, detail=str(resolved))
            if not resolved.is_file():
                raise MarshalError(
                    MarshalErrorCode.FILE_READ_ERROR,
                    detail=f"{resolved} is not a file",
                )
            capped = min(max_bytes, MAX_READ_BYTES)
            with resolved.open("rb") as f:
                raw = f.read(capped)
            content = raw.decode("utf-8", errors="replace")
            result = {
                "path": str(resolved),
                "content": content,
                "size_bytes": len(raw),
                "truncated": resolved.stat().st_size > capped,
            }
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except PermissionError as e:
            err = MarshalError(MarshalErrorCode.PERMISSION_DENIED, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err
        except Exception as e:
            err = MarshalError(MarshalErrorCode.FILE_READ_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def fs_write(
        self,
        action_id: str,
        path: str,
        content: str = "",
        overwrite: bool = False,
        is_directory: bool = False,
        **_: Any,
    ) -> dict:
        resolved = self._authorize(path)
        row_id = self._audit_start(action_id, "WRITE", {"path": path, "overwrite": overwrite})
        try:
            if is_directory:
                # Create directory (and parents)
                resolved.mkdir(parents=True, exist_ok=True)
                result = {"path": str(resolved), "created": "directory"}
                self._audit_end(row_id, result)
                return result

            if resolved.exists() and not overwrite:
                raise MarshalError(
                    MarshalErrorCode.FILE_WRITE_ERROR,
                    detail=f"{resolved} already exists. Set overwrite=True to allow.",
                )
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            result = {"path": str(resolved), "bytes_written": len(content.encode())}
            self._audit_end(row_id, result)
            return result
        except MarshalError:
            raise
        except PermissionError as e:
            err = MarshalError(MarshalErrorCode.PERMISSION_DENIED, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err
        except Exception as e:
            err = MarshalError(MarshalErrorCode.FILE_WRITE_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def fs_delete(
        self,
        action_id: str,
        path: str,
        pattern: Optional[str] = None,
        recursive: bool = False,
        **_: Any,
    ) -> dict:
        """
        Delete a single file (no pattern) or all files matching a glob
        pattern within a directory (with pattern).

        Never deletes directories — only files.
        With pattern: deletes all matching files up to MAX_BULK_OPS.
        """
        resolved = self._authorize(path)
        row_id = self._audit_start(
            action_id, "DELETE",
            {"path": path, "pattern": pattern, "recursive": recursive},
        )
        try:
            # --- Bulk delete: pattern provided ---
            if pattern:
                if not resolved.exists():
                    raise MarshalError(
                        MarshalErrorCode.PATH_DOES_NOT_EXIST,
                        detail=str(resolved),
                    )
                if not resolved.is_dir():
                    raise MarshalError(
                        MarshalErrorCode.FILE_DELETE_ERROR,
                        detail=f"{resolved} is not a directory — cannot use pattern on a file",
                    )
                glob_fn = resolved.rglob if recursive else resolved.glob
                targets = [
                    p for p in glob_fn(pattern)
                    if p.is_file()
                ][:MAX_BULK_OPS]

                deleted = []
                errors = []
                for target in targets:
                    try:
                        target.unlink()
                        deleted.append(str(target))
                    except Exception as e:
                        errors.append({"path": str(target), "error": str(e)})

                result = {
                    "path": str(resolved),
                    "pattern": pattern,
                    "deleted_count": len(deleted),
                    "deleted": deleted,
                    "errors": errors,
                }
                self._audit_end(row_id, result)
                return result

            # --- Single file delete ---
            if not resolved.exists():
                raise MarshalError(MarshalErrorCode.FILE_NOT_FOUND, detail=str(resolved))
            if resolved.is_dir():
                raise MarshalError(
                    MarshalErrorCode.FILE_DELETE_ERROR,
                    detail=(
                        f"{resolved} is a directory. "
                        "FileAgent only deletes files, not directories. "
                        "Use pattern='*' to delete files within a directory."
                    ),
                )
            resolved.unlink()
            result = {"path": str(resolved), "deleted": True}
            self._audit_end(row_id, result)
            return result

        except MarshalError:
            raise
        except PermissionError as e:
            err = MarshalError(MarshalErrorCode.PERMISSION_DENIED, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err
        except Exception as e:
            err = MarshalError(MarshalErrorCode.FILE_DELETE_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def fs_move(
        self,
        action_id: str,
        source: str,
        destination: str,
        pattern: Optional[str] = None,
        recursive: bool = False,
        **_: Any,
    ) -> dict:
        """
        Move a single file/directory (no pattern) or all files matching
        a glob pattern from source directory to destination directory.

        Parameter names are 'source'/'destination' to match GoalSpec schema.
        """
        resolved_src = self._authorize(source)
        resolved_dst = self._authorize(destination)
        row_id = self._audit_start(
            action_id, "MOVE",
            {"source": source, "destination": destination, "pattern": pattern},
        )
        try:
            # --- Bulk move: pattern provided ---
            if pattern:
                if not resolved_src.exists():
                    raise MarshalError(
                        MarshalErrorCode.PATH_DOES_NOT_EXIST,
                        detail=str(resolved_src),
                    )
                if not resolved_src.is_dir():
                    raise MarshalError(
                        MarshalErrorCode.FILE_MOVE_ERROR,
                        detail=f"{resolved_src} is not a directory — cannot use pattern on a file",
                    )
                resolved_dst.mkdir(parents=True, exist_ok=True)
                glob_fn = resolved_src.rglob if recursive else resolved_src.glob
                targets = [
                    p for p in glob_fn(pattern)
                    if p.is_file()
                ][:MAX_BULK_OPS]

                moved = []
                errors = []
                for target in targets:
                    try:
                        dst_path = resolved_dst / target.name
                        shutil.move(str(target), str(dst_path))
                        moved.append({"from": str(target), "to": str(dst_path)})
                    except Exception as e:
                        errors.append({"path": str(target), "error": str(e)})

                result = {
                    "source": str(resolved_src),
                    "destination": str(resolved_dst),
                    "pattern": pattern,
                    "moved_count": len(moved),
                    "moved": moved,
                    "errors": errors,
                }
                self._audit_end(row_id, result)
                return result

            # --- Single file/dir move ---
            if not resolved_src.exists():
                raise MarshalError(MarshalErrorCode.FILE_NOT_FOUND, detail=str(resolved_src))
            resolved_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(resolved_src), str(resolved_dst))
            result = {
                "source": str(resolved_src),
                "destination": str(resolved_dst),
                "moved": True,
            }
            self._audit_end(row_id, result)
            return result

        except MarshalError:
            raise
        except PermissionError as e:
            err = MarshalError(MarshalErrorCode.PERMISSION_DENIED, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err
        except Exception as e:
            err = MarshalError(MarshalErrorCode.FILE_MOVE_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    def fs_copy(
        self,
        action_id: str,
        source: str,
        destination: str,
        pattern: Optional[str] = None,
        recursive: bool = False,
        **_: Any,
    ) -> dict:
        """
        Copy a single file (no pattern) or all files matching a glob
        pattern from source directory to destination directory.

        Parameter names are 'source'/'destination' to match GoalSpec schema.
        """
        resolved_src = self._authorize(source)
        resolved_dst = self._authorize(destination)
        row_id = self._audit_start(
            action_id, "COPY",
            {"source": source, "destination": destination, "pattern": pattern},
        )
        try:
            # --- Bulk copy: pattern provided ---
            if pattern:
                if not resolved_src.exists():
                    raise MarshalError(
                        MarshalErrorCode.PATH_DOES_NOT_EXIST,
                        detail=str(resolved_src),
                    )
                if not resolved_src.is_dir():
                    raise MarshalError(
                        MarshalErrorCode.FILE_MOVE_ERROR,
                        detail=f"{resolved_src} is not a directory — cannot use pattern on a file",
                    )
                resolved_dst.mkdir(parents=True, exist_ok=True)
                glob_fn = resolved_src.rglob if recursive else resolved_src.glob
                targets = [
                    p for p in glob_fn(pattern)
                    if p.is_file()
                ][:MAX_BULK_OPS]

                copied = []
                errors = []
                for target in targets:
                    try:
                        dst_path = resolved_dst / target.name
                        shutil.copy2(str(target), str(dst_path))
                        copied.append({"from": str(target), "to": str(dst_path)})
                    except Exception as e:
                        errors.append({"path": str(target), "error": str(e)})

                result = {
                    "source": str(resolved_src),
                    "destination": str(resolved_dst),
                    "pattern": pattern,
                    "copied_count": len(copied),
                    "copied": copied,
                    "errors": errors,
                }
                self._audit_end(row_id, result)
                return result

            # --- Single file copy ---
            if not resolved_src.exists():
                raise MarshalError(MarshalErrorCode.FILE_NOT_FOUND, detail=str(resolved_src))
            if not resolved_src.is_file():
                raise MarshalError(
                    MarshalErrorCode.FILE_MOVE_ERROR,
                    detail=f"{resolved_src} is not a file. Use pattern to copy files from a directory.",
                )
            resolved_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(resolved_src), str(resolved_dst))
            result = {
                "source": str(resolved_src),
                "destination": str(resolved_dst),
                "copied": True,
            }
            self._audit_end(row_id, result)
            return result

        except MarshalError:
            raise
        except PermissionError as e:
            err = MarshalError(MarshalErrorCode.PERMISSION_DENIED, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err
        except Exception as e:
            err = MarshalError(MarshalErrorCode.FILE_MOVE_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err

    # ------------------------------------------------------------------
    # Authorization — the security core
    # ------------------------------------------------------------------

    def _authorize(self, raw_path: str) -> Path:
        """
        Resolve the path and verify it is inside an authorized root.

        Uses .expanduser().resolve(strict=False) to defeat symlink traversal.
        Uses Path.relative_to() for containment — NOT str.startswith().
        Raises MarshalError(PATH_NOT_AUTHORIZED) if outside all authorized roots.
        """
        try:
            resolved = Path(raw_path).expanduser().resolve(strict=False)
        except Exception as e:
            raise MarshalError(
                MarshalErrorCode.PATH_NOT_AUTHORIZED,
                detail=f"Could not resolve path {raw_path!r}: {e}",
                cause=e,
            )

        for root in self._authorized_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue

        raise MarshalError(
            MarshalErrorCode.PATH_NOT_AUTHORIZED,
            detail=(
                f"Path {resolved!r} is outside all authorized roots: "
                f"{[str(r) for r in self._authorized_roots]}"
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_list(self, directory: Path, pattern: str, recursive: bool) -> list[dict]:
        if not directory.exists():
            raise MarshalError(MarshalErrorCode.PATH_DOES_NOT_EXIST, detail=str(directory))
        if not directory.is_dir():
            raise MarshalError(
                MarshalErrorCode.FILE_READ_ERROR,
                detail=f"{directory} is not a directory",
            )

        results = []
        glob_fn = directory.rglob if recursive else directory.glob
        try:
            for p in glob_fn(pattern):
                if len(results) >= MAX_LIST_RESULTS:
                    break
                try:
                    stat = p.stat()
                    results.append({
                        "name": p.name,
                        "path": str(p),
                        "is_dir": p.is_dir(),
                        "size_bytes": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except (PermissionError, OSError):
                    continue
        except PermissionError as e:
            raise MarshalError(MarshalErrorCode.PERMISSION_DENIED, detail=str(e), cause=e)

        return sorted(results, key=lambda x: x["name"])

    # _audit_start / _audit_end inherited from BaseAgent
