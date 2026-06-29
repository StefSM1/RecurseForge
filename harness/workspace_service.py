"""Safe, revisioned storage for the persistent RecurseForge workspace."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from engine.interfaces import EngineEvent, EventType, WorkspaceLock
from harness.event_bus import get_event_bus


class WorkspaceError(RuntimeError):
    """Base error with a stable API-facing code."""

    def __init__(self, message: str, *, code: str = "workspace_error", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


class WorkspaceConflict(WorkspaceError):
    def __init__(self, message: str, *, code: str = "workspace_conflict"):
        super().__init__(message, code=code, status=409)


@dataclass(frozen=True)
class WorkspaceSettings:
    root: Path
    active_dir: str = "workspace/current"
    history_dir: str = "workspace-history"
    trash_dir: str = "workspace-trash"
    max_file_bytes: int = 1024 * 1024
    max_workspace_bytes: int = 100 * 1024 * 1024


class WorkspaceService:
    """Owns all filesystem access beneath the configured workspace root."""

    def __init__(self, settings: WorkspaceSettings):
        self.settings = settings
        self.root = settings.root.resolve()
        self.active = self.root / settings.active_dir
        self.history = self.root / settings.history_dir
        self.trash = self.root / settings.trash_dir
        self._mutex = threading.RLock()
        self._locks: dict[str, WorkspaceLock] = {}
        self._ensure_layout()
        self._state_path = self.root / "workspace-state.json"
        self._workspace_revision = self._load_revision()

    def _ensure_layout(self) -> None:
        for directory in (self.active, self.history, self.trash):
            directory.mkdir(parents=True, exist_ok=True)

    def _load_revision(self) -> int:
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
            return max(0, int(value.get("workspace_revision", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0

    def _bump_revision(self) -> None:
        self._workspace_revision += 1
        data = json.dumps({"workspace_revision": self._workspace_revision}).encode("utf-8")
        self._atomic_write(self._state_path, data)

    @staticmethod
    def _normalize(relative_path: str) -> str:
        raw = str(relative_path or "").replace("\\", "/").strip()
        pure = PurePosixPath(raw)
        if not raw or raw.startswith("/") or pure.is_absolute() or ":" in pure.parts[0]:
            raise WorkspaceError("path must be relative", code="invalid_path")
        if any(part in {"", ".", ".."} for part in pure.parts):
            raise WorkspaceError("path traversal is not allowed", code="invalid_path")
        return pure.as_posix()

    def _resolve(self, relative_path: str, *, allow_missing: bool = True) -> tuple[str, Path]:
        normalized = self._normalize(relative_path)
        candidate = self.active.joinpath(*PurePosixPath(normalized).parts)
        current = self.active
        for part in PurePosixPath(normalized).parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise WorkspaceError("symbolic links are not allowed", code="symlink_rejected")
        resolved_parent = candidate.parent.resolve()
        if self.active != resolved_parent and self.active not in resolved_parent.parents:
            raise WorkspaceError("path escapes active workspace", code="invalid_path")
        if not allow_missing and not candidate.is_file():
            raise WorkspaceError("file not found", code="not_found", status=404)
        return normalized, candidate

    @staticmethod
    def _hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _decode(content: bytes) -> str:
        if b"\x00" in content:
            raise WorkspaceError("binary files are not supported", code="binary_rejected")
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError("file must be UTF-8 text", code="binary_rejected") from exc

    def _validate_content(self, content: str) -> bytes:
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise WorkspaceError("content must be valid UTF-8", code="invalid_encoding") from exc
        if len(encoded) > self.settings.max_file_bytes:
            raise WorkspaceError("file exceeds configured size limit", code="file_too_large", status=413)
        return encoded

    def _active_size(self, *, replacing: Path | None = None) -> int:
        total = 0
        for path in self.active.rglob("*"):
            if path.is_symlink():
                raise WorkspaceError("symbolic links are not allowed", code="symlink_rejected")
            if path.is_file() and path != replacing:
                total += path.stat().st_size
        return total

    def _check_lock(self, path: str, owner_id: str | None) -> None:
        lock = self._locks.get(path)
        if lock and lock.owner_id != owner_id:
            raise WorkspaceConflict("file is locked by another owner", code="file_locked")

    def _emit_changed(self, operation: str, path: str | None = None, revision: str | None = None) -> None:
        payload: dict[str, Any] = {
            "operation": operation,
            "workspace_revision": self._workspace_revision,
        }
        if path is not None:
            payload["path"] = path
        if revision is not None:
            payload["revision"] = revision
        get_event_bus().emit(EngineEvent(event_type=EventType.WORKSPACE_CHANGED.value, payload=payload))

    def tree(self) -> dict[str, Any]:
        with self._mutex:
            files = []
            for path in sorted(self.active.rglob("*")):
                if path.is_symlink():
                    continue
                if path.is_file():
                    relative = path.relative_to(self.active).as_posix()
                    data = path.read_bytes()
                    files.append({
                        "path": relative,
                        "size_bytes": len(data),
                        "revision": self._hash(data),
                        "locked_by": self._locks.get(relative).owner_id if relative in self._locks else None,
                    })
            return {"files": files, "workspace_revision": self._workspace_revision}

    @property
    def workspace_revision(self) -> int:
        with self._mutex:
            return self._workspace_revision

    def snapshot_to(self, destination: Path) -> dict[str, Any]:
        """Copy the active workspace into destination at one stable revision."""
        with self._mutex:
            destination = destination.resolve()
            if destination.exists() and any(destination.iterdir()):
                raise WorkspaceError("snapshot destination must be empty", code="snapshot_destination_not_empty")
            destination.mkdir(parents=True, exist_ok=True)

            files = []
            for source in sorted(self.active.rglob("*")):
                if source.is_symlink():
                    raise WorkspaceError("symbolic links are not allowed", code="symlink_rejected")
                if not source.is_file():
                    continue
                relative = source.relative_to(self.active)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                files.append(relative.as_posix())

            return {
                "snapshot_path": str(destination),
                "workspace_revision": self._workspace_revision,
                "files": files,
            }

    def read_file(self, relative_path: str) -> dict[str, Any]:
        with self._mutex:
            normalized, path = self._resolve(relative_path, allow_missing=False)
            data = path.read_bytes()
            if len(data) > self.settings.max_file_bytes:
                raise WorkspaceError("file exceeds configured size limit", code="file_too_large", status=413)
            return {
                "path": normalized,
                "content": self._decode(data),
                "revision": self._hash(data),
                "size_bytes": len(data),
                "workspace_revision": self._workspace_revision,
                "locked_by": self._locks.get(normalized).owner_id if normalized in self._locks else None,
            }

    def create_file(self, relative_path: str, content: str, *, owner_id: str | None = None) -> dict[str, Any]:
        with self._mutex:
            normalized, path = self._resolve(relative_path)
            if path.exists():
                raise WorkspaceConflict("file already exists", code="already_exists")
            self._check_lock(normalized, owner_id)
            encoded = self._validate_content(content)
            if self._active_size() + len(encoded) > self.settings.max_workspace_bytes:
                raise WorkspaceError("workspace exceeds configured size limit", code="workspace_too_large", status=413)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(path, encoded)
            return self._mutation_result("create", normalized, encoded)

    def save_file(self, relative_path: str, content: str, expected_revision: str, *, owner_id: str | None = None) -> dict[str, Any]:
        with self._mutex:
            normalized, path = self._resolve(relative_path, allow_missing=False)
            self._check_lock(normalized, owner_id)
            previous = path.read_bytes()
            if self._hash(previous) != expected_revision:
                raise WorkspaceConflict("file revision is stale", code="stale_revision")
            encoded = self._validate_content(content)
            if self._active_size(replacing=path) + len(encoded) > self.settings.max_workspace_bytes:
                raise WorkspaceError("workspace exceeds configured size limit", code="workspace_too_large", status=413)
            self._atomic_write(path, encoded)
            return self._mutation_result("save", normalized, encoded)

    def edit_file(self, relative_path: str, expected_revision: str, replacements: list[dict[str, Any]], *, owner_id: str | None = None) -> dict[str, Any]:
        current = self.read_file(relative_path)
        content = current["content"]
        if current["revision"] != expected_revision:
            raise WorkspaceConflict("file revision is stale", code="stale_revision")
        for replacement in replacements:
            old = str(replacement.get("old_text", ""))
            new = str(replacement.get("new_text", ""))
            expected = int(replacement.get("expected_occurrences", 1))
            actual = content.count(old) if old else 0
            if actual != expected:
                raise WorkspaceConflict(
                    f"expected {expected} occurrences, found {actual}",
                    code="ambiguous_edit",
                )
            content = content.replace(old, new)
        return self.save_file(relative_path, content, expected_revision, owner_id=owner_id)

    def delete_file(self, relative_path: str, *, owner_id: str | None = None) -> dict[str, Any]:
        with self._mutex:
            normalized, path = self._resolve(relative_path, allow_missing=False)
            self._check_lock(normalized, owner_id)
            operation_id = f"{int(time.time() * 1000)}-{uuid4().hex[:8]}"
            destination = self.trash / operation_id / normalized
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            self._prune_empty(path.parent, self.active)
            self._bump_revision()
            self._emit_changed("delete", normalized)
            return {"operation_id": operation_id, "path": normalized, "workspace_revision": self._workspace_revision}

    def list_trash(self) -> dict[str, Any]:
        with self._mutex:
            items = []
            for operation in sorted(self.trash.iterdir(), reverse=True):
                if not operation.is_dir():
                    continue
                for path in operation.rglob("*"):
                    if path.is_file():
                        items.append({"operation_id": operation.name, "path": path.relative_to(operation).as_posix(), "size_bytes": path.stat().st_size})
            return {"items": items}

    def restore_trash(self, operation_id: str) -> dict[str, Any]:
        with self._mutex:
            source_root = self._safe_record_dir(self.trash, operation_id)
            files = [path for path in source_root.rglob("*") if path.is_file()]
            if not files:
                raise WorkspaceError("trash item not found", code="not_found", status=404)
            for source in files:
                relative = source.relative_to(source_root).as_posix()
                _, destination = self._resolve(relative)
                if destination.exists():
                    raise WorkspaceConflict("restore would overwrite an existing file", code="restore_conflict")
            for source in files:
                destination = self.active / source.relative_to(source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
            shutil.rmtree(source_root)
            self._bump_revision()
            self._emit_changed("restore")
            return {"operation_id": operation_id, "workspace_revision": self._workspace_revision}

    def permanently_delete_trash(self, operation_id: str, *, confirmed: bool) -> dict[str, Any]:
        if not confirmed:
            raise WorkspaceError("explicit confirmation is required", code="confirmation_required")
        with self._mutex:
            target = self._safe_record_dir(self.trash, operation_id)
            if not target.exists():
                raise WorkspaceError("trash item not found", code="not_found", status=404)
            shutil.rmtree(target)
            return {"operation_id": operation_id, "deleted": True}

    def archive_reset(self, *, confirmed: bool, run_id: str = "manual") -> dict[str, Any]:
        if not confirmed:
            raise WorkspaceError("explicit confirmation is required", code="confirmation_required")
        with self._mutex:
            archive_id = self._archive_current(run_id)
            self._bump_revision()
            self._emit_changed("archive")
            return {"archive_id": archive_id, "workspace_revision": self._workspace_revision}

    def list_history(self) -> dict[str, Any]:
        with self._mutex:
            archives = []
            for archive in sorted(self.history.iterdir(), reverse=True):
                if archive.is_dir():
                    size = sum(path.stat().st_size for path in archive.rglob("*") if path.is_file())
                    archives.append({"archive_id": archive.name, "size_bytes": size, "modified_at": archive.stat().st_mtime})
            return {"archives": archives}

    def restore_history(self, archive_id: str, *, confirmed: bool) -> dict[str, Any]:
        if not confirmed:
            raise WorkspaceError("explicit confirmation is required", code="confirmation_required")
        with self._mutex:
            source = self._safe_record_dir(self.history, archive_id)
            if not source.is_dir():
                raise WorkspaceError("archive not found", code="not_found", status=404)
            current_archive_id = self._archive_current("pre-restore")
            shutil.copytree(source, self.active, dirs_exist_ok=True)
            self._bump_revision()
            self._emit_changed("restore")
            return {"archive_id": archive_id, "current_archive_id": current_archive_id, "workspace_revision": self._workspace_revision}

    def permanently_delete_history(self, archive_id: str, *, confirmed: bool) -> dict[str, Any]:
        if not confirmed:
            raise WorkspaceError("explicit confirmation is required", code="confirmation_required")
        with self._mutex:
            target = self._safe_record_dir(self.history, archive_id)
            if not target.exists():
                raise WorkspaceError("archive not found", code="not_found", status=404)
            shutil.rmtree(target)
            return {"archive_id": archive_id, "deleted": True}

    def acquire_lock(self, relative_path: str, owner_id: str) -> WorkspaceLock:
        with self._mutex:
            normalized, _ = self._resolve(relative_path)
            existing = self._locks.get(normalized)
            if existing and existing.owner_id != owner_id:
                raise WorkspaceConflict("file is locked by another owner", code="file_locked")
            lock = existing or WorkspaceLock(path=normalized, owner_id=owner_id)
            self._locks[normalized] = lock
            self._emit_lock(lock, True)
            return lock

    def release_lock(self, relative_path: str, owner_id: str) -> None:
        with self._mutex:
            normalized = self._normalize(relative_path)
            existing = self._locks.get(normalized)
            if existing and existing.owner_id != owner_id:
                raise WorkspaceConflict("lock belongs to another owner", code="lock_owner_mismatch")
            if existing:
                del self._locks[normalized]
                self._emit_lock(existing, False)

    def _emit_lock(self, lock: WorkspaceLock, locked: bool) -> None:
        get_event_bus().emit(EngineEvent(
            event_type=EventType.WORKSPACE_LOCK_CHANGED.value,
            payload={"path": lock.path, "owner_id": lock.owner_id, "locked": locked},
        ))

    def _mutation_result(self, operation: str, normalized: str, encoded: bytes) -> dict[str, Any]:
        revision = self._hash(encoded)
        self._bump_revision()
        self._emit_changed(operation, normalized, revision)
        return {"path": normalized, "revision": revision, "size_bytes": len(encoded), "workspace_revision": self._workspace_revision}

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        handle, temp_name = tempfile.mkstemp(prefix=".rf-", dir=str(path.parent))
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _archive_current(self, run_id: str) -> str:
        archive_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id}-{uuid4().hex[:6]}"
        destination = self.history / archive_id
        shutil.move(str(self.active), str(destination))
        self.active.mkdir(parents=True, exist_ok=True)
        self._locks.clear()
        return archive_id

    @staticmethod
    def _safe_record_dir(parent: Path, record_id: str) -> Path:
        if not record_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in record_id):
            raise WorkspaceError("invalid record identifier", code="invalid_identifier")
        return parent / record_id

    @staticmethod
    def _prune_empty(directory: Path, boundary: Path) -> None:
        while directory != boundary and directory.exists():
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent
