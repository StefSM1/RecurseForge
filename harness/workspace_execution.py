"""Manifest-driven Python execution for the persistent workspace."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from harness.workspace_service import WorkspaceError, WorkspaceService


_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_TARGET_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class WorkspaceExecutionSettings:
    python_executable: Path
    timeout_s: int = 30


class WorkspaceExecutionService:
    """Runs declared Python targets from a revision-stable workspace snapshot."""

    def __init__(self, workspace: WorkspaceService, settings: WorkspaceExecutionSettings):
        self.workspace = workspace
        self.settings = settings

    def load_manifest(self) -> dict[str, Any]:
        """Read and validate the visible recurseforge.json manifest."""
        try:
            manifest_file = self.workspace.read_file("recurseforge.json")
        except WorkspaceError as exc:
            if exc.code == "not_found":
                raise WorkspaceError("recurseforge.json was not found", code="manifest_not_found", status=404) from exc
            raise

        try:
            raw = json.loads(manifest_file["content"])
        except json.JSONDecodeError as exc:
            raise WorkspaceError("recurseforge.json must be valid JSON", code="invalid_manifest") from exc
        if not isinstance(raw, dict):
            raise WorkspaceError("recurseforge.json must contain a JSON object", code="invalid_manifest")

        manifest = {
            "schema_version": raw.get("schema_version"),
            "entrypoints": raw.get("entrypoints", []),
            "tests": raw.get("tests", []),
            "workspace_revision": manifest_file["workspace_revision"],
        }
        if manifest["schema_version"] != 1:
            raise WorkspaceError("manifest schema_version must be 1", code="invalid_manifest")

        manifest["entrypoints"] = self._validate_targets(manifest["entrypoints"], group="entrypoint")
        manifest["tests"] = self._validate_targets(manifest["tests"], group="test")
        self._validate_unique_ids(manifest)
        self._require_tests_for_python_projects(manifest)
        return manifest

    def run_python_target(self, target_id: str) -> dict[str, Any]:
        """Run one manifest target by ID inside a disposable snapshot."""
        manifest = self.load_manifest()
        target = self._find_target(manifest, target_id)
        with tempfile.TemporaryDirectory(prefix="recurseforge_workspace_run_") as temp_dir:
            snapshot_dir = Path(temp_dir) / "snapshot"
            snapshot = self.workspace.snapshot_to(snapshot_dir)
            started_at = time.perf_counter()
            command = self._build_command(snapshot_dir, target)

            try:
                proc = subprocess.run(
                    command,
                    cwd=str(snapshot_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.settings.timeout_s,
                    encoding="utf-8",
                    errors="replace",
                    env=self._restricted_env(temp_dir),
                )
                exit_code = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
                status = "success" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired as exc:
                exit_code = -1
                stdout = exc.stdout or ""
                stderr = "TIMEOUT: execution exceeded {}s".format(self.settings.timeout_s)
                status = "timeout"

            return {
                "target_id": target["id"],
                "target_group": target["group"],
                "kind": target["kind"],
                "target": target["target"],
                "args": list(target["args"]),
                "snapshot_revision": snapshot["workspace_revision"],
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "duration_ms": round((time.perf_counter() - started_at) * 1000),
                "status": status,
            }

    def _validate_targets(self, value: Any, *, group: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise WorkspaceError(f"manifest {group}s must be an array", code="invalid_manifest")
        return [self._validate_target(item, group=group) for item in value]

    def _validate_target(self, item: Any, *, group: str) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise WorkspaceError(f"manifest {group} must be an object", code="invalid_manifest")
        target_id = item.get("id")
        kind = item.get("kind")
        target = item.get("target")
        args = item.get("args", [])

        if not isinstance(target_id, str) or not target_id.strip() or not _TARGET_ID_RE.match(target_id):
            raise WorkspaceError("target id must be a simple non-empty string", code="invalid_manifest")
        if kind not in {"script", "module"}:
            raise WorkspaceError("only Python script and module targets are supported", code="unsupported_target")
        if not isinstance(target, str) or not target.strip():
            raise WorkspaceError("target must be a non-empty string", code="invalid_manifest")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise WorkspaceError("target args must be an array of strings", code="invalid_manifest")
        if any("\x00" in arg for arg in args):
            raise WorkspaceError("target args contain invalid characters", code="invalid_manifest")

        if kind == "script":
            normalized = self._validate_script_target(target)
            return {"id": target_id, "group": group, "kind": kind, "target": normalized, "args": args}

        if not _MODULE_RE.match(target):
            raise WorkspaceError("module target must be a Python module name, not a command string", code="invalid_manifest")
        return {"id": target_id, "group": group, "kind": kind, "target": target, "args": args}

    def _validate_script_target(self, target: str) -> str:
        raw = target.replace("\\", "/").strip()
        pure = PurePosixPath(raw)
        if not raw or raw.startswith("/") or pure.is_absolute() or ":" in pure.parts[0]:
            raise WorkspaceError("script target must be a relative workspace path", code="invalid_manifest")
        if any(part in {"", ".", ".."} for part in pure.parts):
            raise WorkspaceError("script target traversal is not allowed", code="invalid_manifest")
        if pure.suffix != ".py":
            raise WorkspaceError("script target must point to a .py file", code="unsupported_target")
        normalized = pure.as_posix()
        self.workspace.read_file(normalized)
        return normalized

    @staticmethod
    def _validate_unique_ids(manifest: dict[str, Any]) -> None:
        ids: set[str] = set()
        for target in manifest["entrypoints"] + manifest["tests"]:
            if target["id"] in ids:
                raise WorkspaceError("manifest target ids must be unique", code="invalid_manifest")
            ids.add(target["id"])

    def _require_tests_for_python_projects(self, manifest: dict[str, Any]) -> None:
        tree = self.workspace.tree()
        has_python = any(file["path"].endswith(".py") for file in tree["files"])
        if has_python and not manifest["tests"]:
            raise WorkspaceError("Python workspaces must declare at least one test target", code="tests_required")

    @staticmethod
    def _find_target(manifest: dict[str, Any], target_id: str) -> dict[str, Any]:
        for target in manifest["entrypoints"] + manifest["tests"]:
            if target["id"] == target_id:
                return target
        raise WorkspaceError("manifest target not found", code="target_not_found", status=404)

    def _build_command(self, snapshot_dir: Path, target: dict[str, Any]) -> list[str]:
        python = str(self.settings.python_executable)
        if target["kind"] == "script":
            return [python, str(snapshot_dir / target["target"]), *target["args"]]
        return [python, "-m", target["target"], *target["args"]]

    @staticmethod
    def _restricted_env(temp_dir: str) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "",
            "HOME": temp_dir,
            "TEMP": temp_dir,
            "TMP": temp_dir,
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }


def default_workspace_python(project_root: Path) -> Path:
    """Return the project-local venv Python, falling back only for tests/dev."""
    candidates = [
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)
