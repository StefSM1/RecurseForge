"""Controlled real-Qwen smoke test for Phase 4 (not run by unittest discovery)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.workspace_agent import WorkspaceAgentService
from harness.workspace_execution import WorkspaceExecutionService, WorkspaceExecutionSettings
from harness.workspace_service import WorkspaceService, WorkspaceSettings


def main() -> None:
    with (PROJECT_ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["llm"]["max_tokens"] = 2048
    config["workspace"]["max_passes"] = 1
    config["workspace"]["max_tool_calls_per_pass"] = 12

    with tempfile.TemporaryDirectory(prefix="rf_live_phase4_") as temp:
        workspace = WorkspaceService(WorkspaceSettings(root=Path(temp) / ".recurseforge"))
        executor = WorkspaceExecutionService(
            workspace,
            WorkspaceExecutionSettings(python_executable=Path(sys.executable), timeout_s=10),
        )
        service = WorkspaceAgentService(workspace, executor, config)
        result = service.run_workspace(
            "live-phase4-smoke",
            "Create hello.txt containing exactly: Hello from RecurseForge! "
            "This is a documentation-only workspace task, so no Python manifest or "
            "tests are needed.",
        )
        print(json.dumps({
            "status": result["status"],
            "pass_number": result["pass_number"],
            "changed_files": [item["path"] for item in result["changed_files"]],
            "tests": [
                {key: item.get(key) for key in ("target_id", "status", "exit_code")}
                for item in result["test_results"]
            ],
            "debug_verdict": result["debug_verdict"],
            "final_summary": result["final_summary"],
            "tree": [item["path"] for item in workspace.tree()["files"]],
        }, indent=2))


if __name__ == "__main__":
    main()
