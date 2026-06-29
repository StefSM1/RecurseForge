"""Bounded workspace tool runtime for the future Worker agent."""

from __future__ import annotations

import fnmatch
import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from engine.interfaces import EngineEvent, EventType
from engine.llm_client import chat_completion_message
from harness.event_bus import get_event_bus
from harness.workspace_execution import WorkspaceExecutionService
from harness.workspace_service import WorkspaceError, WorkspaceService


WORKSPACE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_workspace",
            "description": "List files in the active isolated workspace.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "Read a bounded UTF-8 slice of a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_workspace",
            "description": "Search text files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new UTF-8 workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Apply exact atomic replacements to a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "expected_revision": {"type": "string"},
                    "replacements": {"type": "array"},
                },
                "required": ["path", "expected_revision", "replacements"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Move a workspace file to recoverable trash.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_target",
            "description": "Run a declared recurseforge.json Python target from a snapshot.",
            "parameters": {
                "type": "object",
                "properties": {"target_id": {"type": "string"}},
                "required": ["target_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_helper",
            "description": "Ask a bounded read-only helper for focused findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "relevant_files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_worker",
            "description": "Finish the worker pass with a concise summary and changed file refs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "changed_files": {"type": "array"},
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]


READ_ONLY_TOOLS = {"list_workspace", "view_file", "search_workspace"}
MUTATION_TOOLS = {"create_file", "edit_file", "delete_file"}


@dataclass(frozen=True)
class WorkspaceToolRuntimeSettings:
    max_tool_calls: int = 40
    max_helpers: int = 4
    max_view_lines: int = 200
    max_tool_result_chars: int = 6000
    max_search_results: int = 50


class WorkspaceToolRuntime:
    """Dispatches workspace tools with budgets, locks, events, and caps."""

    def __init__(
        self,
        workspace: WorkspaceService,
        executor: WorkspaceExecutionService,
        *,
        run_id: str,
        actor_id: str = "worker",
        stage: str = "worker",
        pass_number: int = 1,
        settings: WorkspaceToolRuntimeSettings | None = None,
        helper_callback: Callable[[str, list[str]], dict[str, Any]] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        read_only: bool = False,
    ):
        self.workspace = workspace
        self.executor = executor
        self.run_id = run_id
        self.actor_id = actor_id
        self.stage = stage
        self.pass_number = pass_number
        self.settings = settings or WorkspaceToolRuntimeSettings()
        self.helper_callback = helper_callback
        self.cancel_check = cancel_check or (lambda: False)
        self.read_only = read_only
        self.tool_calls_used = 0
        self.helpers_used = 0
        self.finished = False
        self.finish_payload: dict[str, Any] | None = None

    def dispatch_tool(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        args = self._parse_arguments(arguments)
        self._check_canceled()
        self._reserve_tool_call()
        if self.read_only and name not in READ_ONLY_TOOLS:
            raise WorkspaceError("helper runtimes may only use read-only tools", code="tool_denied", status=403)

        tool_run_id = str(uuid4())
        path = str(args.get("path") or "") or None
        revision_before = self._revision_for_path(path)
        started_at = time.perf_counter()
        self._emit_tool(EventType.FILE_TOOL_STARTED.value, {
            "tool_run_id": tool_run_id,
            "operation": name,
            "path": path,
            "status": "running",
            "revision_before": revision_before,
        })

        try:
            result = self._dispatch_locked(name, args)
            status = "success"
            error_preview = None
            return result
        except WorkspaceError as exc:
            status = "failed"
            error_preview = str(exc)[:500]
            raise
        except Exception as exc:
            status = "failed"
            error_preview = str(exc)[:500]
            raise WorkspaceError(str(exc), code="tool_error") from exc
        finally:
            revision_after = self._revision_for_path(path)
            self._emit_tool(EventType.FILE_TOOL_COMPLETED.value, {
                "tool_run_id": tool_run_id,
                "operation": name,
                "path": path,
                "status": status,
                "revision_before": revision_before,
                "revision_after": revision_after,
                "duration_ms": round((time.perf_counter() - started_at) * 1000),
                "error_preview": error_preview,
            })

    def run_worker_loop(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        context_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a native tool-call loop until finish_worker or no tool calls."""
        transcript = list(messages)
        while not self.finished:
            self._check_canceled()
            assistant = chat_completion_message(
                client=client,
                model=model,
                messages=transcript,
                max_tokens=max_tokens,
                temperature=temperature,
                call_kind="workspace_worker",
                context_config=context_config,
                tools=WORKSPACE_TOOL_SCHEMAS,
                tool_choice="auto",
            )
            tool_calls = assistant.get("tool_calls", [])
            transcript.append({
                "role": "assistant",
                "content": assistant.get("content") or "",
                "tool_calls": tool_calls,
            })
            if not tool_calls:
                break
            for tool_call in tool_calls:
                name = tool_call.get("function", {}).get("name", "")
                raw_args = tool_call.get("function", {}).get("arguments", "{}")
                try:
                    result = self.dispatch_tool(name, raw_args)
                except WorkspaceError as exc:
                    result = {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
                transcript.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": self._cap_json_result(result),
                })

        return {
            "status": "finished" if self.finished else "stopped",
            "finish": self.finish_payload,
            "tool_calls_used": self.tool_calls_used,
            "helpers_used": self.helpers_used,
            "messages": transcript,
        }

    def _dispatch_locked(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in {
            "list_workspace", "view_file", "search_workspace", "create_file",
            "edit_file", "delete_file", "run_python_target", "spawn_helper",
            "finish_worker",
        }:
            raise WorkspaceError("unknown workspace tool: {}".format(name), code="unknown_tool")
        if name in MUTATION_TOOLS:
            return self._with_lock(str(args.get("path") or ""), lambda owner: self._dispatch(name, args, owner))
        return self._dispatch(name, args, None)

    def _dispatch(self, name: str, args: dict[str, Any], owner_id: str | None) -> dict[str, Any]:
        self._check_canceled()
        if name == "list_workspace":
            return self.workspace.tree()
        if name == "view_file":
            return self._view_file(args)
        if name == "search_workspace":
            return self._search_workspace(args)
        if name == "create_file":
            return self.workspace.create_file(
                str(args.get("path") or ""),
                str(args.get("content") or ""),
                owner_id=owner_id,
            )
        if name == "edit_file":
            replacements = args.get("replacements", [])
            if not isinstance(replacements, list):
                raise WorkspaceError("replacements must be an array", code="invalid_tool_arguments")
            return self.workspace.edit_file(
                str(args.get("path") or ""),
                str(args.get("expected_revision") or ""),
                replacements,
                owner_id=owner_id,
            )
        if name == "delete_file":
            return self.workspace.delete_file(str(args.get("path") or ""), owner_id=owner_id)
        if name == "run_python_target":
            return self.executor.run_python_target(str(args.get("target_id") or ""))
        if name == "spawn_helper":
            return self._spawn_helper(args)
        if name == "finish_worker":
            self.finished = True
            self.finish_payload = {
                "summary": str(args.get("summary") or ""),
                "changed_files": args.get("changed_files", []),
            }
            return {"status": "finished", **self.finish_payload}
        raise AssertionError("unreachable")

    def _with_lock(self, path: str, operation: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        owner_id = "{}:{}:{}".format(self.actor_id, self.stage, uuid4().hex)
        self.workspace.acquire_lock(path, owner_id)
        try:
            self._check_canceled()
            return operation(owner_id)
        finally:
            self.workspace.release_lock(path, owner_id)

    def _view_file(self, args: dict[str, Any]) -> dict[str, Any]:
        read = self.workspace.read_file(str(args.get("path") or ""))
        lines = read["content"].splitlines()
        start = max(1, int(args.get("start_line") or 1))
        end = int(args.get("end_line") or min(len(lines), start + self.settings.max_view_lines - 1))
        end = min(end, start + self.settings.max_view_lines - 1, len(lines))
        selected = "\n".join(lines[start - 1:end])
        capped = self._cap_text(selected)
        return {
            "path": read["path"],
            "revision": read["revision"],
            "workspace_revision": read["workspace_revision"],
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "content": capped["text"],
            "truncated": capped["truncated"] or end < len(lines),
        }

    def _search_workspace(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        pattern = str(args.get("glob") or "*")
        if not query:
            raise WorkspaceError("search query is required", code="invalid_tool_arguments")
        matches = []
        for file in self.workspace.tree()["files"]:
            path = file["path"]
            if not fnmatch.fnmatch(path, pattern):
                continue
            try:
                read = self.workspace.read_file(path)
            except WorkspaceError:
                continue
            for line_no, line in enumerate(read["content"].splitlines(), start=1):
                if query in line:
                    matches.append({
                        "path": path,
                        "line": line_no,
                        "revision": read["revision"],
                        "preview": line[:300],
                    })
                    if len(matches) >= self.settings.max_search_results:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _spawn_helper(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.helpers_used >= self.settings.max_helpers:
            raise WorkspaceError("helper budget exceeded", code="helper_budget_exceeded", status=429)
        self.helpers_used += 1
        task = str(args.get("task") or "")
        relevant = args.get("relevant_files", [])
        if not isinstance(relevant, list) or not all(isinstance(item, str) for item in relevant):
            raise WorkspaceError("relevant_files must be an array of strings", code="invalid_tool_arguments")
        if self.helper_callback is None:
            return {"status": "skipped", "summary": "no helper callback configured", "relevant_files": relevant}
        result = self.helper_callback(task, relevant)
        return self._cap_mapping(result)

    def _reserve_tool_call(self) -> None:
        if self.tool_calls_used >= self.settings.max_tool_calls:
            raise WorkspaceError("tool call budget exceeded", code="tool_budget_exceeded", status=429)
        self.tool_calls_used += 1

    def _check_canceled(self) -> None:
        if self.cancel_check():
            raise WorkspaceError("workspace run was canceled", code="canceled", status=499)

    def _revision_for_path(self, path: str | None) -> str | None:
        if not path:
            return None
        try:
            return self.workspace.read_file(path)["revision"]
        except WorkspaceError:
            return None

    def _emit_tool(self, event_type: str, payload: dict[str, Any]) -> None:
        clean_payload = {
            "run_id": self.run_id,
            "actor_id": self.actor_id,
            "stage": self.stage,
            "pass_number": self.pass_number,
            **payload,
        }
        get_event_bus().emit(EngineEvent(
            run_id=self.run_id,
            event_type=event_type,
            payload={key: value for key, value in clean_payload.items() if value is not None},
        ))

    @staticmethod
    def _parse_arguments(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise WorkspaceError("tool arguments must be valid JSON", code="invalid_tool_arguments") from exc
        if not isinstance(parsed, dict):
            raise WorkspaceError("tool arguments must decode to an object", code="invalid_tool_arguments")
        return parsed

    def _cap_json_result(self, result: dict[str, Any]) -> str:
        return json.dumps(self._cap_mapping(result), ensure_ascii=False)

    def _cap_mapping(self, result: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(result, ensure_ascii=False)
        if len(text) <= self.settings.max_tool_result_chars:
            return result
        return {
            "truncated": True,
            "preview": text[:self.settings.max_tool_result_chars],
        }

    def _cap_text(self, text: str) -> dict[str, Any]:
        if len(text) <= self.settings.max_tool_result_chars:
            return {"text": text, "truncated": False}
        return {
            "text": text[:self.settings.max_tool_result_chars],
            "truncated": True,
        }
