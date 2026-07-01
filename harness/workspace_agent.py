"""Planner -> Worker -> Debugger orchestration for isolated workspaces."""

from __future__ import annotations

import json
import logging
import threading
import time
from html import escape
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from context.repo_map import RepoMap
from engine.context_governor import count_text_tokens
from engine.interfaces import (
    DebugVerdict,
    EngineEvent,
    EventType,
    FixRequest,
    PlanFrame,
    WorkspaceAgentState,
    WorkspaceFileRef,
    WorkspaceRunStatus,
    WorkspaceStage,
)
from engine.llm_client import chat_completion, get_client
from harness.event_bus import get_event_bus
from harness.workspace_execution import WorkspaceExecutionService
from harness.workspace_service import WorkspaceError, WorkspaceService
from harness.workspace_tools import (
    DEBUG_TOOLS,
    READ_ONLY_TOOLS,
    WORKSPACE_TOOL_SCHEMAS,
    WorkspaceToolRuntime,
    WorkspaceToolRuntimeSettings,
    tool_schemas_for,
)


logger = logging.getLogger("recurseforge.harness.workspace_agent")
TModel = TypeVar("TModel", bound=BaseModel)


PLAN_SYSTEM = """You are the read-only Planner in a coding-agent cycle.
Inspect the workspace with the provided tools before planning when useful. Return only
one JSON object with: objective, steps, relevant_files, success_criteria. Do not edit
files. Make the plan concrete enough for a separate Worker to execute."""

WORKER_SYSTEM = """You are the Worker, the only model stage allowed to mutate files.
Implement the task in the isolated workspace by using native tools. Follow PLAN.md,
inspect before editing, create recurseforge.json for Python work, and run every declared
test target. Repair runtime/test failures before finishing. Source belongs in files, not
your chat response. Call finish_worker only after the implementation and declared tests
are ready, with a concise summary and changed file paths."""

DEBUG_SYSTEM = """You are the read-only Debugger. Review behavior and logic, not just
syntax. Compare the user's task, PLAN.md, changed files, manifest, and test results.
You may inspect files and run declared Python targets, but may not mutate files. Return
only one JSON object with: verdict ('pass' or 'revise'), findings, affected_files,
rationale, required_changes. Pass only when requirements and declared tests are met."""

HELPER_SYSTEM = """You are a focused read-only helper. Inspect only what is needed,
then return short findings with file paths. Do not propose tool calls after the answer."""


class SharedChatHistory:
    """Process-local Main-chat history with deterministic newest-first retention."""

    def __init__(self, max_tokens: int = 12000):
        self.max_tokens = max(1, int(max_tokens))
        self._messages: list[dict[str, str]] = []
        self._lock = threading.RLock()

    def snapshot(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(message) for message in self._messages]

    def append_exchange(self, user: str, assistant: str) -> None:
        with self._lock:
            self._messages.extend([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ])
            self._messages = self._trim(self._messages)

    def messages_with_user(self, user: str) -> list[dict[str, str]]:
        with self._lock:
            return self._trim([*self._messages, {"role": "user", "content": user}])

    def _trim(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        units: list[list[dict[str, str]]] = []
        index = len(messages) - 1
        while index >= 0:
            current = messages[index]
            if (
                current.get("role") == "assistant"
                and index > 0
                and messages[index - 1].get("role") == "user"
            ):
                units.append([messages[index - 1], current])
                index -= 2
            else:
                units.append([current])
                index -= 1

        kept_units: list[list[dict[str, str]]] = []
        used = 0
        for unit in units:
            cost = sum(
                count_text_tokens(json.dumps(message, ensure_ascii=False)) + 4
                for message in unit
            )
            if kept_units and used + cost > self.max_tokens:
                break
            if not kept_units and cost > self.max_tokens:
                message = dict(unit[-1])
                content = str(message.get("content", ""))
                byte_limit = self.max_tokens * 3
                message = {**message, "content": content.encode("utf-8")[-byte_limit:].decode("utf-8", errors="ignore")}
                unit = [message]
                cost = count_text_tokens(json.dumps(message, ensure_ascii=False)) + 4
            kept_units.append([dict(message) for message in unit])
            used += cost
        kept_units.reverse()
        return [message for unit in kept_units for message in unit]


class WorkspaceAgentService:
    """Deterministic outer controller around role-scoped native tool loops."""

    def __init__(
        self,
        workspace: WorkspaceService,
        executor: WorkspaceExecutionService,
        config: dict[str, Any],
        *,
        history: SharedChatHistory | None = None,
        cancel_check: Callable[[str], bool] | None = None,
    ):
        self.workspace = workspace
        self.executor = executor
        self.config = config
        llm = config.get("llm", {})
        self.client = get_client(str(llm.get("base_url", "http://localhost:8080/v1")))
        self.model = str(llm.get("model_name", "qwen"))
        self.max_tokens = int(llm.get("max_tokens", 8192))
        self.temperature = float(llm.get("temperature", 0.3))
        workspace_cfg = config.get("workspace", {})
        self.max_passes = max(1, int(workspace_cfg.get("max_passes", 3)))
        self.tool_settings = WorkspaceToolRuntimeSettings(
            max_tool_calls=int(workspace_cfg.get("max_tool_calls_per_pass", 40)),
            max_helpers=int(workspace_cfg.get("max_helpers_per_pass", 4)),
        )
        self.history = history or SharedChatHistory(workspace_cfg.get("shared_history_tokens", 12000))
        self.cancel_check = cancel_check or (lambda _run_id: False)

    def run_chat(self, run_id: str, prompt: str) -> dict[str, Any]:
        """Direct Qwen conversation: no graph, sandbox, or workspace tools."""
        self._raise_if_canceled(run_id)
        messages = [
            {"role": "system", "content": "Answer the user directly and concisely."},
            *self.history.messages_with_user(prompt),
        ]
        answer = chat_completion(
            self.client,
            self.model,
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            call_kind="dashboard_direct_chat",
            context_config=self.config,
        )
        self._raise_if_canceled(run_id)
        self.history.append_exchange(prompt, answer)
        return {"status": WorkspaceRunStatus.SUCCESS.value, "final_output": answer}

    def run_workspace(self, run_id: str, prompt: str) -> dict[str, Any]:
        state = WorkspaceAgentState(
            run_id=run_id,
            workspace_revision=self.workspace.workspace_revision,
            status=WorkspaceRunStatus.RUNNING.value,
        )
        initial_files = self._file_revisions()
        fix_request: FixRequest | None = None
        unresolved: list[str] = []

        try:
            for pass_number in range(1, self.max_passes + 1):
                state.pass_number = pass_number
                self._raise_if_canceled(run_id)

                plan = self._run_plan(run_id, prompt, pass_number, fix_request)
                state.plan = plan
                self._write_plan(plan)
                state.workspace_revision = self.workspace.workspace_revision

                worker = self._run_worker(run_id, prompt, plan, pass_number)
                state.tool_calls_used += worker["tool_calls_used"]
                state.helpers_used += worker["helpers_used"]
                state.test_results = worker["test_results"]
                state.manifest = self._manifest_or_none()
                state.changed_files = self._changed_files(initial_files)
                state.workspace_revision = self.workspace.workspace_revision

                verdict = self._run_debug_stable(
                    run_id, prompt, plan, pass_number, state.changed_files,
                    state.test_results,
                )
                state.debug_verdict = verdict
                state.workspace_revision = self.workspace.workspace_revision
                tests_pass = all(item.get("status") == "success" for item in state.test_results)
                worker_finished = worker["finished"]

                if verdict.verdict == "pass" and tests_pass and worker_finished:
                    state.status = WorkspaceRunStatus.SUCCESS.value
                    break

                unresolved = verdict.findings or ["Worker did not complete all required checks."]
                fix_request = FixRequest(
                    pass_number=pass_number,
                    findings=verdict.findings,
                    affected_files=verdict.affected_files,
                    required_changes=verdict.required_changes,
                )
            else:
                state.status = WorkspaceRunStatus.INCOMPLETE.value

            state.final_summary = self._run_synthesis(run_id, prompt, state, unresolved)
            self.history.append_exchange(prompt, state.final_summary)
            return state.model_dump(mode="json")
        except WorkspaceError as exc:
            if exc.code != "canceled":
                raise
            state.status = WorkspaceRunStatus.CANCELED.value
            state.cancellation_requested = True
            state.final_summary = "Workspace Agent run canceled. Existing workspace files were preserved."
            return state.model_dump(mode="json")

    def _run_plan(
        self,
        run_id: str,
        prompt: str,
        pass_number: int,
        fix_request: FixRequest | None,
    ) -> PlanFrame:
        self._stage_event(EventType.STAGE_STARTED, run_id, WorkspaceStage.PLAN.value, pass_number, "running")
        started = time.perf_counter()
        try:
            repo_map = self._workspace_repo_map()
            request = "Task:\n{}\n\nWorkspace map:\n{}".format(prompt, repo_map)
            if fix_request is not None:
                request += "\n\nPrevious Debug FixRequest:\n{}".format(fix_request.model_dump_json())
            messages = [
                {"role": "system", "content": PLAN_SYSTEM},
                *self.history.snapshot(),
                {"role": "user", "content": request},
            ]
            runtime = self._runtime(run_id, "planner", WorkspaceStage.PLAN.value, pass_number, READ_ONLY_TOOLS, True)
            result = runtime.run_worker_loop(
                self.client, self.model, messages,
                max_tokens=min(self.max_tokens, 4096),
                temperature=0.15,
                context_config=self.config,
                tool_schemas=tool_schemas_for(READ_ONLY_TOOLS),
            )
            content = self._last_assistant_text(result["messages"])
            plan = self._parse_model(PlanFrame, content, PlanFrame(
                objective=prompt,
                steps=["Inspect the workspace", "Implement the requested change", "Run declared tests"],
                success_criteria=["The requested behavior is implemented", "Declared tests pass"],
            ))
            self._stage_event(EventType.STAGE_COMPLETED, run_id, WorkspaceStage.PLAN.value, pass_number, "success", started)
            return plan
        except Exception:
            self._stage_event(EventType.STAGE_COMPLETED, run_id, WorkspaceStage.PLAN.value, pass_number, "failed", started)
            raise

    def _run_worker(self, run_id: str, prompt: str, plan: PlanFrame, pass_number: int) -> dict[str, Any]:
        self._stage_event(EventType.STAGE_STARTED, run_id, WorkspaceStage.WORKER.value, pass_number, "running")
        started = time.perf_counter()
        try:
            repo_map = self._workspace_repo_map()
            messages = [{"role": "system", "content": WORKER_SYSTEM}, {
                "role": "user",
                "content": "Task:\n{}\n\nPlan:\n{}\n\nWorkspace map:\n{}".format(
                    prompt, plan.model_dump_json(), repo_map),
            }]
            runtime = self._runtime(
                run_id, "worker", WorkspaceStage.WORKER.value, pass_number,
                {schema["function"]["name"] for schema in WORKSPACE_TOOL_SCHEMAS}, False,
            )
            result: dict[str, Any] = {}
            tests: list[dict[str, Any]] = []
            for repair_round in range(3):
                runtime.finished = False
                runtime.finish_payload = None
                result = runtime.run_worker_loop(
                    self.client, self.model, messages,
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    context_config=self.config,
                )
                messages = result["messages"]
                tests = self._run_required_tests()
                if runtime.finished and all(item.get("status") == "success" for item in tests):
                    break
                if repair_round < 2:
                    messages.append({
                        "role": "user",
                        "content": "The pass is not complete. Repair these enforced test/manifest results, rerun checks, then call finish_worker:\n{}".format(
                            json.dumps(tests, ensure_ascii=False)),
                    })
            self._stage_event(
                EventType.STAGE_COMPLETED, run_id, WorkspaceStage.WORKER.value,
                pass_number, "success" if runtime.finished else "failed", started,
            )
            return {
                "finished": runtime.finished,
                "finish": runtime.finish_payload,
                "tool_calls_used": runtime.tool_calls_used,
                "helpers_used": runtime.helpers_used,
                "test_results": tests,
            }
        except Exception:
            self._stage_event(EventType.STAGE_COMPLETED, run_id, WorkspaceStage.WORKER.value, pass_number, "failed", started)
            raise

    def _run_debug_stable(
        self,
        run_id: str,
        prompt: str,
        plan: PlanFrame,
        pass_number: int,
        changed_files: list[WorkspaceFileRef],
        tests: list[dict[str, Any]],
    ) -> DebugVerdict:
        for _attempt in range(3):
            revision = self.workspace.workspace_revision
            verdict = self._run_debug(run_id, prompt, plan, pass_number, changed_files, tests, revision)
            if revision == self.workspace.workspace_revision:
                return verdict
            logger.info("[WORKSPACE] Discarded stale Debug verdict at revision %d", revision)
        raise WorkspaceError("workspace kept changing during Debug review", code="workspace_changed_during_review", status=409)

    def _run_debug(
        self,
        run_id: str,
        prompt: str,
        plan: PlanFrame,
        pass_number: int,
        changed_files: list[WorkspaceFileRef],
        tests: list[dict[str, Any]],
        review_revision: int,
    ) -> DebugVerdict:
        self._stage_event(EventType.STAGE_STARTED, run_id, WorkspaceStage.DEBUG.value, pass_number, "running")
        started = time.perf_counter()
        try:
            manifest = self._manifest_or_none()
            messages = [{"role": "system", "content": DEBUG_SYSTEM}, {
                "role": "user",
                "content": "Task:\n{}\n\nPlan:\n{}\n\nReview revision: {}\nChanged files:\n{}\nManifest:\n{}\nTests:\n{}\nWorkspace map:\n{}".format(
                    prompt,
                    plan.model_dump_json(),
                    review_revision,
                    json.dumps([item.model_dump(mode="json") for item in changed_files]),
                    json.dumps(manifest),
                    json.dumps(tests),
                    self._workspace_repo_map(),
                ),
            }]
            runtime = self._runtime(run_id, "debugger", WorkspaceStage.DEBUG.value, pass_number, DEBUG_TOOLS, True)
            result = runtime.run_worker_loop(
                self.client, self.model, messages,
                max_tokens=min(self.max_tokens, 4096),
                temperature=0.1,
                context_config=self.config,
                tool_schemas=tool_schemas_for(DEBUG_TOOLS),
            )
            content = self._last_assistant_text(result["messages"])
            fallback = DebugVerdict(
                verdict="revise",
                findings=["Debugger did not return a valid structured verdict."],
                rationale=content[:1000],
                required_changes=["Re-review the implementation and return the required JSON verdict."],
            )
            verdict = self._parse_model(DebugVerdict, content, fallback)
            verdict.verdict = "pass" if verdict.verdict.strip().lower() == "pass" else "revise"
            self._stage_event(EventType.STAGE_COMPLETED, run_id, WorkspaceStage.DEBUG.value, pass_number, "success", started)
            return verdict
        except Exception:
            self._stage_event(EventType.STAGE_COMPLETED, run_id, WorkspaceStage.DEBUG.value, pass_number, "failed", started)
            raise

    def _run_synthesis(
        self,
        run_id: str,
        prompt: str,
        state: WorkspaceAgentState,
        unresolved: list[str],
    ) -> str:
        stage = WorkspaceStage.ROOT_SYNTHESIS.value
        self._stage_event(EventType.STAGE_STARTED, run_id, stage, state.pass_number, "running")
        started = time.perf_counter()
        try:
            outcome = {
                WorkspaceRunStatus.SUCCESS.value: "completed successfully",
                WorkspaceRunStatus.INCOMPLETE.value: "stopped incomplete",
                WorkspaceRunStatus.CANCELED.value: "was canceled",
            }.get(state.status, "failed")
            answer_lines = [
                "Workspace Agent {} in {} pass{}.".format(
                    outcome, state.pass_number, "" if state.pass_number == 1 else "es"),
                "",
                "Changed files:",
            ]
            if state.changed_files:
                answer_lines.extend("- `{}`".format(item.path) for item in state.changed_files)
            else:
                answer_lines.append("- None")
            answer_lines.extend(["", "Tests:"])
            if state.test_results:
                for item in state.test_results:
                    answer_lines.append("- `{}`: {} (exit {})".format(
                        item.get("target_id", "unknown"),
                        item.get("status", "unknown"),
                        item.get("exit_code", "?"),
                    ))
            else:
                answer_lines.append("- No declared tests were required for this non-Python workspace.")
            answer_lines.extend([
                "",
                "Debug verdict: {}.".format(
                    state.debug_verdict.verdict if state.debug_verdict else "not available"),
                "",
                "Unresolved issues:",
            ])
            if unresolved:
                answer_lines.extend("- {}".format(item) for item in unresolved)
            else:
                answer_lines.append("- None")
            answer = "\n".join(answer_lines)
            self._stage_event(EventType.STAGE_COMPLETED, run_id, stage, state.pass_number, "success", started)
            return answer
        except Exception:
            self._stage_event(EventType.STAGE_COMPLETED, run_id, stage, state.pass_number, "failed", started)
            raise

    def _runtime(
        self,
        run_id: str,
        actor: str,
        stage: str,
        pass_number: int,
        allowed_tools: set[str],
        read_only: bool,
    ) -> WorkspaceToolRuntime:
        helper = None if read_only else lambda task, files: self._run_helper(run_id, pass_number, task, files)
        return WorkspaceToolRuntime(
            self.workspace,
            self.executor,
            run_id=run_id,
            actor_id=actor,
            stage=stage,
            pass_number=pass_number,
            settings=self.tool_settings,
            helper_callback=helper,
            cancel_check=lambda: self.cancel_check(run_id),
            read_only=read_only,
            allowed_tools=allowed_tools,
        )

    def _run_helper(self, run_id: str, pass_number: int, task: str, relevant_files: list[str]) -> dict[str, Any]:
        runtime = self._runtime(run_id, "helper", "helper", pass_number, READ_ONLY_TOOLS, True)
        result = runtime.run_worker_loop(
            self.client,
            self.model,
            [{"role": "system", "content": HELPER_SYSTEM}, {
                "role": "user",
                "content": "Task: {}\nRelevant files: {}".format(task, json.dumps(relevant_files)),
            }],
            max_tokens=min(self.max_tokens, 2048),
            temperature=0.1,
            context_config=self.config,
            tool_schemas=tool_schemas_for(READ_ONLY_TOOLS),
        )
        return {"summary": self._last_assistant_text(result["messages"])[:6000]}

    def _run_required_tests(self) -> list[dict[str, Any]]:
        try:
            manifest = self.executor.load_manifest()
        except WorkspaceError as exc:
            has_python = any(item["path"].endswith(".py") for item in self.workspace.tree()["files"])
            if exc.code == "manifest_not_found" and not has_python:
                return []
            return [{"target_id": "manifest", "status": "failed", "stderr": str(exc), "exit_code": -1}]
        return [
            self._compact_test_result(self.executor.run_python_target(target["id"]))
            for target in manifest["tests"]
        ]

    @staticmethod
    def _compact_test_result(result: dict[str, Any]) -> dict[str, Any]:
        compact = dict(result)
        for field in ("stdout", "stderr"):
            value = str(compact.get(field) or "")
            if len(value) > 4000:
                compact[field] = value[:2000] + "\n...[output trimmed]...\n" + value[-2000:]
        return compact

    def _manifest_or_none(self) -> dict[str, Any] | None:
        try:
            return self.executor.load_manifest()
        except WorkspaceError:
            return None

    def _write_plan(self, plan: PlanFrame) -> None:
        lines = ["# Workspace Agent Plan", "", "## Objective", "", plan.objective, "", "## Steps", ""]
        lines.extend("{}. {}".format(index, step) for index, step in enumerate(plan.steps, start=1))
        if plan.relevant_files:
            lines.extend(["", "## Relevant Files", "", *("- `{}`".format(path) for path in plan.relevant_files)])
        if plan.success_criteria:
            lines.extend(["", "## Success Criteria", "", *("- {}".format(item) for item in plan.success_criteria)])
        content = "\n".join(lines).rstrip() + "\n"
        try:
            current = self.workspace.read_file("PLAN.md")
            self.workspace.save_file("PLAN.md", content, current["revision"], owner_id="engine:planner")
        except WorkspaceError as exc:
            if exc.code != "not_found":
                raise
            self.workspace.create_file("PLAN.md", content, owner_id="engine:planner")

    def _workspace_repo_map(self) -> str:
        tree = self.workspace.tree()
        symbol_map: dict[str, list[str]] = {}
        try:
            parsed = RepoMap(str(self.workspace.active))
            for raw_path, info in parsed._index.items():
                normalized = str(raw_path).replace("\\", "/")
                symbol_map[normalized] = ["{} {} {}".format(item.kind, item.name, item.signature).strip() for item in info.symbols]
        except Exception as exc:
            logger.debug("Workspace symbol map unavailable: %s", exc)

        lines = ['<workspace revision="{}">'.format(tree["workspace_revision"])]
        for item in tree["files"]:
            path = item["path"]
            lines.append('  <file path="{}" bytes="{}" revision="{}">'.format(
                escape(path, quote=True), item["size_bytes"], item["revision"][:12]))
            for symbol in symbol_map.get(path, []):
                lines.append("    <symbol>{}</symbol>".format(escape(symbol)))
            lines.append("  </file>")
        lines.append("</workspace>")
        return "\n".join(lines)

    def _file_revisions(self) -> dict[str, str]:
        return {item["path"]: item["revision"] for item in self.workspace.tree()["files"]}

    def _changed_files(self, initial: dict[str, str]) -> list[WorkspaceFileRef]:
        refs = []
        for item in self.workspace.tree()["files"]:
            if initial.get(item["path"]) != item["revision"]:
                refs.append(WorkspaceFileRef(path=item["path"], revision=item["revision"], size_bytes=item["size_bytes"]))
        return refs

    def _raise_if_canceled(self, run_id: str) -> None:
        if self.cancel_check(run_id):
            raise WorkspaceError("workspace run was canceled", code="canceled", status=499)

    def _stage_event(
        self,
        event_type: EventType,
        run_id: str,
        stage: str,
        pass_number: int,
        status: str,
        started: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {"stage": stage, "pass_number": pass_number, "status": status}
        if started is not None:
            payload["duration_ms"] = round((time.perf_counter() - started) * 1000)
        get_event_bus().emit(EngineEvent(run_id=run_id, event_type=event_type.value, payload=payload))

    @staticmethod
    def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant" and str(message.get("content") or "").strip():
                return str(message["content"]).strip()
        return ""

    @staticmethod
    def _parse_model(model: type[TModel], text: str, fallback: TModel) -> TModel:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return model.model_validate_json(stripped[start:end + 1])
            except (ValidationError, ValueError, json.JSONDecodeError):
                pass
        return fallback
