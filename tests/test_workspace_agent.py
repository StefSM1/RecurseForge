import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from engine.interfaces import DebugVerdict, PlanFrame
from harness import event_bus
from harness.workspace_agent import SharedChatHistory, WorkspaceAgentService
from harness.workspace_execution import WorkspaceExecutionService, WorkspaceExecutionSettings
from harness.workspace_service import WorkspaceService, WorkspaceSettings


def text_choice(content):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=[]),
        finish_reason="stop",
    )])


def tool_choice(call_id, name, arguments):
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[call]),
        finish_reason="tool_calls",
    )])


class WorkspaceAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        event_bus._bus = event_bus.EventBus()
        self.workspace = WorkspaceService(WorkspaceSettings(
            root=Path(self.temp.name) / ".recurseforge",
        ))
        self.executor = WorkspaceExecutionService(
            self.workspace,
            WorkspaceExecutionSettings(python_executable=Path(sys.executable), timeout_s=3),
        )
        self.config = {
            "llm": {
                "base_url": "http://unused/v1",
                "model_name": "fake",
                "max_tokens": 2048,
                "temperature": 0.1,
            },
            "context_governor": {"enabled": False},
            "workspace": {
                "max_passes": 3,
                "max_tool_calls_per_pass": 40,
                "max_helpers_per_pass": 4,
                "shared_history_tokens": 12000,
            },
        }

    def tearDown(self):
        self.temp.cleanup()

    def service_with_responses(self, responses):
        service = WorkspaceAgentService(self.workspace, self.executor, self.config)
        service.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        return service

    def test_first_pass_native_tool_cycle_creates_files_tests_and_synthesizes(self):
        manifest = json.dumps({
            "schema_version": 1,
            "entrypoints": [{"id": "main", "kind": "script", "target": "main.py", "args": []}],
            "tests": [{"id": "unit", "kind": "module", "target": "unittest", "args": ["discover", "-s", "tests"]}],
        })
        responses = [
            text_choice(json.dumps({
                "objective": "Create a greeting",
                "steps": ["Create files", "Run tests"],
                "relevant_files": ["main.py"],
                "success_criteria": ["Tests pass"],
            })),
            tool_choice("create-main", "create_file", {"path": "main.py", "content": "print('hello')\n"}),
            tool_choice("create-tests", "create_file", {"path": "tests/test_main.py", "content": "import unittest\n"}),
            tool_choice("create-manifest", "create_file", {"path": "recurseforge.json", "content": manifest}),
            tool_choice("finish", "finish_worker", {"summary": "Created greeting", "changed_files": ["main.py"]}),
            text_choice(json.dumps({
                "verdict": "pass",
                "findings": [],
                "affected_files": [],
                "rationale": "Implementation and tests satisfy the task.",
                "required_changes": [],
            })),
        ]
        service = self.service_with_responses(responses)

        result = service.run_workspace("workspace-run", "Create a greeting")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["pass_number"], 1)
        self.assertEqual(result["test_results"][0]["status"], "success")
        self.assertTrue(self.workspace.read_file("PLAN.md")["content"].startswith("# Workspace Agent Plan"))
        self.assertEqual(self.workspace.read_file("main.py")["content"], "print('hello')\n")
        self.assertIn("main.py", [item["path"] for item in result["changed_files"]])
        self.assertNotIn("print('hello')", result["final_summary"])
        self.assertIn("`unit`: success (exit 0)", result["final_summary"])
        self.assertEqual(responses, [])

        events = event_bus.get_event_bus().drain()
        stages = [event for event in events if event.event_type.startswith("stage_")]
        self.assertEqual(
            [event.payload["stage"] for event in stages if event.event_type == "stage_started"],
            ["plan", "worker", "debug", "root_synthesis"],
        )

    def test_shared_history_keeps_newest_messages_inside_budget(self):
        history = SharedChatHistory(max_tokens=80)
        history.append_exchange("old " * 50, "old answer " * 50)
        history.append_exchange("new question", "new answer")
        snapshot = history.snapshot()
        self.assertEqual(snapshot[-2]["content"], "new question")
        self.assertEqual(snapshot[-1]["content"], "new answer")
        self.assertNotIn("old " * 50, [message["content"] for message in snapshot])

    def test_debug_revision_is_retried_when_workspace_changes(self):
        service = WorkspaceAgentService(self.workspace, self.executor, self.config)
        calls = []

        def debug(*args, **kwargs):
            calls.append(kwargs.get("review_revision", args[-1]))
            if len(calls) == 1:
                self.workspace.create_file("user-edit.txt", "changed")
            return DebugVerdict(verdict="pass")

        service._run_debug = debug
        verdict = service._run_debug_stable(
            "run", "task", PlanFrame(objective="task"), 1, [], [],
        )
        self.assertEqual(verdict.verdict, "pass")
        self.assertEqual(len(calls), 2)

    def test_cancellation_preserves_existing_files(self):
        self.workspace.create_file("keep.txt", "safe")
        service = WorkspaceAgentService(
            self.workspace,
            self.executor,
            self.config,
            cancel_check=lambda _run_id: True,
        )
        result = service.run_workspace("cancel-run", "Change files")
        self.assertEqual(result["status"], "canceled")
        self.assertEqual(self.workspace.read_file("keep.txt")["content"], "safe")

    def test_debug_revise_returns_to_planner_and_can_pass_next_cycle(self):
        class ScriptedService(WorkspaceAgentService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fix_requests = []
                self.verdicts = [
                    DebugVerdict(
                        verdict="revise",
                        findings=["Wrong behavior"],
                        affected_files=["main.py"],
                        required_changes=["Correct the branch"],
                    ),
                    DebugVerdict(verdict="pass"),
                ]

            def _run_plan(self, run_id, prompt, pass_number, fix_request):
                self.fix_requests.append(fix_request)
                return PlanFrame(objective=prompt, steps=["Implement pass {}".format(pass_number)])

            def _run_worker(self, run_id, prompt, plan, pass_number):
                if pass_number == 1:
                    self.workspace.create_file("main.txt", "first")
                else:
                    current = self.workspace.read_file("main.txt")
                    self.workspace.save_file("main.txt", "fixed", current["revision"])
                return {"finished": True, "tool_calls_used": 1, "helpers_used": 0, "test_results": []}

            def _run_debug_stable(self, *args, **kwargs):
                return self.verdicts.pop(0)

            def _run_synthesis(self, run_id, prompt, state, unresolved):
                return "Completed in {} passes.".format(state.pass_number)

        service = ScriptedService(self.workspace, self.executor, self.config)
        result = service.run_workspace("revise-run", "Fix behavior")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["pass_number"], 2)
        self.assertIsNone(service.fix_requests[0])
        self.assertEqual(service.fix_requests[1].findings, ["Wrong behavior"])
        self.assertEqual(self.workspace.read_file("main.txt")["content"], "fixed")

    def test_three_revisions_finish_incomplete_without_deleting_files(self):
        class NeverPassingService(WorkspaceAgentService):
            def _run_plan(self, run_id, prompt, pass_number, fix_request):
                return PlanFrame(objective=prompt, steps=["Try again"])

            def _run_worker(self, run_id, prompt, plan, pass_number):
                path = "attempt-{}.txt".format(pass_number)
                self.workspace.create_file(path, "preserved")
                return {"finished": True, "tool_calls_used": 1, "helpers_used": 0, "test_results": []}

            def _run_debug_stable(self, *args, **kwargs):
                return DebugVerdict(verdict="revise", findings=["Still incomplete"])

            def _run_synthesis(self, run_id, prompt, state, unresolved):
                return "Incomplete after three passes."

        service = NeverPassingService(self.workspace, self.executor, self.config)
        result = service.run_workspace("incomplete-run", "Impossible task")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["pass_number"], 3)
        self.assertEqual(self.workspace.read_file("attempt-3.txt")["content"], "preserved")


if __name__ == "__main__":
    unittest.main()
