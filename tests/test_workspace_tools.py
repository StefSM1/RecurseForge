import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from engine.llm_client import chat_completion, chat_completion_message
from harness import event_bus
from harness.workspace_execution import WorkspaceExecutionService, WorkspaceExecutionSettings
from harness.workspace_service import WorkspaceError, WorkspaceService, WorkspaceSettings
from harness.workspace_tools import WorkspaceToolRuntime, WorkspaceToolRuntimeSettings


class WorkspaceToolRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        event_bus._bus = event_bus.EventBus()
        self.workspace = WorkspaceService(WorkspaceSettings(
            root=Path(self.temp.name) / ".recurseforge",
        ))
        self.executor = WorkspaceExecutionService(
            self.workspace,
            WorkspaceExecutionSettings(python_executable=Path(sys.executable), timeout_s=2),
        )
        self.runtime = WorkspaceToolRuntime(
            self.workspace,
            self.executor,
            run_id="run-tools",
            settings=WorkspaceToolRuntimeSettings(max_tool_calls=40, max_helpers=4),
        )

    def tearDown(self):
        self.temp.cleanup()

    def _write_manifest(self):
        self.workspace.create_file("tests/test_smoke.py", "import unittest\n")
        self.workspace.create_file("recurseforge.json", json.dumps({
            "schema_version": 1,
            "entrypoints": [{"id": "main", "kind": "script", "target": "main.py", "args": []}],
            "tests": [{"id": "unit", "kind": "module", "target": "unittest", "args": ["discover", "-s", "tests"]}],
        }))

    def test_inspect_create_edit_delete_run_and_finish(self):
        created = self.runtime.dispatch_tool("create_file", {
            "path": "main.py",
            "content": "print('hello')\n",
        })
        listed = self.runtime.dispatch_tool("list_workspace", {})
        viewed = self.runtime.dispatch_tool("view_file", {"path": "main.py"})
        searched = self.runtime.dispatch_tool("search_workspace", {"query": "hello", "glob": "*.py"})
        edited = self.runtime.dispatch_tool("edit_file", {
            "path": "main.py",
            "expected_revision": created["revision"],
            "replacements": [{
                "old_text": "hello",
                "new_text": "tool",
                "expected_occurrences": 1,
            }],
        })
        self._write_manifest()
        run = self.runtime.dispatch_tool("run_python_target", {"target_id": "main"})
        finished = self.runtime.dispatch_tool("finish_worker", {
            "summary": "done",
            "changed_files": [{"path": "main.py", "revision": edited["revision"]}],
        })
        deleted = self.runtime.dispatch_tool("delete_file", {"path": "main.py"})

        self.assertEqual(listed["files"][0]["path"], "main.py")
        self.assertIn("hello", viewed["content"])
        self.assertEqual(searched["matches"][0]["path"], "main.py")
        self.assertEqual(run["status"], "success")
        self.assertEqual(run["stdout"], "tool\n")
        self.assertEqual(finished["status"], "finished")
        self.assertTrue(deleted["operation_id"])
        self.assertTrue(self.runtime.finished)

        events = event_bus.get_event_bus().drain()
        tool_events = [event for event in events if event.event_type.startswith("file_tool_")]
        self.assertEqual(len(tool_events), 16)
        self.assertTrue(all("content" not in event.payload for event in tool_events))
        self.assertEqual(tool_events[0].payload["operation"], "create_file")
        self.assertEqual(tool_events[-1].payload["status"], "success")

    def test_malformed_unknown_budget_helper_and_cancellation_fail_cleanly(self):
        with self.assertRaises(WorkspaceError) as malformed:
            self.runtime.dispatch_tool("list_workspace", "not-json")
        self.assertEqual(malformed.exception.code, "invalid_tool_arguments")

        with self.assertRaises(WorkspaceError) as unknown:
            self.runtime.dispatch_tool("unknown_tool", {})
        self.assertEqual(unknown.exception.code, "unknown_tool")

        limited = WorkspaceToolRuntime(
            self.workspace,
            self.executor,
            run_id="limited",
            settings=WorkspaceToolRuntimeSettings(max_tool_calls=1),
        )
        limited.dispatch_tool("list_workspace", {})
        with self.assertRaises(WorkspaceError) as budget:
            limited.dispatch_tool("list_workspace", {})
        self.assertEqual(budget.exception.code, "tool_budget_exceeded")

        helpers = WorkspaceToolRuntime(
            self.workspace,
            self.executor,
            run_id="helpers",
            settings=WorkspaceToolRuntimeSettings(max_helpers=1),
            helper_callback=lambda task, files: {"summary": task, "files": files},
        )
        self.assertEqual(helpers.dispatch_tool("spawn_helper", {"task": "inspect"})["summary"], "inspect")
        with self.assertRaises(WorkspaceError) as helper_budget:
            helpers.dispatch_tool("spawn_helper", {"task": "again"})
        self.assertEqual(helper_budget.exception.code, "helper_budget_exceeded")

        canceled = WorkspaceToolRuntime(
            self.workspace,
            self.executor,
            run_id="canceled",
            cancel_check=lambda: True,
        )
        with self.assertRaises(WorkspaceError) as cancel:
            canceled.dispatch_tool("list_workspace", {})
        self.assertEqual(cancel.exception.code, "canceled")

    def test_helper_runtime_denies_writes_and_locks_are_released_on_failure(self):
        read_only = WorkspaceToolRuntime(
            self.workspace,
            self.executor,
            run_id="helper",
            read_only=True,
        )
        with self.assertRaises(WorkspaceError) as denied:
            read_only.dispatch_tool("create_file", {"path": "x.py", "content": ""})
        self.assertEqual(denied.exception.code, "tool_denied")

        created = self.workspace.create_file("main.py", "value = 1")
        with self.assertRaises(WorkspaceError):
            self.runtime.dispatch_tool("edit_file", {
                "path": "main.py",
                "expected_revision": created["revision"],
                "replacements": [{"old_text": "missing", "new_text": "x", "expected_occurrences": 1}],
            })
        saved = self.workspace.save_file("main.py", "value = 2", created["revision"])
        self.assertTrue(saved["revision"])

    def test_worker_loop_preserves_native_tool_ids_and_tool_responses(self):
        tool_call = SimpleNamespace(
            id="call-1",
            type="function",
            function=SimpleNamespace(
                name="finish_worker",
                arguments=json.dumps({"summary": "done", "changed_files": []}),
            ),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[tool_call]),
            finish_reason="tool_calls",
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(choices=[choice])
                )
            )
        )

        result = self.runtime.run_worker_loop(
            client,
            "fake",
            [{"role": "user", "content": "finish"}],
            context_config={"context_governor": {"enabled": False}},
        )

        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["messages"][1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(result["messages"][2]["role"], "tool")
        self.assertEqual(result["messages"][2]["tool_call_id"], "call-1")


class LlmNativeToolProtocolTests(unittest.TestCase):
    def test_chat_completion_message_returns_tool_calls_and_plain_text_still_works(self):
        tool_call = SimpleNamespace(
            id="tool-1",
            type="function",
            function=SimpleNamespace(name="list_workspace", arguments="{}"),
        )
        tool_choice = SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[tool_call]),
            finish_reason="tool_calls",
        )
        text_choice = SimpleNamespace(
            message=SimpleNamespace(content="plain"),
            finish_reason="stop",
        )
        calls = [SimpleNamespace(choices=[tool_choice]), SimpleNamespace(choices=[text_choice])]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: calls.pop(0))
            )
        )

        full = chat_completion_message(
            client,
            "fake",
            [{"role": "user", "content": "use tool"}],
            context_config={"context_governor": {"enabled": False}},
        )
        plain = chat_completion(
            client,
            "fake",
            [{"role": "user", "content": "answer"}],
            context_config={"context_governor": {"enabled": False}},
        )

        self.assertEqual(full["tool_calls"][0]["id"], "tool-1")
        self.assertEqual(full["tool_calls"][0]["function"]["name"], "list_workspace")
        self.assertEqual(plain, "plain")


if __name__ == "__main__":
    unittest.main()
