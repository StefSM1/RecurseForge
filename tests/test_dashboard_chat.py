import unittest
from unittest.mock import patch

from fastapi import HTTPException

from harness import dashboard_server


class FakeThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False

    def start(self):
        self.started = True


class DashboardChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with dashboard_server._chat_runs_lock:
            dashboard_server._chat_runs.clear()

    async def test_start_chat_run_rejects_empty_message(self):
        with self.assertRaises(HTTPException) as ctx:
            await dashboard_server.start_chat_run({"message": "  "})
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_start_chat_run_creates_record_without_invoking_llm_inline(self):
        with patch.object(dashboard_server.threading, "Thread", FakeThread):
            response = await dashboard_server.start_chat_run({"message": "Solve it"})

        self.assertEqual(response["prompt"], "Solve it")
        self.assertEqual(response["status"], "pending")
        self.assertTrue(response["run_id"].startswith("chat-"))
        with dashboard_server._chat_runs_lock:
            self.assertIn(response["run_id"], dashboard_server._chat_runs)

    async def test_only_one_active_chat_run_is_allowed(self):
        with dashboard_server._chat_runs_lock:
            dashboard_server._chat_runs["chat-active"] = {
                "run_id": "chat-active",
                "prompt": "First",
                "status": "running",
                "created_at": 1,
                "started_at": 1,
                "completed_at": None,
                "final_output": None,
                "error": None,
                "stop_requested": False,
            }

        with self.assertRaises(HTTPException) as ctx:
            await dashboard_server.start_chat_run({"message": "Second"})
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_stop_marks_active_run_as_stopping(self):
        with dashboard_server._chat_runs_lock:
            dashboard_server._chat_runs["chat-run"] = {
                "run_id": "chat-run",
                "prompt": "Task",
                "status": "running",
                "created_at": 1,
                "started_at": 1,
                "completed_at": None,
                "final_output": None,
                "error": None,
                "stop_requested": False,
            }

        response = await dashboard_server.stop_chat_run("chat-run")
        self.assertTrue(response["stop_requested"])
        self.assertEqual(response["status"], "stopping")

    def test_format_chat_output_prefers_direct_answer(self):
        output = dashboard_server._format_chat_output({
            "direct_answer": "Final answer",
            "results": [{"result": "ignored"}],
        })
        self.assertEqual(output, "Final answer")

    def test_format_chat_output_summarizes_delegated_results(self):
        output = dashboard_server._format_chat_output({
            "results": [{
                "task": "Subtask",
                "result": "Done",
                "result_frame": {"summary": "Compact done"},
                "success": True,
                "code_executed": True,
                "exit_code": 0,
                "stdout": "ok",
            }],
        })
        self.assertIn("[1] Subtask (success)", output)
        self.assertIn("Compact done", output)
        self.assertNotIn("\nDone\n", output)
        self.assertIn("Sandbox exit: 0", output)
        self.assertIn("stdout:\nok", output)

    def test_run_chat_graph_stores_final_output_with_mocked_graph(self):
        class FakeGraph:
            def invoke(self, state):
                self.state = state
                return {"status": "done", "direct_answer": "Mocked final"}

        fake_graph = FakeGraph()
        with dashboard_server._chat_runs_lock:
            dashboard_server._chat_runs["chat-run"] = {
                "run_id": "chat-run",
                "prompt": "Task",
                "status": "pending",
                "created_at": 1,
                "started_at": None,
                "completed_at": None,
                "final_output": None,
                "error": None,
                "stop_requested": False,
            }

        with patch.object(dashboard_server, "_load_dashboard_config", return_value={}), \
                patch("engine.graph.build_graph", return_value=fake_graph):
            dashboard_server._run_chat_graph("chat-run", "Task")

        with dashboard_server._chat_runs_lock:
            record = dashboard_server._chat_runs["chat-run"]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["final_output"], "Mocked final")
        self.assertEqual(fake_graph.state["run_id"], "chat-run")


if __name__ == "__main__":
    unittest.main()
