import json
import sys
import unittest
from unittest.mock import patch

from engine import graph
from engine.interfaces import EventType
from harness import event_bus
from harness.sandbox import SandboxPool


def config(*, textgrad=False, retries=1, timeout=2):
    return {
        "llm": {
            "base_url": "http://unused/v1",
            "model_name": "fake",
            "max_tokens": 256,
            "temperature": 0.0,
        },
        "recursion": {
            "max_depth": 1,
            "max_children": 2,
            "max_retries": retries,
        },
        "sandbox": {"pool_size": 1, "timeout_s": timeout},
        "textgrad": {
            "enabled": textgrad,
            "max_iterations": 1,
            "eval_temperature": 0.1,
            "update_temperature": 0.2,
        },
    }


def state(cfg, *, children=None):
    return {
        "task_id": "run-test",
        "task_description": "test task",
        "status": "executing",
        "children": children or [],
        "depth": 0,
        "results": [],
        "direct_answer": "",
        "config": cfg,
    }


class BackendTelemetryTests(unittest.TestCase):
    def setUp(self):
        event_bus._bus = event_bus.EventBus()
        graph._sandbox_pool = None

    def tearDown(self):
        if graph._sandbox_pool is not None:
            graph._sandbox_pool.shutdown()
            graph._sandbox_pool = None

    def events(self):
        return event_bus.get_event_bus().drain()

    def event_types(self):
        return [event.event_type for event in self.events()]

    def run_child(self, response, cfg=None):
        child = {"node_id": "child-1", "parent_id": "root", "task": "run code"}
        with patch.object(graph, "get_client", return_value=object()), \
                patch.object(graph, "_fetch_repo_map", return_value=""), \
                patch.object(graph, "chat_completion", return_value=response):
            return graph.execute_node(state(cfg or config(), children=[child]))

    def test_first_attempt_success_event_order(self):
        result = self.run_child("```python\nprint('ok')\n```")
        self.assertTrue(result["results"][0]["success"])
        self.assertEqual(self.event_types(), [
            EventType.SANDBOX_STARTED.value,
            EventType.SANDBOX_COMPLETED.value,
            EventType.NODE_COMPLETE.value,
        ])

    def test_failure_textgrad_success_event_order(self):
        def fake_gradient_fix(**kwargs):
            callback = kwargs["progress_callback"]
            callback("evaluating_loss", {"iteration": 1})
            callback("gradient_ready", {
                "iteration": 1, "severity": 0.5, "num_mutations": 1,
                "mutations": [{"line": 1, "cause": "boom", "suggestion": "print"}],
            })
            callback("applying_update", {"iteration": 1})
            callback("iteration_complete", {"iteration": 1})
            return "print('fixed')", [{
                "iteration": 1, "severity": 0.5, "num_mutations": 1,
                "mutations": [],
            }]

        with patch.object(graph, "gradient_fix", side_effect=fake_gradient_fix):
            result = self.run_child(
                "```python\nraise RuntimeError('boom')\n```",
                config(textgrad=True),
            )

        self.assertTrue(result["results"][0]["success"])
        self.assertEqual(self.event_types(), [
            "sandbox_started", "sandbox_completed", "correction_started",
            "correction_progress", "correction_progress", "correction_progress",
            "correction_progress", "gradient_flow", "correction_completed",
            "sandbox_started", "sandbox_completed", "node_complete",
        ])

    def test_textgrad_failure_falls_back_to_llm_retry(self):
        child = {"node_id": "child-1", "parent_id": "root", "task": "run code"}
        responses = [
            "```python\nraise RuntimeError('bad')\n```",
            "```python\nprint('retry fixed')\n```",
        ]
        with patch.object(graph, "get_client", return_value=object()), \
                patch.object(graph, "_fetch_repo_map", return_value=""), \
                patch.object(graph, "chat_completion", side_effect=responses), \
                patch.object(graph, "gradient_fix", side_effect=RuntimeError("tg unavailable")):
            result = graph.execute_node(
                state(config(textgrad=True), children=[child]))

        self.assertTrue(result["results"][0]["success"])
        types = self.event_types()
        self.assertEqual(types.count("correction_started"), 2)
        self.assertEqual(types.count("correction_completed"), 2)
        self.assertEqual(types[-3:], [
            "sandbox_started", "sandbox_completed", "node_complete"])

    def test_final_failure_is_terminal(self):
        result = self.run_child(
            "```python\nraise RuntimeError('final')\n```",
            config(retries=0),
        )
        self.assertFalse(result["results"][0]["success"])
        events = self.events()
        self.assertEqual(events[-1].event_type, "node_complete")
        self.assertFalse(events[-1].payload["success"])
        self.assertIn("final", events[-1].payload["failure_reason"])

    def test_timeout_emits_failed_completion(self):
        pool = SandboxPool(timeout_s=1, python_executable=sys.executable)
        try:
            result = pool.execute(
                "root", "import time\ntime.sleep(2)", run_id="timeout-run",
                execution_id="timeout-exec", attempt=1, trigger="initial")
        finally:
            pool.shutdown()
        self.assertEqual(result.exit_code, -1)
        events = self.events()
        self.assertEqual([e.event_type for e in events], [
            "sandbox_started", "sandbox_completed"])
        self.assertEqual(events[-1].payload["status"], "failed")
        self.assertEqual(events[-1].payload["exit_code"], -1)

    def test_direct_root_code_has_run_and_sandbox_lifecycle(self):
        answer = "```python\nprint('root ok')\n```"
        llm_output = json.dumps({"delegate": False, "answer": answer})
        with patch.object(graph, "get_client", return_value=object()), \
                patch.object(graph, "chat_completion", return_value=llm_output):
            direct_state = state(config())
            direct_state.update(graph.init_node(direct_state))
            result = graph.plan_node(direct_state)

        self.assertIn("root ok", result["direct_answer"])
        events = self.events()
        self.assertEqual([event.event_type for event in events], [
            "run_started", "sandbox_started", "sandbox_completed", "run_completed"])
        self.assertEqual(len({event.run_id for event in events}), 1)
        self.assertEqual(events[-1].payload["result"],
                         result["direct_answer"])

    def test_graph_initialization_creates_unique_run_ids(self):
        first = graph.init_node(state(config()))["run_id"]
        second = graph.init_node(state(config()))["run_id"]
        self.assertNotEqual(first, second)

    def test_direct_root_textgrad_failure_uses_llm_retry(self):
        answers = [
            json.dumps({
                "delegate": False,
                "answer": "```python\nraise RuntimeError('root bad')\n```",
            }),
            "```python\nprint('root retry fixed')\n```",
        ]
        with patch.object(graph, "get_client", return_value=object()), \
                patch.object(graph, "chat_completion", side_effect=answers), \
                patch.object(graph, "gradient_fix", side_effect=RuntimeError("tg failed")):
            result = graph.plan_node(state(config(textgrad=True)))

        self.assertIn("root retry fixed", result["direct_answer"])
        events = self.events()
        self.assertEqual([e.event_type for e in events].count("correction_started"), 2)
        self.assertTrue(events[-1].payload["success"])
        self.assertEqual(events[-1].payload["mode"], "direct")

    def test_delegated_validation_emits_run_completion(self):
        cfg = config()
        validation_state = state(cfg)
        validation_state["results"] = [{
            "result": "done", "success": False, "code_executed": True,
            "attempts": 1,
        }]
        graph.validate_node(validation_state)
        event = self.events()[0]
        self.assertEqual(event.event_type, "run_completed")
        self.assertEqual(event.payload["mode"], "delegated")
        self.assertFalse(event.payload["success"])

    def test_text_only_child_bypasses_sandbox(self):
        result = self.run_child("A plain-language answer.")
        self.assertTrue(result["results"][0]["success"])
        events = self.events()
        self.assertEqual([event.event_type for event in events], ["node_complete"])
        event = events[0]
        self.assertEqual(event.payload["result"], "A plain-language answer.")

    def test_engine_events_have_identity_run_and_engine_timestamp(self):
        self.run_child("A plain-language answer.")
        event = self.events()[0]
        self.assertTrue(event.event_id)
        self.assertEqual(event.run_id, "run-test")
        self.assertGreater(event.timestamp, 0)


if __name__ == "__main__":
    unittest.main()
