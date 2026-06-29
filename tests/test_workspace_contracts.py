import unittest

from engine.interfaces import (
    DebugVerdict,
    EventType,
    FileToolRun,
    PlanFrame,
    RunMode,
    StageRun,
    WorkspaceAgentState,
    WorkspaceFileRef,
)


class WorkspaceContractTests(unittest.TestCase):
    def test_workspace_state_round_trip(self):
        state = WorkspaceAgentState(
            run_id="run-1",
            pass_number=2,
            workspace_revision=7,
            plan=PlanFrame(objective="Build", steps=["Inspect", "Edit"]),
            changed_files=[WorkspaceFileRef(path="main.py", revision="abc", size_bytes=12)],
            debug_verdict=DebugVerdict(verdict="revise", findings=["Missing branch"]),
        )
        restored = WorkspaceAgentState.from_json(state.to_json())
        self.assertEqual(restored, state)

    def test_run_and_tool_contract_defaults_are_stable(self):
        self.assertEqual(RunMode.WORKSPACE_AGENT.value, "workspace_agent")
        tool = FileToolRun(
            run_id="run-1", actor_id="worker", stage="worker",
            pass_number=1, operation="view", path="main.py",
        )
        stage = StageRun(run_id="run-1", stage="plan", pass_number=1)
        self.assertEqual(tool.status, "pending")
        self.assertEqual(stage.status, "pending")
        self.assertTrue(tool.tool_run_id)

    def test_event_additions_are_additive(self):
        values = {item.value for item in EventType}
        self.assertIn("node_spawn", values)
        self.assertTrue({
            "stage_started", "stage_completed", "file_tool_started",
            "file_tool_completed", "workspace_changed",
            "workspace_lock_changed",
        }.issubset(values))


if __name__ == "__main__":
    unittest.main()
