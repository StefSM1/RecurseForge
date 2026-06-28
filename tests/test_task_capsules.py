import json
import unittest

from engine.interfaces import TaskCapsule
from engine.redel import (
    build_execute_sections,
    normalize_task_capsule,
    parse_plan_response,
    render_task_capsule,
    spawn_children,
)


class TaskCapsuleTests(unittest.TestCase):
    def test_structured_capsule_parses_and_spawns(self):
        response = json.dumps({
            "delegate": True,
            "subtasks": [{
                "task": "Inspect retry handling",
                "role": "debugger",
                "goal": "Find overwritten history",
                "known_facts": ["Retries already exist"],
                "constraints": ["Keep event names"],
                "success_criteria": ["Name the responsible function"],
                "requested_files": ["engine/graph.py"],
                "requested_symbols": ["execute_node"],
                "return_format": "concise findings",
            }],
        })
        parsed = parse_plan_response(response)
        children = spawn_children(
            {"task_id": "root", "depth": 0}, parsed, 3, 4)
        self.assertEqual(children[0]["task"], "Inspect retry handling")
        capsule = TaskCapsule.model_validate(children[0]["task_capsule"])
        self.assertEqual(capsule.role, "debugger")
        self.assertEqual(capsule.requested_files, ["engine/graph.py"])

    def test_legacy_string_subtask_becomes_minimal_capsule(self):
        parsed = parse_plan_response(
            '{"delegate": true, "subtasks": ["Inspect graph.py"]}')
        capsule = TaskCapsule.model_validate(parsed["subtasks"][0])
        self.assertEqual(capsule.task, "Inspect graph.py")
        self.assertEqual(capsule.constraints, [])

    def test_mixed_and_imperfect_fields_are_normalized(self):
        response = json.dumps({
            "delegate": True,
            "subtasks": [
                "Legacy task",
                {"task": "Structured", "known_facts": "one fact",
                 "constraints": 42},
            ],
        })
        parsed = parse_plan_response(response)
        first = TaskCapsule.model_validate(parsed["subtasks"][0])
        second = TaskCapsule.model_validate(parsed["subtasks"][1])
        self.assertEqual(first.task, "Legacy task")
        self.assertEqual(second.known_facts, ["one fact"])
        self.assertEqual(second.constraints, [])

    def test_missing_task_falls_back_to_goal(self):
        capsule = normalize_task_capsule({"goal": "Find the defect"})
        self.assertEqual(capsule.task, "Find the defect")

    def test_max_children_still_applies_to_capsules(self):
        children = spawn_children(
            {"task_id": "root", "depth": 0},
            {"subtasks": [{"task": "one"}, {"task": "two"}]},
            max_depth=3,
            max_children=1,
        )
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["task"], "one")

    def test_capsule_render_omits_empty_fields(self):
        rendered = render_task_capsule(TaskCapsule(
            task="Inspect code", role="reviewer",
            success_criteria=["Report the faulty symbol"]))
        self.assertIn("TASK: Inspect code", rendered)
        self.assertIn("ROLE: reviewer", rendered)
        self.assertNotIn("KNOWN FACTS", rendered)

    def test_execute_sections_use_capsule_and_optional_repo_map(self):
        sections = build_execute_sections(
            "Legacy label",
            repo_map="<repo />",
            task_capsule={"task": "Focused task", "goal": "Precise outcome"},
        )
        by_name = {section.name: section for section in sections}
        self.assertTrue(by_name["task_capsule"].required)
        self.assertIn("GOAL: Precise outcome", by_name["task_capsule"].content)
        self.assertFalse(by_name["repo_map"].required)


if __name__ == "__main__":
    unittest.main()
