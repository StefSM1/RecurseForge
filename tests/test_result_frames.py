import json
import unittest

from engine.context_governor import count_text_tokens
from engine.result_frames import build_result_frame


class ResultFrameTests(unittest.TestCase):
    def test_clean_json_parses(self):
        raw = json.dumps({
            "status": "success",
            "summary": "Found the parser.",
            "evidence": [{
                "file_path": "engine/redel.py",
                "symbol_name": "parse_plan_response",
                "line_start": 243,
                "line_end": 280,
                "finding": "This function parses planning JSON.",
            }],
            "changes_needed": [],
            "risks": [{"description": "Malformed JSON", "severity": "low"}],
            "open_questions": [],
            "confidence": 0.9,
        })
        frame = build_result_frame(raw, "child-1", True)
        self.assertEqual(frame.summary, "Found the parser.")
        self.assertEqual(frame.evidence[0].symbol_name, "parse_plan_response")
        self.assertEqual(frame.confidence, 0.9)

    def test_fenced_and_embedded_json_parse(self):
        fenced = "Before\n```json\n" + json.dumps({
            "status": "success", "summary": "Fenced",
            "evidence": [],
        }) + "\n```"
        embedded = "Solution text\nRESULT_FRAME=" + json.dumps({
            "status": "success", "summary": "Embedded", "risks": [],
        })
        self.assertEqual(
            build_result_frame(fenced, "a", True).summary, "Fenced")
        self.assertEqual(
            build_result_frame(embedded, "b", True).summary, "Embedded")

    def test_prose_fallback_is_bounded(self):
        raw = "ordinary prose " * 500
        frame = build_result_frame(raw, "child-1", True, {
            "result_frames": {"max_summary_chars": 120}})
        self.assertTrue(frame.summary.startswith("ordinary prose"))
        self.assertLessEqual(len(frame.summary), 120)

    def test_sandbox_status_is_authoritative(self):
        raw = json.dumps({
            "status": "success", "summary": "Model claimed success",
            "evidence": [],
        })
        frame = build_result_frame(raw, "child-1", False)
        self.assertEqual(frame.status, "failed")

    def test_oversized_frame_is_deterministically_limited(self):
        raw = json.dumps({
            "status": "success",
            "summary": "s" * 4000,
            "evidence": [
                {"file_path": "file.py", "finding": "f" * 1000}
                for _ in range(20)
            ],
            "changes_needed": ["c" * 500 for _ in range(20)],
            "risks": [
                {"description": "r" * 500, "severity": "high"}
                for _ in range(20)
            ],
            "open_questions": ["q" * 500 for _ in range(20)],
        })
        cfg = {"result_frames": {
            "max_tokens": 140,
            "max_summary_chars": 1800,
            "max_evidence": 6,
            "max_finding_chars": 500,
            "max_changes": 6,
            "max_risks": 4,
            "max_questions": 4,
            "max_item_chars": 400,
        }}
        first = build_result_frame(raw, "child-1", True, cfg)
        second = build_result_frame(raw, "child-1", True, cfg)
        self.assertLessEqual(count_text_tokens(first.model_dump_json()), 140)
        self.assertEqual(first.model_dump(), second.model_dump())


if __name__ == "__main__":
    unittest.main()
