import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from engine.context_governor import (
    ContextBudgetError,
    ContextEstimationError,
    extract_server_context_window,
    get_context_budget,
    preflight_messages,
    validate_context_config,
)
from engine.interfaces import ContextBudget
from engine.llm_client import chat_completion
from engine.textgrad import TextGradient, TextLoss, TextVariable, TGD


def config(**overrides):
    governor = {
        "enabled": True,
        "context_window": 100,
        "max_prompt_tokens": 70,
        "reserved_output_tokens": 20,
        "safety_buffer_tokens": 10,
    }
    governor.update(overrides)
    return {
        "llm": {"context_window": 100, "max_tokens": 20},
        "context_governor": governor,
    }


class ContextGovernorTests(unittest.TestCase):
    def test_budget_rejects_invalid_total(self):
        with self.assertRaises(ValidationError):
            ContextBudget(
                context_window=100,
                max_prompt_tokens=71,
                reserved_output_tokens=20,
                safety_buffer_tokens=10,
            )

    @patch("engine.context_governor.estimate_message_tokens", return_value=70)
    def test_exact_prompt_boundary_is_accepted(self, _estimate):
        report = preflight_messages([], 20, "root_plan", config())
        self.assertTrue(report.within_budget)
        self.assertEqual(report.remaining_prompt_tokens, 0)

    @patch("engine.context_governor.estimate_message_tokens", return_value=71)
    def test_one_token_over_is_rejected(self, _estimate):
        with self.assertRaises(ContextBudgetError) as ctx:
            preflight_messages([], 20, "child_execute", config())
        self.assertEqual(ctx.exception.report.call_kind, "child_execute")
        self.assertEqual(ctx.exception.report.remaining_prompt_tokens, -1)

    def test_larger_requested_output_reduces_prompt_ceiling(self):
        budget = get_context_budget(config(), requested_output_tokens=30)
        self.assertEqual(budget.reserved_output_tokens, 30)
        self.assertEqual(budget.max_prompt_tokens, 60)

    @patch("engine.context_governor.estimate_message_tokens",
           side_effect=RuntimeError("tokenizer unavailable"))
    def test_estimator_failure_is_controlled(self, _estimate):
        with self.assertRaises(ContextEstimationError) as ctx:
            preflight_messages([], 20, "root_plan", config())
        self.assertIn("request was not sent", str(ctx.exception))

    def test_duplicate_context_windows_must_match(self):
        cfg = config(context_window=99)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_context_config(cfg)

    def test_invalid_configured_total_is_not_silently_normalized(self):
        cfg = config(max_prompt_tokens=71)
        with self.assertRaises(ValidationError):
            validate_context_config(cfg)

    def test_llama_props_context_extraction(self):
        props = {"default_generation_settings": {"n_ctx": 65536}}
        self.assertEqual(extract_server_context_window(props), 65536)

    @patch("engine.context_governor.estimate_message_tokens", return_value=71)
    def test_chat_completion_rejects_before_client_call(self, _estimate):
        client = Mock()
        with self.assertRaises(ContextBudgetError):
            chat_completion(
                client=client,
                model="fake",
                messages=[{"role": "user", "content": "too large"}],
                max_tokens=20,
                call_kind="root_plan",
                context_config=config(),
            )
        client.chat.completions.create.assert_not_called()

    @patch("engine.context_governor.estimate_message_tokens", return_value=5)
    def test_length_finish_reason_returns_partial_content_with_warning(
        self, _estimate,
    ):
        choice = SimpleNamespace(
            message=SimpleNamespace(content="partial"),
            finish_reason="length",
        )
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[choice])
        with self.assertLogs("recurseforge.engine.llm_client", "WARNING") as logs:
            result = chat_completion(
                client=client,
                model="fake",
                messages=[{"role": "user", "content": "answer"}],
                max_tokens=20,
                call_kind="child_execute",
                context_config=config(),
            )
        self.assertEqual(result, "partial")
        self.assertIn("generation limit", " ".join(logs.output))

    @patch("engine.textgrad.chat_completion")
    def test_textgrad_evaluator_uses_governed_call_kind(self, completion):
        completion.return_value = "LINE: 1\nCAUSE: bad\nFIX: repair"
        loss = TextLoss(
            client=Mock(),
            model="fake",
            context_config=config(),
        )
        loss(
            TextVariable("print('x')", requires_grad=True),
            {"task": "fix", "stdout": "", "stderr": "bad"},
        )
        self.assertEqual(
            completion.call_args.kwargs["call_kind"], "textgrad_evaluate")
        self.assertIs(completion.call_args.kwargs["context_config"],
                      loss.context_config)

    @patch("engine.textgrad.chat_completion")
    def test_textgrad_updater_uses_governed_call_kind(self, completion):
        completion.return_value = "```python\nprint('fixed')\n```"
        variable = TextVariable("print('old')", requires_grad=True)
        variable.grad = TextGradient(
            node_id="textgrad",
            loss_description="bad",
            mutations=[],
        )
        optimizer = TGD(
            client=Mock(),
            model="fake",
            parameters=[variable],
            context_config=config(),
        )
        optimizer.step()
        self.assertEqual(
            completion.call_args.kwargs["call_kind"], "textgrad_update")
        self.assertEqual(variable.value, "print('fixed')")


if __name__ == "__main__":
    unittest.main()
