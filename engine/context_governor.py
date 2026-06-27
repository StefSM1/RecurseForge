"""Token measurement and hard context limits for every LLM request."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from engine.interfaces import ContextBudget, ContextBudgetReport

logger = logging.getLogger("recurseforge.engine.context_governor")

_MESSAGE_OVERHEAD_TOKENS = 4
_REPLY_OVERHEAD_TOKENS = 3


class ContextGovernorError(RuntimeError):
    """Base class for context preflight failures."""


class ContextBudgetError(ContextGovernorError):
    """Raised before inference when a prompt cannot fit its hard budget."""

    def __init__(self, report: ContextBudgetReport):
        self.report = report
        super().__init__(
            "Context budget exceeded for '{}': estimated prompt {} tokens, "
            "prompt ceiling {}, output reserve {}, safety reserve {}, context "
            "window {}. Phase 2 deterministic trimming is required for this "
            "request.".format(
                report.call_kind,
                report.estimated_prompt_tokens,
                report.max_prompt_tokens,
                report.reserved_output_tokens,
                report.safety_buffer_tokens,
                report.effective_context_window,
            )
        )


class ContextEstimationError(ContextGovernorError):
    """Raised when prompt measurement fails; inference must not bypass it."""


def count_text_tokens(text: str) -> int:
    """Conservatively estimate tokens without network tokenizer assets."""
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate serialized chat tokens, including a wrapper allowance."""
    total = _REPLY_OVERHEAD_TOKENS
    for message in messages:
        total += _MESSAGE_OVERHEAD_TOKENS
        for key, value in message.items():
            total += count_text_tokens(str(key))
            if isinstance(value, str):
                content = value
            else:
                content = json.dumps(
                    value, ensure_ascii=True, sort_keys=True, default=str)
            total += count_text_tokens(content)
    return total


def context_governor_enabled(config: dict[str, Any] | None) -> bool:
    return bool((config or {}).get("context_governor", {}).get("enabled", False))


def get_context_budget(
    config: dict[str, Any],
    requested_output_tokens: int | None = None,
) -> ContextBudget:
    """Build and validate the effective budget from application config."""
    llm_cfg = config.get("llm", {})
    governor_cfg = config.get("context_governor", {})
    context_window = int(
        governor_cfg.get("context_window", llm_cfg.get("context_window", 65536))
    )
    configured_output = int(governor_cfg.get(
        "reserved_output_tokens", llm_cfg.get("max_tokens", 8192)))
    reserved_output = max(configured_output, int(requested_output_tokens or 0))
    safety_buffer = int(governor_cfg.get("safety_buffer_tokens", 8192))
    default_prompt_limit = context_window - reserved_output - safety_buffer

    configured_prompt_limit = int(governor_cfg.get(
        "max_prompt_tokens", default_prompt_limit))
    effective_prompt_limit = min(configured_prompt_limit, default_prompt_limit)

    return ContextBudget(
        context_window=context_window,
        max_prompt_tokens=effective_prompt_limit,
        reserved_output_tokens=reserved_output,
        safety_buffer_tokens=safety_buffer,
    )


def validate_context_config(config: dict[str, Any]) -> ContextBudget | None:
    """Validate an enabled governor and detect duplicated window mismatches."""
    if not context_governor_enabled(config):
        return None

    llm_window = config.get("llm", {}).get("context_window")
    governor_window = config.get("context_governor", {}).get("context_window")
    if llm_window is not None and governor_window is not None:
        if int(llm_window) != int(governor_window):
            raise ValueError(
                "llm.context_window ({}) does not match "
                "context_governor.context_window ({})".format(
                    llm_window, governor_window))
    governor_cfg = config.get("context_governor", {})
    return ContextBudget(
        context_window=int(governor_cfg.get(
            "context_window", config.get("llm", {}).get("context_window", 65536))),
        max_prompt_tokens=int(governor_cfg.get("max_prompt_tokens", 49152)),
        reserved_output_tokens=int(governor_cfg.get(
            "reserved_output_tokens", config.get("llm", {}).get("max_tokens", 8192))),
        safety_buffer_tokens=int(governor_cfg.get("safety_buffer_tokens", 8192)),
    )


def preflight_messages(
    messages: list[dict[str, Any]],
    max_tokens: int,
    call_kind: str,
    config: dict[str, Any] | None,
) -> ContextBudgetReport | None:
    """Measure and reject an oversized prompt before inference."""
    if not context_governor_enabled(config):
        return None

    budget = get_context_budget(config or {}, max_tokens)
    try:
        estimated = estimate_message_tokens(messages)
    except Exception as exc:
        raise ContextEstimationError(
            "Could not estimate prompt tokens for '{}'; request was not sent: "
            "{}".format(call_kind, exc)) from exc
    prompt_remaining = budget.max_prompt_tokens - estimated
    window_remaining = (
        budget.context_window
        - budget.reserved_output_tokens
        - budget.safety_buffer_tokens
        - estimated
    )
    remaining = min(prompt_remaining, window_remaining)
    within_budget = remaining >= 0
    report = ContextBudgetReport(
        call_kind=call_kind,
        estimated_prompt_tokens=estimated,
        reserved_output_tokens=budget.reserved_output_tokens,
        safety_buffer_tokens=budget.safety_buffer_tokens,
        effective_context_window=budget.context_window,
        max_prompt_tokens=budget.max_prompt_tokens,
        remaining_prompt_tokens=remaining,
        within_budget=within_budget,
        estimator_name="utf8_bytes_per_3_conservative",
    )

    logger.info(
        "[CONTEXT] call=%s prompt=%d/%d (%.1f%%) output=%d safety=%d "
        "remaining=%d",
        call_kind,
        estimated,
        budget.max_prompt_tokens,
        (estimated / budget.max_prompt_tokens) * 100,
        budget.reserved_output_tokens,
        budget.safety_buffer_tokens,
        remaining,
    )
    if not within_budget:
        raise ContextBudgetError(report)
    return report


def extract_server_context_window(props: dict[str, Any]) -> int | None:
    """Read n_ctx from known llama.cpp `/props` response locations."""
    candidates = [
        props.get("n_ctx"),
        props.get("context_size"),
        props.get("default_generation_settings", {}).get("n_ctx"),
    ]
    for candidate in candidates:
        if isinstance(candidate, (int, float)) and int(candidate) > 0:
            return int(candidate)
    return None
