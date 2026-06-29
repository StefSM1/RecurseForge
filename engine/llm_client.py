"""
engine/llm_client.py
====================
Thin wrapper around the OpenAI SDK pointed at a local llama.cpp server.
Both graph.py and redel.py use this to talk to the model.
"""

import logging
from typing import Any

from openai import OpenAI

from engine.context_governor import (
    build_context_bundle,
    context_governor_enabled,
    preflight_messages,
)
from engine.interfaces import ContextSection

logger = logging.getLogger("recurseforge.engine.llm_client")


def _message_has_native_tool_fields(message: dict[str, Any]) -> bool:
    return "tool_calls" in message or "tool_call_id" in message


def _prepare_messages(
    messages: list[dict],
    max_tokens: int,
    call_kind: str,
    context_config: dict[str, Any] | None,
    context_sections: list[ContextSection] | None,
) -> list[dict]:
    """Apply context checks while preserving native tool protocol messages."""
    has_tool_protocol = any(_message_has_native_tool_fields(message) for message in messages)
    if context_governor_enabled(context_config) and not has_tool_protocol:
        sections = context_sections or [
            ContextSection(
                name="legacy_message_{}".format(index),
                role=message.get("role", "user"),
                content=str(message.get("content", "")),
                required=True,
                priority=100,
            )
            for index, message in enumerate(messages)
        ]
        return build_context_bundle(
            call_kind=call_kind,
            sections=sections,
            max_tokens=max_tokens,
            config=context_config or {},
        ).messages

    preflight_messages(
        messages=messages,
        max_tokens=max_tokens,
        call_kind=call_kind,
        config=context_config,
    )
    return messages


def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    function = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": {
            "name": getattr(function, "name", "") if function is not None else "",
            "arguments": getattr(function, "arguments", "") if function is not None else "",
        },
    }


def get_client(base_url: str) -> OpenAI:
    """Create an OpenAI client pointing at the local llama.cpp server."""
    # llama.cpp's OpenAI-compatible endpoint does not require a real key,
    # but the SDK insists on something non-empty.
    return OpenAI(base_url=base_url, api_key="not-needed")


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 8192,
    temperature: float = 0.3,
    no_think: bool = False,
    call_kind: str = "unspecified",
    context_config: dict[str, Any] | None = None,
    context_sections: list[ContextSection] | None = None,
) -> str:
    """
    Send a chat completion request and return the assistant's text response.

    Handles Qwen 3.5's thinking mode: if the model spends tokens on
    reasoning_content and leaves content empty, we log a warning and
    return the reasoning as a fallback.

    Args:
        client: OpenAI client (from get_client).
        model: Model name string (llama.cpp accepts anything).
        messages: List of {"role": ..., "content": ...} dicts.
        max_tokens: Max tokens to generate (shared between reasoning + content).
        temperature: Sampling temperature.
        no_think: If True, attempt to disable thinking mode via extra params.

    Returns:
        The assistant's response text, stripped.
    """
    message = chat_completion_message(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        no_think=no_think,
        call_kind=call_kind,
        context_config=context_config,
        context_sections=context_sections,
    )

    # Primary: content field
    content = (message.get("content") or "").strip()

    if message.get("finish_reason") == "length":
        logger.warning(
            "[LLM] Response for call=%s was truncated at max_tokens=%d "
            "(partial content: %d chars). The context window and generation "
            "limit are separate budgets.",
            call_kind, max_tokens, len(content),
        )

    # Qwen 3.5 thinking mode: model may put reasoning in reasoning_content
    # and leave content empty if max_tokens was consumed by reasoning.
    if not content:
        reasoning = message.get("reasoning_content")
        if reasoning:
            logger.warning(
                "[LLM] Model used all %d tokens for reasoning, "
                "content is empty. finish_reason=%s. "
                "Increase max_tokens or add /no_think to disable thinking.",
                max_tokens, message.get("finish_reason"),
            )
            # Return reasoning as fallback so we don't lose the output entirely
            return reasoning.strip()

    return content


def chat_completion_message(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 8192,
    temperature: float = 0.3,
    no_think: bool = False,
    call_kind: str = "unspecified",
    context_config: dict[str, Any] | None = None,
    context_sections: list[ContextSection] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the full assistant message, including native tool calls."""
    prepared_messages = _prepare_messages(
        messages=messages,
        max_tokens=max_tokens,
        call_kind=call_kind,
        context_config=context_config,
        context_sections=context_sections,
    )

    kwargs = dict(
        model=model,
        messages=prepared_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if no_think:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    msg = choice.message
    tool_calls = getattr(msg, "tool_calls", None) or []
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [_serialize_tool_call(tool_call) for tool_call in tool_calls],
        "reasoning_content": getattr(msg, "reasoning_content", None),
        "finish_reason": choice.finish_reason,
    }
