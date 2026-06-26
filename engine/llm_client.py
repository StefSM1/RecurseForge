"""
engine/llm_client.py
====================
Thin wrapper around the OpenAI SDK pointed at a local llama.cpp server.
Both graph.py and redel.py use this to talk to the model.
"""

import logging
from openai import OpenAI

logger = logging.getLogger("recurseforge.engine.llm_client")


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
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Try to disable thinking for planning steps
    if no_think:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    msg = choice.message

    # Primary: content field
    content = (msg.content or "").strip()

    # Qwen 3.5 thinking mode: model may put reasoning in reasoning_content
    # and leave content empty if max_tokens was consumed by reasoning.
    if not content:
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            logger.warning(
                "[LLM] Model used all %d tokens for reasoning, "
                "content is empty. finish_reason=%s. "
                "Increase max_tokens or add /no_think to disable thinking.",
                max_tokens, choice.finish_reason,
            )
            # Return reasoning as fallback so we don't lose the output entirely
            return reasoning.strip()

        if choice.finish_reason == "length":
            logger.warning(
                "[LLM] Response truncated at max_tokens=%d. "
                "Content is empty. Consider increasing max_tokens.",
                max_tokens,
            )

    return content
