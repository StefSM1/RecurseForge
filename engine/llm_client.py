"""
engine/llm_client.py
====================
Thin wrapper around the OpenAI SDK pointed at a local llama.cpp server.
Both graph.py and redel.py use this to talk to the model.
"""

from openai import OpenAI


def get_client(base_url: str) -> OpenAI:
    """Create an OpenAI client pointing at the local llama.cpp server."""
    # llama.cpp's OpenAI-compatible endpoint does not require a real key,
    # but the SDK insists on something non-empty.
    return OpenAI(base_url=base_url, api_key="not-needed")


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """
    Send a chat completion request and return the assistant's text response.

    Args:
        client: OpenAI client (from get_client).
        model: Model name string (llama.cpp accepts anything).
        messages: List of {"role": ..., "content": ...} dicts.
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text, stripped.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
