"""
engine/redel.py
===============
ReDel (Recursive Delegation) layer.

This module handles:
  1. Building the delegation prompt that asks the LLM to plan.
  2. Parsing the LLM's JSON response (delegate or answer directly).
  3. Spawning child node dicts with max_depth / max_children enforcement.
  4. Building execution prompts for sub-agents.
"""

import json
import uuid

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """\
You are a task decomposition agent inside a recursive agent framework.
Your job is to decide whether a given task should be broken into sub-tasks
or answered directly.

RULES:
- Current recursion depth: {depth}/{max_depth}.
  If depth equals max_depth, you MUST answer directly -- no more delegation.
- Maximum sub-tasks per level: {max_children}.
- Each sub-task must be independently solvable without referencing siblings.
- Keep sub-task descriptions clear and self-contained.

Respond in this EXACT JSON format (no markdown, no code fences, raw JSON):
  Option A (delegate): {{"delegate": true, "subtasks": ["task 1", "task 2"]}}
  Option B (answer):   {{"delegate": false, "answer": "your complete answer here"}}
"""

EXECUTE_SYSTEM_PROMPT = """\
You are a sub-agent inside a recursive agent framework.
You have been assigned a single, specific task. Complete it thoroughly.
Do NOT ask clarifying questions -- work with what you have.
Return your complete result as plain text.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_plan_messages(
    task_description: str,
    depth: int,
    max_depth: int,
    max_children: int,
) -> list[dict]:
    """
    Build the message list for the planning step.

    Returns:
        A list of {"role": ..., "content": ...} dicts ready for the LLM.
    """
    system = PLAN_SYSTEM_PROMPT.format(
        depth=depth,
        max_depth=max_depth,
        max_children=max_children,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task_description},
    ]


def build_execute_messages(task: str) -> list[dict]:
    """
    Build the message list for executing a single sub-task.

    Returns:
        A list of {"role": ..., "content": ...} dicts ready for the LLM.
    """
    return [
        {"role": "system", "content": EXECUTE_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_plan_response(llm_output: str) -> dict:
    """
    Parse the LLM's planning response into a structured dict.

    The LLM should return raw JSON. We try to extract it even if the model
    wraps it in markdown code fences or adds surrounding text.

    Returns:
        {"delegate": True,  "subtasks": [...]}  -- if the LLM wants to delegate
        {"delegate": False, "answer": "..."}     -- if the LLM answers directly
        {"delegate": False, "answer": llm_output} -- fallback on parse failure
    """
    # Try to extract JSON from the response
    text = llm_output.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)

    # Try to find JSON object in the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_str = text[brace_start:brace_end + 1]
        try:
            parsed = json.loads(json_str)
            if "delegate" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: treat entire output as a direct answer
    return {"delegate": False, "answer": llm_output}


# ---------------------------------------------------------------------------
# Child spawning
# ---------------------------------------------------------------------------

def spawn_children(
    state: dict,
    plan_response: dict,
    max_depth: int,
    max_children: int,
) -> list[dict]:
    """
    Create child node dicts from a delegation plan.

    Enforces:
      - max_depth: returns empty list if current depth >= max_depth.
      - max_children: truncates the subtask list.

    Args:
        state: The current graph state (needs "task_id" and "depth").
        plan_response: Parsed LLM response from parse_plan_response().
        max_depth: Hard ceiling on recursion depth.
        max_children: Max sub-agents per level.

    Returns:
        A list of child node dicts. Empty list if delegation is blocked.
    """
    current_depth = state.get("depth", 0)

    # Block delegation at max depth
    if current_depth >= max_depth:
        return []

    subtasks = plan_response.get("subtasks", [])
    if not subtasks:
        return []

    # Enforce max_children
    subtasks = subtasks[:max_children]

    parent_id = state.get("task_id", "root")
    children = []
    for task_text in subtasks:
        children.append({
            "node_id": str(uuid.uuid4())[:8],  # short unique id
            "parent_id": parent_id,
            "task": task_text,
            "depth": current_depth + 1,
            "result": None,
        })

    return children
