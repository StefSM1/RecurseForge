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
import re
import uuid

from engine.interfaces import ContextSection, TaskCapsule

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """\
/no_think
You are a task decomposition agent inside a recursive agent framework.
Your job is to decide whether a given task should be broken into sub-tasks
or answered directly.

RULES:
- Current recursion depth: {depth}/{max_depth}.
  If depth equals max_depth, you MUST answer directly -- no more delegation.
- Maximum sub-tasks per level: {max_children}.
- Each sub-task must be independently solvable without referencing siblings.
- Return every sub-task as a compact object with: task, role, goal,
  known_facts, constraints, success_criteria, requested_files,
  requested_symbols, and return_format.
- Use empty arrays or empty strings when optional details are unavailable.
- Keep assignments clear, factual, and self-contained. Do not copy your
  reasoning transcript into a sub-task.

Respond in this EXACT JSON format (no markdown, no code fences, raw JSON):
  Option A (delegate): {{"delegate": true, "subtasks": [{{"task": "task 1", "role": "researcher", "goal": "specific outcome", "known_facts": [], "constraints": [], "success_criteria": ["objective check"], "requested_files": [], "requested_symbols": [], "return_format": "concise findings"}}]}}
  Option B (answer):   {{"delegate": false, "answer": "your complete answer here"}}
"""

EXECUTE_SYSTEM_PROMPT = """\
You are a coding sub-agent inside a recursive agent framework.
You have been assigned a single, specific task. Complete it thoroughly.

RULES:
- If your task involves writing code, wrap it in a single ```python ... ``` block.
- Your code WILL be executed automatically in a sandbox. Make sure it runs without errors.
- Include a brief explanation before the code block.
- Do NOT ask clarifying questions -- work with what you have.
- If a codebase map is provided below, use it to understand the project structure
  and write code that fits naturally with the existing code.

{repo_map_section}
"""

REPO_MAP_TEMPLATE = """\
=== CODEBASE MAP ===
Below is an overview of the target project. Use it to understand the project
structure, existing functions/classes, and write code that integrates well.

{repo_map}
=== END CODEBASE MAP ===
"""

RETRY_PROMPT_TEMPLATE = """\
Your previous code was executed but FAILED with this error:

```
{error_output}
```

Please fix the code and provide the corrected version.
Wrap the fixed code in a ```python ... ``` block.
Explain what went wrong and how you fixed it.
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


def build_plan_sections(
    task_description: str,
    depth: int,
    max_depth: int,
    max_children: int,
) -> list[ContextSection]:
    messages = build_plan_messages(
        task_description, depth, max_depth, max_children)
    return [
        ContextSection(
            name="system_instructions", role="system",
            content=messages[0]["content"], required=True, priority=100),
        ContextSection(
            name="current_task", role="user",
            content=messages[1]["content"], required=True, priority=95),
    ]


def build_execute_messages(task: str, repo_map: str = "") -> list[dict]:
    """
    Build the message list for executing a single sub-task.

    Args:
        task: The sub-task description.
        repo_map: Optional XML-packed codebase overview to inject.

    Returns:
        A list of {"role": ..., "content": ...} dicts ready for the LLM.
    """
    if repo_map:
        repo_section = REPO_MAP_TEMPLATE.format(repo_map=repo_map)
    else:
        repo_section = ""
    system = EXECUTE_SYSTEM_PROMPT.format(repo_map_section=repo_section)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]


def render_task_capsule(capsule: TaskCapsule) -> str:
    """Render a capsule compactly, omitting empty optional fields."""
    lines = ["TASK: {}".format(capsule.task)]
    scalar_fields = [
        ("ROLE", capsule.role),
        ("GOAL", capsule.goal),
        ("RETURN FORMAT", capsule.return_format),
    ]
    list_fields = [
        ("KNOWN FACTS", capsule.known_facts),
        ("CONSTRAINTS", capsule.constraints),
        ("SUCCESS CRITERIA", capsule.success_criteria),
        ("REQUESTED FILES", capsule.requested_files),
        ("REQUESTED SYMBOLS", capsule.requested_symbols),
    ]
    for label, value in scalar_fields:
        if value:
            lines.append("{}: {}".format(label, value))
    for label, values in list_fields:
        if values:
            lines.append("{}:\n- {}".format(label, "\n- ".join(values)))
    return "\n".join(lines)


def build_execute_sections(
    task: str,
    repo_map: str = "",
    task_capsule: TaskCapsule | dict | None = None,
) -> list[ContextSection]:
    capsule = normalize_task_capsule(task_capsule or task)
    sections = [
        ContextSection(
            name="system_instructions", role="system",
            content=EXECUTE_SYSTEM_PROMPT.format(repo_map_section=""),
            required=True, priority=100),
    ]
    if repo_map:
        sections.append(ContextSection(
            name="repo_map", role="system",
            content=REPO_MAP_TEMPLATE.format(repo_map=repo_map),
            required=False, priority=30, trim_strategy="head"))
    sections.append(ContextSection(
        name="task_capsule", role="user", content=render_task_capsule(capsule),
        required=True, priority=95))
    return sections


def build_retry_messages(original_task: str, previous_code: str,
                         error_output: str) -> list[dict]:
    """
    Build messages for retrying after a sandbox execution failure.

    Args:
        original_task: The original sub-task description.
        previous_code: The code that failed.
        error_output: The stderr/timeout message from the sandbox.

    Returns:
        Message list that asks the LLM to fix the code.
    """
    return [
        {"role": "system", "content": EXECUTE_SYSTEM_PROMPT.format(
            repo_map_section="")},
        {"role": "user", "content": original_task},
        {"role": "assistant", "content": previous_code},
        {"role": "user", "content": RETRY_PROMPT_TEMPLATE.format(
            error_output=error_output[:2000])},
    ]


def build_retry_sections(
    original_task: str,
    previous_code: str,
    error_output: str,
) -> list[ContextSection]:
    return [
        ContextSection(
            name="system_instructions", role="system",
            content=EXECUTE_SYSTEM_PROMPT.format(repo_map_section=""),
            required=True, priority=100),
        ContextSection(
            name="current_task", role="user", content=original_task,
            required=True, priority=95),
        ContextSection(
            name="previous_code", role="assistant", content=previous_code,
            required=True, priority=90, trim_strategy="head"),
        ContextSection(
            name="sandbox_error", role="user",
            content=RETRY_PROMPT_TEMPLATE.format(error_output=error_output),
            required=True, priority=92, trim_strategy="head_tail"),
    ]


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_python_code(text: str) -> str | None:
    """
    Extract the first Python code block from the LLM's response.

    Looks for ```python ... ``` fenced blocks. Returns the code inside
    the block, or None if no code block is found.
    """
    # Match ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try ``` ... ``` (without language tag)
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Only return if it looks like Python (has def/class/import/indentation)
        if any(kw in code for kw in ["def ", "class ", "import ", "    ", "print("]):
            return code

    return None


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
                if parsed.get("delegate"):
                    subtasks = parsed.get("subtasks", [])
                    if not isinstance(subtasks, list):
                        subtasks = [subtasks] if subtasks else []
                    parsed["subtasks"] = [
                        normalize_task_capsule(item).model_dump()
                        for item in subtasks
                    ]
                return parsed
            # Model returned {"answer": "..."} without "delegate" field
            if "answer" in parsed:
                return {"delegate": False, "answer": parsed["answer"]}
        except json.JSONDecodeError:
            pass

    # Fallback: treat entire output as a direct answer
    return {"delegate": False, "answer": llm_output}


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_task_capsule(value: object) -> TaskCapsule:
    """Normalize legacy or imperfect planner output without losing its task."""
    if isinstance(value, TaskCapsule):
        return value
    if isinstance(value, str):
        return TaskCapsule(task=value.strip() or "Complete the assigned sub-task")
    if isinstance(value, dict):
        raw_task = value.get("task")
        if not isinstance(raw_task, str) or not raw_task.strip():
            raw_goal = value.get("goal")
            raw_task = raw_goal if isinstance(raw_goal, str) else json.dumps(
                value, ensure_ascii=True, default=str)
        return TaskCapsule(
            task=raw_task.strip(),
            role=str(value.get("role", "") or "").strip(),
            goal=str(value.get("goal", "") or "").strip(),
            known_facts=_string_list(value.get("known_facts")),
            constraints=_string_list(value.get("constraints")),
            success_criteria=_string_list(value.get("success_criteria")),
            requested_files=_string_list(value.get("requested_files")),
            requested_symbols=_string_list(value.get("requested_symbols")),
            return_format=str(value.get("return_format", "") or "").strip(),
        )
    return TaskCapsule(task=str(value).strip() or "Complete the assigned sub-task")


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
    for subtask in subtasks:
        capsule = normalize_task_capsule(subtask)
        children.append({
            "node_id": str(uuid.uuid4())[:8],  # short unique id
            "parent_id": parent_id,
            "task": capsule.task,
            "task_capsule": capsule.model_dump(),
            "depth": current_depth + 1,
            "result": None,
        })

    return children
