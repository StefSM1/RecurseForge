# RecurseForge -- Agent Instructions

## Identity
Recursive LLM agent framework. Runs Qwen 3.5 9B locally via llama.cpp on
8GB VRAM. LangGraph state machine with dynamic sub-agent spawning (ReDel),
token-efficient context loading (Phase 2), and textual backpropagation (Phase 3).

## Reference Documents
- `README.md` -- Human-readable project guide with diagrams and tutorials
- `LLMRecursionPlan_v2.txt` -- Architecture blueprint, interface contracts, phase plan
- `HarnessPlan.txt` -- Harness specification, file structure, build order

## Mandatory Conventions

### Python Environment
- ALL dependencies go in `.venv`. NEVER install system-wide.
- Run: `.venv\Scripts\python -m harness.cli`
- Install: `.venv\Scripts\pip install <pkg>`
- Source of truth: `requirements.txt`

### File Ownership
- NEVER delete files the user added that were not created by you.
- NEVER delete anything in `.devcontainer/` unless explicitly asked.
- Platform target: Windows (native). `.devcontainer/` is for future Docker use.

### Post-Build Explanations
- After building each module, explain to the user **in chat** (not in code files):
  why it was built, how it works, what problems it solves.
- Pitch: someone learning about recursion/agentic loops for the first time.
- Use analogies and step-by-step walkthroughs.

## Architecture (Phase 1 -- Built)

### Graph Flow (`engine/graph.py`)
```
START -> init_node -> plan_node --[has children?]--> execute_node -> validate_node -> END
                                  |
                             (no children)
                                  |
                                  +-> END
```
- `StateGraph(RecursionState)` with TypedDict state.
- `route_after_plan`: conditional edge checking `state["children"]`.
- Config dict injected into state at invoke time (not hardcoded).

### Spawning (`engine/redel.py`)
- `build_plan_messages()`: constructs delegation prompt with depth/max_depth/max_children.
- `parse_plan_response()`: extracts JSON from LLM output. Handles markdown fences,
  embedded JSON, and garbage fallback (treats unparseable output as direct answer).
- `spawn_children()`: creates child node dicts. Enforces `max_depth` (blocks if
  depth >= max) and `max_children` (truncates subtask list).
- LLM contract: `{"delegate": bool, "subtasks": [...] | "answer": "..."}`

### LLM Client (`engine/llm_client.py`)
- `openai.OpenAI(base_url=..., api_key="not-needed")` pointed at llama.cpp.
- Single `chat_completion()` helper used by both plan and execute nodes.

### Harness (`harness/cli.py`)
- Loads config.yaml, builds graph via `engine.graph.build_graph(config)`.
- Invokes `graph.invoke(initial_state)` and formats output.
- Flags: `--task` (one-shot), `--verbose` (debug logs to stderr), `--json` (raw output).
- Harness NEVER imports LLM inference code. Engine/harness boundary is strict.

### State Shape (RecursionState TypedDict)
```python
{
    "task_id": str,            # unique node identifier
    "task_description": str,   # natural-language task
    "status": str,             # init | planning | executing | validating | done
    "children": list,          # child node dicts from redel.spawn_children()
    "depth": int,              # current recursion depth (0 = root)
    "results": list,           # collected ExecutionResult dicts
    "direct_answer": str,      # filled when LLM answers without delegating
    "config": dict,            # runtime config from config.yaml
}
```

### Child Node Dict
```python
{
    "node_id": str,      # uuid[:8]
    "parent_id": str,    # task_id of parent
    "task": str,         # sub-task description
    "depth": int,        # parent depth + 1
    "result": str|None,  # filled during execute_node
}
```

## Interface Contracts (All Layers)
Full field definitions in `LLMRecursionPlan_v2.txt` section "INTERFACE CONTRACTS".
Summary of the 5 interfaces:

1. **LangGraph <-> ReDel**: `GraphState` dict with children list.
2. **ReDel <-> Context**: `NodeFrame` + `ContextRequest`/`ContextPayload`.
3. **Context <-> TextGrad**: `ExecutionResult` (stdout, stderr, exit_code).
4. **TextGrad <-> LangGraph**: `TextGradient` with mutations list.
5. **Harness <-> Engine**: `HarnessCommand` (in) / `EngineEvent` (out) via queue.

## Upcoming Phases

### Phase 2: VRAM Shield (`context/`)
- Tree-sitter repo map server (FastAPI) -- `context/repo_map.py`
- L0/L1/L2 tiered memory manager -- `context/vram_manager.py`
- Surgical code loading: agents request specific files/symbols, never full codebases.
- XML packing (Repomix-style) for token-efficient context.

### Phase 3: TextGrad (`engine/textgrad.py`)
- Textual autograd: treat text strings as mutable variables with "gradients."
- Loss function: terminal errors / failed validations.
- Gradient: structured critique `[line] -> [cause] -> [suggestion]` (FedTextGrad UID).
- Dynamic graph traversal: walk execution tree backward to route gradients.

### Post-Phase 1: `engine/interfaces.py`
- Extract current dict shapes into Pydantic v2 models.
- Add `schema_version` field for forward compatibility.
- All models get `.to_json()` / `.from_json()`.

## File Structure
```
RecurseForge/
  engine/           # Recursion engine
    graph.py        # LangGraph StateGraph (built)
    redel.py        # Spawning logic (built)
    llm_client.py   # OpenAI SDK wrapper (built)
    textgrad.py     # Textual backprop (Phase 3)
    interfaces.py   # Pydantic models (post-Phase 1)
  context/          # Context optimization (Phase 2)
  harness/          # Runtime harness
    cli.py          # CLI entry point (built)
  .venv/            # Virtual environment (DO NOT commit or delete)
  .devcontainer/    # Docker config (user-managed, DO NOT delete)
  config.yaml       # Runtime config (llm endpoint, recursion limits)
  requirements.txt  # Python dependencies
```

## Running
```bash
.venv\Scripts\activate
python -m harness.cli --task "..." --verbose --json
```
