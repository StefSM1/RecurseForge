# RecurseForge -- Agent Instructions

## Identity
Recursive LLM agent framework. Runs Qwen 3.5 9B locally via llama.cpp on
8GB VRAM. LangGraph state machine with dynamic sub-agent spawning (ReDel),
token-efficient context loading, sandbox code execution with TextGrad
self-correction (Phase 3a), and full textual backpropagation (Phase 3b -- future).

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

## Architecture (Phases 1-2 -- Built)

### Graph Flow (`engine/graph.py`)
```
START -> init_node -> plan_node --[has children?]--> execute_node -> validate_node -> END
                                  |                        |
                             (no children)                 | (per child)
                                  |                   1. Fetch repo-map
                                  +-> sandbox exec     2. Call LLM (with context)
                                       of direct       3. Extract Python code
                                       answer          4. Run in sandbox
                                       (TextGrad if    5. If fails + textgrad enabled:
                                        it fails)         gradient_fix() -> re-execute
                                                       6. Else simple retry (up to max_retries)
```
- `StateGraph(RecursionState)` with TypedDict state.
- `route_after_plan`: conditional edge checking `state["children"]`.
- Config dict injected into state at invoke time (not hardcoded).
- Direct answers also get sandbox-executed if they contain Python code.

### Spawning (`engine/redel.py`)
- `build_plan_messages()`: delegation prompt with depth/max_depth/max_children.
  Includes `/no_think` directive to disable Qwen thinking mode for plan step.
- `parse_plan_response()`: extracts JSON from LLM output. Handles markdown fences,
  embedded JSON, missing `"delegate"` key, and garbage fallback.
- `spawn_children()`: creates child node dicts. Enforces `max_depth` and `max_children`.
- `build_execute_messages(task, repo_map)`: code-aware prompt with optional codebase
  map injection. Tells agent its code will be sandbox-executed.
- `build_retry_messages(task, previous_code, error)`: asks LLM to fix failed code.
- `extract_python_code(text)`: regex extraction of ```python ... ``` blocks.
- LLM contract: `{"delegate": bool, "subtasks": [...] | "answer": "..."}`

### LLM Client (`engine/llm_client.py`)
- `openai.OpenAI(base_url=..., api_key="not-needed")` pointed at llama.cpp.
- `chat_completion()` with `no_think` parameter: passes
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` for plan steps.
- Handles Qwen 3.5 thinking mode: if content is empty but reasoning_content exists,
  returns reasoning as fallback with warning.

### Context Layer (`context/`)
- `repo_map.py`: Tree-sitter FastAPI server. Parses .py/.js/.ts files, extracts
  symbols, generates XML-packed repo map (~4096 tokens in 65k-context mode).
  Endpoints: GET /map,
  POST /lookup, POST /refresh. Path normalization for cross-platform compatibility.
- `vram_manager.py`: L0/L1/L2 tiered memory. Auto-demotes oldest L0 blocks to L1,
  L1 to L2 (serialized to disk as JSON). Promotes on access.

### Harness Layer (`harness/`)
- `cli.py`: CLI entry point. Starts event bus, VRAM monitor, builds graph,
  invokes, formats output (shows [code executed]/[text only] tags, sandbox
  stdout/stderr, retry counts). Shuts down sandbox pool on exit.
- `sandbox.py`: SandboxPool. Writes code to temp file, runs in subprocess with
  restricted env, captures stdout/stderr/exit_code, cleans up. Configurable timeout.
- `event_bus.py`: Pub/sub via queue.Queue. `emit()` non-blocking, `subscribe()`
  with callbacks, background dispatcher thread. Singleton via `get_event_bus()`.
- `vram_monitor.py`: Background thread polling GPU memory via pynvml (or stub).
  Emits VRAM_ALERT events at warning/critical thresholds.

### Engine Event Emissions
- `NODE_SPAWN`: emitted in plan_node for each child created.
- `NODE_COMPLETE`: emitted in execute_node after each child finishes.
  Payload includes: result_summary, code_executed, sandbox_exit_code, attempts.
- `GRADIENT_FLOW`: emitted when TextGrad runs a gradient fix iteration.
  Payload includes: node_id, iteration, severity, num_mutations.

### State Shape (RecursionState TypedDict)
```python
{
    "task_id": str,            # unique node identifier
    "task_description": str,   # natural-language task
    "status": str,             # init | planning | executing | validating | done
    "children": list,          # child node dicts from redel.spawn_children()
    "depth": int,              # current recursion depth (0 = root)
    "results": list,           # collected result entries with sandbox data
    "direct_answer": str,      # filled when LLM answers without delegating
    "config": dict,            # runtime config from config.yaml
}
```

### Result Entry (per child)
```python
{
    "node_id": str,
    "task": str,
    "result": str,             # LLM text response
    "success": bool,           # True if sandbox exit_code == 0 (or no code)
    "code_executed": bool,     # True if Python code was found and run
    "attempts": int,           # number of sandbox execution attempts (1 + retries)
    "stdout": str,             # sandbox stdout (if code was executed)
    "stderr": str,             # sandbox stderr (if code was executed)
    "exit_code": int,          # sandbox exit code
}
```

## Interface Contracts
Full field definitions in `LLMRecursionPlan_v2.txt` section "INTERFACE CONTRACTS".
Pydantic models in `engine/interfaces.py`: GraphState, NodeFrame, ContextRequest,
ContextPayload, ExecutionResult, Mutation, TextGradient, HarnessCommand, EngineEvent.
All extend VersionedModel with schema_version=1, .to_json(), .from_json().

## Qwen 3.5 Configuration
- Model: Qwen3.5-9B-DeepSeek-V4-Flash-MTP (IQ4_XS quantization, ~5GB weights)
- Server target: `llama-server -m <model>.gguf -ngl 99 --ctx-size 65536 --flash-attn on`
- KV cache recommendation for 8GB VRAM: Q8 K + Q4 V if available/stable.
- No MTP (speculative decoding disabled -- overhead exceeds benefit for this model)
- `context_window: 65536` records the intended server context size.
- `max_tokens: 8192` is the output budget per call, not the full prompt window.
- `no_think: true` on plan step via `enable_thinking: false` extra_body param
- Avoid filling the whole 65k window; keep a safety buffer for generated output,
  Qwen reasoning, retries, and recursive branches.

## Phase Status
- [x] Phase 1: Spawning Graph (LangGraph + ReDel)
- [x] Phase 2: VRAM Shield (repo-map, sandbox, VRAM manager, event bus, retry loop)
- [x] Phase 3a: The Diagnostician (TextGrad single-variable backpropagation)
- [ ] Phase 3b: Full TextGrad (multi-variable backprop, dynamic graph traversal)
- [ ] Dashboard (visualization for debugging Phase 3b)

### Phase 3a: The Diagnostician (`engine/textgrad.py`) -- BUILT
- `TextVariable`: mutable text with requires_grad, grad storage, history tracking.
- `TextLoss`: LLM-as-judge evaluator. Calls LLM with CODE_EVAL_PROMPT requesting
  LINE/CAUSE/FIX format. Parses critique into TextGradient with Mutations.
  Low temperature (0.1) for precise critique.
- `TGD`: Textual Gradient Descent optimizer. Applies gradients via LLM with
  UPDATE_PROMPT. Low temperature (0.2) for faithful application.
- `gradient_fix()`: one-shot API wrapping evaluate -> backward -> step loop.
  Returns (fixed_code, gradient_log).
- Integration: execute_node checks `config.textgrad.enabled`. When enabled,
  failed sandbox calls use gradient_fix() instead of simple retry. Falls back
  to simple retry if TextGrad itself fails. Also wired into plan_node for
  direct answers with code.
- `TextGradient.to_formatted_string()` produces UID format:
  `[L4] CAUSE: missing import -> FIX: add 'import os' at top`
- Config: `textgrad.enabled` (default false), `max_iterations` (1),
  `eval_temperature` (0.1), `update_temperature` (0.2).
- Evaluator and updater use DIFFERENT prompts (separation of concerns).

### Phase 3b: Full TextGrad -- FUTURE
- Multi-variable backprop: gradients flow from child nodes to parent prompts.
- Dynamic graph traversal: walk execution tree backward to route gradients.
- Information density scoring: compute UID scores on gradient text.
- Prompt optimization: can also mutate system_prompt, not just code.

## File Structure
```
RecurseForge/
  engine/
    graph.py          # LangGraph StateGraph + sandbox/repo-map wiring
    redel.py          # Spawning, code extraction, retry prompts
    llm_client.py     # OpenAI SDK wrapper with thinking mode control
    interfaces.py     # Pydantic v2 models (all contracts)
    textgrad.py       # TextGrad engine (TextVariable, TextLoss, TGD) -- Phase 3a
  context/
    repo_map.py       # Tree-sitter repo map FastAPI server
    vram_manager.py   # L0/L1/L2 tiered memory manager
  harness/
    cli.py            # CLI orchestrator entry point
    sandbox.py        # Sandbox executor pool (subprocess workers)
    vram_monitor.py   # VRAM monitor daemon (pynvml polling)
    event_bus.py      # Engine-harness event bus (queue.Queue)
    dashboard.py      # (Future) Gradient log viewer (Streamlit)
  .venv/              # Virtual environment (DO NOT commit or delete)
  .devcontainer/      # Docker config (user-managed, DO NOT delete)
  config.yaml         # Runtime config (llm, recursion, context, vram, sandbox)
  requirements.txt    # Python dependencies
```

## Running
```bash
.venv\Scripts\activate
python -m harness.cli --task "..." --verbose --json
```
