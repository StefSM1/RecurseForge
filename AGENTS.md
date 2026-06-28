# RecurseForge -- Agent Instructions

## Identity
Recursive LLM agent framework. Runs Qwen 3.5 9B locally via llama.cpp on
8GB VRAM. LangGraph state machine with dynamic sub-agent spawning (ReDel),
bounded context assembly, structured child communication, sandbox code execution
with TextGrad self-correction (Phase 3a), and an optional real-time dashboard.
Full textual backpropagation remains Phase 3b future work.

## Reference Documents
- `README.md` -- Human-readable project guide with diagrams and tutorials
- `.plans/LLMRecursionPlan_v2.txt` -- Original architecture blueprint and contracts
- `.plans/HarnessPlan.txt` -- Harness specification and original build order
- `.plans/Context/CodexContextOptPlan.md` -- Context optimization roadmap and gates

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

## Architecture (Current Built System)

### Graph Flow (`engine/graph.py`)
```
START -> init_node -> plan_node --[has children?]--> execute_node -> validate_node -> END
                                  |                        |
                             (no children)                 | (per child)
                                  |                   1. Fetch optional repo-map
                                  +-> sandbox exec     2. Build governed context bundle
                                       of direct       3. Call LLM with TaskCapsule
                                       answer          4. Extract/run Python code
                                       (TextGrad if    5. TextGrad or simple retry
                                        it fails)      6. Build bounded ResultFrame
```
- `StateGraph(RecursionState)` with TypedDict state.
- `route_after_plan`: conditional edge checking `state["children"]`.
- Config dict injected into state at invoke time (not hardcoded).
- Direct answers also get sandbox-executed if they contain Python code.
- The current compiled graph plans once at the root and executes one child layer.
  Depth fields and `max_depth` guards exist, but recursive child re-planning is not
  currently wired into `build_graph()`; do not assume an arbitrary-depth runtime tree.

### Spawning (`engine/redel.py`)
- `build_plan_messages()`: delegation prompt with depth/max_depth/max_children.
  Includes `/no_think` directive to disable Qwen thinking mode for plan step.
- `parse_plan_response()`: extracts JSON from LLM output. Handles markdown fences,
  embedded JSON, missing `"delegate"`, structured capsules, legacy string subtasks,
  and garbage fallback.
- `spawn_children()`: creates child dicts with both legacy `task` and structured
  `task_capsule`. Enforces `max_depth` and `max_children`.
- `build_execute_sections()`: produces required instructions/capsule sections and an
  optional repo-map section for the deterministic governor.
- `build_retry_messages(task, previous_code, error)`: asks LLM to fix failed code.
- `extract_python_code(text)`: regex extraction of ```python ... ``` blocks.
- Planner contract: `{"delegate": bool, "subtasks": [TaskCapsule, ...] | "answer": "..."}`.
  String subtasks remain accepted for backward compatibility.

### LLM Client (`engine/llm_client.py`)
- `openai.OpenAI(base_url=..., api_key="not-needed")` pointed at llama.cpp.
- `chat_completion()` with `no_think` parameter: passes
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` for plan steps.
- Every enabled call passes through `build_context_bundle()` before the OpenAI SDK.
- `context_sections` is the structured path; legacy `messages` is retained as the
  rollback path when `context_governor.enabled` is false.
- Handles Qwen 3.5 thinking mode: if content is empty but reasoning_content exists,
  returns reasoning as fallback with warning.

### Context Layer (`context/`)
- `repo_map.py`: Tree-sitter FastAPI server. Parses .py/.js/.ts files, extracts
  symbols, generates XML-packed repo map (~4096 tokens in 65k-context mode).
  Endpoints: GET /map,
  POST /lookup, POST /refresh. Path normalization for cross-platform compatibility.
- `vram_manager.py`: L0/L1/L2 tiered memory. Auto-demotes oldest L0 blocks to L1,
  L1 to L2 (serialized to disk as JSON). Promotes on access.

### Context Contracts (`engine/context_governor.py`, `engine/result_frames.py`)
- `context_governor.py`: conservative UTF-8 token estimation, hard preflight limits,
  named context sections, deterministic section caps/omission, and inclusion telemetry.
- `result_frames.py`: parses clean, fenced, or embedded frame JSON; falls back to
  bounded prose; applies deterministic count/length/token limits without another LLM call.
- Required sections are never omitted. Required-only overflow raises
  `ContextBudgetError` before contacting llama.cpp.
- Sandbox execution status is authoritative over an LLM-provided frame status.

### Harness Layer (`harness/`)
- `cli.py`: CLI entry point. Starts event bus, VRAM monitor, builds graph,
  invokes, formats output (shows [code executed]/[text only] tags, sandbox
  stdout/stderr, retry counts). Shuts down sandbox pool on exit.
- `sandbox.py`: SandboxPool. Writes code to temp file, runs in subprocess with
  restricted env, captures stdout/stderr/exit_code, cleans up. Configurable timeout.
  This is process isolation for testing trusted local model output, not a hardened
  security boundary against malicious code.
- `event_bus.py`: Pub/sub via queue.Queue. `emit()` non-blocking, `subscribe()`
  with callbacks, background dispatcher thread. Singleton via `get_event_bus()`.
- `vram_monitor.py`: Background thread polling GPU memory via pynvml (or stub).
  Emits VRAM_ALERT events at warning/critical thresholds.
- `dashboard_server.py`: FastAPI/WebSocket bridge, resource/history APIs, and chat-run
  endpoints. It invokes the same graph; it must not duplicate engine behavior.

### Dashboard Integration (`dashboard/`) -- BUILT
- React 19 + TypeScript + Vite frontend; React Flow/Dagre renders execution topology.
- The dashboard is an observability/client layer, not a second agent engine.
- Engine events and REST/WebSocket payload compatibility are user-facing contracts.
- Agent Monitor represents root/child/sandbox/retry/output state from real telemetry.
- Chat prompts call the normal graph through `POST /api/chat/runs`; delegated chat
  aggregation prefers `result_frame.summary`, while Agent Details retains full `result`.
- Do not move decision, retry, sandbox, validation, or context-governor logic into React.

### Engine Event Emissions
- `RUN_STARTED` / `RUN_COMPLETED`: run-scoped lifecycle with stable `run_id`.
- `NODE_SPAWN`: emitted in plan_node for each child created.
- `NODE_COMPLETE`: emitted in execute_node after each child finishes.
  Payload includes full `result`, compact `result_frame`, execution status, and attempts.
- `SANDBOX_STARTED` / `SANDBOX_COMPLETED`: one pair per sandbox attempt.
- `CORRECTION_STARTED` / `CORRECTION_PROGRESS` / `CORRECTION_COMPLETED`: retry or
  TextGrad lifecycle tied to owner, failed execution, attempt, and strategy.
- `GRADIENT_FLOW`: emitted when TextGrad runs a gradient fix iteration.
  Payload includes: node_id, iteration, severity, num_mutations.

### State Shape (RecursionState TypedDict)
```python
{
    "task_id": str,            # unique node identifier
    "run_id": str,             # stable identity shared by every event in one run
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
    "result": str,             # legacy/full LLM response (dashboard compatibility)
    "raw_result": str,         # explicit full debug artifact
    "result_frame": dict,      # bounded parent-facing ResultFrame
    "success": bool,           # True if sandbox exit_code == 0 (or no code)
    "code_executed": bool,     # True if Python code was found and run
    "attempts": int,           # number of sandbox execution attempts (1 + retries)
    "stdout": str,             # sandbox stdout (if code was executed)
    "stderr": str,             # sandbox stderr (if code was executed)
    "exit_code": int,          # sandbox exit code
}
```

## Interface Contracts
Original field definitions are in `.plans/LLMRecursionPlan_v2.txt`; the Pydantic
models in `engine/interfaces.py` are the current source of truth.
Pydantic models in `engine/interfaces.py`: GraphState, NodeFrame, ContextRequest,
ContextPayload, ContextBudget, ContextBudgetReport, ContextSection, ContextBundle,
TaskCapsule, EvidenceRef, RiskItem, ResultFrame, ExecutionResult, Mutation,
TextGradient, HarnessCommand, and EngineEvent.
All extend VersionedModel with schema_version=1, .to_json(), .from_json().

## Qwen 3.5 Configuration
- Model: Qwen3.5-9B-DeepSeek-V4-Flash-MTP (IQ4_XS quantization, ~5GB weights)
- Tested server profile: `llama-server -m <model>.gguf -ngl 99 --ctx-size 65536
  --flash-attn on --cache-type-k q8_0 --cache-type-v q4_0`.
- No MTP (speculative decoding disabled -- overhead exceeds benefit for this model)
- `context_window: 65536` records the intended server context size.
- `max_tokens: 8192` is the output budget per call, not the full prompt window.
- Plan calls request no-thinking via `enable_thinking: false`, but some Qwen/llama.cpp
  combinations may still return reasoning or consume generation budget on it.
- Avoid filling the whole 65k window; keep a safety buffer for generated output,
  Qwen reasoning, retries, and recursive branches.

## Context Optimization Task 1 -- BUILT

The four phases form one directional protocol:

```text
parent task -> measured/assembled prompt -> TaskCapsule -> child
parent      <- bounded ResultFrame       <- raw child response
```

### Phase 1: Measurement and Hard Limits
- Models: `ContextBudget` and `ContextBudgetReport`.
- Default invariant: `49152 prompt + 8192 output + 8192 safety <= 65536`.
- `count_text_tokens()` uses a conservative UTF-8 byte estimate. It is deliberately
  dependency-free and protected by the safety reserve; it is not an exact Qwen tokenizer.
- `preflight_messages()` measures the complete request immediately before inference.
- Every graph, retry, and TextGrad call has a stable `call_kind`, including
  `root_plan`, `child_execute`, `direct_retry`, `child_retry`,
  `textgrad_evaluate`, and `textgrad_update`.
- Estimation failure raises `ContextEstimationError`; over-budget input raises
  `ContextBudgetError`. Neither failure may silently bypass the governor.
- Startup diagnostics compare the configured application window with llama.cpp
  `/props` when available and log prompt/output/safety budgets separately.

### Phase 2: Deterministic Context Assembly
- Models: `ContextSection` and `ContextBundle`.
- A section carries `name`, `role`, `content`, `required`, `priority`,
  `trim_strategy`, and optional `max_tokens`.
- Current priority order preserves system rules, task/capsule, output schema, and
  current sandbox errors before optional repo maps or older background.
- Section caps come from `context_governor.sections` in `config.yaml`.
- Trimming is deterministic: cap configured sections, preserve traceback head/tail,
  then omit the lowest-priority optional section until the request fits.
- Required sections are never dropped. If they cannot fit, fail before inference.
- `ContextBundle` records exact messages, included/omitted section names,
  per-section estimates, and the final budget report. Logs must not dump full prompts.
- This phase controls prompt selection; it does NOT implement KV-cache or prefix reuse.

### Phase 3: Task Capsules (Downward Boundary)
- Model: `TaskCapsule` with `task`, `role`, `goal`, `known_facts`, `constraints`,
  `success_criteria`, `requested_files`, `requested_symbols`, and `return_format`.
- Planner prompts request capsule objects. `normalize_task_capsule()` accepts valid
  objects, legacy strings, mixed lists, missing optional fields, and imperfect types.
- Every child keeps the old `task` string for labels/events and a serialized
  `task_capsule` for internal execution.
- `render_task_capsule()` omits empty headings and creates the required child context
  section. Children do not inherit a raw parent reasoning transcript.
- Requested files/symbols are declarations for later targeted retrieval; current
  execution still uses the optional repo map.

### Phase 4: Result Frames (Upward Boundary)
- Models: `ResultFrame`, `EvidenceRef`, and `RiskItem`.
- Child prompts append a compact JSON Result Frame after their normal answer/code.
  Python fences remain available to `extract_python_code()` and sandbox execution.
- `build_result_frame()` tries clean JSON, fenced JSON, embedded JSON, then bounded
  prose fallback. A malformed frame must never fail an otherwise valid child run.
- Each result stores `result` (compatibility), `raw_result` (full debug text), and
  `result_frame` (compact parent payload).
- Default frame target is about 800 estimated tokens. Limits cap evidence, findings,
  changes, risks, questions, and summary length; collections are reduced before the
  summary. No second summarization inference is made.
- Validation, CLI summaries, and dashboard chat aggregation prefer frame summaries.
  Full event `result` remains available to Agent Details.
- Engine/sandbox truth wins: a failed execution produces frame status `failed` even
  when the model's JSON claims success.

### Context Test Commands
```text
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m harness.cli --task "..." --verbose --json
```

In verbose output, inspect `[CONTEXT]` records for prompt estimates and
included/omitted sections. In JSON output, inspect child `task_capsule`, `raw_result`,
and `result_frame` fields.

## Phase Status
- [x] Phase 1: Spawning Graph (LangGraph + ReDel)
- [x] Phase 2: VRAM Shield (repo-map, sandbox, VRAM manager, event bus, retry loop)
- [x] Phase 3a: The Diagnostician (TextGrad single-variable backpropagation)
- [x] Dashboard: React/FastAPI execution observability and chat client
- [x] Context Task 1: measurement, deterministic governor, Task Capsules, Result Frames
- [ ] Phase 3b: Full TextGrad (multi-variable backprop, dynamic graph traversal)

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
    context_governor.py # token budgets + deterministic context assembly
    graph.py          # LangGraph StateGraph + sandbox/repo-map wiring
    redel.py          # Spawning, code extraction, retry prompts
    llm_client.py     # OpenAI SDK wrapper with thinking mode control
    interfaces.py     # Pydantic v2 models (all contracts)
    result_frames.py  # parse/bound compact child-to-parent payloads
    textgrad.py       # TextGrad engine (TextVariable, TextLoss, TGD) -- Phase 3a
  context/
    repo_map.py       # Tree-sitter repo map FastAPI server
    vram_manager.py   # L0/L1/L2 tiered memory manager
  harness/
    cli.py            # CLI orchestrator entry point
    sandbox.py        # Sandbox executor pool (subprocess workers)
    vram_monitor.py   # VRAM monitor daemon (pynvml polling)
    event_bus.py      # Engine-harness event bus (queue.Queue)
    dashboard_server.py # FastAPI REST/WebSocket/chat bridge
  dashboard/          # React/TypeScript observability client
  .venv/              # Virtual environment (DO NOT commit or delete)
  .devcontainer/      # Docker config (user-managed, DO NOT delete)
  .plans/             # architecture and implementation plans
  config.yaml         # Runtime config, budgets, frame limits, sandbox/TextGrad
  requirements.txt    # Python dependencies
```

## Running
```bash
.venv\Scripts\activate
python -m harness.cli --task "..." --verbose --json
python -m harness.cli --dashboard --verbose
```
