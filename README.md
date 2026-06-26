# RecurseForge

**A recursive LLM agent framework that spawns sub-agents, self-corrects, and runs on 8GB of VRAM.**

RecurseForge is a summer project that combines four powerful ideas from the open-source AI world into a single recursive engine: **LangGraph** (deterministic state machines), **ReDel** (dynamic task decomposition), **Aider/Repomix** (token-efficient code context), and **TextGrad** (textual backpropagation). It runs entirely on a local Qwen 3.5 9B model served by llama.cpp.

---

## Table of Contents

- [How to Read This Document](#how-to-read-this-document)
- [The Big Picture](#the-big-picture)
- [Core Concepts Explained](#core-concepts-explained)
  - [What Is Recursion?](#what-is-recursion)
  - [What Is an Agentic Loop?](#what-is-an-agentic-loop)
  - [What Is Sub-Agent Spawning?](#what-is-sub-agent-spawning)
  - [Why LangGraph?](#why-langgraph)
- [How Each Module Works](#how-each-module-works)
  - [The Spawning Logic (redel.py)](#the-spawning-logic-redelpy)
  - [The State Machine (graph.py)](#the-state-machine-graphpy)
  - [The LLM Client (llm_client.py)](#the-llm-client-llm_clientpy)
  - [The Harness (cli.py)](#the-harness-clipy)
  - [The Sandbox (sandbox.py)](#the-sandbox-sandboxpy)
  - [The Event Bus (event_bus.py)](#the-event-bus-event_buspy)
  - [The VRAM Manager (vram_manager.py)](#the-vram-manager-vram_managerpy)
  - [The Repo Map Server (repo_map.py)](#the-repo-map-server-repo_mappy)
  - [The Retry Loop](#the-retry-loop)
  - [The Diagnostician (textgrad.py)](#the-diagnostician-textgradpy----phase-3a)
- [Data Flow Diagram](#data-flow-diagram)
- [Architecture Overview](#architecture-overview)
- [Project Roadmap](#project-roadmap)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

---

## How to Read This Document

This README is written for **humans** who want to learn how the project works and why it was built this way. Every concept is explained from scratch with analogies and diagrams.

If you are an **AI agent** working on this codebase, read `AGENTS.md` instead -- it is optimized for quick structural understanding without tutorials.

---

## The Big Picture

Imagine you give a complex task to a single AI. It tries to do everything at once and produces a mediocre result. Now imagine a different approach:

```
                    "Build me a web scraper"
                              |
                              v
                    +-------------------+
                    |    ROOT AGENT     |
                    |  (Should I split  |
                    |   this task?)     |
                    +-------------------+
                         /     |     \
                        /      |      \
                       v       v       v
              +----------+ +---------+ +---------+
              | Agent A  | | Agent B | | Agent C |
              | Fetch    | | Parse   | | Format  |
              | the HTML | | the data| | to JSON |
              +----------+ +---------+ +---------+
                  |            |           |
                  v            v           v
                "HTML..."  "parsed..."  "JSON..."
                   \           |          /
                    \          |         /
                     v         v        v
                    +-------------------+
                    |    ROOT AGENT     |
                    |  (Collect results,|
                    |   validate, done) |
                    +-------------------+
```

The **Root Agent** acts like a manager. It decides whether a task is too big to handle alone. If so, it spawns independent sub-agents, each with their own focused task. Each sub-agent works in isolation, returns its result, and the root agent collects everything.

This is the core of RecurseForge. And it can go deeper -- a sub-agent can spawn its own sub-agents, creating a tree of recursive delegation. A safety limit (`max_depth`) prevents infinite recursion.

---

## Core Concepts Explained

### What Is Recursion?

Recursion is when something **calls itself** to solve a smaller piece of a bigger problem.

**Analogy: The CEO and the org chart**

```
         CEO: "Launch a new product"
          |
    +-----+--------+
    |              |
   VP Eng:    VP Sales:
   "Build     "Plan the
    the app"   launch event"
    |              |
 +--+--+         +-+--+
 |     |         |    |
Dev1  Dev2    Sales1 Sales2
```

1. The CEO gets a big task: "Launch a new product."
2. She breaks it into two sub-tasks and assigns them to VPs.
3. Each VP further breaks their sub-task and assigns it to directors.
4. Directors do the actual work and report results back up.
5. The CEO collects all reports and delivers the final answer.

**The critical rule:** There must be a maximum depth. You cannot have infinite layers of managers delegating to more managers. In our code, this is `max_depth` in `config.yaml` (default: 3).

---

### What Is an Agentic Loop?

An agentic loop is when an AI model doesn't just answer once - it runs in a **cycle**: think, act, observe, repeat.

**Analogy: A chef cooking a recipe**

```
    +------->  THINK: "What should I do next?"
    |            |
    |            v
    |           ACT: Chop onion / stir pot / add spice
    |            |
    |            v
    |         OBSERVE: Taste the dish / check the timer
    |            |
    +------------+
         |
    (dish is ready)
         v
       DONE
```

A normal chatbot is like someone who reads a recipe and describes the dish without cooking it. An agent actually **does** the work step by step, checks the result, and adjusts.

In RecurseForge, the agentic loop is the **state machine** in `engine/graph.py`. Each step (Plan, Execute, Validate) is a phase of the loop, and the graph decides what comes next based on the current state.

---

### What Is Sub-Agent Spawning?

Spawning means creating a new, **independent** AI instance to handle a sub-task.

**Analogy: A teacher grading essays**

```
    Teacher: "Grade 30 essays"
                |
            Spawns 3 TAs
                |
    +-----------+-----------+
    |           |           |
   TA-1       TA-2       TA-3
   Essays     Essays     Essays
   1-10       11-20      21-30
    |           |           |
    v           v           v
  Grades      Grades      Grades
  for 1-10    for 11-20   for 21-30
    \           |          /
     \          |         /
      v         v        v
    Teacher collects all grades
```

Key points:
- Each TA works **independently** -- they don't see each other's work.
- Each TA gets their own **instructions** (system prompt) and **assignment** (task).
- The teacher **collects** results and produces the final output.

In our code, `engine/redel.py` handles spawning. Each sub-agent is a plain dictionary with a unique ID, a task, and a depth counter.

---

### Why LangGraph?

Without LangGraph, you would need to manually code all the branching logic:

```python
# Without LangGraph -- manual spaghetti
result = llm.plan(task)
if result.wants_to_delegate:
    children = spawn(result.subtasks)
    for child in children:
        child.result = llm.execute(child.task)
    results = collect(children)
    if any_failed(results):
        # now what? retry? which one? how?
        ...
else:
    return result.answer
```

LangGraph replaces this with a **graph** -- a visual, testable flow:

```
              START
                |
                v
            +------+
            | Init |  Set up clean state
            +------+
                |
                v
            +------+
            | Plan |  LLM decides: delegate or answer?
            +------+
                |
    +--[has children?]--+
    |                    |
    v                    v
+---------+          +------+
| Execute |          | END  |  (direct answer)
+---------+          +------+
    |
    v
+----------+
| Validate |  Check results
+----------+
    |
    v
  +------+
  | END  |
  +------+
```

**The key insight:** The LLM does NOT control the flow. The graph does. The LLM only makes decisions *inside* specific nodes. This prevents the AI from going off the rails -- it can't skip the Validate step or loop forever.

---

## How Each Module Works

### The Spawning Logic (`redel.py`)

This file handles the **Plan** step -- the moment the LLM decides whether to delegate or answer directly.

**Step-by-step flow:**

```
  1. Build prompt
     |  "You are at depth 0/3. Should this task be split?"
     v
  2. Send to LLM
     |  (via llama.cpp on localhost:8080)
     v
  3. Parse JSON response
     |  {"delegate": true, "subtasks": ["task A", "task B"]}
     |  OR
     |  {"delegate": false, "answer": "The answer is 42"}
     v
  4. Spawn children (if delegating)
     |  Each subtask becomes: {node_id, parent_id, task, depth+1}
     |  Safety checks: max_depth, max_children
     v
  5. Return to graph
```

**What happens when parsing fails?** If the LLM returns garbage instead of valid JSON, the parser falls back to treating the entire response as a direct answer. The system never crashes -- it just degrades gracefully.

---

### The State Machine (`graph.py`)

This is the **outer loop** -- the deterministic graph that controls everything.

**States and transitions:**

```
  START --> init_node --> plan_node -----> execute_node --> validate_node --> END
                            |
                     (no children spawned)
                            |
                            +----------------------------------------------> END
```

Each node is a simple Python function that receives the current state and returns an update:

| Node | What it does |
|------|-------------|
| `init_node` | Resets counters, sets status to "planning" |
| `plan_node` | Calls the LLM (with thinking disabled for speed), parses the response, spawns children or stores the direct answer. If the answer contains code, runs it in the sandbox |
| `execute_node` | For each child: fetches repo-map context, calls the LLM, extracts Python code, runs it in the sandbox, retries on failure (up to 2x) |
| `validate_node` | Checks sandbox results, reports code execution stats and retry counts |

**The routing function** `route_after_plan` is what makes the graph dynamic. After the Plan node runs, it checks: did we spawn children? If yes -> go to Execute. If no -> go to END.

---

### The LLM Client (`llm_client.py`)

A thin wrapper around the OpenAI Python SDK that points at a **local** llama.cpp server.

```
  +-----------------+     HTTP      +---------------------+
  |  Python code    | ------------> |  llama.cpp          |
  |  openai.Chat... |  localhost    |  llama-server       |
  |                 |   :8080/v1    |  (your Qwen model)  |
  +-----------------+               +---------------------+
```

Why use the OpenAI SDK for a local model? Because llama.cpp exposes an **OpenAI-compatible API**. We just change the `base_url` from `api.openai.com` to `localhost:8080/v1`. This means:

- No llama.cpp-specific Python bindings needed
- If you later switch to a cloud provider, you only change the URL
- The same code works with any OpenAI-compatible endpoint

---

### The Harness (`cli.py`)

The harness is the "shell" that wraps the engine. It does NOT contain any AI logic.

```
  +--------------------+          +---------------------+
  |     HARNESS        |          |       ENGINE        |
  |     (cli.py)       |          |  (graph.py +        |
  |                    |          |   redel.py +        |
  | - Loads config     |  calls   |   llm_client.py)    |
  | - Reads user input | -------> |                     |
  | - Invokes graph    |          | - All LLM calls     |
  | - Formats output   |  result  | - All spawning      |
  |                    | <------- | - All execution     |
  +--------------------+          +---------------------+
```

The separation matters:
- The **engine** knows how to think.
- The **harness** knows how to run.
- You could swap the engine's model tomorrow without changing one line of harness code.

**CLI arguments explained:**

| Command | What it does |
|---------|-------------|
| `python -m harness.cli` | Interactive mode -- prints a banner and waits for you to type a task |
| `--task "..."` | One-shot mode -- runs the task immediately without prompting |
| `--verbose` / `-v` | Enables DEBUG-level logging to stderr (shows every LLM call, every state transition) |
| `--json` | Outputs the raw graph state as JSON instead of formatted text |

You can combine flags: `--task "..." --verbose --json` runs a task with full debug logging AND raw JSON output.

---

### The Sandbox (`sandbox.py`)

The sandbox is a clean room where sub-agents can safely run their generated code.

```
  Agent generates code
         |
         v
  +------------------+
  |    SANDBOX       |
  |                  |
  | 1. Write code    |
  |    to temp file  |
  | 2. Run in fresh  |
  |    subprocess    |
  | 3. Capture       |
  |    stdout/stderr |
  | 4. Clean up      |
  +------------------+
         |
         v
  ExecutionResult:
  {exit_code, stdout, stderr}
```

Each execution is **isolated** -- the subprocess has a restricted environment (no PYTHONPATH, limited HOME). If the code crashes, loops forever, or does something unexpected, only the temp subprocess is affected. After execution, the temp file is deleted.

**Why not Docker?** Docker adds ~500ms startup latency per container. For our local setup, Python subprocesses are faster and simpler. Docker support can be added later as an optional backend.

---

### The Event Bus (`event_bus.py`)

The event bus is like a bulletin board. When something important happens, the engine pins a note. Anyone who cares reads the board and reacts.

```
  Engine                    Event Bus                 Subscribers
  ------                    ---------                 -----------
  plan_node                 +------------+
  spawns child  --------->  | NODE_SPAWN |  ------->  Dashboard
                            +------------+              VRAM Monitor
                                                       CLI Logger
 
  execute_node              +---------------+
  child finishes ---------> | NODE_COMPLETE | ------->  Dashboard
                            +---------------+            CLI Logger
                           
  VRAM monitor              +------------+
  detects spike --------->  | VRAM_ALERT | ------->  VRAM Manager
                            +------------+            (demotes L1->L2)
```

The engine doesn't need to know who's reading the notes. It just posts them. This is called **pub/sub** (publish/subscribe) and keeps components loosely connected.

---

### The VRAM Manager (`vram_manager.py`)

Manages context data across three tiers, like a desk with drawers and a filing cabinet:

```
  L0 (Desk)           L1 (Drawer)         L2 (Filing Cabinet)
  +-----------+       +-----------+       +-----------+
  | Active    |       | Recently  |       | Archived  |
  | context   | demote| used      | demote| to disk   |
  | (in RAM)  | ----> | (in RAM)  | ----> | (JSON on  |
  |           |       |           |       |  disk)    |
  +-----------+       +-----------+       +-----------+
       ^                   |                    |
       |    promote        |    promote         |
       +-------------------+--------------------+
```

- **L0**: The code and variables the agent is working with right now
- **L1**: Recently accessed file summaries (tree-sitter representations)
- **L2**: Full history serialized to disk as JSON files

When the VRAM monitor detects memory pressure, it triggers automatic demotion. When an agent needs old context back, it gets promoted from disk.

---

### The Repo Map Server (`repo_map.py`)

A standalone FastAPI service that gives sub-agents a "table of contents" for your codebase.

```
  Your Codebase                Repo Map Server
  +-----------+               +-----------------+
  | engine/   |  tree-sitter  |  AST Index      |
  |  graph.py | ----------->  |  (in memory)    |
  |  redel.py |  parse on     |                 |
  | context/  |  startup      |  GET /map       |
  |  repo_... |               |  POST /lookup   |
  +-----------+               |  POST /refresh  |
                              +-----------------+
                                     |
                                     v
                              XML-packed output:
                              <codebase_summary>
                                <file path="engine/graph.py">
                                  <function name="build_graph">
                                  ...
```

Instead of dumping an entire codebase into the agent's prompt (which would blow up VRAM), the agent sees a compressed **XML map** of class names, function signatures, and line numbers. In 65k-context mode, the default map budget is 4096 tokens: large enough to be useful, still small enough to avoid turning every sub-agent call into a full-codebase prompt. When it needs actual code, it requests specific files or symbols via the `/lookup` endpoint.

---

### The Retry Loop

When a sub-agent writes buggy code, the system doesn't just fail -- it tries to fix it:

```
  Agent writes code
         |
         v
  Sandbox executes
         |
    +----+----+
    |         |
  exit 0    exit != 0
    |         |
    v         v
  SUCCESS   Send stderr back
    |       to the same agent
    |         |
    v         v
  Done      Agent fixes code
              |
              v
            Sandbox re-executes
              |
         (up to 2 retries)
```

This is a **basic feedback loop** -- the agent sees its own error and tries again. Phase 3 (TextGrad) makes this much more sophisticated.

---

### The Diagnostician (`textgrad.py`) -- Phase 3a

The Diagnostician is a new expert that replaces the blunt "fix it" retry with a **structured medical examination** of failed code. Instead of handing the agent a wall of error text, it performs a precise diagnosis and writes a targeted prescription.

**The problem with the basic retry:**

```
  Phase 2 retry (blunt):
    Code fails with: "NameError: name 'os' is not defined"
    Agent gets: "Here's the error, fix it"
    Agent: "uh... let me try again?"
```

**What the Diagnostician does instead:**

```
  Phase 3 TextGrad (precise):
    Code fails with: "NameError: name 'os' is not defined"
    
    DIAGNOSTICIAN (Evaluator):
      "I've examined the code and its execution.
       LINE: 3
       CAUSE: The os.path.join() call uses 'os' but it was never imported.
       FIX: Add 'import os' at the top of the file."
    
    PRESCRIBER (Updater):
      "Applying the diagnosis to produce fixed code..."
      -> Returns code with 'import os' added
```

The Diagnostician has three components, inspired by PyTorch's autograd:

```
  +--------------------+     +--------------------+     +--------------------+
  |   TextVariable     |     |     TextLoss       |     |       TGD          |
  |   (the patient)    |     |   (the doctor)     |     |  (the pharmacist)  |
  |                    |     |                    |     |                    |
  | Holds the code     | --> | Examines the code  | --> | Applies the fix    |
  | Tracks gradients   |     | + execution output |     | based on the       |
  | Records history    |     | Writes a structured|     | doctor's           |
  |                    |     | diagnosis (LINE/   |     | prescription       |
  |                    |     | CAUSE/FIX format)  |     |                    |
  +--------------------+     +--------------------+     +--------------------+
         |                          |                          |
         |    backward(grad)        |   loss = evaluate()      |  optimizer.step()
         |  <-----------------      |  <---------------        |  <-------------
         |                          |                          |
         v                          v                          v
    Updated code             Structured gradient         Fixed code
    (new version)            (LINE/CAUSE/FIX)            (ready to re-test)
```

**Why two separate LLM calls (doctor + pharmacist)?**

Separation of concerns. The evaluator's job is to **diagnose** (find what's wrong). The updater's job is to **prescribe** (fix it). If one LLM tries to do both at once, it tends to be vague ("your code has some issues, try fixing them"). With separate calls:
- The evaluator is forced to be **specific** (it must name line numbers and causes)
- The updater is forced to **follow instructions** (it applies each fix precisely)

This mirrors how real medicine works: the doctor who diagnoses you is not the same person who fills your prescription.

**The gradient format (UID principle):**

Each mutation is compressed into a single dense line, following the Uniform Information Density principle from the FedTextGrad paper:

```
[L5] CAUSE: Missing import statement -> FIX: Add 'import os' at top
[L12] CAUSE: Off-by-one in range -> FIX: Change range(n) to range(n+1)
[L20] CAUSE: Unhandled None return -> FIX: Add 'if result is None: return False'
```

No conversational fluff. Maximum signal per token. This is critical because the gradient is fed back into the LLM's context window, and every wasted token is wasted VRAM.

**When does the Diagnostician activate?**

Only when code **fails in the sandbox**. If the code runs successfully (exit code 0), the Diagnostician is never called. This is by design -- you don't need a doctor when you're healthy.

**Current limitations (Phase 3a vs full Phase 3):**

| Feature | Phase 3a (Built) | Phase 3b (Future) |
|---------|------------------|-------------------|
| Single-variable fix | Yes -- fix one piece of code at a time | -- |
| Multi-variable backprop | No | Yes -- gradients flow from child to parent nodes |
| Dynamic graph traversal | No | Yes -- walk the execution tree backward |
| Information density scoring | Format enforced, not scored | Yes -- compute UID scores |
| Prompt optimization | No -- only fixes code | Yes -- can also fix system prompts |

---

## Data Flow Diagram

Here is the complete journey of a task through the system:

```
  User types: "Write a sort and filter function"
                    |
                    v
          +---------------------+
          |    GraphState       |
          | {                   | 
          |   task_id: "root",  |
          |   description: ..,  |
          |   status: "init",   |
          |   children: [],     |
          |   depth: 0          |
          | }                   |
          +---------------------+
                    |
                    v  (init_node sets status = "planning")
                    |
                    v  (plan_node calls LLM)
                    |
          +---------------------+
          |  LLM Response:      |
          |  {                  |
          |   delegate: true,   |
          |   subtasks: [       |
          |     "Write sort",   |
          |     "Write filter"  |
          |   ]                 |
          |  }                  |
          +---------------------+
                    |
                    v  (redel.py spawns 2 children)
                    |
          +---------------------+
          |  children: [        |
          |    {id: "a1",       |
          |     task: "sort",   |
          |     depth: 1},      |
          |    {id: "b2",       |
          |     task: "filter"  |
          |     depth: 1}       |
          |  ]                  |
          +---------------------+
                    |
                    v  (execute_node per child)
                    |
          +--------------------+
          |  1. Fetch repo-map |  (codebase structure from server)
          |  2. Call LLM       |  (with context injected)
          |  3. Extract code   |  (regex for ```python blocks)
          |  4. Run in sandbox |  (isolated subprocess)
          |  5. If fails:      |
          |     send error     |
          |     back to LLM    |
          |     retry (up 2x)  |
          +--------------------+
                    |
                    v  (results with sandbox output)
                    |
          +---------------------+
          |  results: [         |
          |    {id: "a1",       |
          |     result: "def    |
          |       sort(l):...", |
          |     success: true,  |
          |     code_executed:  |
          |       true,         |
          |     stdout: "...",  |
          |     attempts: 1},   |
          |    ...              |
          |  ]                  |
          +---------------------+
                    |
                    v  (validate_node checks sandbox results)
                    |
                    v  (output to user with sandbox output)
```

---

## Architecture Overview

The full architecture (across all phases) has four layers:

```
  +--------------------------------------------------------+
  |              LANGGRAPH: Deterministic Graph            |
  |          (Controls the outer state machine)            |
  +--------------------------------------------------------+
                          |
                          v
  +-------------------------------------------------------+
  |            REDEL: Dynamic Tree Spawning               |
  |       (Spawns ephemeral sub-agent nodes)              |
  +-------------------------------------------------------+
                          |
                          v
  +----------------------------------------------------=---+
  |          AIDER/REPOMIX: Context Optimization           |
  |    (Token-efficient AST repo maps, XML packing)        |
  +-----------------------------------------------------=--+
                          |
                          v
  +-----------------------------------------------------=--+
  |          TEXTGRAD: Textual Backpropagation             |
  |    (Computes linguistic gradients, self-corrects)      |
  +-----------------------------------------------------=--+
```

| Layer           | File                      | Phase   | Status        |
|-----------------|---------------------------|---------|---------------|
| LangGraph       | `engine/graph.py`         | 1       | Built         |
| ReDel           | `engine/redel.py`         | 1       | Built         |
| LLM Client      | `engine/llm_client.py`    | 1       | Built         |
| CLI Harness     | `harness/cli.py`          | 1       | Built         |
| Interfaces      | `engine/interfaces.py`    | 2       | Built         |
| Repo Map Server | `context/repo_map.py`     | 2       | Built         |
| VRAM Manager    | `context/vram_manager.py` | 2       | Built         |
| Sandbox Pool    | `harness/sandbox.py`      | 2       | Built         |
| VRAM Monitor    | `harness/vram_monitor.py` | 2       | Built         |
| Event Bus       | `harness/event_bus.py`    | 2       | Built         |
| TextGrad Engine | `engine/textgrad.py`      | 3a      | Built         |
| Dashboard       | `harness/dashboard.py`    | Future  | Not yet built |

---

## Project Roadmap

### Phase 1: The Spawning Graph -- DONE
LangGraph state machine + ReDel spawning logic. The root agent can decide to delegate, spawn children, execute them, and collect results.

### Phase 2: The VRAM Shield -- DONE
Tree-sitter-based repository maps (Repomix-style XML packing). Sandbox code execution with automatic retry on failure. Three-tier memory manager (L0/L1/L2). VRAM monitoring daemon. Event bus for engine-harness communication. Qwen 3.5 thinking mode handling.

### Phase 3a: The Diagnostician (TextGrad) -- DONE
Single-variable textual backpropagation. When sandbox execution fails, the Diagnostician examines the code and error output, produces a structured LINE/CAUSE/FIX diagnosis, and applies the fix to produce corrected code. Uses PyTorch-inspired primitives (TextVariable, TextLoss, TGD optimizer). Works for both direct answers and delegated children.

### Phase 3b: Full TextGrad -- FUTURE
Multi-variable backpropagation through the execution tree. Gradients flow from child nodes back to parent prompts. Dynamic graph traversal to route gradients to the correct variable. Information density scoring on gradients.

### Dashboard -- FUTURE
Streamlit visualization for the execution tree, gradient flows, VRAM timeline, and per-node inspector. Critical for debugging Phase 3b.

---

## Getting Started

### Prerequisites

1. **llama.cpp server** running with your Qwen model on port 8080:
   ```
   llama-server.exe -m path\to\qwen.gguf --port 8080 --ctx-size 65536 --n-gpu-layers 99 -t 8
   ```
   If your llama.cpp build supports KV cache quantization, your tested 65k profile
   of Q8 K + Q4 V cache is the practical target for this project. Keep MTP off
   unless you have measured that it helps on your hardware.

2. **Python 3.10+** with a virtual environment:
   ```
   .venv\Scripts\activate
   ```

### Run It

```bash
# Interactive prompt
python -m harness.cli

# One-shot task
python -m harness.cli --task "Write a sort function"

# Debug mode (full logging to stderr)
python -m harness.cli --task "What is 2+2?" --verbose

# Raw JSON output
python -m harness.cli --task "Explain recursion" --json

# Combine flags
python -m harness.cli --task "Complex task here" --verbose --json
```

---

## Project Structure

```
RecurseForge/
|
+-- engine/                     # The recursion engine
|   +-- graph.py                # LangGraph state machine (outer loop + sandbox/repo-map wiring)
|   +-- redel.py                # Task decomposition, code extraction, retry prompts
|   +-- llm_client.py           # OpenAI SDK wrapper with thinking mode control
|   +-- interfaces.py           # Pydantic v2 models (all interface contracts)
|   +-- textgrad.py             # TextGrad engine (TextVariable, TextLoss, TGD)
|
+-- context/                    # Context optimization layer
|   +-- repo_map.py             # Tree-sitter repo map server (FastAPI)
|   +-- vram_manager.py         # L0/L1/L2 tiered memory manager
|
+-- harness/                    # The custom harness
|   +-- cli.py                  # CLI orchestrator entry point
|   +-- sandbox.py              # Sandbox executor pool (subprocess workers)
|   +-- vram_monitor.py         # VRAM monitor daemon (pynvml polling)
|   +-- event_bus.py            # Engine-harness event bus (queue.Queue)
|   +-- dashboard.py            # (Future) Gradient log viewer (Streamlit)
|
+-- .venv/                      # Python virtual environment
+-- .devcontainer/              # Docker dev container config
+-- config.yaml                 # Runtime configuration
+-- requirements.txt            # Python dependencies
+-- AGENTS.md                   # AI agent development guide
+-- README.md                   # This file (human-readable)
+-- LLMRecursionPlan_v2.txt     # Architecture theory + interface contracts
+-- HarnessPlan.txt             # Custom harness specification
```
