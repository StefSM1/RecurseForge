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
- [Keeping Context Under Control](#keeping-context-under-control)
  - [Phase 1: Measure Before Sending](#phase-1-measure-before-sending)
  - [Phase 2: Pack Context by Priority](#phase-2-pack-context-by-priority)
  - [Phase 3: Task Capsules](#phase-3-task-capsules)
  - [Phase 4: Result Frames](#phase-4-result-frames)
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

The **Root Agent** acts like a manager. It decides whether a task is too big to handle alone. If so, it spawns independent sub-agents, each with their own focused task. Each sub-agent works on its assignment and returns a compact report for validation and aggregation.

This is the core direction of RecurseForge. The data structures include parent IDs,
depth, and `max_depth` for recursive delegation, but the graph compiled today plans
once at the root and executes one child layer. Recursive child re-planning remains a
future engine extension rather than something the current runtime already does.

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

In our code, `engine/redel.py` handles spawning. Each child is represented by a
dictionary containing its unique ID, parent, legacy task label, structured Task
Capsule, and depth counter.

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

**The key insight:** The LLM does not directly wire the flow; the graph does. The LLM
makes decisions inside specific nodes, and the graph selects a declared route. A direct
answer intentionally ends after Plan, while delegated work must pass through Execute
and Validate.

---

## Keeping Context Under Control

A 65K context window is a large desk, but a recursive system can still bury that
desk under repeated instructions, code maps, error logs, and child responses.
RecurseForge therefore treats context as a **budgeted pipeline**, not a bucket to
fill completely.

```
  Parent has a task
          |
          v
  +-------------------+
  | Measure + pack    |  Keep required facts, trim optional background
  +-------------------+
          |
          v
  +-------------------+
  | Task Capsule      |  Small assignment sent to a child
  +-------------------+
          |
          v
       Child works
          |
          v
  +-------------------+
  | Result Frame      |  Small report returned to the parent
  +-------------------+
          |
          v
  Full response remains available for debugging
```

This is split into four phases so each safety boundary remains understandable.

### Phase 1: Measure Before Sending

Before any prompt reaches llama.cpp, RecurseForge estimates its size and checks:

```
  prompt + reserved answer + safety buffer <= model context window

  49,152 + 8,192 + 8,192 = 65,536 tokens
```

The numbers are ceilings, not goals. If required input cannot fit, the request is
stopped before inference with a useful error instead of failing mysteriously inside
the model server.

**Analogy:** An airline weighs a suitcase before loading it onto the plane.

### Phase 2: Pack Context by Priority

Prompts are divided into labeled sections rather than glued into one giant string:

```
  MUST KEEP                         OPTIONAL
  +----------------------+          +----------------------+
  | System rules         |          | Repository map       |
  | Current task         |          | Older background     |
  | Required format      |          | Sibling summaries    |
  | Current error        |          | Extra context        |
  +----------------------+          +----------------------+
```

When space gets tight, optional low-priority sections are removed first. Long error
tracebacks keep their beginning and ending, where the exception type and useful call
sites usually live. The same input always produces the same packed prompt; no extra
LLM call is used for trimming.

This phase does **not** reuse the model's KV cache. It decides what information belongs
in each request.

### Phase 3: Task Capsules

Children no longer receive vague instructions or a copied parent transcript. They
receive a compact assignment card:

```
  TASK: Inspect retry handling
  ROLE: Debugger
  GOAL: Find why failed attempts overwrite history
  KNOWN FACTS: Retries already exist
  CONSTRAINTS: Keep event names unchanged
  SUCCESS: Identify the responsible function
  REQUESTED FILES: engine/graph.py
  RETURN FORMAT: Concise findings
```

Old string subtasks still work, but structured capsules make delegation clearer and
prevent context from multiplying as work moves down the agent tree.

**Analogy:** A manager gives a specialist a focused work order, not a recording of
every meeting that happened before it.

### Phase 4: Result Frames

The return path uses the same discipline. A child may produce a long explanation or
code response, but its parent receives a compact report:

```json
{
  "status": "success",
  "summary": "The retry parser is in parse_plan_response().",
  "evidence": [],
  "changes_needed": [],
  "risks": [],
  "open_questions": [],
  "confidence": 0.9
}
```

```
  Full child response -----------------> kept for debugging
                |
                +----> Result Frame ---> parent and validation
                           (~800-token target)
```

If the model forgets the JSON or formats it badly, RecurseForge creates a bounded
fallback from the prose. Sandbox truth always wins: failed code cannot become a
successful frame merely because the model claimed it worked.

Together, the four phases stop context growth in both directions:

- measurement prevents invisible overflow;
- deterministic packing removes low-value input;
- Task Capsules keep downward delegation focused;
- Result Frames keep upward aggregation compact.

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
     |  {"delegate": true, "subtasks": [{"task": "task A",
     |    "role": "researcher", "goal": "...", ...}]}
     |  OR
     |  {"delegate": false, "answer": "The answer is 42"}
     v
  4. Spawn children (if delegating)
     |  Each becomes: {node_id, parent_id, task, task_capsule, depth+1}
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
| `plan_node` | Calls the LLM (requesting no-thinking mode), parses the response, spawns children or stores the direct answer. If the answer contains code, runs it in the sandbox |
| `execute_node` | For each child: fetches repo-map context, calls the LLM, extracts Python code, runs it in the sandbox, retries on failure (up to 2x) |
| `validate_node` | Checks sandbox truth, reports execution stats, and aggregates compact Result Frame summaries |

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

Before the SDK sends anything, the Context Governor measures and assembles the
request. This keeps token budgeting in one place instead of relying on every graph
node to remember the rules.

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
| `--dashboard` | Starts the optional visual dashboard alongside the same engine |

You can combine flags: `--task "..." --verbose --json` runs a task with full debug logging AND raw JSON output.

---

### The Sandbox (`sandbox.py`)

The sandbox is a lightweight test room where sub-agents run generated code.

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

Each execution gets a fresh subprocess, restricted environment, timeout, captured
output, and temporary file cleanup. This contains ordinary crashes and runaway loops,
but it is **not a hardened security sandbox**: generated Python still runs with the
current user's operating-system permissions. Use it for trusted local-model output,
not hostile code.

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
  detects spike --------->  | VRAM_ALERT | ------->  Dashboard / logger
                            +------------+            (recommended action)
```

Sandbox attempts and corrections also emit lifecycle events, so failed execution,
TextGrad diagnosis, and retries can be reconstructed without changing graph logic.

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

The manager automatically demotes blocks when its own L0 capacity is full. The VRAM
monitor separately emits warning/critical events and recommended actions. Those alerts
are observable today, but they are not yet wired to call the manager's demotion methods
automatically. When a stored block is requested, the manager can promote it again.

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

Instead of dumping an entire codebase into the agent's prompt, the agent sees a
compressed **XML map** of class names, function signatures, and line numbers. In
65K-context mode, the default map budget is 4096 tokens: large enough to be useful,
still small enough to avoid turning every child call into a full-codebase prompt.
The server exposes `/lookup` for specific files and symbols, but the current graph does
not invoke targeted lookup automatically yet; that belongs to the later retrieval work.

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
          |   run_id: "...",    |
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
          |     {task: "sort",  |
          |      goal: "..."},  |
          |     {task: "filter",|
          |      goal: "..."}   |
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
          |     task_capsule: {},|
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
          |  2. Govern context |  (measure, pack, preserve required sections)
          |  3. Call LLM       |  (with Task Capsule + selected context)
          |  4. Extract code   |  (regex for ```python blocks)
          |  5. Run in sandbox |  (isolated subprocess)
          |  6. If fails:      |
          |     send error     |
          |     back to LLM    |
          |     retry (up 2x)  |
          +--------------------+
                    |
                    v  (raw results + compact Result Frames)
                    |
          +---------------------+
          |  results: [         |
          |    {id: "a1",       |
          |     result: "def    |
          |       sort(l):...", |
          |     raw_result: ...,|
          |     result_frame:   |
          |       {summary: ...},|
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

The built engine combines five cooperating layers:

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
  +--------------------------------------------------------+
  |          AIDER/REPOMIX: Context Optimization           |
  |    (Token-efficient AST repo maps, XML packing)        |
  +--------------------------------------------------------+
                          |
                          v
  +--------------------------------------------------------+
  |          TEXTGRAD: Textual Backpropagation             |
  |    (Computes linguistic gradients, self-corrects)      |
  +--------------------------------------------------------+
                          |
                          v
  +--------------------------------------------------------+
  |       CONTEXT PROTOCOL: Governor + Capsules + Frames   |
  |     (Bounds requests and child-to-parent communication) |
  +--------------------------------------------------------+
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
| Context Governor| `engine/context_governor.py` | Context 1-2 | Built    |
| Task Capsules   | `engine/redel.py`         | Context 3 | Built       |
| Result Frames   | `engine/result_frames.py` | Context 4 | Built       |
| Dashboard       | `dashboard/`              | Visual client | Built    |

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

### Dashboard -- BUILT
An optional visual representation of the existing engine. It monitors agents,
sandbox attempts, retries, resources, and chat runs; it does not define RecurseForge's
reasoning or execution rules.

### Context Optimization Task 1 -- DONE
Hard prompt budgets, deterministic context packing, Task Capsules, and Result Frames.
Together they bound both downward assignments and upward child results while preserving
full debug output.

### Workspace Agent -- PLANNED
A persistent workspace and editor-oriented Plan/Worker/Debug workflow. This is the next
major product architecture step; advanced retrieval and error-ledger work can be added
where real workspace measurements justify it.

---

## Getting Started

### Prerequisites

1. **llama.cpp server** running with your Qwen model on port 8080:
   ```
   llama-server.exe -m path\to\qwen.gguf -ngl 99 --ctx-size 65536 --flash-attn on --cache-type-k q8_0 --cache-type-v q4_0
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

# Start the optional visual dashboard and interactive chat
python -m harness.cli --dashboard --verbose
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
|   +-- context_governor.py     # Prompt measurement + deterministic context assembly
|   +-- result_frames.py        # Compact child-to-parent result parsing and limits
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
|   +-- dashboard_server.py     # Dashboard REST/WebSocket/chat bridge
|
+-- dashboard/                  # Optional React/TypeScript visualization client
|
+-- .venv/                      # Python virtual environment
+-- .devcontainer/              # Docker dev container config
+-- config.yaml                 # Runtime configuration
+-- requirements.txt            # Python dependencies
+-- AGENTS.md                   # AI agent development guide
+-- README.md                   # This file (human-readable)
+-- .plans/                     # Architecture and implementation plans
```
