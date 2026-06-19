"""
engine/graph.py
===============
LangGraph state machine -- the outer deterministic loop.

Graph flow:
    Init --> Plan --> [route] --> Execute --> Validate --> END
                         |
                     (no children)
                         |
                         v
                        END

The Plan node calls the LLM to decide: delegate (spawn children) or
answer directly. The routing function checks whether children exist
and either continues to Execute or skips to the end.
"""

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from engine import redel
from engine.llm_client import get_client, chat_completion
from engine.interfaces import EngineEvent, EventType
from harness.event_bus import get_event_bus
from harness.sandbox import SandboxPool
from engine.textgrad import gradient_fix

logger = logging.getLogger("recurseforge.engine")

# ---------------------------------------------------------------------------
# Repo-map client (optional -- used if repo-map server is running)
# ---------------------------------------------------------------------------

def _fetch_repo_map(config: dict) -> str:
    """Try to fetch the repo map from the server. Returns empty string on failure."""
    ctx_cfg = config.get("context", {})
    if not ctx_cfg:
        return ""
    port = ctx_cfg.get("repo_map_port", 8001)
    max_tokens = ctx_cfg.get("max_map_tokens", 1024)
    try:
        import httpx
        resp = httpx.get(
            "http://127.0.0.1:{}/map".format(port),
            params={"max_tokens": max_tokens},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            xml = data.get("map", "")
            logger.info("[REPO-MAP] Fetched map (%d tokens, %d files indexed)",
                        data.get("token_count", 0),
                        data.get("files_indexed", 0))
            return xml
    except Exception as e:
        logger.debug("[REPO-MAP] Server not available (%s). "
                     "Running without codebase context.", e)
    return ""


# Global sandbox pool (created once, reused across calls)
_sandbox_pool: SandboxPool | None = None


def _get_sandbox(config: dict) -> SandboxPool:
    """Get or create the global sandbox pool."""
    global _sandbox_pool
    if _sandbox_pool is None:
        sbx_cfg = config.get("sandbox", {})
        _sandbox_pool = SandboxPool(
            pool_size=sbx_cfg.get("pool_size", 4),
            timeout_s=sbx_cfg.get("timeout_s", 30),
        )
    return _sandbox_pool


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class RecursionState(TypedDict):
    """The state object that flows through the LangGraph state machine."""
    task_id: str
    task_description: str
    status: str               # "init" | "planning" | "executing" | "validating" | "done"
    children: list            # list of child node dicts (from redel.spawn_children)
    depth: int                # current recursion depth (0 = root)
    results: list             # collected results from executed children
    direct_answer: str        # filled when the LLM answers without delegating
    config: dict              # runtime config (from config.yaml)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def init_node(state: RecursionState) -> dict:
    """
    Entry point. Just ensures clean initial values.
    """
    logger.info("[INIT] Task: %s", state["task_description"][:80])
    return {
        "status": "planning",
        "children": [],
        "results": [],
        "direct_answer": "",
    }


def plan_node(state: RecursionState) -> dict:
    """
    Ask the LLM to plan: delegate into sub-tasks or answer directly.

    Calls the LLM with the delegation prompt (from redel), parses the
    response, and either spawns children or stores the direct answer.
    """
    config = state["config"]
    llm_cfg = config["llm"]
    rec_cfg = config["recursion"]

    client = get_client(llm_cfg["base_url"])
    messages = redel.build_plan_messages(
        task_description=state["task_description"],
        depth=state["depth"],
        max_depth=rec_cfg["max_depth"],
        max_children=rec_cfg["max_children"],
    )

    logger.info("[PLAN] Depth %d/%d -- Asking LLM to plan...",
                state["depth"], rec_cfg["max_depth"])

    llm_output = chat_completion(
        client=client,
        model=llm_cfg["model_name"],
        messages=messages,
        max_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
        no_think=True,  # Planning step: disable thinking, we just need JSON
    )

    plan_response = redel.parse_plan_response(llm_output)

    if plan_response.get("delegate") and state["depth"] < rec_cfg["max_depth"]:
        children = redel.spawn_children(
            state=state,
            plan_response=plan_response,
            max_depth=rec_cfg["max_depth"],
            max_children=rec_cfg["max_children"],
        )
        if children:
            child_ids = [c["node_id"] for c in children]
            logger.info("[PLAN] Delegating to %d children: %s",
                        len(children), child_ids)
            # Emit NODE_SPAWN events for each child
            bus = get_event_bus()
            for child in children:
                bus.emit(EngineEvent(
                    event_type=EventType.NODE_SPAWN.value,
                    payload={
                        "node_id": child["node_id"],
                        "parent_id": child["parent_id"],
                        "task": child["task"],
                    },
                ))
            return {"status": "executing", "children": children}

    # Direct answer -- no delegation
    answer = plan_response.get("answer", llm_output)
    logger.info("[PLAN] Answering directly (%d chars)", len(answer))

    # Even for direct answers, extract and execute code if present
    code = redel.extract_python_code(answer)
    if code:
        logger.info("[PLAN] Direct answer contains code (%d chars), "
                    "executing in sandbox...", len(code))
        sandbox = _get_sandbox(state["config"])
        exec_result = sandbox.execute("direct", code)
        if exec_result.exit_code == 0:
            logger.info("[PLAN] Direct code execution OK (stdout: %d chars)",
                        len(exec_result.stdout))
            answer += "\n\n--- Sandbox Output ---\n" + exec_result.stdout
        else:
            logger.warning("[PLAN] Direct code execution failed: %s",
                           exec_result.stderr[:200])

            # Try TextGrad if enabled
            tg_cfg = state["config"].get("textgrad", {})
            if tg_cfg.get("enabled", False):
                logger.info("[PLAN] Using TextGrad to fix direct answer code...")
                try:
                    error_msg = exec_result.stderr or "Exit code: {}".format(
                        exec_result.exit_code)
                    fixed_code, grad_log = gradient_fix(
                        client=client,
                        model=llm_cfg["model_name"],
                        code=code,
                        task=state["task_description"],
                        stdout=exec_result.stdout,
                        stderr=error_msg,
                        max_iterations=tg_cfg.get("max_iterations", 1),
                        eval_temperature=tg_cfg.get("eval_temperature", 0.1),
                        update_temperature=tg_cfg.get("update_temperature", 0.2),
                        max_tokens=llm_cfg["max_tokens"],
                    )
                    # Re-execute the fixed code
                    exec_result2 = sandbox.execute("direct_fixed", fixed_code)
                    if exec_result2.exit_code == 0:
                        logger.info("[PLAN] TextGrad fixed the code! (stdout: %d chars)",
                                    len(exec_result2.stdout))
                        answer += ("\n\n--- TextGrad Fixed Code ---\n"
                                   + fixed_code
                                   + "\n\n--- Sandbox Output (after fix) ---\n"
                                   + exec_result2.stdout)
                    else:
                        logger.warning("[PLAN] TextGrad fix still failed: %s",
                                       exec_result2.stderr[:200])
                        answer += ("\n\n--- Sandbox Error (after TextGrad attempt) ---\n"
                                   + exec_result2.stderr)
                except Exception as e:
                    logger.error("[PLAN] TextGrad failed: %s", e)
                    answer += "\n\n--- Sandbox Error ---\n" + exec_result.stderr
            else:
                answer += "\n\n--- Sandbox Error ---\n" + exec_result.stderr

    return {"status": "done", "direct_answer": answer}


def execute_node(state: RecursionState) -> dict:
    """
    Run each child sub-task through the LLM, extract code, execute in
    sandbox, and retry on failure.

    Flow per child:
      1. Build messages (with repo-map if available)
      2. Call LLM -> get text response
      3. Extract Python code from response
      4. If code found -> run in sandbox
      5. If sandbox fails -> send error back to LLM, retry (up to max_retries)
      6. Store result (LLM text + sandbox output)
    """
    config = state["config"]
    llm_cfg = config["llm"]
    client = get_client(llm_cfg["base_url"])
    children = state.get("children", [])
    bus = get_event_bus()

    # Fetch repo map once for all children in this batch
    repo_map = _fetch_repo_map(config)

    # Get sandbox pool
    sandbox = _get_sandbox(config)

    # Retry settings
    max_retries = config.get("recursion", {}).get("max_retries", 2)

    logger.info("[EXECUTE] Running %d children (repo_map: %s, sandbox: ready)...",
                len(children),
                "available" if repo_map else "not available")
    results = []

    for i, child in enumerate(children):
        logger.info("[EXECUTE] Child %d/%d [%s]: %s",
                    i + 1, len(children),
                    child["node_id"],
                    child["task"][:60])

        # --- LLM call with repo-map context ---
        messages = redel.build_execute_messages(child["task"], repo_map=repo_map)
        try:
            llm_response = chat_completion(
                client=client,
                model=llm_cfg["model_name"],
                messages=messages,
                max_tokens=llm_cfg["max_tokens"],
                temperature=llm_cfg["temperature"],
            )
        except Exception as e:
            child["result"] = None
            results.append({
                "node_id": child["node_id"],
                "task": child["task"],
                "result": str(e),
                "success": False,
                "code_executed": False,
            })
            logger.error("[EXECUTE] Child [%s] LLM call failed: %s",
                         child["node_id"], e)
            continue

        # --- Extract and execute code ---
        code = redel.extract_python_code(llm_response)
        exec_result = None
        success = True
        attempts = 0

        if code:
            logger.info("[EXECUTE] Child [%s]: extracted %d chars of code, "
                        "running in sandbox...", child["node_id"], len(code))

            # Execute in sandbox (with retries on failure)
            for attempt in range(max_retries + 1):
                attempts = attempt + 1
                exec_result = sandbox.execute(child["node_id"], code)

                if exec_result.exit_code == 0:
                    logger.info("[EXECUTE] Child [%s]: code OK (attempt %d, "
                                "stdout: %d chars)",
                                child["node_id"], attempts,
                                len(exec_result.stdout))
                    break
                else:
                    logger.warning("[EXECUTE] Child [%s]: code failed (attempt %d/%d): %s",
                                   child["node_id"], attempts, max_retries + 1,
                                   exec_result.stderr[:200])

                    if attempt < max_retries:
                        # Check if TextGrad is enabled
                        tg_cfg = config.get("textgrad", {})
                        use_textgrad = tg_cfg.get("enabled", False)

                        if use_textgrad:
                            # TextGrad: structured gradient fix
                            logger.info("[EXECUTE] Child [%s]: using TextGrad to fix code...",
                                        child["node_id"])
                            error_msg = exec_result.stderr or "Exit code: {}".format(
                                exec_result.exit_code)
                            try:
                                fixed_code, grad_log = gradient_fix(
                                    client=client,
                                    model=llm_cfg["model_name"],
                                    code=code,
                                    task=child["task"],
                                    stdout=exec_result.stdout,
                                    stderr=error_msg,
                                    max_iterations=tg_cfg.get("max_iterations", 1),
                                    eval_temperature=tg_cfg.get("eval_temperature", 0.1),
                                    update_temperature=tg_cfg.get("update_temperature", 0.2),
                                    max_tokens=llm_cfg["max_tokens"],
                                )
                                code = fixed_code
                                logger.info("[EXECUTE] Child [%s]: TextGrad applied "
                                            "(%d iterations, %d chars)",
                                            child["node_id"], len(grad_log), len(code))
                                # Emit gradient flow event
                                for g in grad_log:
                                    bus.emit(EngineEvent(
                                        event_type=EventType.GRADIENT_FLOW.value,
                                        payload={
                                            "node_id": child["node_id"],
                                            "iteration": g["iteration"],
                                            "severity": g["severity"],
                                            "num_mutations": g["num_mutations"],
                                        },
                                    ))
                            except Exception as e:
                                logger.error("[EXECUTE] Child [%s]: TextGrad failed: %s",
                                             child["node_id"], e)
                                # Fall back to simple retry
                                error_msg = exec_result.stderr or "Exit code: {}".format(
                                    exec_result.exit_code)
                                retry_messages = redel.build_retry_messages(
                                    child["task"], code, error_msg)
                                try:
                                    llm_response = chat_completion(
                                        client=client,
                                        model=llm_cfg["model_name"],
                                        messages=retry_messages,
                                        max_tokens=llm_cfg["max_tokens"],
                                        temperature=llm_cfg["temperature"],
                                    )
                                    new_code = redel.extract_python_code(llm_response)
                                    if new_code:
                                        code = new_code
                                except Exception:
                                    break
                        else:
                            # Simple retry: send error back to LLM
                            error_msg = exec_result.stderr or "Exit code: {}".format(
                                exec_result.exit_code)
                            retry_messages = redel.build_retry_messages(
                                child["task"], code, error_msg)
                            try:
                                llm_response = chat_completion(
                                    client=client,
                                    model=llm_cfg["model_name"],
                                    messages=retry_messages,
                                    max_tokens=llm_cfg["max_tokens"],
                                    temperature=llm_cfg["temperature"],
                                )
                                new_code = redel.extract_python_code(llm_response)
                                if new_code:
                                    code = new_code
                                    logger.info("[EXECUTE] Child [%s]: retrying with "
                                                "fixed code (%d chars)",
                                                child["node_id"], len(code))
                                else:
                                    logger.warning("[EXECUTE] Child [%s]: retry response "
                                                   "had no code block", child["node_id"])
                                    break
                            except Exception as e:
                                logger.error("[EXECUTE] Child [%s]: retry LLM call "
                                             "failed: %s", child["node_id"], e)
                                break
        else:
            logger.info("[EXECUTE] Child [%s]: no code block found, "
                        "treating as text-only response", child["node_id"])

        # Determine success
        if exec_result and exec_result.exit_code != 0:
            success = False

        child["result"] = llm_response
        result_entry = {
            "node_id": child["node_id"],
            "task": child["task"],
            "result": llm_response,
            "success": success,
            "code_executed": code is not None,
            "attempts": attempts,
        }
        if exec_result:
            result_entry["stdout"] = exec_result.stdout[:500]
            result_entry["stderr"] = exec_result.stderr[:500]
            result_entry["exit_code"] = exec_result.exit_code
        results.append(result_entry)

        # Emit event
        bus.emit(EngineEvent(
            event_type=EventType.NODE_COMPLETE.value,
            payload={
                "node_id": child["node_id"],
                "result_summary": llm_response[:200],
                "token_usage": len(llm_response.split()),
                "code_executed": code is not None,
                "sandbox_exit_code": exec_result.exit_code if exec_result else None,
                "attempts": attempts,
            },
        ))

    return {"status": "validating", "results": results}


def validate_node(state: RecursionState) -> dict:
    """
    Check execution results and produce the final status.

    Reports sandbox execution outcomes: code execution success/failure,
    retry counts, and stdout/stderr summaries.
    """
    results = state.get("results", [])
    all_success = all(r.get("success", False) for r in results)
    failed = [r for r in results if not r.get("success")]
    code_runs = [r for r in results if r.get("code_executed")]
    retried = [r for r in results if r.get("attempts", 1) > 1]

    if all_success:
        logger.info("[VALIDATE] All %d children succeeded. "
                    "%d had code execution, %d required retries.",
                    len(results), len(code_runs), len(retried))
    else:
        logger.warning("[VALIDATE] %d/%d children failed. "
                       "%d had code execution, %d required retries.",
                       len(failed), len(results),
                       len(code_runs), len(retried))

    return {"status": "done"}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_plan(state: RecursionState) -> str:
    """
    Conditional edge after the Plan node.

    If the planner spawned children -> go to Execute.
    If the planner answered directly -> go to END (we're done).
    """
    if state.get("children"):
        return "execute"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(config: dict):
    """
    Compile and return the LangGraph state machine.

    Args:
        config: Parsed config.yaml dict with "llm" and "recursion" keys.

    Returns:
        A compiled LangGraph graph ready for .invoke().
    """
    graph = StateGraph(RecursionState)

    # Register nodes
    graph.add_node("init", init_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("validate", validate_node)

    # Edges
    graph.add_edge(START, "init")           # entry -> init
    graph.add_edge("init", "plan")          # init -> plan
    graph.add_conditional_edges(            # plan -> execute OR end
        "plan",
        route_after_plan,
        {"execute": "execute", "__end__": END},
    )
    graph.add_edge("execute", "validate")   # execute -> validate
    graph.add_edge("validate", END)         # validate -> end

    compiled = graph.compile()
    logger.info("[GRAPH] Compiled: START -> init -> plan -> [execute -> validate] -> END")
    return compiled
