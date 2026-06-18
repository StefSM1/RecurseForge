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

logger = logging.getLogger("recurseforge.engine")


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
            return {"status": "executing", "children": children}

    # Direct answer -- no delegation
    answer = plan_response.get("answer", llm_output)
    logger.info("[PLAN] Answering directly (%d chars)", len(answer))
    return {"status": "done", "direct_answer": answer}


def execute_node(state: RecursionState) -> dict:
    """
    Run each child sub-task through the LLM and collect results.

    Each child is an independent LLM call. Results are stored in the
    child's "result" field and also aggregated into the top-level
    "results" list.
    """
    config = state["config"]
    llm_cfg = config["llm"]
    client = get_client(llm_cfg["base_url"])
    children = state.get("children", [])

    logger.info("[EXECUTE] Running %d children...", len(children))
    results = []

    for i, child in enumerate(children):
        logger.info("[EXECUTE] Child %d/%d [%s]: %s",
                    i + 1, len(children),
                    child["node_id"],
                    child["task"][:60])

        messages = redel.build_execute_messages(child["task"])
        try:
            result = chat_completion(
                client=client,
                model=llm_cfg["model_name"],
                messages=messages,
                max_tokens=llm_cfg["max_tokens"],
                temperature=llm_cfg["temperature"],
            )
            child["result"] = result
            results.append({
                "node_id": child["node_id"],
                "task": child["task"],
                "result": result,
                "success": True,
            })
            logger.info("[EXECUTE] Child [%s] succeeded (%d chars)",
                        child["node_id"], len(result))
        except Exception as e:
            child["result"] = None
            results.append({
                "node_id": child["node_id"],
                "task": child["task"],
                "result": str(e),
                "success": False,
            })
            logger.error("[EXECUTE] Child [%s] failed: %s",
                         child["node_id"], e)

    return {"status": "validating", "results": results}


def validate_node(state: RecursionState) -> dict:
    """
    Check execution results and produce the final status.

    Phase 1: simple validation -- if all children succeeded, mark done.
    Phase 3 will add TextGrad-based re-planning for failed children.
    """
    results = state.get("results", [])
    all_success = all(r.get("success", False) for r in results)
    failed = [r for r in results if not r.get("success")]

    if all_success:
        logger.info("[VALIDATE] All %d children succeeded.", len(results))
    else:
        logger.warning("[VALIDATE] %d/%d children failed.",
                       len(failed), len(results))

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
