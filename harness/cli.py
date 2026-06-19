"""
harness/cli.py
===============
Minimal CLI entry point for RecurseForge.

Usage:
    python -m harness.cli                          # interactive prompt
    python -m harness.cli --task "Write a poem"     # one-shot task
    python -m harness.cli --config path/to/cfg.yaml # custom config

The CLI loads config.yaml, builds the LangGraph state machine, sends
your task through it, and prints the result.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Ensure the project root is on sys.path so "engine" imports work
# when running as `python harness/cli.py` from the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.graph import build_graph
from harness.event_bus import get_event_bus
from harness.vram_monitor import VRAMMonitor


def setup_logging(verbose: bool = False):
    """Configure structured logging to stderr (keeps stdout clean for output)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def load_config(config_path: str) -> dict:
    """Load and return the YAML config."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_result(result: dict) -> str:
    """
    Format the final graph state into a human-readable summary.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  RecurseForge Result")
    lines.append("=" * 60)
    lines.append(f"Task: {result['task_description']}")
    lines.append(f"Status: {result['status']}")
    lines.append("")

    # Direct answer (no delegation)
    if result.get("direct_answer"):
        lines.append("--- Direct Answer ---")
        lines.append(result["direct_answer"])

    # Delegated results
    children = result.get("children", [])
    results = result.get("results", [])
    if children:
        lines.append(f"--- {len(children)} Sub-tasks Executed ---")
        for i, r in enumerate(results):
            status = "OK" if r.get("success") else "FAILED"
            code_tag = " [code executed]" if r.get("code_executed") else " [text only]"
            attempts = r.get("attempts", 1)
            retry_tag = f" ({attempts} attempts)" if attempts > 1 else ""
            lines.append(f"\n[{i + 1}] {r['task']}  ({status}){code_tag}{retry_tag}")

            # Show LLM response (truncated)
            llm_text = r.get("result", "") or "(no output)"
            lines.append(f"    {llm_text[:400]}")

            # Show sandbox output if code was executed
            if r.get("code_executed"):
                stdout = r.get("stdout", "").strip()
                stderr = r.get("stderr", "").strip()
                exit_code = r.get("exit_code", "?")
                lines.append(f"    --- Sandbox (exit {exit_code}) ---")
                if stdout:
                    lines.append(f"    stdout: {stdout[:300]}")
                if stderr:
                    lines.append(f"    stderr: {stderr[:300]}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="RecurseForge -- Recursive LLM Agent Framework",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Task to execute (if omitted, enters interactive prompt)",
    )
    parser.add_argument(
        "--config", type=str, default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output raw JSON state instead of formatted text",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    config = load_config(args.config)

    logger = logging.getLogger("recurseforge.cli")
    logger.info("Config loaded from %s", args.config)
    logger.info("LLM endpoint: %s", config["llm"]["base_url"])
    logger.info("Recursion: max_depth=%d, max_children=%d",
                config["recursion"]["max_depth"],
                config["recursion"]["max_children"])

    graph = build_graph(config)

    # Start event bus
    bus = get_event_bus()
    bus.start()

    # Start VRAM monitor if config has vram section
    vram_monitor = None
    vram_cfg = config.get("vram", {})
    if vram_cfg:
        vram_monitor = VRAMMonitor(
            warning_mb=vram_cfg.get("warning_mb", 6000),
            critical_mb=vram_cfg.get("critical_mb", 7500),
            poll_interval_s=vram_cfg.get("poll_interval_s", 2.0),
        )
        vram_monitor.start()
        logger.info("VRAM monitor started (warning: %d MB, critical: %d MB)",
                    vram_cfg.get("warning_mb", 6000),
                    vram_cfg.get("critical_mb", 7500))

    # Get task from --flag or interactive prompt
    task = args.task
    if task is None:
        print("RecurseForge v0.1 (Phase 1)")
        print(f"  LLM: {config['llm']['base_url']} ({config['llm']['model_name']})")
        print(f"  Max depth: {config['recursion']['max_depth']}")
        print()
        task = input("Enter task> ").strip()
        if not task:
            print("No task provided. Exiting.", file=sys.stderr)
            sys.exit(0)

    # Build initial state
    initial_state = {
        "task_id": "root",
        "task_description": task,
        "status": "init",
        "children": [],
        "depth": 0,
        "results": [],
        "direct_answer": "",
        "config": config,
    }

    # Run the graph
    logger.info("Invoking graph...")
    result = graph.invoke(initial_state)

    # Output
    if args.output_json:
        # Remove config from JSON output (too noisy)
        output = {k: v for k, v in result.items() if k != "config"}
        print(json.dumps(output, indent=2, default=str))
    else:
        print(format_result(result))

    # Shutdown
    from engine.graph import _sandbox_pool
    if _sandbox_pool is not None:
        _sandbox_pool.shutdown()
    if vram_monitor:
        vram_monitor.stop()
    bus.stop()


if __name__ == "__main__":
    main()
