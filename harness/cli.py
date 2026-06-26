"""
harness/cli.py
===============
CLI entry point for RecurseForge.

Usage:
    python -m harness.cli                              # interactive prompt
    python -m harness.cli --task "Write a poem"          # one-shot task
    python -m harness.cli --task "..." --dashboard       # full dashboard (auto-starts frontend + backend + browser)
    python -m harness.cli --config path/to/cfg.yaml      # custom config

The --dashboard flag automatically:
    1. Kills any existing sessions on ports 8100 and 5173
    2. Starts the backend (FastAPI) on port 8100 in a background thread
    3. Starts the frontend (Vite) on port 5173 as a subprocess
    4. Opens the dashboard in your default browser
    5. Cleans up both servers on exit
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import webbrowser
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
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Start the dashboard server alongside the engine (shared event bus)",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    config = load_config(args.config)

    logger = logging.getLogger("recurseforge.cli")
    logger.info("Config loaded from %s", args.config)
    logger.info("LLM endpoint: %s", config["llm"]["base_url"])
    logger.info("LLM context window target: %s tokens; output budget: %s tokens",
                config["llm"].get("context_window", "server-default"),
                config["llm"].get("max_tokens", "server-default"))
    logger.info("Recursion: max_depth=%d, max_children=%d",
                config["recursion"]["max_depth"],
                config["recursion"]["max_children"])

    graph = build_graph(config)

    # Start event bus
    bus = get_event_bus()
    bus.start()

    # Start dashboard (backend + frontend + browser) if requested
    dashboard_thread = None
    vite_process = None
    if args.dashboard:
        import threading
        import uvicorn
        from harness.dashboard_server import app as dashboard_app

        # --- Kill old sessions ---
        _kill_port(8100)
        _kill_port(5173)

        # --- Start backend (FastAPI) in a background thread ---
        def run_dashboard():
            uvicorn.run(dashboard_app, host="127.0.0.1", port=8100,
                        log_level="warning")

        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True,
                                            name="dashboard-server")
        dashboard_thread.start()
        logger.info("Dashboard backend started on http://127.0.0.1:8100")

        # --- Start frontend (Vite dev server) as a subprocess ---
        dashboard_dir = PROJECT_ROOT / "dashboard"
        if dashboard_dir.exists():
            try:
                # On Windows, npx is a .cmd file and needs shell=True
                use_shell = sys.platform == "win32"
                vite_process = subprocess.Popen(
                    ["npx", "vite", "--host", "--port", "5173"],
                    cwd=str(dashboard_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=use_shell,
                    creationflags=subprocess.CREATE_NO_WINDOW
                    if sys.platform == "win32" and not use_shell else 0,
                )
                # Wait for Vite to be ready
                _wait_for_port(5173, timeout=15)
                logger.info("Dashboard frontend started on http://localhost:5173")

                # Open browser
                webbrowser.open("http://localhost:5173")
                logger.info("Opened dashboard in browser")
            except Exception as e:
                logger.warning("Could not start Vite frontend: %s", e)
                logger.info("Open http://localhost:5173 manually to view the dashboard")
        else:
            logger.warning("Dashboard directory not found: %s", dashboard_dir)

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

    # If dashboard mode, keep running until user exits via dashboard Exit button or Ctrl+C
    if args.dashboard:
        logger.info("Task complete. Dashboard remains active.")
        logger.info("Use the Exit button in the dashboard or press Ctrl+C to stop.")
        try:
            # Block until interrupted
            import signal
            signal.pause() if sys.platform != "win32" else input()
        except (KeyboardInterrupt, EOFError):
            pass

    # Shutdown
    from engine.graph import _sandbox_pool
    if _sandbox_pool is not None:
        _sandbox_pool.shutdown()
    if vram_monitor:
        vram_monitor.stop()
    bus.stop()
    if vite_process:
        vite_process.terminate()
        try:
            vite_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            vite_process.kill()
        logger.info("Dashboard frontend stopped")


def _kill_port(port: int):
    """Kill any process listening on the given port (Windows and Linux)."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if ":{} ".format(port) in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = int(parts[-1])
                        subprocess.run(
                            ["taskkill", "/f", "/pid", str(pid)],
                            capture_output=True, timeout=5,
                        )
        else:
            subprocess.run(
                ["fuser", "-k", "{}/tcp".format(port)],
                capture_output=True, timeout=5,
            )
    except Exception:
        pass


def _wait_for_port(port: int, timeout: float = 15):
    """Wait until a port is accepting connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    raise TimeoutError("Port {} not ready after {}s".format(port, timeout))


if __name__ == "__main__":
    main()
