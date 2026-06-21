"""
harness/sandbox.py
==================
Sandbox executor pool for running sub-agent generated code safely.

Maintains a pool of subprocess workers. Each worker runs code in an
isolated Python subprocess with restricted environment.  Workers are
recycled after every execution to prevent state leakage.

Usage:
    pool = SandboxPool(pool_size=4, timeout_s=30)
    result = pool.execute("node-abc", "print(2+2)")
    print(result.stdout)  # "4\n"
    print(result.exit_code)  # 0
    pool.shutdown()
"""

import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from engine.interfaces import EngineEvent, EventType, ExecutionResult
from harness.event_bus import get_event_bus

logger = logging.getLogger("recurseforge.harness.sandbox")


class SandboxPool:
    """
    Pool of ephemeral subprocess workers for code execution.

    Each execution:
    1. Writes the code to a temp file.
    2. Runs it in a fresh Python subprocess.
    3. Captures stdout, stderr, exit_code.
    4. Cleans up the temp file.
    5. Returns an ExecutionResult.

    No persistent state between executions -- each run is isolated.
    """

    def __init__(self, pool_size: int = 4, timeout_s: int = 30,
                 python_executable: str | None = None):
        """
        Args:
            pool_size: Max concurrent executions (currently sequential,
                       reserved for future threading).
            timeout_s: Seconds before killing the subprocess.
            python_executable: Path to Python binary. Defaults to sys.executable.
        """
        self.pool_size = pool_size
        self.timeout_s = timeout_s
        self.python = python_executable or sys.executable
        self._temp_dir = Path(tempfile.mkdtemp(prefix="recurseforge_sandbox_"))
        self._execution_count = 0
        logger.info("[Sandbox] Pool ready. Python: %s, Timeout: %ds, Temp: %s",
                    self.python, self.timeout_s, self._temp_dir)

    def execute(self, node_id: str, code: str,
                timeout_s: int | None = None, *,
                run_id: str | None = None,
                execution_id: str | None = None,
                attempt: int = 1,
                trigger: str = "initial") -> ExecutionResult:
        """
        Execute Python code in an isolated subprocess.

        Args:
            node_id: Identifier for the calling sub-agent.
            code: Python source code to execute.
            timeout_s: Override default timeout for this execution.

        Returns:
            ExecutionResult with stdout, stderr, exit_code.
        """
        self._execution_count += 1
        timeout = timeout_s or self.timeout_s
        execution_id = execution_id or str(uuid.uuid4())
        bus = get_event_bus()
        bus.emit(EngineEvent(
            run_id=run_id,
            event_type=EventType.SANDBOX_STARTED.value,
            payload={
                "execution_id": execution_id,
                "owner_node_id": node_id,
                "attempt": attempt,
                "trigger": trigger,
                "timeout_s": timeout,
                "code_preview": code[:500],
            },
        ))
        started_at = time.perf_counter()

        script_path = self._temp_dir / "sandbox_{}.py".format(node_id)

        try:
            # File preparation is part of the attempt so internal errors still
            # produce a matching sandbox_completed event.
            script_path.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                [self.python, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._temp_dir),
                encoding="utf-8",
                errors="replace",
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": "",
                    "HOME": str(self._temp_dir),
                    "TEMP": str(self._temp_dir),
                    "TMP": str(self._temp_dir),
                    "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                },
            )
            result = ExecutionResult(
                node_id=node_id,
                code_output=code,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                token_usage=len(code.split()),  # rough estimate
            )
            if proc.returncode == 0:
                logger.info("[Sandbox] %s executed OK (stdout: %d chars)",
                            node_id, len(proc.stdout))
            else:
                logger.warning("[Sandbox] %s failed (exit %d): %s",
                               node_id, proc.returncode,
                               proc.stderr[:200])
            return result

        except subprocess.TimeoutExpired:
            logger.error("[Sandbox] %s timed out after %ds", node_id, timeout)
            result = ExecutionResult(
                node_id=node_id,
                code_output=code,
                stdout="",
                stderr="TIMEOUT: execution exceeded {}s".format(timeout),
                exit_code=-1,
            )
            return result
        except Exception as e:
            logger.error("[Sandbox] %s error: %s", node_id, e)
            result = ExecutionResult(
                node_id=node_id,
                code_output=code,
                stdout="",
                stderr=str(e),
                exit_code=-2,
            )
            return result
        finally:
            result = locals().get("result")
            if result is not None:
                bus.emit(EngineEvent(
                    run_id=run_id,
                    event_type=EventType.SANDBOX_COMPLETED.value,
                    payload={
                        "execution_id": execution_id,
                        "owner_node_id": node_id,
                        "attempt": attempt,
                        "status": "success" if result.exit_code == 0 else "failed",
                        "exit_code": result.exit_code,
                        "duration_ms": round((time.perf_counter() - started_at) * 1000),
                        "stdout_preview": result.stdout[:500],
                        "stderr_preview": result.stderr[:500],
                    },
                ))
            # Clean up temp file
            try:
                script_path.unlink(missing_ok=True)
            except Exception:
                pass

    def shutdown(self):
        """Clean up the temp directory."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
        logger.info("[Sandbox] Pool shut down. %d executions completed.",
                    self._execution_count)

    @property
    def stats(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "timeout_s": self.timeout_s,
            "execution_count": self._execution_count,
            "temp_dir": str(self._temp_dir),
        }
