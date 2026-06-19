"""
harness/vram_monitor.py
=======================
Background daemon that monitors GPU VRAM usage and triggers tiered
memory demotion when thresholds are exceeded.

Uses pynvml (NVIDIA) if available, falls back to a stub that reports
zero usage (for development without a GPU).

Emits VRAM_ALERT events on the event bus when thresholds are crossed.

Usage:
    monitor = VRAMMonitor(warning_mb=6000, critical_mb=7500)
    monitor.start()   # starts background polling thread
    ...
    monitor.stop()
"""

import logging
import threading
import time

from engine.interfaces import EngineEvent, EventType
from harness.event_bus import get_event_bus

logger = logging.getLogger("recurseforge.harness.vram_monitor")

# ---------------------------------------------------------------------------
# GPU memory reading (pynvml or stub)
# ---------------------------------------------------------------------------

def _try_import_pynvml():
    """Try to import pynvml. Return getter function or stub."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def read_vram_mb():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used // (1024 * 1024)

        logger.info("[VRAM] Using pynvml for GPU monitoring.")
        return read_vram_mb

    except ImportError:
        logger.warning("[VRAM] pynvml not installed. "
                       "Using stub (reports 0 MB). "
                       "Install with: .venv\\Scripts\\pip install pynvml")

        def stub():
            return 0

        return stub
    except Exception as e:
        logger.warning("[VRAM] pynvml init failed (%s). Using stub.", e)

        def stub():
            return 0

        return stub


# ---------------------------------------------------------------------------
# Monitor daemon
# ---------------------------------------------------------------------------

class VRAMMonitor:
    """
    Background thread that polls GPU memory and emits alerts.

    Args:
        warning_mb: VRAM usage (MB) that triggers a WARNING demotion.
        critical_mb: VRAM usage (MB) that triggers CRITICAL eviction.
        poll_interval_s: Seconds between polls (default 2).
    """

    def __init__(self, warning_mb: int = 6000, critical_mb: int = 7500,
                 poll_interval_s: float = 2.0):
        self.warning_mb = warning_mb
        self.critical_mb = critical_mb
        self.poll_interval_s = poll_interval_s
        self._read_vram = _try_import_pynvml()
        self._bus = get_event_bus()
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_alert_level = "ok"  # "ok" | "warning" | "critical"
        self._history: list[tuple[float, int]] = []

    def start(self):
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="vram-monitor",
        )
        self._thread.start()
        logger.info("[VRAM] Monitor started. Warning: %d MB, Critical: %d MB",
                    self.warning_mb, self.critical_mb)

    def stop(self):
        """Stop the background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[VRAM] Monitor stopped.")

    def _poll_loop(self):
        while self._running:
            try:
                current_mb = self._read_vram()
                self._history.append((time.time(), current_mb))

                if current_mb >= self.critical_mb:
                    if self._last_alert_level != "critical":
                        self._emit_alert(current_mb, "critical",
                                         "force_evict_l0")
                        self._last_alert_level = "critical"
                elif current_mb >= self.warning_mb:
                    if self._last_alert_level == "ok":
                        self._emit_alert(current_mb, "warning",
                                         "demote_l1_to_l2")
                        self._last_alert_level = "warning"
                else:
                    self._last_alert_level = "ok"

            except Exception as e:
                logger.error("[VRAM] Poll error: %s", e)

            time.sleep(self.poll_interval_s)

    def _emit_alert(self, current_mb: int, level: str, action: str):
        event = EngineEvent(
            event_type=EventType.VRAM_ALERT.value,
            payload={
                "current_vram_mb": current_mb,
                "threshold_mb": (self.critical_mb if level == "critical"
                                 else self.warning_mb),
                "level": level,
                "action_taken": action,
            },
        )
        self._bus.emit(event)
        logger.warning("[VRAM] ALERT [%s]: %d MB (threshold %d MB) -> %s",
                       level, current_mb,
                       event.payload["threshold_mb"], action)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def current_vram_mb(self) -> int:
        return self._read_vram()

    @property
    def history(self) -> list[tuple[float, int]]:
        """Return (timestamp, vram_mb) pairs."""
        return list(self._history)

    @property
    def stats(self) -> dict:
        return {
            "current_mb": self.current_vram_mb,
            "warning_mb": self.warning_mb,
            "critical_mb": self.critical_mb,
            "alert_level": self._last_alert_level,
            "history_points": len(self._history),
        }
