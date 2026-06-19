"""
harness/event_bus.py
====================
Pub/sub event bus for engine-harness communication.

The engine emits EngineEvent objects at key moments (node spawned, node
completed, gradient flowed, VRAM alert).  Harness components (dashboard,
VRAM monitor, CLI logger) subscribe to specific event types.

Thread-safe via queue.Queue.  One singleton instance per process.
"""

import logging
import queue
import threading
from typing import Callable, Iterator

from engine.interfaces import EngineEvent, EventType

logger = logging.getLogger("recurseforge.harness.event_bus")


class EventBus:
    """
    Simple pub/sub event bus backed by a thread-safe queue.

    Usage:
        bus = EventBus()

        # Subscribe (blocking consumer in a thread)
        def on_spawn(event):
            print("Spawned:", event.payload)
        bus.subscribe(EventType.NODE_SPAWN, on_spawn)

        # Emit
        bus.emit(EngineEvent(
            event_type=EventType.NODE_SPAWN.value,
            payload={"node_id": "abc", "parent_id": "root", "task": "..."},
        ))
    """

    def __init__(self, maxsize: int = 0):
        """
        Args:
            maxsize: Queue capacity. 0 = unlimited.
        """
        self._queue: queue.Queue[EngineEvent] = queue.Queue(maxsize=maxsize)
        self._subscribers: dict[str, list[Callable[[EngineEvent], None]]] = {}
        self._worker_thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, event: EngineEvent) -> None:
        """Put an event onto the queue (non-blocking)."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event bus queue full, dropping event: %s",
                           event.event_type)

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: str | EventType,
        callback: Callable[[EngineEvent], None],
    ) -> None:
        """
        Register a callback for a specific event type.

        Args:
            event_type: The EventType value to listen for.
            callback: Function called with the EngineEvent when it arrives.
        """
        key = event_type.value if isinstance(event_type, EventType) else event_type
        self._subscribers.setdefault(key, []).append(callback)
        logger.debug("Subscribed to %s", key)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background dispatcher thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="event-bus-dispatcher",
        )
        self._worker_thread.start()
        logger.info("Event bus dispatcher started.")

    def stop(self) -> None:
        """Signal the dispatcher to stop and wait for it."""
        self._running = False
        # Push a sentinel to unblock the queue.get()
        self._queue.put(EngineEvent(event_type="__stop__"))
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Event bus dispatcher stopped.")

    def _dispatch_loop(self) -> None:
        """Internal loop: pull events and fan-out to subscribers."""
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event.event_type == "__stop__":
                break

            callbacks = self._subscribers.get(event.event_type, [])
            for cb in callbacks:
                try:
                    cb(event)
                except Exception as e:
                    logger.error("Subscriber error for %s: %s",
                                 event.event_type, e)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def drain(self) -> list[EngineEvent]:
        """Non-blocking drain of all pending events (useful for testing)."""
        events = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-wide EventBus singleton (creates on first call)."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
