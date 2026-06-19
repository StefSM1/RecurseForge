"""
engine/interfaces.py
====================
Pydantic v2 models -- the shared contracts between every layer.

Every message flowing through the system is one of these models.
Both the engine and the harness import from here.

Models:
    GraphState        -- the outer LangGraph state object
    NodeFrame         -- a single sub-agent's task + result
    ContextRequest    -- sub-agent asking the repo-map for code
    ContextPayload    -- repo-map server's reply
    ExecutionResult   -- sandbox output after running agent code
    Mutation          -- a single fix suggestion inside a TextGradient
    TextGradient      -- structured critique from TextGrad
    HarnessCommand    -- CLI -> engine control signal
    EngineEvent       -- engine -> harness telemetry signal
"""

from __future__ import annotations

import enum
import json
from typing import Any, Optional


from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base mixin
# ---------------------------------------------------------------------------

class VersionedModel(BaseModel):
    """Every model carries a schema version for forward compatibility."""
    schema_version: int = Field(default=1, frozen=True)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str | dict) -> "VersionedModel":
        if isinstance(data, str):
            return cls.model_validate_json(data)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GraphStatus(str, enum.Enum):
    INIT = "init"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    DONE = "done"


class CommandType(str, enum.Enum):
    RUN = "run"
    STEP = "step"
    STATUS = "status"
    ABORT = "abort"


class EventType(str, enum.Enum):
    NODE_SPAWN = "node_spawn"
    NODE_COMPLETE = "node_complete"
    GRADIENT_FLOW = "gradient_flow"
    VRAM_ALERT = "vram_alert"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class NodeFrame(VersionedModel):
    """A single sub-agent's task assignment and result."""
    node_id: str
    parent_id: str | None = None
    system_prompt: str = ""
    task: str
    code_context_request: list[str] = Field(default_factory=list)
    depth: int = 0
    result: str | None = None


class ContextRequest(VersionedModel):
    """Sub-agent asking the repo-map server for a code fragment."""
    node_id: str
    file_path: str
    symbol_name: str | None = None
    line_range: tuple[int, int] | None = None


class ContextPayload(VersionedModel):
    """Repo-map server's reply to a ContextRequest."""
    node_id: str
    file_path: str
    content: str
    token_count: int


class ExecutionResult(VersionedModel):
    """Sandbox output after running agent-generated code."""
    node_id: str
    code_output: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    token_usage: int = 0


# ---------------------------------------------------------------------------
# TextGrad models (Phase 3 -- defined now for completeness)
# ---------------------------------------------------------------------------

class Mutation(VersionedModel):
    """A single fix suggestion inside a TextGradient."""
    line: int
    cause: str
    suggestion: str


class TextGradient(VersionedModel):
    """Structured critique from the TextGrad layer."""
    node_id: str
    loss_description: str
    mutations: list[Mutation] = Field(default_factory=list)
    target_variable: str = "code_output"  # or "system_prompt"
    severity: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_formatted_string(self) -> str:
        """
        Format gradient as dense, structured text (FedTextGrad UID principle).

        Each mutation is compressed into a single line:
            [L<line>] CAUSE: <cause> -> FIX: <suggestion>

        This maximizes information density per token, stripping
        conversational fluff -- the core idea behind Uniform Information
        Density from the FedTextGrad paper.
        """
        lines = []
        for m in self.mutations:
            if m.line > 0:
                lines.append("[L{}] CAUSE: {} -> FIX: {}".format(
                    m.line, m.cause, m.suggestion))
            else:
                lines.append("CAUSE: {} -> FIX: {}".format(
                    m.cause, m.suggestion))
        if lines:
            return "\n".join(lines)
        return self.loss_description


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(VersionedModel):
    """The outer LangGraph state object.

    Note: LangGraph requires a TypedDict for StateGraph. We use this
    Pydantic model for validation and serialization, but the graph
    itself operates on plain dicts (converted via model_dump()).
    """
    task_id: str = "root"
    task_description: str = ""
    status: str = GraphStatus.INIT.value
    children: list[dict[str, Any]] = Field(default_factory=list)
    depth: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    direct_answer: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Harness-engine communication
# ---------------------------------------------------------------------------

class HarnessCommand(VersionedModel):
    """CLI -> engine control signal."""
    command: str = CommandType.RUN.value  # run | step | status | abort
    task_description: str | None = None
    config: dict[str, Any] | None = None


class EngineEvent(VersionedModel):
    """Engine -> harness telemetry signal."""
    event_type: str  # EventType value
    payload: dict[str, Any] = Field(default_factory=dict)
