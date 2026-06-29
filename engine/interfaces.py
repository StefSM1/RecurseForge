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
    ContextBudget     -- hard input/output/safety token limits
    ContextBudgetReport -- per-call context preflight telemetry
    ContextSection     -- one named, prioritized prompt contribution
    ContextBundle      -- assembled messages plus inclusion telemetry
    TaskCapsule       -- focused downward assignment for one child agent
    ResultFrame       -- compact upward payload from a completed child
    ExecutionResult   -- sandbox output after running agent code
    Mutation          -- a single fix suggestion inside a TextGradient
    TextGradient      -- structured critique from TextGrad
    HarnessCommand    -- CLI -> engine control signal
    EngineEvent       -- engine -> harness telemetry signal
"""

from __future__ import annotations

import enum
import json
import time
import uuid
from typing import Any, Optional


from pydantic import BaseModel, Field, model_validator


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
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    NODE_SPAWN = "node_spawn"
    NODE_COMPLETE = "node_complete"
    SANDBOX_STARTED = "sandbox_started"
    SANDBOX_COMPLETED = "sandbox_completed"
    CORRECTION_STARTED = "correction_started"
    CORRECTION_PROGRESS = "correction_progress"
    CORRECTION_COMPLETED = "correction_completed"
    GRADIENT_FLOW = "gradient_flow"
    VRAM_ALERT = "vram_alert"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    FILE_TOOL_STARTED = "file_tool_started"
    FILE_TOOL_COMPLETED = "file_tool_completed"
    WORKSPACE_CHANGED = "workspace_changed"
    WORKSPACE_LOCK_CHANGED = "workspace_lock_changed"


class RunMode(str, enum.Enum):
    CHAT = "chat"
    WORKSPACE_AGENT = "workspace_agent"


class WorkspaceStage(str, enum.Enum):
    PLAN = "plan"
    WORKER = "worker"
    DEBUG = "debug"
    ROOT_SYNTHESIS = "root_synthesis"


class WorkspaceRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    CANCELED = "canceled"


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


class ContextBudget(VersionedModel):
    """Hard token limits applied before an LLM request is sent."""
    context_window: int = Field(gt=0)
    max_prompt_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(ge=0)
    safety_buffer_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total_budget(self) -> "ContextBudget":
        total = (
            self.max_prompt_tokens
            + self.reserved_output_tokens
            + self.safety_buffer_tokens
        )
        if total > self.context_window:
            raise ValueError(
                "Context budget exceeds context window: "
                "{} prompt + {} output + {} safety = {} > {}".format(
                    self.max_prompt_tokens,
                    self.reserved_output_tokens,
                    self.safety_buffer_tokens,
                    total,
                    self.context_window,
                )
            )
        return self


class ContextBudgetReport(VersionedModel):
    """Serializable preflight result for one LLM call."""
    call_kind: str
    estimated_prompt_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(ge=0)
    safety_buffer_tokens: int = Field(ge=0)
    effective_context_window: int = Field(gt=0)
    max_prompt_tokens: int = Field(gt=0)
    remaining_prompt_tokens: int
    within_budget: bool
    estimator_name: str = "utf8_bytes_per_3_conservative"


class ContextSection(VersionedModel):
    """A named prompt contribution with deterministic retention rules."""
    name: str
    role: str
    content: str
    required: bool = False
    priority: int = 0
    trim_strategy: str = "none"
    max_tokens: int | None = Field(default=None, gt=0)


class ContextBundle(VersionedModel):
    """The exact messages selected for an LLM call and why."""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    included_sections: list[str] = Field(default_factory=list)
    omitted_sections: list[str] = Field(default_factory=list)
    section_token_counts: dict[str, int] = Field(default_factory=dict)
    budget_report: ContextBudgetReport | None = None


class TaskCapsule(VersionedModel):
    """A compact, self-contained assignment passed from parent to child."""
    task: str
    role: str = ""
    goal: str = ""
    known_facts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    requested_files: list[str] = Field(default_factory=list)
    requested_symbols: list[str] = Field(default_factory=list)
    return_format: str = ""


class EvidenceRef(VersionedModel):
    """A bounded source reference supporting a result-frame finding."""
    file_path: str = ""
    symbol_name: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    finding: str = ""


class RiskItem(VersionedModel):
    """A concise risk reported by a child agent."""
    description: str
    severity: str = "unknown"


class ResultFrame(VersionedModel):
    """Compact parent-facing result; raw model output is stored separately."""
    node_id: str
    status: str
    summary: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)
    changes_needed: list[str] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ExecutionResult(VersionedModel):
    """Sandbox output after running agent-generated code."""
    node_id: str
    code_output: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    token_usage: int = 0


# ---------------------------------------------------------------------------
# Workspace-agent contracts
# ---------------------------------------------------------------------------

class WorkspaceFileRef(VersionedModel):
    path: str
    revision: str
    size_bytes: int = Field(ge=0)


class WorkspaceFileRevision(VersionedModel):
    path: str
    content_hash: str
    workspace_revision: int = Field(ge=0)
    modified_at: float = Field(default_factory=time.time)


class WorkspaceLock(VersionedModel):
    path: str
    owner_id: str
    acquired_at: float = Field(default_factory=time.time)


class PlanFrame(VersionedModel):
    objective: str
    steps: list[str] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class WorkerCompletion(VersionedModel):
    summary: str
    changed_files: list[WorkspaceFileRef] = Field(default_factory=list)
    test_results: list[dict[str, Any]] = Field(default_factory=list)


class DebugVerdict(VersionedModel):
    verdict: str
    findings: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    rationale: str = ""
    required_changes: list[str] = Field(default_factory=list)


class FixRequest(VersionedModel):
    pass_number: int = Field(ge=1)
    findings: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)


class FileToolRun(VersionedModel):
    tool_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    actor_id: str
    stage: str
    pass_number: int = Field(ge=1)
    operation: str
    path: str | None = None
    status: str = WorkspaceRunStatus.PENDING.value
    revision_before: str | None = None
    revision_after: str | None = None
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    error: str | None = None


class StageRun(VersionedModel):
    stage_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    stage: str
    pass_number: int = Field(ge=1)
    status: str = WorkspaceRunStatus.PENDING.value
    started_at: float | None = None
    completed_at: float | None = None


class WorkspaceAgentState(VersionedModel):
    run_id: str
    pass_number: int = Field(default=1, ge=1)
    workspace_revision: int = Field(default=0, ge=0)
    status: str = WorkspaceRunStatus.PENDING.value
    plan: PlanFrame | None = None
    changed_files: list[WorkspaceFileRef] = Field(default_factory=list)
    manifest: dict[str, Any] | None = None
    test_results: list[dict[str, Any]] = Field(default_factory=list)
    debug_verdict: DebugVerdict | None = None
    tool_calls_used: int = Field(default=0, ge=0)
    helpers_used: int = Field(default=0, ge=0)
    cancellation_requested: bool = False
    final_summary: str | None = None


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
    run_id: str | None = None
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
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str | None = None
    timestamp: float = Field(default_factory=time.time)
    event_type: str  # EventType value
    payload: dict[str, Any] = Field(default_factory=dict)
