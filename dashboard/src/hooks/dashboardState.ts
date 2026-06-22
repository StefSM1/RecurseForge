import type {
  AgentNode,
  CorrectionCompletedPayload,
  CorrectionProgressPayload,
  CorrectionRun,
  CorrectionStartedPayload,
  EngineEvent,
  GradientFlowPayload,
  NodeCompletePayload,
  NodeSpawnPayload,
  RunCompletedPayload,
  RunStartedPayload,
  RunState,
  SandboxCompletedPayload,
  SandboxRun,
  SandboxStartedPayload,
} from '../types/events'

export interface DashboardDataState {
  run: RunState | null
  nodes: Map<string, AgentNode>
  sandboxRuns: Map<string, SandboxRun>
  corrections: Map<string, CorrectionRun>
  events: EngineEvent[]
  seenEventIds: Set<string>
}

export function createDashboardDataState(): DashboardDataState {
  return {
    run: null,
    nodes: new Map(),
    sandboxRuns: new Map(),
    corrections: new Map(),
    events: [],
    seenEventIds: new Set(),
  }
}

function eventTime(event: EngineEvent): number {
  const timestamp = event.timestamp
  if (timestamp === undefined) return Date.now()
  return timestamp < 1_000_000_000_000 ? timestamp * 1000 : timestamp
}

function eventIdentity(event: EngineEvent): string {
  return event.event_id ?? [
    event.run_id ?? '',
    event.timestamp ?? '',
    event.event_type,
    JSON.stringify(event.payload),
  ].join(':')
}

export function reduceEngineEvent(
  state: DashboardDataState,
  event: EngineEvent,
): DashboardDataState {
  const identity = eventIdentity(event)
  if (state.seenEventIds.has(identity)) return state

  let nodes = new Map(state.nodes)
  let sandboxRuns = new Map(state.sandboxRuns)
  let corrections = new Map(state.corrections)
  const seenEventIds = new Set(state.seenEventIds).add(identity)
  const timestamp = eventTime(event)
  let run = state.run

  switch (event.event_type) {
    case 'run_started': {
      const payload = event.payload as RunStartedPayload
      const nextRunId = event.run_id ?? 'current'
      if (run?.id !== nextRunId) {
        nodes = new Map()
        sandboxRuns = new Map()
        corrections = new Map()
      }
      run = {
        id: nextRunId,
        task: payload.task,
        mode: payload.mode ?? 'unknown',
        status: 'running',
        startTime: timestamp,
      }
      break
    }
    case 'run_completed': {
      const payload = event.payload as RunCompletedPayload
      run = {
        id: event.run_id ?? run?.id ?? 'current',
        task: run?.task ?? '',
        mode: payload.mode,
        status: payload.success ? 'success' : 'failed',
        startTime: run?.startTime ?? timestamp,
        completeTime: timestamp,
        resultSummary: payload.result_summary,
      }
      break
    }
    case 'node_spawn': {
      const payload = event.payload as NodeSpawnPayload
      nodes.set(payload.node_id, {
        id: payload.node_id,
        runId: event.run_id,
        parentId: payload.parent_id,
        task: payload.task,
        status: 'running',
        spawnTime: timestamp,
      })
      break
    }
    case 'node_complete': {
      const payload = event.payload as NodeCompletePayload
      const existing = nodes.get(payload.node_id)
      if (existing) {
        const success = payload.success
          ?? (payload.sandbox_exit_code === 0 || payload.sandbox_exit_code === null)
        nodes.set(payload.node_id, {
          ...existing,
          status: success ? 'success' : 'failed',
          codeExecuted: payload.code_executed,
          result: payload.result_summary,
          exitCode: payload.sandbox_exit_code,
          attempts: payload.attempts,
          completeTime: timestamp,
        })
      }
      break
    }
    case 'sandbox_started': {
      const payload = event.payload as SandboxStartedPayload
      sandboxRuns.set(payload.execution_id, {
        id: payload.execution_id,
        runId: event.run_id,
        ownerId: payload.owner_node_id,
        attempt: payload.attempt,
        trigger: payload.trigger,
        status: 'running',
        timeoutS: payload.timeout_s,
        codePreview: payload.code_preview,
        startTime: timestamp,
      })
      break
    }
    case 'sandbox_completed': {
      const payload = event.payload as SandboxCompletedPayload
      const existing = sandboxRuns.get(payload.execution_id)
      sandboxRuns.set(payload.execution_id, {
        id: payload.execution_id,
        runId: event.run_id ?? existing?.runId,
        ownerId: payload.owner_node_id,
        attempt: payload.attempt,
        trigger: existing?.trigger ?? 'initial',
        status: payload.status,
        timeoutS: existing?.timeoutS ?? 0,
        codePreview: existing?.codePreview ?? '',
        startTime: existing?.startTime ?? timestamp - payload.duration_ms,
        completeTime: timestamp,
        durationMs: payload.duration_ms,
        exitCode: payload.exit_code,
        stdout: payload.stdout_preview,
        stderr: payload.stderr_preview,
      })
      break
    }
    case 'correction_started': {
      const payload = event.payload as CorrectionStartedPayload
      corrections.set(payload.correction_id, {
        id: payload.correction_id,
        runId: event.run_id,
        ownerId: payload.owner_node_id,
        failedExecutionId: payload.failed_execution_id,
        attempt: payload.attempt,
        strategy: payload.strategy,
        status: 'running',
        phase: payload.strategy === 'textgrad' ? 'evaluating_loss' : 'requesting_retry',
        startTime: timestamp,
        mutations: [],
      })
      break
    }
    case 'correction_progress': {
      const payload = event.payload as CorrectionProgressPayload
      const existing = corrections.get(payload.correction_id)
      if (existing) {
        corrections.set(payload.correction_id, {
          ...existing,
          phase: payload.phase,
          iterations: payload.iteration ?? existing.iterations,
          severity: payload.severity ?? existing.severity,
          numMutations: payload.num_mutations ?? existing.numMutations,
          mutations: payload.mutations ?? existing.mutations,
        })
      }
      break
    }
    case 'correction_completed': {
      const payload = event.payload as CorrectionCompletedPayload
      const existing = corrections.get(payload.correction_id)
      if (existing) {
        corrections.set(payload.correction_id, {
          ...existing,
          phase: 'completed',
          status: payload.success ? 'success' : 'failed',
          iterations: payload.iterations ?? existing.iterations,
          error: payload.error,
          completeTime: timestamp,
        })
      }
      break
    }
    case 'gradient_flow': {
      const payload = event.payload as GradientFlowPayload
      const existing = nodes.get(payload.node_id)
      if (existing) {
        nodes.set(payload.node_id, {
          ...existing,
          gradientSeverity: payload.severity,
        })
      }
      if (payload.correction_id) {
        const correction = corrections.get(payload.correction_id)
        if (correction) {
          corrections.set(payload.correction_id, {
            ...correction,
            phase: 'gradient_ready',
            iterations: payload.iteration,
            severity: payload.severity,
            numMutations: payload.num_mutations,
          })
        }
      }
      break
    }
  }

  return {
    run,
    nodes,
    sandboxRuns,
    corrections,
    events: [...state.events, event],
    seenEventIds,
  }
}
