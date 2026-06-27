// TypeScript contracts for current engine events and the frontend execution prototype.

export type EventType =
  | 'run_started'
  | 'run_completed'
  | 'node_spawn'
  | 'node_complete'
  | 'sandbox_started'
  | 'sandbox_completed'
  | 'correction_started'
  | 'correction_progress'
  | 'correction_completed'
  | 'gradient_flow'
  | 'vram_alert'

export interface RunStartedPayload {
  task: string
  mode?: 'direct' | 'delegated' | 'unknown'
}

export interface RunCompletedPayload {
  success: boolean
  mode: 'direct' | 'delegated'
  result_summary?: string
  result?: string
}

export interface NodeSpawnPayload {
  node_id: string
  parent_id: string
  task: string
}

export interface NodeCompletePayload {
  node_id: string
  result_summary: string
  result?: string
  token_usage: number
  code_executed: boolean
  sandbox_exit_code: number | null
  attempts: number
  success?: boolean
  failure_reason?: string
}

export type SandboxTrigger = 'initial' | 'textgrad' | 'llm_retry'
export type SandboxStatus = 'running' | 'success' | 'failed'

export interface SandboxStartedPayload {
  execution_id: string
  owner_node_id: string
  attempt: number
  trigger: SandboxTrigger
  timeout_s: number
  code_preview: string
}

export interface SandboxCompletedPayload {
  execution_id: string
  owner_node_id: string
  attempt: number
  status: Exclude<SandboxStatus, 'running'>
  exit_code: number
  duration_ms: number
  stdout_preview?: string
  stderr_preview?: string
}

export type CorrectionStrategy = 'textgrad' | 'llm_retry'
export type CorrectionPhase =
  | 'evaluating_loss'
  | 'gradient_ready'
  | 'applying_update'
  | 'requesting_retry'
  | 'completed'

export interface CorrectionStartedPayload {
  correction_id: string
  owner_node_id: string
  failed_execution_id: string
  attempt: number
  strategy: CorrectionStrategy
}

export interface CorrectionProgressPayload {
  correction_id: string
  phase: CorrectionPhase
  iteration?: number
  severity?: number
  num_mutations?: number
  mutations?: Array<{ line: number; cause: string; suggestion: string }>
}

export interface CorrectionCompletedPayload {
  correction_id: string
  success: boolean
  iterations?: number
  error?: string
}

export interface GradientFlowPayload {
  node_id: string
  correction_id?: string
  iteration: number
  severity: number
  num_mutations: number
}

export interface VRAMAlertPayload {
  current_vram_mb: number
  threshold_mb: number
  level: 'warning' | 'critical'
  action_taken: string
}

export type EventPayload =
  | RunStartedPayload
  | RunCompletedPayload
  | NodeSpawnPayload
  | NodeCompletePayload
  | SandboxStartedPayload
  | SandboxCompletedPayload
  | CorrectionStartedPayload
  | CorrectionProgressPayload
  | CorrectionCompletedPayload
  | GradientFlowPayload
  | VRAMAlertPayload

export interface EngineEvent {
  event_id?: string
  run_id?: string
  timestamp?: number
  event_type: EventType
  payload: EventPayload
}

export type AgentStatus = 'running' | 'success' | 'failed' | 'retrying' | 'gradient'

export interface AgentNode {
  id: string
  runId?: string
  parentId: string
  task: string
  status: AgentStatus
  codeExecuted?: boolean
  result?: string
  stdout?: string
  stderr?: string
  exitCode?: number | null
  attempts?: number
  gradientSeverity?: number
  spawnTime: number
  completeTime?: number
}

export interface SandboxRun {
  id: string
  runId?: string
  ownerId: string
  attempt: number
  trigger: SandboxTrigger
  status: SandboxStatus
  timeoutS: number
  codePreview: string
  startTime: number
  completeTime?: number
  durationMs?: number
  exitCode?: number
  stdout?: string
  stderr?: string
}

export type CorrectionStatus = 'running' | 'success' | 'failed'

export interface CorrectionRun {
  id: string
  runId?: string
  ownerId: string
  failedExecutionId: string
  attempt: number
  strategy: CorrectionStrategy
  status: CorrectionStatus
  phase: CorrectionPhase
  startTime: number
  completeTime?: number
  iterations?: number
  severity?: number
  numMutations?: number
  mutations: Array<{ line: number; cause: string; suggestion: string }>
  error?: string
}

export interface RunState {
  id: string
  task: string
  mode: 'direct' | 'delegated' | 'unknown'
  status: 'running' | 'success' | 'failed'
  startTime: number
  completeTime?: number
  resultSummary?: string
  result?: string
}

export interface RootAgentView {
  id: 'root'
  task: string
  status: 'offline' | 'running' | 'retrying' | 'success' | 'error'
  result?: string
}

export type GraphEntity =
  | { kind: 'root'; value: RootAgentView }
  | { kind: 'agent'; value: AgentNode }
  | { kind: 'sandbox'; value: SandboxRun }
  | { kind: 'correction'; value: CorrectionRun }
