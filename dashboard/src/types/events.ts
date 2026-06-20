// TypeScript types matching the Python EngineEvent payloads

export type EventType = 'node_spawn' | 'node_complete' | 'gradient_flow' | 'vram_alert'

export interface NodeSpawnPayload {
  node_id: string
  parent_id: string
  task: string
}

export interface NodeCompletePayload {
  node_id: string
  result_summary: string
  token_usage: number
  code_executed: boolean
  sandbox_exit_code: number | null
  attempts: number
}

export interface GradientFlowPayload {
  node_id: string
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
  | NodeSpawnPayload
  | NodeCompletePayload
  | GradientFlowPayload
  | VRAMAlertPayload

export interface EngineEvent {
  event_type: EventType
  payload: EventPayload
}

// Agent node state for the visualization
export type AgentStatus = 'running' | 'success' | 'failed' | 'retrying' | 'gradient'

export interface AgentNode {
  id: string
  parentId: string
  task: string
  status: AgentStatus
  result?: string
  stdout?: string
  stderr?: string
  exitCode?: number | null
  attempts?: number
  gradientSeverity?: number
  spawnTime: number
  completeTime?: number
}
