export type ChatRunStatus =
  | 'pending'
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'success'
  | 'failed'

export interface ChatRunRecord {
  run_id: string
  prompt: string
  status: ChatRunStatus
  created_at: number
  started_at: number | null
  completed_at: number | null
  final_output: string | null
  error: string | null
  stop_requested: boolean
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: number
  runId?: string
  status?: ChatRunStatus
}

export interface ChatTab {
  id: string
  label: 'Main' | 'Sub'
  kind: 'main' | 'sub-agent'
  agentId?: string
}
