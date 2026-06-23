import type { AgentNode, RunState, SandboxRun } from '../types/events'

export type VramSeverity = 'unknown' | 'normal' | 'warning' | 'critical'

export interface SummaryStats {
  taskName: string
  elapsedMs: number
  totalNodes: number
  runningCount: number
  successCount: number
  failedCount: number
  sandboxCount: number
  eventCount: number
  vramSeverity: VramSeverity
}

const VRAM_WARNING_MB = 6500
const VRAM_CRITICAL_MB = 7000

export function getVramSeverity(vramMb: number): VramSeverity {
  if (vramMb <= 0) return 'unknown'
  if (vramMb > VRAM_CRITICAL_MB) return 'critical'
  if (vramMb > VRAM_WARNING_MB) return 'warning'
  return 'normal'
}

export function formatElapsedTime(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export function buildSummaryStats({
  run,
  nodes,
  sandboxRuns,
  eventCount,
  rootTask,
  retryingOwnerIds = new Set<string>(),
  now = Date.now(),
  vramMb = 0,
}: {
  run: RunState | null
  nodes: AgentNode[]
  sandboxRuns: SandboxRun[]
  eventCount: number
  rootTask: string
  retryingOwnerIds?: Set<string>
  now?: number
  vramMb?: number
}): SummaryStats {
  const completeOrNow = run?.completeTime ?? now
  const elapsedMs = run?.startTime ? completeOrNow - run.startTime : 0
  const runningCount = nodes.filter(node =>
    node.status === 'running' || retryingOwnerIds.has(node.id),
  ).length + (retryingOwnerIds.has('root') ? 1 : 0)

  return {
    taskName: run?.task || rootTask || 'No active task',
    elapsedMs,
    totalNodes: nodes.length,
    runningCount,
    successCount: nodes.filter(node => node.status === 'success').length,
    failedCount: nodes.filter(node => node.status === 'failed').length,
    sandboxCount: sandboxRuns.length,
    eventCount,
    vramSeverity: getVramSeverity(vramMb),
  }
}
