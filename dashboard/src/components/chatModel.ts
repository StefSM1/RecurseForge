import type { AgentNode, EngineEvent } from '../types/events'
import type { ChatTab } from '../types/chat'

export function shouldSubmitChatInput(event: {
  key: string
  shiftKey: boolean
  isComposing?: boolean
}): boolean {
  return event.key === 'Enter' && !event.shiftKey && !event.isComposing
}

export function buildChatTabs(nodes: AgentNode[]): ChatTab[] {
  const activeSubAgents = nodes
    .filter(node => node.status === 'running' || node.status === 'retrying')
    .sort((left, right) => right.spawnTime - left.spawnTime)

  return [
    { id: 'main', label: 'Main', kind: 'main' },
    ...activeSubAgents.map(node => ({
      id: `sub-${node.id}`,
      label: 'Sub' as const,
      kind: 'sub-agent' as const,
      agentId: node.id,
    })),
  ]
}

export function isTerminalChatRunStatus(status: string): boolean {
  return status === 'success' || status === 'failed' || status === 'stopped'
}

export function hasRunCompletedEvent(events: EngineEvent[], runId: string): boolean {
  return events.some(event =>
    event.run_id === runId && event.event_type === 'run_completed',
  )
}
