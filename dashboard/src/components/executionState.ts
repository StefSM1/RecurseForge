import type {
  AgentNode,
  CorrectionRun,
  RunState,
  SandboxRun,
  SandboxStatus,
} from '../types/events'

export type SandboxNodeStatus = SandboxStatus | 'offline'
export type FeedbackState = 'receiving' | 'withdrawing' | 'none'

export interface SandboxNodeView extends Omit<SandboxRun, 'id' | 'status'> {
  id: string
  executionId: string
  status: SandboxNodeStatus
}

export interface OwnerExecutionView {
  ownerId: string
  sandbox: SandboxNodeView
  retrying: boolean
  feedback: FeedbackState
  feedbackAttempt: number
  failedExecutionId?: string
}

export interface RetryCycle {
  attempt: SandboxRun
  corrections: CorrectionRun[]
}

function byAttempt(a: SandboxRun, b: SandboxRun): number {
  return a.attempt - b.attempt || a.startTime - b.startTime
}

function isOwnerTerminal(
  ownerId: string,
  agents: Map<string, AgentNode>,
  run: RunState | null,
): boolean {
  if (ownerId === 'root') return Boolean(run && run.status !== 'running')
  const owner = agents.get(ownerId)
  return owner?.status === 'success' || owner?.status === 'failed'
}

export function buildOwnerExecutionViews(
  agentNodes: AgentNode[],
  sandboxRuns: SandboxRun[],
  corrections: CorrectionRun[],
  run: RunState | null,
): OwnerExecutionView[] {
  const agents = new Map(agentNodes.map(agent => [agent.id, agent]))
  const runsByOwner = new Map<string, SandboxRun[]>()
  for (const sandbox of sandboxRuns) {
    if (sandbox.ownerId !== 'root' && !agents.has(sandbox.ownerId)) continue
    const ownerRuns = runsByOwner.get(sandbox.ownerId) ?? []
    ownerRuns.push(sandbox)
    runsByOwner.set(sandbox.ownerId, ownerRuns)
  }

  return Array.from(runsByOwner, ([ownerId, ownerRuns]) => {
    const sortedRuns = [...ownerRuns].sort(byAttempt)
    const latest = sortedRuns[sortedRuns.length - 1]
    const relatedCorrections = corrections
      .filter(correction => correction.failedExecutionId === latest.id)
      .sort((a, b) => a.startTime - b.startTime)
    const terminal = isOwnerTerminal(ownerId, agents, run)
    const correctionStarted = relatedCorrections.length > 0
    const retrying = latest.status === 'failed' && correctionStarted && !terminal
    const feedback: FeedbackState = latest.status !== 'failed' || terminal
      ? 'none'
      : correctionStarted
        ? 'withdrawing'
        : 'receiving'

    return {
      ownerId,
      retrying,
      feedback,
      feedbackAttempt: latest.attempt,
      failedExecutionId: latest.status === 'failed' ? latest.id : undefined,
      sandbox: {
        ...latest,
        id: `sandbox-${ownerId}`,
        executionId: latest.id,
        status: retrying ? 'offline' : latest.status,
      },
    }
  })
}

export function buildRetryCycles(
  ownerId: string,
  sandboxRuns: SandboxRun[],
  corrections: CorrectionRun[],
): RetryCycle[] {
  return sandboxRuns
    .filter(sandbox => sandbox.ownerId === ownerId && sandbox.status === 'failed')
    .sort(byAttempt)
    .map(attempt => ({
      attempt,
      corrections: corrections
        .filter(correction => correction.failedExecutionId === attempt.id)
        .sort((a, b) => a.startTime - b.startTime),
    }))
}

export function retryingOwnerIds(views: OwnerExecutionView[]): Set<string> {
  return new Set(views.filter(view => view.retrying).map(view => view.ownerId))
}
