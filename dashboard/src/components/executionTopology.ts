import type {
  AgentNode,
  CorrectionRun,
  RunState,
  SandboxRun,
} from '../types/events'
import {
  EDGE_DRAW_DURATION_MS,
  buildExecutionEdges,
  type ExecutionEdge,
  type ExecutionEdgePhase,
  type ExecutionEdgeTone,
} from './agentTopology'

export interface WorkflowTopology {
  renderEdges: ExecutionEdge[]
  layoutEdges: ExecutionEdge[]
}

function workflowEdge(
  source: string,
  target: string,
  phase: ExecutionEdgePhase,
  tone: ExecutionEdgeTone = 'orange',
): ExecutionEdge {
  return {
    id: `flow-${source}-${target}`,
    source,
    target,
    type: 'infoLine',
    data: { phase, tone, delayMs: 0, durationMs: EDGE_DRAW_DURATION_MS },
  }
}

function sortedSandboxes(sandboxRuns: SandboxRun[]): SandboxRun[] {
  return [...sandboxRuns].sort((a, b) => a.attempt - b.attempt || a.startTime - b.startTime)
}

export function buildWorkflowTopology(
  agents: AgentNode[],
  sandboxRuns: SandboxRun[],
  corrections: CorrectionRun[],
  run: RunState | null,
): WorkflowTopology {
  const renderEdges = buildExecutionEdges(agents)
    .filter(edge => edge.data.phase === 'spawn')
  const correctionByOwnerAttempt = new Map(
    corrections.map(correction => [
      `${correction.ownerId}:${correction.attempt}`,
      correction,
    ]),
  )
  const agentsWithChildren = new Set(
    renderEdges
      .filter(edge => edge.data.phase === 'spawn' && edge.source !== 'root')
      .map(edge => edge.source),
  )
  const agentById = new Map(agents.map(agent => [agent.id, agent]))
  const sandboxesByOwner = new Map<string, SandboxRun[]>()

  for (const sandbox of sandboxRuns) {
    const ownerRuns = sandboxesByOwner.get(sandbox.ownerId) ?? []
    ownerRuns.push(sandbox)
    sandboxesByOwner.set(sandbox.ownerId, ownerRuns)
  }

  for (const correction of corrections) {
    renderEdges.push(workflowEdge(
      correction.failedExecutionId,
      correction.id,
      'diagnostic',
      correction.strategy === 'textgrad' ? 'purple' : 'amber',
    ))
  }

  for (const [ownerId, ownerRuns] of sandboxesByOwner) {
    for (const sandbox of sortedSandboxes(ownerRuns)) {
      const correction = sandbox.attempt > 1
        ? correctionByOwnerAttempt.get(`${ownerId}:${sandbox.attempt - 1}`)
        : undefined
      const previousSandbox = ownerRuns.find(item => item.attempt === sandbox.attempt - 1)
      const source = correction?.id ?? previousSandbox?.id ?? ownerId
      const tone: ExecutionEdgeTone = correction
        ? correction.strategy === 'textgrad' ? 'purple' : 'amber'
        : 'orange'
      renderEdges.push(workflowEdge(
        source,
        sandbox.id,
        correction ? 'correction' : 'sandbox',
        tone,
      ))
    }
  }

  const layoutEdges = [...renderEdges]
  const outputCandidates: Array<{ ownerId: string; source: string; ready: boolean; failed: boolean }> = []

  for (const agent of agents) {
    if (agentsWithChildren.has(agent.id)) continue
    const ownerRuns = sortedSandboxes(sandboxesByOwner.get(agent.id) ?? [])
    const finalSandbox = ownerRuns[ownerRuns.length - 1]
    outputCandidates.push({
      ownerId: agent.id,
      source: finalSandbox?.id ?? agent.id,
      ready: agent.status === 'success' || agent.status === 'failed',
      failed: finalSandbox
        ? finalSandbox.status === 'failed'
        : agent.status === 'failed',
    })
  }

  const rootRuns = sortedSandboxes(sandboxesByOwner.get('root') ?? [])
  const finalRootSandbox = rootRuns[rootRuns.length - 1]
  if (agents.length === 0) {
    outputCandidates.push({
      ownerId: 'root',
      source: finalRootSandbox?.id ?? 'root',
      ready: run ? run.status !== 'running' : true,
      failed: finalRootSandbox
        ? finalRootSandbox.status === 'failed'
        : run?.status === 'failed',
    })
  }

  for (const candidate of outputCandidates) {
    const edge = workflowEdge(
      candidate.source,
      'output',
      candidate.failed ? 'failure' : 'result',
      candidate.failed ? 'red' : 'orange',
    )
    layoutEdges.push(edge)
    if (candidate.ready) renderEdges.push(edge)
  }

  // Ignore sandbox owners that do not correspond to a visible root or agent.
  return {
    renderEdges: renderEdges.filter(edge => (
      edge.source === 'root'
      || edge.target === 'output'
      || agentById.has(edge.source)
      || sandboxRuns.some(runItem => runItem.id === edge.source || runItem.id === edge.target)
      || corrections.some(item => item.id === edge.source || item.id === edge.target)
    )),
    layoutEdges,
  }
}
