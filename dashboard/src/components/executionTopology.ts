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
  type ExecutionEdgeRoute,
  type ExecutionEdgeTone,
} from './agentTopology'
import {
  buildOwnerExecutionViews,
  type OwnerExecutionView,
  type SandboxNodeView,
} from './executionState'

export interface WorkflowTopology {
  renderEdges: ExecutionEdge[]
  layoutEdges: ExecutionEdge[]
  sandboxNodes: SandboxNodeView[]
  ownerViews: OwnerExecutionView[]
}

interface WorkflowEdgeOptions {
  tone?: ExecutionEdgeTone
  route: ExecutionEdgeRoute
  sourceHandle: string
  targetHandle: string
  feedbackOffset?: number
  feedbackState?: ExecutionEdge['data']['feedbackState']
}

function workflowEdge(
  id: string,
  source: string,
  target: string,
  phase: ExecutionEdgePhase,
  options: WorkflowEdgeOptions,
): ExecutionEdge {
  return {
    id,
    source,
    target,
    sourceHandle: options.sourceHandle,
    targetHandle: options.targetHandle,
    type: 'infoLine',
    data: {
      phase,
      route: options.route,
      tone: options.tone ?? 'orange',
      delayMs: 0,
      durationMs: EDGE_DRAW_DURATION_MS,
      feedbackOffset: options.feedbackOffset,
      feedbackState: options.feedbackState,
    },
  }
}

function isTerminal(status: AgentNode['status']): boolean {
  return status === 'success' || status === 'failed'
}

export function buildWorkflowTopology(
  agents: AgentNode[],
  sandboxRuns: SandboxRun[],
  corrections: CorrectionRun[],
  run: RunState | null,
): WorkflowTopology {
  const ownerViews = buildOwnerExecutionViews(agents, sandboxRuns, corrections, run)
  const sandboxNodes = ownerViews.map(view => view.sandbox)
  const ownerViewById = new Map(ownerViews.map(view => [view.ownerId, view]))
  const spawnEdges = buildExecutionEdges(agents)
    .filter(edge => edge.data.phase === 'spawn')
  const renderEdges = [...spawnEdges]
  const agentsWithChildren = new Set(
    spawnEdges
      .filter(edge => edge.source !== 'root')
      .map(edge => edge.source),
  )
  for (const view of ownerViews) {
    const directRoot = view.ownerId === 'root' && agents.length === 0
    renderEdges.push(workflowEdge(
      `sandbox-input-${view.ownerId}`,
      view.ownerId,
      view.sandbox.id,
      'sandbox',
      {
        route: directRoot ? 'horizontal' : 'vertical',
        sourceHandle: directRoot ? 'direct-out' : 'forward-out',
        targetHandle: directRoot ? 'direct-in' : 'forward-in',
      },
    ))
    if (view.feedback !== 'none') {
      renderEdges.push(workflowEdge(
        `feedback-${view.ownerId}-attempt-${view.feedbackAttempt}`,
        view.sandbox.id,
        view.ownerId,
        'feedback',
        {
          tone: 'purple',
          route: directRoot ? 'feedback-horizontal' : 'feedback',
          sourceHandle: directRoot ? 'feedback-top' : 'feedback-out',
          targetHandle: directRoot ? 'feedback-top' : 'feedback-in',
          feedbackOffset: 34,
          feedbackState: view.feedback,
        },
      ))
    }
  }

  const layoutEdges = [...renderEdges]
  const outputCandidates: Array<{
    ownerId: string
    source: string
    ready: boolean
    failed: boolean
    directRoot: boolean
  }> = []

  for (const agent of agents) {
    if (agentsWithChildren.has(agent.id)) continue
    const ownerView = ownerViewById.get(agent.id)
    const finalSandbox = ownerView?.sandbox
    outputCandidates.push({
      ownerId: agent.id,
      source: finalSandbox?.id ?? agent.id,
      ready: isTerminal(agent.status),
      failed: finalSandbox?.status === 'failed' || agent.status === 'failed',
      directRoot: false,
    })
  }

  if (agents.length === 0) {
    const finalRootSandbox = ownerViewById.get('root')?.sandbox
    outputCandidates.push({
      ownerId: 'root',
      source: finalRootSandbox?.id ?? 'root',
      ready: run ? run.status !== 'running' : true,
      failed: finalRootSandbox?.status === 'failed' || run?.status === 'failed',
      directRoot: true,
    })
  }

  for (const candidate of outputCandidates) {
    const sourceIsSandbox = sandboxNodes.some(sandbox => sandbox.id === candidate.source)
    const edge = workflowEdge(
      `terminal-${candidate.ownerId}-${candidate.source}-output`,
      candidate.source,
      'output',
      candidate.failed ? 'failure' : candidate.source === 'root' ? 'direct' : 'result',
      {
        tone: candidate.failed ? 'red' : 'orange',
        route: candidate.directRoot ? 'horizontal' : 'gutter',
        sourceHandle: candidate.directRoot
          ? sourceIsSandbox ? 'direct-out' : 'direct-out'
          : sourceIsSandbox ? 'result-out' : 'forward-out',
        targetHandle: 'result-in',
      },
    )
    layoutEdges.push(edge)
    if (candidate.ready) renderEdges.push(edge)
  }

  return { renderEdges, layoutEdges, sandboxNodes, ownerViews }
}
