import type { AgentNode } from '../types/events'
import type { FeedbackState } from './executionState'

export interface ExecutionEdge {
  id: string
  source: string
  target: string
  sourceHandle?: string
  targetHandle?: string
  type: 'infoLine'
  data: ExecutionEdgeData
}

export type ExecutionEdgePhase =
  | 'direct'
  | 'spawn'
  | 'result'
  | 'sandbox'
  | 'feedback'
  | 'failure'

export type ExecutionEdgeTone = 'orange' | 'purple' | 'amber' | 'red'
export type ExecutionEdgeRoute =
  | 'branch'
  | 'vertical'
  | 'horizontal'
  | 'feedback'
  | 'feedback-horizontal'
  | 'gutter'

export interface ExecutionEdgeData extends Record<string, unknown> {
  phase: ExecutionEdgePhase
  delayMs: number
  durationMs: number
  tone?: ExecutionEdgeTone
  route: ExecutionEdgeRoute
  feedbackOffset?: number
  feedbackState?: FeedbackState
}

const ROOT_ID = 'root'
const OUTPUT_ID = 'output'
export const EDGE_DRAW_DURATION_MS = 900
export const EDGE_STAGGER_MS = 80
export const NODE_REVEAL_LEAD_MS = 40

export class AnimationRegistry {
  private readonly drawnEdgeIds = new Set<string>()
  private readonly revealedNodeIds = new Set<string>()

  shouldAnimateEdge(edgeId: string): boolean {
    return !this.drawnEdgeIds.has(edgeId)
  }

  shouldRevealNode(nodeId: string): boolean {
    return !this.revealedNodeIds.has(nodeId)
  }

  markEdgeDrawn(edgeId: string): void {
    this.drawnEdgeIds.add(edgeId)
  }

  markNodeRevealed(nodeId: string): void {
    this.revealedNodeIds.add(nodeId)
  }
}

function edge(
  source: string,
  target: string,
  phase: ExecutionEdgePhase,
  delayMs = 0,
): ExecutionEdge {
  return {
    id: `flow-${source}-${target}`,
    source,
    target,
    type: 'infoLine',
    sourceHandle: source === ROOT_ID ? 'spawn-out' : 'forward-out',
    targetHandle: 'spawn-in',
    data: {
      phase,
      route: phase === 'spawn' ? 'branch' : 'gutter',
      delayMs,
      durationMs: EDGE_DRAW_DURATION_MS,
    },
  }
}

function hasValidParentChain(
  nodeId: string,
  parentId: string,
  agentsById: Map<string, AgentNode>,
): boolean {
  if (parentId === ROOT_ID) return true
  if (!agentsById.has(parentId) || parentId === nodeId) return false

  const visited = new Set([nodeId])
  let currentId = parentId

  while (currentId !== ROOT_ID) {
    if (visited.has(currentId)) return false
    visited.add(currentId)

    const current = agentsById.get(currentId)
    if (!current) return false
    currentId = current.parentId
  }

  return true
}

function buildEdges(agents: AgentNode[], includePendingResults: boolean): ExecutionEdge[] {
  if (agents.length === 0) return [edge(ROOT_ID, OUTPUT_ID, 'direct')]

  const agentsById = new Map(agents.map(agent => [agent.id, agent]))
  const resolvedParents = new Map<string, string>()

  for (const agent of agents) {
    const parentId = hasValidParentChain(agent.id, agent.parentId, agentsById)
      ? agent.parentId
      : ROOT_ID
    resolvedParents.set(agent.id, parentId)
  }

  const siblingIndexes = new Map<string, number>()
  const edges = agents.map(agent => {
    const parentId = resolvedParents.get(agent.id)!
    const siblingIndex = siblingIndexes.get(parentId) ?? 0
    siblingIndexes.set(parentId, siblingIndex + 1)
    return edge(parentId, agent.id, 'spawn', siblingIndex * EDGE_STAGGER_MS)
  })
  const agentsWithChildren = new Set(
    Array.from(resolvedParents.values()).filter(parentId => parentId !== ROOT_ID),
  )

  let resultIndex = 0
  for (const agent of agents) {
    if (
      !agentsWithChildren.has(agent.id)
      && (includePendingResults || agent.status === 'success')
    ) {
      edges.push(edge(
        agent.id,
        OUTPUT_ID,
        'result',
        resultIndex * EDGE_STAGGER_MS,
      ))
      resultIndex += 1
    }
  }

  return edges
}

/** Build the forward execution flow shown by the orange dashboard edges. */
export function buildExecutionEdges(agents: AgentNode[]): ExecutionEdge[] {
  return buildEdges(agents, false)
}

/** Keep output positioned beneath leaves before their result wires are visible. */
export function buildLayoutEdges(agents: AgentNode[]): ExecutionEdge[] {
  return buildEdges(agents, true)
}
