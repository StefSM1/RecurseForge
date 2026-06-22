import dagre from 'dagre'
import type { Node } from '@xyflow/react'
import type { AgentNode } from '../types/events'
import type { ExecutionEdge } from './agentTopology'
import type { SandboxNodeView } from './executionState'

export const ROOT_POSITION = { x: 0, y: 0 }
export const OUTPUT_POSITION = { x: 900, y: 0 }

const NODE_SIZES: Record<string, { width: number; height: number }> = {
  root: { width: 200, height: 80 },
  output: { width: 200, height: 80 },
  agent: { width: 180, height: 80 },
  sandbox: { width: 170, height: 70 },
}

const AGENT_RANK_SEPARATION = 120
const AGENT_NODE_SEPARATION = 90
const SANDBOX_LANE_GAP = 52
const DIRECT_SANDBOX_X = 360

export function nodeDimensions(node: Node): { width: number; height: number } {
  return NODE_SIZES[node.type ?? 'agent'] ?? NODE_SIZES.agent
}

function layoutAgentTree(
  agents: AgentNode[],
  spawnEdges: ExecutionEdge[],
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>()
  if (agents.length === 0) return positions

  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  graph.setGraph({
    rankdir: 'TB',
    nodesep: AGENT_NODE_SEPARATION,
    ranksep: AGENT_RANK_SEPARATION,
    marginx: 0,
    marginy: 0,
  })
  // Dagre mutates node labels with x/y coordinates, so every node needs its
  // own dimensions object rather than a shared reference.
  graph.setNode('root', { ...NODE_SIZES.root })
  for (const agent of agents) graph.setNode(agent.id, { ...NODE_SIZES.agent })
  for (const edge of spawnEdges) graph.setEdge(edge.source, edge.target)
  dagre.layout(graph)

  const rootCenter = graph.node('root')
  const offsetX = ROOT_POSITION.x + NODE_SIZES.root.width / 2 - rootCenter.x
  const offsetY = ROOT_POSITION.y + NODE_SIZES.root.height / 2 - rootCenter.y

  for (const agent of agents) {
    const center = graph.node(agent.id)
    positions.set(agent.id, {
      x: center.x + offsetX - NODE_SIZES.agent.width / 2,
      y: center.y + offsetY - NODE_SIZES.agent.height / 2,
    })
  }
  return positions
}

export function layoutExecutionGraph(
  nodes: Node[],
  edges: ExecutionEdge[],
  agents: AgentNode[],
  sandboxNodes: SandboxNodeView[],
): Node[] {
  const spawnEdges = edges.filter(edge => edge.data.phase === 'spawn')
  const agentPositions = layoutAgentTree(agents, spawnEdges)
  const sandboxPositions = new Map<string, { x: number; y: number }>()
  for (const sandbox of sandboxNodes) {
    const ownerId = sandbox.ownerId
    if (ownerId === 'root' && agents.length === 0) {
      sandboxPositions.set(sandbox.id, {
        x: DIRECT_SANDBOX_X,
        y: ROOT_POSITION.y + 5,
      })
      continue
    }

    const ownerPosition = ownerId === 'root'
      ? ROOT_POSITION
      : agentPositions.get(ownerId)
    if (!ownerPosition) continue
    const ownerSize = ownerId === 'root' ? NODE_SIZES.root : NODE_SIZES.agent
    sandboxPositions.set(sandbox.id, {
      x: ownerPosition.x + (ownerSize.width - NODE_SIZES.sandbox.width) / 2,
      y: ownerPosition.y + ownerSize.height + SANDBOX_LANE_GAP,
    })
  }

  return nodes.map(node => {
    let position = node.position
    if (node.id === 'root') position = ROOT_POSITION
    else if (node.id === 'output') position = OUTPUT_POSITION
    else if (node.type === 'agent') position = agentPositions.get(node.id) ?? position
    else if (node.type === 'sandbox') position = sandboxPositions.get(node.id) ?? position
    return { ...node, position }
  })
}
