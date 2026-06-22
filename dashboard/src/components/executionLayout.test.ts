import { describe, expect, it } from 'vitest'
import type { Node } from '@xyflow/react'
import type { AgentNode, SandboxRun } from '../types/events'
import { buildWorkflowTopology } from './executionTopology'
import {
  layoutExecutionGraph,
  OUTPUT_POSITION,
  ROOT_POSITION,
} from './executionLayout'

const agents: AgentNode[] = [
  { id: 'agent-a', parentId: 'root', task: 'A', status: 'running', spawnTime: 1 },
  { id: 'agent-b', parentId: 'root', task: 'B', status: 'running', spawnTime: 2 },
]
const sandboxes: SandboxRun[] = [
  {
    id: 'sandbox-a-1', ownerId: 'agent-a', attempt: 1, trigger: 'initial',
    status: 'running', timeoutS: 30, codePreview: '', startTime: 3,
  },
  {
    id: 'sandbox-b-1', ownerId: 'agent-b', attempt: 1, trigger: 'initial',
    status: 'running', timeoutS: 30, codePreview: '', startTime: 4,
  },
]

function graphNodes(extra: Node[] = []): Node[] {
  return [
    { id: 'root', type: 'root', data: {}, position: { x: 0, y: 0 } },
    { id: 'output', type: 'output', data: {}, position: { x: 0, y: 0 } },
    ...extra,
  ]
}

describe('anchored execution layout', () => {
  it('keeps root and output at stable anchors', () => {
    const topology = buildWorkflowTopology(agents, sandboxes, [], null)
    const nodes = graphNodes([
      ...agents.map(agent => ({
        id: agent.id, type: 'agent', data: {}, position: { x: 0, y: 0 },
      })),
      ...topology.sandboxNodes.map(sandbox => ({
        id: sandbox.id, type: 'sandbox', data: {}, position: { x: 0, y: 0 },
      })),
    ])
    const layout = layoutExecutionGraph(
      nodes, topology.layoutEdges, agents, topology.sandboxNodes,
    )
    const siblingPositions = agents.map(agent => (
      layout.find(node => node.id === agent.id)!.position.x
    ))
    expect(layout.find(node => node.id === 'root')?.position).toEqual(ROOT_POSITION)
    expect(layout.find(node => node.id === 'output')?.position).toEqual(OUTPUT_POSITION)
    expect(new Set(siblingPositions).size).toBe(agents.length)
  })

  it('aligns each delegated sandbox vertically beneath its owner', () => {
    const topology = buildWorkflowTopology(agents, sandboxes, [], null)
    const nodes = graphNodes([
      ...agents.map(agent => ({
        id: agent.id, type: 'agent', data: {}, position: { x: 0, y: 0 },
      })),
      ...topology.sandboxNodes.map(sandbox => ({
        id: sandbox.id, type: 'sandbox', data: {}, position: { x: 0, y: 0 },
      })),
    ])
    const layout = layoutExecutionGraph(
      nodes, topology.layoutEdges, agents, topology.sandboxNodes,
    )
    const owner = layout.find(node => node.id === 'agent-a')!
    const sandbox = layout.find(node => node.id === 'sandbox-agent-a')!
    expect(sandbox.position.x + 85).toBe(owner.position.x + 90)
    expect(sandbox.position.y).toBeGreaterThan(owner.position.y)
  })

  it('places direct-root attempts in the reserved root-output corridor', () => {
    const directSandboxes = sandboxes.map((sandbox, index) => ({
      ...sandbox,
      id: `root-${index + 1}`,
      ownerId: 'root',
      attempt: index + 1,
    }))
    const topology = buildWorkflowTopology([], directSandboxes, [], null)
    const nodes = graphNodes(topology.sandboxNodes.map(sandbox => ({
      id: sandbox.id, type: 'sandbox', data: {}, position: { x: 0, y: 0 },
    })))
    const layout = layoutExecutionGraph(
      nodes, topology.layoutEdges, [], topology.sandboxNodes,
    )
    const attempts = layout.filter(node => node.type === 'sandbox')
    expect(attempts).toHaveLength(1)
    expect(attempts.every(node => (
      node.position.x > ROOT_POSITION.x && node.position.x < OUTPUT_POSITION.x
    ))).toBe(true)
    expect(topology.sandboxNodes[0]).toMatchObject({
      id: 'sandbox-root', executionId: 'root-2', attempt: 2,
    })
  })
})
