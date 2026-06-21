import { describe, expect, it } from 'vitest'
import type { AgentNode, AgentStatus } from '../types/events'
import {
  EDGE_DRAW_DURATION_MS,
  EDGE_STAGGER_MS,
  buildExecutionEdges,
  buildLayoutEdges,
  isAnimationPending,
} from './agentTopology'

function agent(
  id: string,
  parentId: string,
  status: AgentStatus = 'running',
): AgentNode {
  return {
    id,
    parentId,
    task: `Task ${id}`,
    status,
    spawnTime: 1,
  }
}

function connections(agents: AgentNode[]): string[] {
  return buildExecutionEdges(agents).map(edge => `${edge.source}->${edge.target}`)
}

describe('buildExecutionEdges', () => {
  it('connects an idle root directly to output', () => {
    expect(connections([])).toEqual(['root->output'])
  })

  it('shows only the incoming wire while an agent is running', () => {
    expect(connections([agent('a', 'root')])).toEqual([
      'root->a',
    ])
  })

  it('connects successful sibling leaves to output', () => {
    expect(connections([
      agent('a', 'root', 'success'),
      agent('b', 'root', 'success'),
    ])).toEqual([
      'root->a',
      'root->b',
      'a->output',
      'b->output',
    ])
  })

  it('only connects the terminal agent in a nested chain to output', () => {
    expect(connections([
      agent('a', 'root', 'success'),
      agent('b', 'a', 'success'),
    ])).toEqual([
      'root->a',
      'a->b',
      'b->output',
    ])
  })

  it.each([
    ['missing', ''],
    ['unknown', 'does-not-exist'],
    ['self', 'a'],
  ])('falls back to root for a %s parent', (_case, parentId) => {
    expect(connections([agent('a', parentId, 'success')])).toEqual([
      'root->a',
      'a->output',
    ])
  })

  it('breaks cyclic parent chains by falling both nodes back to root', () => {
    expect(connections([
      agent('a', 'b', 'success'),
      agent('b', 'a', 'success'),
    ])).toEqual([
      'root->a',
      'root->b',
      'a->output',
      'b->output',
    ])
  })

  it.each<AgentStatus>(['failed', 'retrying', 'gradient'])(
    'does not send a %s leaf to output',
    status => {
      expect(connections([agent('a', 'root', status)])).toEqual(['root->a'])
    },
  )

  it('keeps the incoming edge ID stable across status changes', () => {
    const running = buildExecutionEdges([agent('a', 'root', 'running')])
    const failed = buildExecutionEdges([agent('a', 'root', 'failed')])
    const retrying = buildExecutionEdges([agent('a', 'root', 'retrying')])

    expect(failed[0]).toEqual(running[0])
    expect(retrying[0]).toEqual(running[0])
  })

  it('adds phase and stagger metadata without changing stable IDs', () => {
    const edges = buildExecutionEdges([
      agent('a', 'root', 'success'),
      agent('b', 'root', 'success'),
    ])

    expect(edges.map(edge => ({ id: edge.id, data: edge.data }))).toEqual([
      {
        id: 'flow-root-a',
        data: { phase: 'spawn', delayMs: 0, durationMs: EDGE_DRAW_DURATION_MS },
      },
      {
        id: 'flow-root-b',
        data: {
          phase: 'spawn',
          delayMs: EDGE_STAGGER_MS,
          durationMs: EDGE_DRAW_DURATION_MS,
        },
      },
      {
        id: 'flow-a-output',
        data: { phase: 'result', delayMs: 0, durationMs: EDGE_DRAW_DURATION_MS },
      },
      {
        id: 'flow-b-output',
        data: {
          phase: 'result',
          delayMs: EDGE_STAGGER_MS,
          durationMs: EDGE_DRAW_DURATION_MS,
        },
      },
    ])
  })

  it('guards completed animation IDs from replaying', () => {
    const completed = new Set<string>()
    expect(isAnimationPending(completed, 'flow-root-a')).toBe(true)
    completed.add('flow-root-a')
    expect(isAnimationPending(completed, 'flow-root-a')).toBe(false)
  })

  it('keeps pending result connections available to the layout only', () => {
    const agents = [agent('a', 'root'), agent('b', 'root')]
    expect(connections(agents)).toEqual(['root->a', 'root->b'])
    expect(buildLayoutEdges(agents).map(edge => `${edge.source}->${edge.target}`))
      .toEqual(['root->a', 'root->b', 'a->output', 'b->output'])
  })
})
