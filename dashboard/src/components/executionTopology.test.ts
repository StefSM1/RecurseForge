import { describe, expect, it } from 'vitest'
import type { AgentNode, CorrectionRun, RunState, SandboxRun } from '../types/events'
import { buildWorkflowTopology } from './executionTopology'

const agent = (status: AgentNode['status'] = 'running'): AgentNode => ({
  id: 'agent-a', parentId: 'root', task: 'A', status, spawnTime: 1,
})
const sandbox = (
  id: string,
  attempt: number,
  status: SandboxRun['status'],
  ownerId = 'agent-a',
): SandboxRun => ({
  id, ownerId, attempt, status, trigger: attempt === 1 ? 'initial' : 'textgrad',
  timeoutS: 30, codePreview: '', startTime: attempt,
})
const correction: CorrectionRun = {
  id: 'correction-agent-a-1', ownerId: 'agent-a', failedExecutionId: 'sandbox-a-1',
  attempt: 1, strategy: 'textgrad', status: 'success', phase: 'completed',
  startTime: 2, mutations: [],
}

function links(
  agents: AgentNode[],
  sandboxes: SandboxRun[],
  corrections: CorrectionRun[],
  run: RunState | null = null,
) {
  return buildWorkflowTopology(agents, sandboxes, corrections, run)
    .renderEdges.map(edge => `${edge.source}->${edge.target}`)
}

describe('execution workflow topology', () => {
  it('shows a running sandbox without prematurely connecting output', () => {
    expect(links([agent()], [sandbox('sandbox-a-1', 1, 'running')], []))
      .toEqual(['root->agent-a', 'agent-a->sandbox-a-1'])
  })

  it('preserves failure, correction, retry, and successful output order', () => {
    expect(links(
      [agent('success')],
      [sandbox('sandbox-a-1', 1, 'failed'), sandbox('sandbox-a-2', 2, 'success')],
      [correction],
    )).toEqual([
      'root->agent-a',
      'sandbox-a-1->correction-agent-a-1',
      'agent-a->sandbox-a-1',
      'correction-agent-a-1->sandbox-a-2',
      'sandbox-a-2->output',
    ])
  })

  it('routes a text-only completed agent directly to output', () => {
    expect(links([agent('success')], [], [])).toEqual([
      'root->agent-a', 'agent-a->output',
    ])
  })

  it('attaches direct root sandbox attempts to root', () => {
    const run: RunState = {
      id: 'run-1', task: 'Direct', mode: 'direct', status: 'success', startTime: 1,
    }
    expect(links([], [sandbox('sandbox-root-1', 1, 'success', 'root')], [], run))
      .toEqual(['root->sandbox-root-1', 'sandbox-root-1->output'])
  })

  it('keeps output constrained below a running lane without rendering its edge', () => {
    const topology = buildWorkflowTopology(
      [agent()], [sandbox('sandbox-a-1', 1, 'running')], [], null,
    )
    expect(topology.renderEdges.some(edge => edge.target === 'output')).toBe(false)
    expect(topology.layoutEdges.some(edge => edge.source === 'sandbox-a-1' && edge.target === 'output'))
      .toBe(true)
  })

  it('uses amber retry flow and red terminal failure routing', () => {
    const retryCorrection: CorrectionRun = {
      ...correction,
      id: 'retry-agent-a-1',
      strategy: 'llm_retry',
    }
    const topology = buildWorkflowTopology(
      [agent('failed')],
      [
        sandbox('sandbox-a-1', 1, 'failed'),
        { ...sandbox('sandbox-a-2', 2, 'failed'), trigger: 'llm_retry' },
      ],
      [retryCorrection],
      null,
    )

    expect(topology.renderEdges.find(edge => edge.source === 'sandbox-a-1')?.data.tone)
      .toBe('amber')
    expect(topology.renderEdges.find(edge => edge.source === 'retry-agent-a-1')?.data.tone)
      .toBe('amber')
    expect(topology.renderEdges.find(edge => edge.target === 'output')?.data.tone)
      .toBe('red')
  })
})
