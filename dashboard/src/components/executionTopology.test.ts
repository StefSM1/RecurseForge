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
      .toEqual(['root->agent-a', 'agent-a->sandbox-agent-a'])
  })

  it('preserves failure, correction, retry, and successful output order', () => {
    expect(links(
      [agent('success')],
      [sandbox('sandbox-a-1', 1, 'failed'), sandbox('sandbox-a-2', 2, 'success')],
      [correction],
    )).toEqual([
      'root->agent-a',
      'agent-a->sandbox-agent-a',
      'sandbox-agent-a->output',
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
      .toEqual(['root->sandbox-root', 'sandbox-root->output'])
  })

  it('keeps output constrained below a running lane without rendering its edge', () => {
    const topology = buildWorkflowTopology(
      [agent()], [sandbox('sandbox-a-1', 1, 'running')], [], null,
    )
    expect(topology.renderEdges.some(edge => edge.target === 'output')).toBe(false)
    expect(topology.layoutEdges.some(edge => edge.source === 'sandbox-agent-a' && edge.target === 'output'))
      .toBe(true)
  })

  it('uses purple owner feedback and red terminal failure routing', () => {
    const retryCorrection: CorrectionRun = {
      ...correction,
      id: 'retry-agent-a-1',
      strategy: 'llm_retry',
    }
    const feedbackTopology = buildWorkflowTopology(
      [agent('running')],
      [sandbox('sandbox-a-1', 1, 'failed')],
      [retryCorrection],
      null,
    )
    const feedback = feedbackTopology.renderEdges.find(edge => edge.data.phase === 'feedback')
    expect(feedback).toMatchObject({
      source: 'sandbox-agent-a', target: 'agent-a',
      sourceHandle: 'feedback-out', targetHandle: 'feedback-in',
    })
    expect(feedback?.data).toMatchObject({
      tone: 'purple', feedbackState: 'withdrawing',
    })

    const topology = buildWorkflowTopology(
      [agent('failed')],
      [
        sandbox('sandbox-a-1', 1, 'failed'),
        { ...sandbox('sandbox-a-2', 2, 'failed'), trigger: 'llm_retry' },
      ],
      [retryCorrection],
      null,
    )

    expect(topology.renderEdges.find(edge => edge.target === 'output')?.data.tone)
      .toBe('red')
  })

  it('matches corrections by failed execution id rather than attempt arithmetic', () => {
    const backendNumberedCorrection = { ...correction, attempt: 2 }
    const topology = buildWorkflowTopology(
      [agent()],
      [sandbox('sandbox-a-1', 1, 'failed')],
      [backendNumberedCorrection],
      null,
    )
    expect(topology.renderEdges.find(edge => edge.data.phase === 'feedback'))
      .toMatchObject({ source: 'sandbox-agent-a', target: 'agent-a' })
  })

  it('does not render correction records as nodes or edge endpoints', () => {
    const topology = buildWorkflowTopology(
      [agent()], [sandbox('sandbox-a-1', 1, 'failed')], [correction], null,
    )
    expect(topology.renderEdges.some(edge => (
      edge.source === correction.id || edge.target === correction.id
    ))).toBe(false)
  })
})
