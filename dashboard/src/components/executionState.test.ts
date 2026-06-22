import { describe, expect, it } from 'vitest'
import type { AgentNode, CorrectionRun, SandboxRun } from '../types/events'
import { buildOwnerExecutionViews, buildRetryCycles } from './executionState'

const agent: AgentNode = {
  id: 'agent-a', parentId: 'root', task: 'A', status: 'running', spawnTime: 1,
}
const attempt = (
  id: string,
  number: number,
  status: SandboxRun['status'],
): SandboxRun => ({
  id, ownerId: 'agent-a', attempt: number, status,
  trigger: number === 1 ? 'initial' : 'textgrad', timeoutS: 30,
  codePreview: '', startTime: number,
})
const correction: CorrectionRun = {
  id: 'correction-a-2', ownerId: 'agent-a', failedExecutionId: 'attempt-1',
  attempt: 2, strategy: 'textgrad', status: 'running',
  phase: 'evaluating_loss', startTime: 3, mutations: [],
}

describe('owner execution projection', () => {
  it('projects every owner to one stable reusable sandbox node', () => {
    const views = buildOwnerExecutionViews(
      [agent], [attempt('attempt-1', 1, 'failed'), attempt('attempt-2', 2, 'running')],
      [correction], null,
    )
    expect(views).toHaveLength(1)
    expect(views[0].sandbox).toMatchObject({
      id: 'sandbox-agent-a', executionId: 'attempt-2', attempt: 2, status: 'running',
    })
  })

  it('shows failed feedback before correction inference starts', () => {
    const view = buildOwnerExecutionViews(
      [agent], [attempt('attempt-1', 1, 'failed')], [], null,
    )[0]
    expect(view).toMatchObject({ retrying: false, feedback: 'receiving' })
    expect(view.sandbox.status).toBe('failed')
  })

  it('makes the sandbox offline and owner retrying during correction', () => {
    const view = buildOwnerExecutionViews(
      [agent], [attempt('attempt-1', 1, 'failed')], [correction], null,
    )[0]
    expect(view).toMatchObject({ retrying: true, feedback: 'withdrawing' })
    expect(view.sandbox.status).toBe('offline')
  })

  it('preserves every failed attempt in retry history after later success', () => {
    const runs = [
      attempt('attempt-1', 1, 'failed'),
      attempt('attempt-2', 2, 'failed'),
      attempt('attempt-3', 3, 'success'),
    ]
    const fallback = { ...correction, id: 'retry-a-2', strategy: 'llm_retry' as const }
    const cycles = buildRetryCycles('agent-a', runs, [correction, fallback])
    expect(cycles.map(cycle => cycle.attempt.id)).toEqual(['attempt-1', 'attempt-2'])
    expect(cycles[0].corrections.map(item => item.strategy))
      .toEqual(['textgrad', 'llm_retry'])
  })

  it('applies the reusable retry lifecycle to direct root execution', () => {
    const rootAttempt = { ...attempt('root-attempt-1', 1, 'failed'), ownerId: 'root' }
    const rootCorrection = {
      ...correction,
      ownerId: 'root',
      failedExecutionId: rootAttempt.id,
    }
    const run = {
      id: 'run-1', task: 'Direct', mode: 'direct' as const,
      status: 'running' as const, startTime: 1,
    }
    const view = buildOwnerExecutionViews([], [rootAttempt], [rootCorrection], run)[0]
    expect(view).toMatchObject({
      ownerId: 'root', retrying: true, feedback: 'withdrawing',
    })
    expect(view.sandbox).toMatchObject({ id: 'sandbox-root', status: 'offline' })
  })
})
