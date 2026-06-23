import { describe, expect, it } from 'vitest'
import type { AgentNode, SandboxRun } from '../types/events'
import {
  buildSummaryStats,
  formatElapsedTime,
  getVramSeverity,
} from './summaryBarModel'

const agent = (
  id: string,
  status: AgentNode['status'],
): AgentNode => ({
  id,
  parentId: 'root',
  task: `Task ${id}`,
  status,
  spawnTime: 1000,
})

const sandbox = (id: string): SandboxRun => ({
  id,
  ownerId: 'agent-a',
  attempt: 1,
  trigger: 'initial',
  status: 'success',
  timeoutS: 30,
  codePreview: '',
  startTime: 1000,
})

describe('summary bar helpers', () => {
  it('formats elapsed time as minutes and seconds', () => {
    expect(formatElapsedTime(0)).toBe('0:00')
    expect(formatElapsedTime(59_999)).toBe('0:59')
    expect(formatElapsedTime(61_000)).toBe('1:01')
  })

  it('classifies VRAM severity with dashboard thresholds', () => {
    expect(getVramSeverity(0)).toBe('unknown')
    expect(getVramSeverity(6400)).toBe('normal')
    expect(getVramSeverity(6600)).toBe('warning')
    expect(getVramSeverity(7100)).toBe('critical')
  })

  it('derives task, counts, elapsed time, and VRAM state', () => {
    const stats = buildSummaryStats({
      run: {
        id: 'run-1',
        task: 'Analyze recursion',
        mode: 'delegated',
        status: 'running',
        startTime: 10_000,
      },
      nodes: [
        agent('agent-a', 'running'),
        agent('agent-b', 'success'),
        agent('agent-c', 'failed'),
      ],
      sandboxRuns: [sandbox('sandbox-a-1'), sandbox('sandbox-c-1')],
      eventCount: 9,
      rootTask: '',
      retryingOwnerIds: new Set(['agent-c']),
      now: 75_000,
      vramMb: 6600,
    })

    expect(stats).toMatchObject({
      taskName: 'Analyze recursion',
      elapsedMs: 65_000,
      totalNodes: 3,
      runningCount: 2,
      successCount: 1,
      failedCount: 1,
      sandboxCount: 2,
      eventCount: 9,
      vramSeverity: 'warning',
    })
  })

  it('uses the root task fallback and counts root retrying as active', () => {
    const stats = buildSummaryStats({
      run: null,
      nodes: [],
      sandboxRuns: [],
      eventCount: 1,
      rootTask: 'Direct root task',
      retryingOwnerIds: new Set(['root']),
      now: 10_000,
      vramMb: 0,
    })

    expect(stats.taskName).toBe('Direct root task')
    expect(stats.runningCount).toBe(1)
    expect(stats.elapsedMs).toBe(0)
    expect(stats.vramSeverity).toBe('unknown')
  })
})
