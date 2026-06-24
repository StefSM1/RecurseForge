import { describe, expect, it } from 'vitest'
import type { AgentNode, EngineEvent } from '../types/events'
import {
  buildChatTabs,
  hasRunCompletedEvent,
  isTerminalChatRunStatus,
  shouldSubmitChatInput,
} from './chatModel'

const agent = (
  id: string,
  status: AgentNode['status'],
  spawnTime: number,
): AgentNode => ({
  id,
  parentId: 'root',
  task: `Task ${id}`,
  status,
  spawnTime,
})

describe('chat model helpers', () => {
  it('submits only bare Enter', () => {
    expect(shouldSubmitChatInput({ key: 'Enter', shiftKey: false })).toBe(true)
    expect(shouldSubmitChatInput({ key: 'Enter', shiftKey: true })).toBe(false)
    expect(shouldSubmitChatInput({ key: 'a', shiftKey: false })).toBe(false)
    expect(shouldSubmitChatInput({
      key: 'Enter',
      shiftKey: false,
      isComposing: true,
    })).toBe(false)
  })

  it('keeps Main first and shows only active sub-agent tabs newest first', () => {
    const tabs = buildChatTabs([
      agent('done', 'success', 1),
      agent('older', 'running', 2),
      agent('newer', 'retrying', 3),
      agent('failed', 'failed', 4),
    ])

    expect(tabs).toEqual([
      { id: 'main', label: 'Main', kind: 'main' },
      { id: 'sub-newer', label: 'Sub', kind: 'sub-agent', agentId: 'newer' },
      { id: 'sub-older', label: 'Sub', kind: 'sub-agent', agentId: 'older' },
    ])
  })

  it('recognizes terminal chat run statuses', () => {
    expect(isTerminalChatRunStatus('success')).toBe(true)
    expect(isTerminalChatRunStatus('failed')).toBe(true)
    expect(isTerminalChatRunStatus('stopped')).toBe(true)
    expect(isTerminalChatRunStatus('running')).toBe(false)
  })

  it('detects run completion events by run id', () => {
    const events: EngineEvent[] = [
      {
        event_type: 'run_completed',
        run_id: 'other',
        timestamp: 1,
        payload: { success: true, mode: 'direct', result_summary: 'no' },
      },
      {
        event_type: 'run_completed',
        run_id: 'target',
        timestamp: 2,
        payload: { success: true, mode: 'direct', result_summary: 'yes' },
      },
    ]

    expect(hasRunCompletedEvent(events, 'target')).toBe(true)
    expect(hasRunCompletedEvent(events, 'missing')).toBe(false)
  })
})
