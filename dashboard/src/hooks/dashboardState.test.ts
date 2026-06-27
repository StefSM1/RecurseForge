import { describe, expect, it } from 'vitest'
import type { EngineEvent, EventPayload } from '../types/events'
import { createDashboardDataState, reduceEngineEvent } from './dashboardState'

function event(
  event_type: EngineEvent['event_type'],
  payload: EventPayload,
  timestamp: number,
): EngineEvent {
  return { event_id: `${event_type}-${timestamp}`, run_id: 'run-1', event_type, payload, timestamp }
}

describe('dashboard execution reducer', () => {
  it('tracks a sandbox failure, TextGrad correction, and successful retry', () => {
    let state = createDashboardDataState()
    const events: EngineEvent[] = [
      event('run_started', { task: 'Fix code', mode: 'direct' }, 1),
      event('sandbox_started', {
        execution_id: 'sandbox-root-1', owner_node_id: 'root', attempt: 1,
        trigger: 'initial', timeout_s: 30, code_preview: 'bad()',
      }, 2),
      event('sandbox_completed', {
        execution_id: 'sandbox-root-1', owner_node_id: 'root', attempt: 1,
        status: 'failed', exit_code: 1, duration_ms: 20, stderr_preview: 'boom',
      }, 3),
      event('correction_started', {
        correction_id: 'correction-root-1', owner_node_id: 'root',
        failed_execution_id: 'sandbox-root-1', attempt: 1, strategy: 'textgrad',
      }, 4),
      event('correction_progress', {
        correction_id: 'correction-root-1', phase: 'gradient_ready',
        iteration: 1, severity: 0.5, num_mutations: 2,
      }, 5),
      event('correction_completed', {
        correction_id: 'correction-root-1', success: true, iterations: 1,
      }, 6),
      event('sandbox_started', {
        execution_id: 'sandbox-root-2', owner_node_id: 'root', attempt: 2,
        trigger: 'textgrad', timeout_s: 30, code_preview: 'fixed()',
      }, 7),
      event('sandbox_completed', {
        execution_id: 'sandbox-root-2', owner_node_id: 'root', attempt: 2,
        status: 'success', exit_code: 0, duration_ms: 15, stdout_preview: 'ok',
      }, 8),
      event('run_completed', { success: true, mode: 'direct', result_summary: 'ok' }, 9),
    ]

    for (const item of events) state = reduceEngineEvent(state, item)

    expect(state.sandboxRuns.get('sandbox-root-1')?.status).toBe('failed')
    expect(state.corrections.get('correction-root-1')).toMatchObject({
      status: 'success', phase: 'completed', severity: 0.5, numMutations: 2,
    })
    expect(state.sandboxRuns.get('sandbox-root-2')?.status).toBe('success')
    expect(state.run?.status).toBe('success')
  })

  it('deduplicates replayed events', () => {
    const spawn = event('node_spawn', {
      node_id: 'agent-a', parent_id: 'root', task: 'A',
    }, 1)
    const once = reduceEngineEvent(createDashboardDataState(), spawn)
    const replayed = reduceEngineEvent(once, spawn)
    expect(replayed).toBe(once)
    expect(replayed.events).toHaveLength(1)
  })

  it('isolates graph maps when a new run starts', () => {
    let state = reduceEngineEvent(createDashboardDataState(), event(
      'run_started', { task: 'First', mode: 'delegated' }, 1,
    ))
    state = reduceEngineEvent(state, event('node_spawn', {
      node_id: 'agent-a', parent_id: 'root', task: 'A',
    }, 2))
    const secondRun: EngineEvent = {
      event_id: 'second-run',
      run_id: 'run-2',
      event_type: 'run_started',
      payload: { task: 'Second', mode: 'direct' },
      timestamp: 3,
    }
    state = reduceEngineEvent(state, secondRun)
    expect(state.run?.id).toBe('run-2')
    expect(state.nodes.size).toBe(0)
    expect(state.sandboxRuns.size).toBe(0)
    expect(state.corrections.size).toBe(0)
  })

  it('stores the run id on spawned agents', () => {
    const state = reduceEngineEvent(createDashboardDataState(), event('node_spawn', {
      node_id: 'agent-a', parent_id: 'root', task: 'A',
    }, 1))
    expect(state.nodes.get('agent-a')?.runId).toBe('run-1')
  })

  it('keeps complete root and agent results instead of their summaries', () => {
    let state = reduceEngineEvent(createDashboardDataState(), event(
      'run_started', { task: 'Build it', mode: 'delegated' }, 1,
    ))
    state = reduceEngineEvent(state, event('node_spawn', {
      node_id: 'agent-a', parent_id: 'root', task: 'Write code',
    }, 2))
    state = reduceEngineEvent(state, event('node_complete', {
      node_id: 'agent-a', result_summary: 'short',
      result: '# Full result\n\n```python\nprint("complete")\n```',
      token_usage: 20, code_executed: true, sandbox_exit_code: 0,
      attempts: 1, success: true,
    }, 3))
    state = reduceEngineEvent(state, event('run_completed', {
      success: true, mode: 'direct', result_summary: 'root short',
      result: '# Complete root result',
    }, 4))

    expect(state.nodes.get('agent-a')?.result).toContain('print("complete")')
    expect(state.run?.result).toBe('# Complete root result')
  })

  it('does not overwrite agent status when legacy gradient telemetry arrives', () => {
    let state = reduceEngineEvent(createDashboardDataState(), event('node_spawn', {
      node_id: 'agent-a', parent_id: 'root', task: 'A',
    }, 1))
    state = reduceEngineEvent(state, event('gradient_flow', {
      node_id: 'agent-a', iteration: 1, severity: 0.75, num_mutations: 3,
    }, 2))
    expect(state.nodes.get('agent-a')).toMatchObject({
      status: 'running', gradientSeverity: 0.75,
    })
  })

  it('tracks an LLM retry that ends in terminal failure', () => {
    let state = createDashboardDataState()
    const sequence: EngineEvent[] = [
      event('node_spawn', { node_id: 'agent-a', parent_id: 'root', task: 'A' }, 1),
      event('sandbox_started', {
        execution_id: 'sandbox-a-1', owner_node_id: 'agent-a', attempt: 1,
        trigger: 'initial', timeout_s: 30, code_preview: 'bad()',
      }, 2),
      event('sandbox_completed', {
        execution_id: 'sandbox-a-1', owner_node_id: 'agent-a', attempt: 1,
        status: 'failed', exit_code: 1, duration_ms: 5,
      }, 3),
      event('correction_started', {
        correction_id: 'retry-a-1', owner_node_id: 'agent-a',
        failed_execution_id: 'sandbox-a-1', attempt: 1, strategy: 'llm_retry',
      }, 4),
      event('correction_completed', { correction_id: 'retry-a-1', success: true }, 5),
      event('sandbox_started', {
        execution_id: 'sandbox-a-2', owner_node_id: 'agent-a', attempt: 2,
        trigger: 'llm_retry', timeout_s: 30, code_preview: 'still_bad()',
      }, 6),
      event('sandbox_completed', {
        execution_id: 'sandbox-a-2', owner_node_id: 'agent-a', attempt: 2,
        status: 'failed', exit_code: 1, duration_ms: 5,
      }, 7),
      event('node_complete', {
        node_id: 'agent-a', result_summary: 'failed', token_usage: 1,
        code_executed: true, sandbox_exit_code: 1, attempts: 2, success: false,
      }, 8),
    ]
    for (const item of sequence) state = reduceEngineEvent(state, item)
    expect(state.corrections.get('retry-a-1')?.strategy).toBe('llm_retry')
    expect(state.sandboxRuns.get('sandbox-a-2')?.status).toBe('failed')
    expect(state.nodes.get('agent-a')?.status).toBe('failed')
  })
})
