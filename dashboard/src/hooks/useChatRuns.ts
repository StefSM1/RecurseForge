import { useCallback, useEffect, useMemo, useState } from 'react'
import { BACKEND_BASE_URL } from '../config'
import type { ChatMessage, ChatRunRecord } from '../types/chat'
import type { EngineEvent } from '../types/events'
import { hasRunCompletedEvent, isTerminalChatRunStatus } from '../components/chatModel'

function messageId(prefix: string): string {
  return `${prefix}-${globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2)}`
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail ?? `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function useChatRuns(events: EngineEvent[]) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [runs, setRuns] = useState<Map<string, ChatRunRecord>>(() => new Map())

  const activeRun = activeRunId ? runs.get(activeRunId) ?? null : null
  const isInferenceActive = !!activeRun &&
    !isTerminalChatRunStatus(activeRun.status)

  const mergeRun = useCallback((record: ChatRunRecord) => {
    setRuns(previous => {
      const next = new Map(previous)
      next.set(record.run_id, record)
      return next
    })
    return record
  }, [])

  const updateAssistantMessage = useCallback((record: ChatRunRecord) => {
    setMessages(previous => previous.map(message => {
      if (message.role !== 'assistant' || message.runId !== record.run_id) {
        return message
      }
      if (record.status === 'stopped') {
        return { ...message, status: record.status, content: 'Run stopped.' }
      }
      if (record.status === 'failed') {
        return {
          ...message,
          status: record.status,
          content: record.error || record.final_output || 'Run failed.',
        }
      }
      if (record.status === 'success') {
        return {
          ...message,
          status: record.status,
          content: record.final_output || 'Run completed without a final answer.',
        }
      }
      return { ...message, status: record.status }
    }))
  }, [])

  const refreshRun = useCallback(async (runId: string) => {
    const record = await fetchJson<ChatRunRecord>(
      `${BACKEND_BASE_URL}/api/chat/runs/${runId}`,
    )
    mergeRun(record)
    updateAssistantMessage(record)
    if (isTerminalChatRunStatus(record.status)) {
      setActiveRunId(current => current === runId ? null : current)
    }
    return record
  }, [mergeRun, updateAssistantMessage])

  const sendPrompt = useCallback(async (message: string) => {
    const prompt = message.trim()
    if (!prompt || isInferenceActive) return

    const createdAt = Date.now()
    setMessages(previous => [
      ...previous,
      {
        id: messageId('user'),
        role: 'user',
        content: prompt,
        createdAt,
      },
    ])

    try {
      const record = await fetchJson<ChatRunRecord>(
        `${BACKEND_BASE_URL}/api/chat/runs`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: prompt }),
        },
      )
      mergeRun(record)
      setActiveRunId(record.run_id)
      setMessages(previous => [
        ...previous,
        {
          id: messageId('assistant'),
          role: 'assistant',
          content: 'Running agent...',
          createdAt: Date.now(),
          runId: record.run_id,
          status: record.status,
        },
      ])
    } catch (error) {
      setMessages(previous => [
        ...previous,
        {
          id: messageId('assistant-error'),
          role: 'assistant',
          content: error instanceof Error ? error.message : 'Failed to start run.',
          createdAt: Date.now(),
          status: 'failed',
        },
      ])
    }
  }, [isInferenceActive, mergeRun])

  const stopActiveRun = useCallback(async () => {
    if (!activeRunId) return
    try {
      const record = await fetchJson<ChatRunRecord>(
        `${BACKEND_BASE_URL}/api/chat/runs/${activeRunId}/stop`,
        { method: 'POST' },
      )
      mergeRun(record)
      updateAssistantMessage(record)
    } catch (error) {
      setMessages(previous => [
        ...previous,
        {
          id: messageId('stop-error'),
          role: 'assistant',
          content: error instanceof Error ? error.message : 'Failed to stop run.',
          createdAt: Date.now(),
          status: 'failed',
        },
      ])
    }
  }, [activeRunId, mergeRun, updateAssistantMessage])

  useEffect(() => {
    if (!activeRunId) return
    if (!hasRunCompletedEvent(events, activeRunId)) return
    void refreshRun(activeRunId)
  }, [activeRunId, events, refreshRun])

  useEffect(() => {
    if (!activeRunId || !activeRun || isTerminalChatRunStatus(activeRun.status)) {
      return
    }
    const timer = setInterval(() => {
      void refreshRun(activeRunId)
    }, 2000)
    return () => clearInterval(timer)
  }, [activeRunId, activeRun, refreshRun])

  return useMemo(() => ({
    messages,
    activeRun,
    isInferenceActive,
    sendPrompt,
    stopActiveRun,
  }), [messages, activeRun, isInferenceActive, sendPrompt, stopActiveRun])
}
