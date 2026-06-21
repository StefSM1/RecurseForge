import { useState, useEffect, useRef, useCallback } from 'react'
import type { EngineEvent, AgentNode, AgentStatus, EventType } from '../types/events'
import { BACKEND_WS_URL } from '../config'

export interface WebSocketState {
  connected: boolean
  nodes: Map<string, AgentNode>
  events: EngineEvent[]
  error: string | null
}

export function useWebSocket(url: string = '/ws') {
  const [state, setState] = useState<WebSocketState>({
    connected: false,
    nodes: new Map(),
    events: [],
    error: null,
  })

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  // Backend WebSocket URL (direct connection, bypasses Vite proxy buffering)
  const backendWsUrl = `${BACKEND_WS_URL}${url}`

  const handleEvent = useCallback((event: EngineEvent) => {
    setState(prev => {
      const newNodes = new Map(prev.nodes)
      const newEvents = [...prev.events, event]

      switch (event.event_type as EventType) {
        case 'node_spawn': {
          const payload = event.payload as {
            node_id: string
            parent_id: string
            task: string
          }
          newNodes.set(payload.node_id, {
            id: payload.node_id,
            parentId: payload.parent_id,
            task: payload.task,
            status: 'running',
            spawnTime: Date.now(),
          })
          break
        }

        case 'node_complete': {
          const payload = event.payload as {
            node_id: string
            result_summary: string
            token_usage: number
            code_executed: boolean
            sandbox_exit_code: number | null
            attempts: number
          }
          const existing = newNodes.get(payload.node_id)
          if (existing) {
            const status: AgentStatus = payload.sandbox_exit_code === 0 || payload.sandbox_exit_code === null
              ? 'success'
              : payload.attempts > 1
                ? 'retrying'
                : 'failed'

            newNodes.set(payload.node_id, {
              ...existing,
              status,
              result: payload.result_summary,
              exitCode: payload.sandbox_exit_code,
              attempts: payload.attempts,
              completeTime: Date.now(),
            })
          }
          break
        }

        case 'gradient_flow': {
          const payload = event.payload as {
            node_id: string
            iteration: number
            severity: number
            num_mutations: number
          }
          const existing = newNodes.get(payload.node_id)
          if (existing) {
            newNodes.set(payload.node_id, {
              ...existing,
              status: 'gradient',
              gradientSeverity: payload.severity,
              attempts: (existing.attempts || 0) + 1,
            })
          }
          break
        }
      }

      return {
        ...prev,
        nodes: newNodes,
        events: newEvents,
      }
    })
  }, [])

  useEffect(() => {
    let active = true

    function connect() {
      if (!active) return

      const ws = new WebSocket(backendWsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setState(prev => ({ ...prev, connected: true, error: null }))
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as EngineEvent
          handleEvent(data)
        } catch {
          // Skip malformed messages
        }
      }

      ws.onerror = () => {
        setState(prev => ({ ...prev, error: 'WebSocket error' }))
      }

      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null
        if (!active) return

        setState(prev => ({ ...prev, connected: false }))
        reconnectTimerRef.current = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      active = false
      clearTimeout(reconnectTimerRef.current)
      const ws = wsRef.current
      wsRef.current = null
      ws?.close()
    }
  }, [backendWsUrl, handleEvent])

  const clearHistory = useCallback(() => {
    setState(prev => ({
      ...prev,
      nodes: new Map(),
      events: [],
    }))
  }, [])

  return { ...state, clearHistory }
}
