import { useState, useEffect, useRef, useCallback } from 'react'
import type { EngineEvent } from '../types/events'
import { BACKEND_WS_URL } from '../config'
import {
  createDashboardDataState,
  reduceEngineEvent,
  type DashboardDataState,
} from './dashboardState'

export interface WebSocketState extends DashboardDataState {
  connected: boolean
  error: string | null
}

function createWebSocketState(): WebSocketState {
  return {
    ...createDashboardDataState(),
    connected: false,
    error: null,
  }
}

export function useWebSocket(url: string = '/ws') {
  const [state, setState] = useState<WebSocketState>(createWebSocketState)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const backendWsUrl = `${BACKEND_WS_URL}${url}`

  const handleEvent = useCallback((event: EngineEvent) => {
    setState(previous => ({
      ...previous,
      ...reduceEngineEvent(previous, event),
    }))
  }, [])

  useEffect(() => {
    let active = true

    function connect() {
      if (!active) return
      const ws = new WebSocket(backendWsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setState(previous => ({ ...previous, connected: true, error: null }))
      }
      ws.onmessage = event => {
        try {
          handleEvent(JSON.parse(event.data) as EngineEvent)
        } catch {
          // Malformed development events are ignored.
        }
      }
      ws.onerror = () => {
        setState(previous => ({ ...previous, error: 'WebSocket error' }))
      }
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null
        if (!active) return
        setState(previous => ({ ...previous, connected: false }))
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
    setState(previous => ({
      ...previous,
      ...createDashboardDataState(),
    }))
  }, [])

  return { ...state, clearHistory }
}
