import { useEffect, useMemo, useState } from 'react'
import { BACKEND_BASE_URL } from '../config'
import type { AgentNode, RunState, SandboxRun } from '../types/events'
import { buildSummaryStats, formatElapsedTime } from './summaryBarModel'

interface SummaryBarProps {
  run: RunState | null
  nodes: Map<string, AgentNode>
  sandboxRuns: Map<string, SandboxRun>
  eventCount: number
  rootTask: string
  retryingOwnerIds?: Set<string>
  pollIntervalMs?: number
}

export default function SummaryBar({
  run,
  nodes,
  sandboxRuns,
  eventCount,
  rootTask,
  retryingOwnerIds = new Set<string>(),
  pollIntervalMs = 3000,
}: SummaryBarProps) {
  const [vramMb, setVramMb] = useState(0)
  const [now, setNow] = useState(0)

  useEffect(() => {
    const tick = () => setNow(Date.now())
    tick()
    const timer = setInterval(tick, 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE_URL}/api/resources`)
        if (!res.ok) return
        const data = await res.json()
        setVramMb(data.vram_mb ?? 0)
      } catch {
        // Resource polling is best-effort; the dashboard can run without it.
      }
    }

    poll()
    const timer = setInterval(poll, pollIntervalMs)
    return () => clearInterval(timer)
  }, [pollIntervalMs])

  const stats = useMemo(() => buildSummaryStats({
    run,
    nodes: Array.from(nodes.values()),
    sandboxRuns: Array.from(sandboxRuns.values()),
    eventCount,
    rootTask,
    retryingOwnerIds,
    now,
    vramMb,
  }), [run, nodes, sandboxRuns, eventCount, rootTask, retryingOwnerIds, now, vramMb])

  const vramClass = {
    unknown: 'text-text-secondary',
    normal: 'text-accent-green',
    warning: 'text-accent-yellow',
    critical: 'text-accent-red',
  }[stats.vramSeverity]

  return (
    <div className="h-14 bg-panel border-t border-border flex items-center px-4 gap-5 min-w-0">
      <div className="min-w-0 max-w-[32%]">
        <span className="text-xs text-text-secondary mr-2">Task:</span>
        <span className="text-xs text-text-primary truncate inline-block align-bottom max-w-[16rem]">
          {stats.taskName}
        </span>
      </div>

      <SummaryMetric label="Elapsed" value={formatElapsedTime(stats.elapsedMs)} />
      <SummaryMetric label="Nodes" value={stats.totalNodes.toString()} accent="text-accent-blue" />

      <div className="flex items-center gap-2">
        <span className="text-xs text-accent-green font-mono">OK {stats.successCount}</span>
        <span className="text-xs text-accent-red font-mono">FAIL {stats.failedCount}</span>
        {stats.runningCount > 0 && (
          <span className="text-xs text-accent-blue font-mono">
            RUN {stats.runningCount}
          </span>
        )}
      </div>

      <SummaryMetric label="Events" value={stats.eventCount.toString()} />
      <SummaryMetric label="Sandbox" value={stats.sandboxCount.toString()} accent="text-accent-yellow" />

      <div className="ml-auto flex items-center gap-2">
        <span className="text-xs text-text-secondary">VRAM:</span>
        <span className={`text-xs font-mono ${vramClass}`}>
          {vramMb > 0 ? `${vramMb} MB` : '-- MB'}
        </span>
      </div>
    </div>
  )
}

function SummaryMetric({
  label,
  value,
  accent = 'text-text-primary',
}: {
  label: string
  value: string
  accent?: string
}) {
  return (
    <div className="flex items-center gap-2 shrink-0">
      <span className="text-xs text-text-secondary">{label}:</span>
      <span className={`text-xs font-mono ${accent}`}>{value}</span>
    </div>
  )
}
