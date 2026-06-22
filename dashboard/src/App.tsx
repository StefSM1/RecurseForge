import { useState, useCallback, useEffect, useMemo } from 'react'
import AgentTree from './components/AgentTree'
import NodeDetailPanel from './components/NodeDetailPanel'
import ResourceMonitor from './components/ResourceMonitor'
import { useWebSocket } from './hooks/useWebSocket'
import { BACKEND_BASE_URL } from './config'
import type { GraphEntity } from './types/events'
import { buildOwnerExecutionViews } from './components/executionState'

type Tab = 'agents' | 'resources'
type RootStatus = 'offline' | 'running' | 'success' | 'error'
type OutputStatus = 'waiting' | 'success' | 'error'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('agents')
  const [selection, setSelection] = useState<{
    kind: GraphEntity['kind']; id: string
  } | null>(null)
  const [showOutputDetail, setShowOutputDetail] = useState(false)

  const { connected, run, nodes, sandboxRuns, corrections, events } = useWebSocket()
  const retryingOwners = useMemo(() => new Set(
    buildOwnerExecutionViews(
      Array.from(nodes.values()),
      Array.from(sandboxRuns.values()),
      Array.from(corrections.values()),
      run,
    ).filter(view => view.retrying).map(view => view.ownerId),
  ), [nodes, sandboxRuns, corrections, run])

  // Poll VRAM for summary bar
  const [vramMb, setVramMb] = useState<number>(0)
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE_URL}/api/resources`)
        if (res.ok) {
          const data = await res.json()
          setVramMb(data.vram_mb ?? 0)
        }
      } catch { /* ignore */ }
    }
    poll()
    const timer = setInterval(poll, 3000)
    return () => clearInterval(timer)
  }, [])

  // Derive root and output state from events
  const { rootStatus, rootTask, outputStatus, outputSummary } = useMemo(() => {
    const hasSpawnEvents = events.some(e => e.event_type === 'node_spawn')
    const hasCompleteEvents = events.some(e => e.event_type === 'node_complete')
    const allComplete = nodes.size > 0 &&
      Array.from(nodes.values()).every(n => n.status === 'success' || n.status === 'failed')

    // Root status
    let rootStatus: RootStatus = 'offline'
    if (run) {
      rootStatus = run.status === 'failed' ? 'error' : run.status
    } else if (hasSpawnEvents && !allComplete) rootStatus = 'running'
    else if (hasCompleteEvents || allComplete) rootStatus = 'success'
    else if (events.length > 0) rootStatus = 'running'

    // Root task (from first spawn event's parent context)
    const rootTask = run?.task || (events.length > 0 ? 'Task received' : '')

    // Output status
    let outputStatus: OutputStatus = 'waiting'
    if (run && run.status !== 'running') {
      outputStatus = run.status === 'failed' ? 'error' : 'success'
    } else if (allComplete) {
      const anyFailed = Array.from(nodes.values()).some(n => n.status === 'failed')
      outputStatus = anyFailed ? 'error' : 'success'
    }

    // Output summary
    const successNodes = Array.from(nodes.values()).filter(n => n.status === 'success')
    const outputSummary = run?.resultSummary || (successNodes.length > 0
      ? `${successNodes.length} task${successNodes.length > 1 ? 's' : ''} completed`
      : '')

    return { rootStatus, rootTask, outputStatus, outputSummary }
  }, [events, nodes, run])

  const selectedEntity: GraphEntity | null = useMemo(() => {
    if (!selection) return null
    if (selection.kind === 'root') {
      return {
        kind: 'root',
        value: {
          id: 'root', task: rootTask,
          status: retryingOwners.has('root') ? 'retrying' : rootStatus,
        },
      }
    }
    if (selection.kind === 'agent') {
      const value = nodes.get(selection.id)
      return value ? {
        kind: 'agent',
        value: retryingOwners.has(value.id) ? { ...value, status: 'retrying' } : value,
      } : null
    }
    if (selection.kind === 'sandbox') {
      const value = sandboxRuns.get(selection.id)
      return value ? { kind: 'sandbox', value } : null
    }
    const value = corrections.get(selection.id)
    return value ? { kind: 'correction', value } : null
  }, [
    selection, nodes, sandboxRuns, corrections, rootTask, rootStatus,
    retryingOwners,
  ])

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelection(previous => previous?.kind === 'agent' && previous.id === nodeId
      ? null
      : { kind: 'agent', id: nodeId })
    setShowOutputDetail(false)
  }, [])

  const handleSandboxClick = useCallback((sandboxId: string) => {
    setSelection(previous => previous?.kind === 'sandbox' && previous.id === sandboxId
      ? null
      : { kind: 'sandbox', id: sandboxId })
    setShowOutputDetail(false)
  }, [])

  const handleRootClick = useCallback(() => {
    setSelection(previous => previous?.kind === 'root'
      ? null
      : { kind: 'root', id: 'root' })
    setShowOutputDetail(false)
  }, [])

  const handleOutputClick = useCallback(() => {
    setShowOutputDetail(prev => !prev)
    setSelection(null)
  }, [])

  // Summary stats
  const totalNodes = nodes.size
  const successCount = Array.from(nodes.values()).filter(n => n.status === 'success').length
  const failedCount = Array.from(nodes.values()).filter(n => n.status === 'failed').length
  const runningCount = Array.from(nodes.values()).filter(n => n.status === 'running').length

  return (
    <div className="flex h-full bg-surface text-text-primary font-sans">
      {/* Left Panel: Tab Navigation */}
      <aside className="w-56 bg-panel border-r border-border flex flex-col">
        <div className="p-4 border-b border-border">
          <h1 className="text-sm font-semibold tracking-wide text-text-primary">
            RecurseForge
          </h1>
          <p className="text-xs text-text-secondary mt-1">Dashboard</p>
        </div>
        <nav className="flex-1 p-2 space-y-1">
          <button
            onClick={() => setActiveTab('agents')}
            className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
              activeTab === 'agents'
                ? 'bg-panel-light text-text-primary'
                : 'text-text-secondary hover:text-text-primary hover:bg-panel-light/50'
            }`}
          >
            <span className="mr-2">◉</span>
            Agent Monitor
            {runningCount > 0 && (
              <span className="ml-2 px-1.5 py-0.5 text-xs bg-accent-blue/20 text-accent-blue rounded">
                {runningCount}
              </span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('resources')}
            className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
              activeTab === 'resources'
                ? 'bg-panel-light text-text-primary'
                : 'text-text-secondary hover:text-text-primary hover:bg-panel-light/50'
            }`}
          >
            <span className="mr-2">◈</span>
            Resources
          </button>
        </nav>

        {/* Event log mini */}
        <div className="p-2 border-t border-border">
          <p className="text-xs text-text-secondary px-1 mb-1">
            Events: {events.length}
          </p>
          <div className="max-h-32 overflow-y-auto space-y-0.5">
            {events.slice(-10).reverse().map((e, i) => (
              <div key={i} className="text-xs font-mono text-text-secondary/60 px-1 truncate">
                {e.event_type}
              </div>
            ))}
          </div>
        </div>

        {/* Connection status + Exit button */}
        <div className="p-3 border-t border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                connected ? 'bg-accent-green animate-pulse' : 'bg-accent-red'
              }`} />
              <span className="text-xs text-text-secondary">
                {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <button
              onClick={() => {
                fetch(`${BACKEND_BASE_URL}/api/exit`, { method: 'POST' })
                  .catch(() => {})
              }}
              className="px-2 py-1 text-xs text-accent-red/70 hover:text-accent-red
                         hover:bg-accent-red/10 rounded transition-colors"
              title="Shut down dashboard server"
            >
              Exit
            </button>
          </div>
        </div>
      </aside>

      {/* Middle Panel: Main View */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Top: Main Content Area */}
        <div className="flex-1 relative overflow-hidden">
          {activeTab === 'agents' ? (
            <>
              <AgentTree
                nodes={nodes}
                sandboxRuns={sandboxRuns}
                corrections={corrections}
                run={run}
                onNodeClick={handleNodeClick}
                onSandboxClick={handleSandboxClick}
                rootStatus={rootStatus}
                rootTask={rootTask}
                outputStatus={outputStatus}
                outputSummary={outputSummary}
                onRootClick={handleRootClick}
                onOutputClick={handleOutputClick}
              />
              <NodeDetailPanel
                entity={selectedEntity}
                sandboxRuns={Array.from(sandboxRuns.values())}
                corrections={Array.from(corrections.values())}
                onClose={() => setSelection(null)}
              />
              {/* Output detail overlay */}
              {showOutputDetail && (
                <div className="absolute right-0 top-0 bottom-0 w-80 bg-panel border-l border-border
                                overflow-y-auto z-10 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-text-primary">Output Result</h3>
                    <button onClick={() => setShowOutputDetail(false)}
                      className="text-text-secondary hover:text-text-primary w-6 h-6 flex items-center justify-center rounded">×</button>
                  </div>
                  <div className="space-y-3">
                    <div>
                      <span className="text-xs text-text-secondary">Status</span>
                      <p className={`text-sm capitalize ${
                        outputStatus === 'success' ? 'text-accent-green' :
                        outputStatus === 'error' ? 'text-accent-red' : 'text-text-secondary'
                      }`}>{outputStatus}</p>
                    </div>
                    <div>
                      <span className="text-xs text-text-secondary">Summary</span>
                      <p className="text-sm text-text-primary">{outputSummary || 'No output yet'}</p>
                    </div>
                    {Array.from(nodes.values()).filter(n => n.result).map(n => (
                      <div key={n.id}>
                        <span className="text-xs text-text-secondary">Node {n.id}</span>
                        <pre className="text-xs text-text-primary bg-surface rounded p-2 mt-1
                                       overflow-x-auto whitespace-pre-wrap font-mono">
                          {n.result?.slice(0, 500)}
                          {(n.result?.length ?? 0) > 500 ? '...' : ''}
                        </pre>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <ResourceMonitor />
          )}
        </div>

        {/* Bottom: Summary Bar */}
        <div className="h-14 bg-panel border-t border-border flex items-center px-4 gap-6">
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary">Nodes:</span>
            <span className="text-xs text-accent-blue font-mono">{totalNodes}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-accent-green font-mono">
              ✓ {successCount}
            </span>
            <span className="text-xs text-accent-red font-mono">
              ✗ {failedCount}
            </span>
            {runningCount > 0 && (
              <span className="text-xs text-accent-blue font-mono">
                ● {runningCount} running
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary">Events:</span>
            <span className="text-xs text-text-primary font-mono">{events.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary">Sandbox:</span>
            <span className="text-xs text-accent-yellow font-mono">{sandboxRuns.size}</span>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-text-secondary">VRAM:</span>
            <span className={`text-xs font-mono ${
              vramMb > 7000 ? 'text-accent-red'
              : vramMb > 6500 ? 'text-accent-yellow'
              : vramMb > 0 ? 'text-accent-green'
              : 'text-text-secondary'
            }`}>
              {vramMb > 0 ? `${vramMb} MB` : '— MB'}
            </span>
          </div>
        </div>
      </main>

      {/* Right Panel: Chat Interface */}
      <aside className="w-80 bg-panel border-l border-border flex flex-col">
        <div className="p-3 border-b border-border">
          <h2 className="text-sm font-medium text-text-primary">Chat</h2>
          <p className="text-xs text-text-secondary">
            Send tasks to the engine
          </p>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-2">
            <span className="text-2xl text-text-secondary/40">💬</span>
            <p className="text-xs text-text-secondary">
              Chat interface placeholder
            </p>
            <p className="text-xs text-text-secondary/60">Stage 5</p>
          </div>
        </div>
        <div className="p-3 border-t border-border">
          <div className="flex gap-2">
            <input
              type="text"
              disabled
              placeholder="Chat coming in Stage 5..."
              className="flex-1 bg-surface rounded-md px-3 py-2 text-xs text-text-primary
                         placeholder-text-secondary/40 border border-border disabled:opacity-50"
            />
            <button
              disabled
              className="px-3 py-2 bg-accent-blue/20 text-accent-blue text-xs rounded-md
                         disabled:opacity-50"
            >
              Send
            </button>
          </div>
        </div>
      </aside>
    </div>
  )
}

export default App
