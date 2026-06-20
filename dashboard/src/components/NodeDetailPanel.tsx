import { motion, AnimatePresence } from 'framer-motion'
import type { AgentNode } from '../types/events'

interface NodeDetailPanelProps {
  node: AgentNode | null
  onClose: () => void
}

export default function NodeDetailPanel({ node, onClose }: NodeDetailPanelProps) {
  return (
    <AnimatePresence>
      {node && (
        <motion.div
          initial={{ x: 300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 300, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          className="absolute right-0 top-0 bottom-0 w-80 bg-panel border-l border-border
                     overflow-y-auto z-10"
        >
          <div className="p-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-text-primary">
                Node Details
              </h3>
              <button
                onClick={onClose}
                className="text-text-secondary hover:text-text-primary transition-colors
                           w-6 h-6 flex items-center justify-center rounded"
              >
                ×
              </button>
            </div>

            {/* Node ID */}
            <div className="mb-3">
              <span className="text-xs text-text-secondary">Node ID</span>
              <p className="text-sm font-mono text-text-primary">{node.id}</p>
            </div>

            {/* Status */}
            <div className="mb-3">
              <span className="text-xs text-text-secondary">Status</span>
              <p className="text-sm text-text-primary capitalize">{node.status}</p>
            </div>

            {/* Task */}
            <div className="mb-3">
              <span className="text-xs text-text-secondary">Task</span>
              <p className="text-sm text-text-primary">{node.task}</p>
            </div>

            {/* Timing */}
            <div className="mb-3">
              <span className="text-xs text-text-secondary">Timing</span>
              <p className="text-sm text-text-primary">
                Spawned: {new Date(node.spawnTime).toLocaleTimeString()}
                {node.completeTime && (
                  <>
                    {' | '}
                    Completed: {new Date(node.completeTime).toLocaleTimeString()}
                    {' | '}
                    Duration: {((node.completeTime - node.spawnTime) / 1000).toFixed(1)}s
                  </>
                )}
              </p>
            </div>

            {/* Attempts */}
            {node.attempts !== undefined && (
              <div className="mb-3">
                <span className="text-xs text-text-secondary">Attempts</span>
                <p className="text-sm text-text-primary">{node.attempts}</p>
              </div>
            )}

            {/* Exit Code */}
            {node.exitCode !== undefined && node.exitCode !== null && (
              <div className="mb-3">
                <span className="text-xs text-text-secondary">Exit Code</span>
                <p className={`text-sm font-mono ${
                  node.exitCode === 0 ? 'text-accent-green' : 'text-accent-red'
                }`}>
                  {node.exitCode}
                </p>
              </div>
            )}

            {/* Gradient */}
            {node.gradientSeverity !== undefined && (
              <div className="mb-3">
                <span className="text-xs text-text-secondary">Gradient Severity</span>
                <p className="text-sm text-accent-purple font-mono">
                  {node.gradientSeverity.toFixed(2)}
                </p>
              </div>
            )}

            {/* Result */}
            {node.result && (
              <div className="mb-3">
                <span className="text-xs text-text-secondary">Result</span>
                <pre className="text-xs text-text-primary bg-surface rounded p-2 mt-1
                               overflow-x-auto whitespace-pre-wrap font-mono">
                  {node.result.slice(0, 500)}
                  {node.result.length > 500 && '...'}
                </pre>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
