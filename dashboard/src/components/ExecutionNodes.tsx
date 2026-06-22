import { useState } from 'react'
import { Handle, Position } from '@xyflow/react'
import { motion, useReducedMotion } from 'framer-motion'
import type { SandboxNodeView } from './executionState'

const hiddenHandleStyle = {
  width: 1,
  height: 1,
  minWidth: 1,
  minHeight: 1,
  opacity: 0,
  pointerEvents: 'none' as const,
}

interface RevealData {
  revealDelayMs: number
  shouldAnimateReveal: boolean
  onRevealComplete: (nodeId: string) => void
}

export interface SandboxNodeData extends RevealData {
  sandbox: SandboxNodeView
  onClick: () => void
}

function WaitingDots() {
  const reducedMotion = useReducedMotion()
  return (
    <span className="execution-waiting" aria-label="Running">
      {[0, 1, 2].map(index => (
        <motion.span
          key={index}
          animate={reducedMotion ? { opacity: 0.65 } : { opacity: [0.2, 1, 0.2] }}
          transition={reducedMotion ? { duration: 0 } : {
            duration: 1.15,
            delay: index * 0.16,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </span>
  )
}

export function SandboxNodeComponent({ data }: { data: SandboxNodeData }) {
  const { sandbox } = data
  const reducedMotion = useReducedMotion()
  const shouldReveal = data.shouldAnimateReveal && !reducedMotion
  const [revealed, setRevealed] = useState(!shouldReveal)
  const statusClass = sandbox.status === 'success'
    ? 'execution-node--success'
    : sandbox.status === 'failed'
      ? 'execution-node--failed'
      : sandbox.status === 'offline'
        ? 'execution-node--offline'
        : 'execution-node--running'

  return (
    <motion.div
      initial={shouldReveal ? { opacity: 0, scale: 0.95 } : { opacity: 1, scale: 1 }}
      animate={{
        opacity: 1,
        scale: 1,
        x: sandbox.status === 'failed' && !reducedMotion ? [0, -2, 2, -1, 0] : 0,
      }}
      transition={{
        opacity: { duration: shouldReveal ? 0.42 : 0, delay: data.revealDelayMs / 1000 },
        scale: { duration: shouldReveal ? 0.42 : 0, delay: data.revealDelayMs / 1000 },
        x: { duration: 0.28 },
      }}
      onAnimationComplete={() => {
        if (!revealed) {
          setRevealed(true)
          data.onRevealComplete(sandbox.id)
        }
      }}
      onClick={data.onClick}
      style={{ pointerEvents: revealed || reducedMotion ? 'auto' : 'none' }}
      className={`execution-node execution-node--sandbox ${statusClass}`}
    >
      <Handle id="forward-in" type="target" position={Position.Top} style={hiddenHandleStyle} />
      <Handle id="direct-in" type="target" position={Position.Left} style={hiddenHandleStyle} />
      <Handle id="feedback-out" type="source" position={Position.Left} style={hiddenHandleStyle} />
      <Handle id="feedback-top" type="source" position={Position.Top} style={hiddenHandleStyle} />
      <Handle id="result-out" type="source" position={Position.Bottom} style={hiddenHandleStyle} />
      <Handle id="direct-out" type="source" position={Position.Right} style={hiddenHandleStyle} />
      <div className="execution-node__header">
        <span className="execution-node__icon">&gt;_</span>
        <span>Sandbox</span>
        {sandbox.status === 'running' && <WaitingDots />}
        {sandbox.status === 'success' && <span className="execution-node__result">OK</span>}
        {sandbox.status === 'failed' && <span className="execution-node__result">FAIL</span>}
        {sandbox.status === 'offline' && <span className="execution-node__result">OFFLINE</span>}
      </div>
      <div className="execution-node__meta">
        <span>attempt {sandbox.attempt}</span>
        <span className="capitalize">{sandbox.status === 'offline' ? 'waiting for retry' : sandbox.trigger.replace('_', ' ')}</span>
        {sandbox.exitCode !== undefined && <span>exit {sandbox.exitCode}</span>}
        {sandbox.durationMs !== undefined && <span>{sandbox.durationMs}ms</span>}
      </div>
    </motion.div>
  )
}
