import { useMemo, useCallback } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
  type EdgeTypes,
  type EdgeProps,
  type OnNodesChange,
  type OnEdgesChange,
  Position,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { motion, AnimatePresence } from 'framer-motion'
import dagre from 'dagre'
import type { AgentNode } from '../types/events'

// ---------------------------------------------------------------------------
// Custom Edge: Info Line (thin orange rectangle)
// ---------------------------------------------------------------------------

function InfoLineEdge({ id, sourceX, sourceY, targetX, targetY }: EdgeProps) {
  // Orthogonal path: down from source, horizontal, then down to target
  const midY = (sourceY + targetY) / 2
  const path = `M ${sourceX},${sourceY} L ${sourceX},${midY} L ${targetX},${midY} L ${targetX},${targetY}`

  return (
    <g className="react-flow__edge">
      <path
        d={path}
        fill="none"
        stroke="#f97316"
        strokeWidth={4}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="react-flow__edge-path"
      />
    </g>
  )
}

// ---------------------------------------------------------------------------
// Custom Node: Root Agent (always visible)
// ---------------------------------------------------------------------------

interface RootNodeData {
  status: 'offline' | 'running' | 'success'
  task: string
  onClick: () => void
}

function RootNodeComponent({ data }: { data: RootNodeData }) {
  const isOffline = data.status === 'offline'
  const isRunning = data.status === 'running'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      onClick={data.onClick}
      className={`
        px-5 py-3 rounded-xl border-2 cursor-pointer min-w-[200px] text-center
        ${isOffline
          ? 'border-gray-600 bg-gray-800/80'
          : isRunning
            ? 'border-accent-green bg-accent-green/10 shadow-[0_0_20px_rgba(34,197,94,0.3)]'
            : 'border-accent-green/50 bg-accent-green/5'
        }
      `}
    >
      <div className="flex items-center justify-center gap-2 mb-1">
        {isRunning ? (
          <motion.div
            className="w-3 h-3 rounded-full bg-accent-green"
            animate={{
              boxShadow: [
                '0 0 0 0 rgba(34, 197, 94, 0.6)',
                '0 0 0 8px rgba(34, 197, 94, 0)',
              ],
            }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        ) : (
          <div className={`w-3 h-3 rounded-full ${
            isOffline ? 'bg-gray-500' : 'bg-accent-green'
          }`} />
        )}
        <span className="text-sm font-semibold text-text-primary">
          Root Agent
        </span>
      </div>
      <p className="text-xs text-text-secondary truncate max-w-[180px]">
        {data.task || 'Waiting for task...'}
      </p>
      <p className={`text-xs mt-1 capitalize ${
        isOffline ? 'text-gray-500' : isRunning ? 'text-accent-green' : 'text-accent-green/60'
      }`}>
        {data.status}
      </p>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Custom Node: Output/Result (always visible)
// ---------------------------------------------------------------------------

interface OutputNodeData {
  status: 'waiting' | 'success' | 'error'
  summary: string
  onClick: () => void
}

function OutputNodeComponent({ data }: { data: OutputNodeData }) {
  const isWaiting = data.status === 'waiting'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      onClick={data.onClick}
      className={`
        px-5 py-3 rounded-xl border-2 cursor-pointer min-w-[200px] text-center
        ${isWaiting
          ? 'border-gray-600 bg-gray-800'
          : data.status === 'error'
            ? 'border-accent-red/50 bg-gray-800'
            : 'border-accent-blue/50 bg-gray-800'
        }
      `}
      style={{ background: 'rgb(30, 30, 46)' }}
    >
      <div className="flex items-center justify-center gap-2 mb-1">
        <div className={`w-3 h-3 rounded-full ${
          isWaiting ? 'bg-gray-500' : data.status === 'error' ? 'bg-accent-red' : 'bg-accent-blue'
        }`} />
        <span className="text-sm font-semibold text-text-primary">
          Output
        </span>
      </div>
      <p className="text-xs text-text-secondary truncate max-w-[180px]">
        {data.summary || 'Awaiting result...'}
      </p>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Custom Node: Sub-Agent
// ---------------------------------------------------------------------------

const statusColors: Record<string, string> = {
  running: 'bg-accent-blue',
  success: 'bg-accent-green',
  failed: 'bg-accent-red',
  retrying: 'bg-accent-yellow',
  gradient: 'bg-accent-purple',
}

const statusBorders: Record<string, string> = {
  running: 'border-accent-blue/50',
  success: 'border-accent-green/50',
  failed: 'border-accent-red/50',
  retrying: 'border-accent-yellow/50',
  gradient: 'border-accent-purple/50',
}

function SubAgentNodeComponent({ data }: { data: {
  status: string; task: string; node_id: string; onClick: () => void
} }) {
  const dotColor = statusColors[data.status] || 'bg-gray-500'
  const borderColor = statusBorders[data.status] || 'border-gray-600'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      onClick={data.onClick}
      className={`
        px-4 py-3 rounded-lg border bg-panel cursor-pointer
        min-w-[180px] max-w-[260px] ${borderColor}
      `}
    >
      <div className="flex items-center gap-2 mb-1">
        {data.status === 'running' ? (
          <motion.div
            className={`w-2.5 h-2.5 rounded-full ${dotColor}`}
            animate={{
              boxShadow: [
                '0 0 0 0 rgba(59, 130, 246, 0.5)',
                '0 0 0 6px rgba(59, 130, 246, 0)',
              ],
            }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        ) : (
          <div className={`w-2.5 h-2.5 rounded-full ${dotColor}`} />
        )}
        <span className="text-xs font-mono text-text-secondary">
          {data.node_id}
        </span>
      </div>
      <p className="text-sm text-text-primary leading-tight truncate">
        {data.task}
      </p>
      <p className="text-xs text-text-secondary mt-1 capitalize">
        {data.status}
      </p>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Layout engine (dagre)
// ---------------------------------------------------------------------------

const NODE_WIDTH = 220
const NODE_HEIGHT = 80

function getLayoutedElements(
  nodes: Node[],
  edges: Edge[]
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', nodesep: 60, ranksep: 100 })

  nodes.forEach(node => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  })
  edges.forEach(edge => {
    g.setEdge(edge.source, edge.target)
  })

  dagre.layout(g)

  const layoutedNodes = nodes.map(node => {
    const pos = g.node(node.id)
    return {
      ...node,
      position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    }
  })

  return { nodes: layoutedNodes, edges }
}

// ---------------------------------------------------------------------------
// AgentTree Component
// ---------------------------------------------------------------------------

interface AgentTreeProps {
  nodes: Map<string, AgentNode>
  onNodeClick: (nodeId: string) => void
  rootStatus: 'offline' | 'running' | 'success'
  rootTask: string
  outputStatus: 'waiting' | 'success' | 'error'
  outputSummary: string
  onRootClick: () => void
  onOutputClick: () => void
}

const nodeTypes: NodeTypes = {
  root: RootNodeComponent,
  output: OutputNodeComponent,
  agent: SubAgentNodeComponent,
}

const edgeTypes: EdgeTypes = {
  infoLine: InfoLineEdge,
}

export default function AgentTree({
  nodes, onNodeClick, rootStatus, rootTask,
  outputStatus, outputSummary, onRootClick, onOutputClick,
}: AgentTreeProps) {
  const { flowNodes, flowEdges } = useMemo(() => {
    const rawNodes: Node[] = []
    const rawEdges: Edge[] = []

    // Always-present root node
    rawNodes.push({
      id: 'root',
      type: 'root',
      data: { status: rootStatus, task: rootTask, onClick: onRootClick },
      position: { x: 0, y: 0 },
    })

    // Always-present output node
    rawNodes.push({
      id: 'output',
      type: 'output',
      data: { status: outputStatus, summary: outputSummary, onClick: onOutputClick },
      position: { x: 0, y: 0 },
    })

    // Static test edge from root to output (always present)
    rawEdges.push({
      id: 'e-static-root-output',
      source: 'root',
      target: 'output',
      type: 'infoLine',
    })

    // Sub-agent nodes
    const agentNodes = Array.from(nodes.values())
    agentNodes.forEach(agent => {
      rawNodes.push({
        id: agent.id,
        type: 'agent',
        data: {
          status: agent.status,
          task: agent.task,
          node_id: agent.id,
          onClick: () => onNodeClick(agent.id),
        },
        position: { x: 0, y: 0 },
      })

      // Edge from root to sub-agent (orange info line)
      rawEdges.push({
        id: `e-root-${agent.id}`,
        source: 'root',
        target: agent.id,
        type: 'infoLine',
      })
    })

    const { nodes: layoutedNodes, edges: layoutedEdges } =
      getLayoutedElements(rawNodes, rawEdges)

    return { flowNodes: layoutedNodes, flowEdges: layoutedEdges }
  }, [nodes, rootStatus, rootTask, outputStatus, outputSummary, onNodeClick, onRootClick, onOutputClick])

  const onNodesChange: OnNodesChange = useCallback(() => {}, [])
  const onEdgesChange: OnEdgesChange = useCallback(() => {}, [])

  return (
    <AnimatePresence>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        proOptions={{ hideAttribution: true }}
        className="bg-surface"
      >
        <Background color="#2a2a3a" gap={20} />
        <Controls showInteractive={false} className="!bg-panel !border-border" />
      </ReactFlow>
    </AnimatePresence>
  )
}
