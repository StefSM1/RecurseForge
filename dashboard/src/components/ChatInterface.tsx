import { useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { AgentNode, EngineEvent } from '../types/events'
import { useChatRuns } from '../hooks/useChatRuns'
import { buildChatTabs, shouldSubmitChatInput } from './chatModel'

interface ChatInterfaceProps {
  nodes: Map<string, AgentNode>
  events: EngineEvent[]
}

export default function ChatInterface({ nodes, events }: ChatInterfaceProps) {
  const [input, setInput] = useState('')
  const [activeTabId, setActiveTabId] = useState('main')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { messages, activeRun, isInferenceActive, sendPrompt, stopActiveRun } =
    useChatRuns(events)

  const nodeList = useMemo(() => Array.from(nodes.values()), [nodes])
  const tabs = useMemo(() => buildChatTabs(nodeList), [nodeList])
  const visibleActiveTabId = tabs.some(tab => tab.id === activeTabId)
    ? activeTabId
    : 'main'
  const activeTab = tabs.find(tab => tab.id === visibleActiveTabId) ?? tabs[0]
  const selectedAgent = activeTab.kind === 'sub-agent' && activeTab.agentId
    ? nodes.get(activeTab.agentId)
    : null

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const submit = () => {
    const prompt = input.trim()
    if (!prompt || isInferenceActive) return
    setInput('')
    void sendPrompt(prompt)
  }

  return (
    <aside className="w-80 bg-panel border-l border-border flex flex-col min-w-0">
      <div className="border-b border-border">
        <div className="px-3 pt-3">
          <h2 className="text-sm font-medium text-text-primary">Chat</h2>
          <p className="text-xs text-text-secondary">
            Send prompts to the root agent
          </p>
        </div>

        <div className="flex items-end gap-2 px-2 pt-3 overflow-hidden">
          <button
            onClick={() => setActiveTabId('main')}
            className={`shrink-0 px-3 py-1.5 rounded-t-md text-xs transition-colors ${
              visibleActiveTabId === 'main'
                ? 'bg-surface text-text-primary border border-border border-b-surface'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            Main
          </button>

          <div className="ml-auto flex flex-row-reverse items-end gap-1 min-w-0">
            <AnimatePresence initial={false}>
              {tabs.filter(tab => tab.kind === 'sub-agent').map(tab => (
                <motion.button
                  key={tab.id}
                  initial={{ opacity: 0, x: 18, scale: 0.96 }}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: 18, scale: 0.96 }}
                  transition={{ duration: 0.18 }}
                  onClick={() => setActiveTabId(tab.id)}
                  className={`shrink-0 px-2.5 py-1.5 rounded-t-md text-xs transition-colors ${
                    visibleActiveTabId === tab.id
                      ? 'bg-surface text-text-primary border border-border border-b-surface'
                      : 'text-text-secondary hover:text-text-primary bg-panel-light/35'
                  }`}
                  title={tab.agentId}
                >
                  Sub
                </motion.button>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {activeTab.kind === 'main' ? (
        <div className="flex-1 min-h-0 flex flex-col">
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-center px-5">
                <div>
                  <p className="text-sm text-text-primary">Main</p>
                  <p className="text-xs text-text-secondary mt-1">
                    Enter a prompt to start a root-agent run.
                  </p>
                </div>
              </div>
            ) : (
              messages.map(message => (
                <div
                  key={message.id}
                  className={`rounded-lg border p-3 ${
                    message.role === 'user'
                      ? 'bg-accent-blue/10 border-accent-blue/25'
                      : 'bg-surface border-border'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] uppercase tracking-wide text-text-secondary">
                      {message.role === 'user' ? 'You' : 'Root Agent'}
                    </span>
                    {message.status && (
                      <span className="text-[10px] text-text-secondary uppercase">
                        {message.status}
                      </span>
                    )}
                  </div>
                  <pre className={`text-xs whitespace-pre-wrap font-sans leading-relaxed ${
                    message.role === 'assistant' && message.status === 'success'
                      ? 'text-text-primary'
                      : message.role === 'assistant'
                        ? 'text-text-secondary'
                        : 'text-text-primary'
                  }`}>
                    {message.content}
                  </pre>
                </div>
              ))
            )}
            {activeRun && !['success', 'failed', 'stopped'].includes(activeRun.status) && (
              <div className="text-xs text-text-secondary px-1">
                Root agent is {activeRun.status}...
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-3 border-t border-border">
            <div className="flex gap-2 items-end">
              <textarea
                value={input}
                disabled={isInferenceActive}
                onChange={event => setInput(event.target.value)}
                onKeyDown={event => {
                  if (!shouldSubmitChatInput(event)) return
                  event.preventDefault()
                  submit()
                }}
                placeholder={isInferenceActive ? 'Agent is running...' : 'Send a prompt...'}
                rows={2}
                className="flex-1 resize-none bg-surface rounded-md px-3 py-2 text-xs text-text-primary
                           placeholder-text-secondary/40 border border-border disabled:opacity-50
                           focus:outline-none focus:border-accent-blue/60"
              />
              <button
                onClick={() => isInferenceActive ? void stopActiveRun() : submit()}
                disabled={!isInferenceActive && !input.trim()}
                className={`w-9 h-9 shrink-0 rounded-md text-sm transition-colors ${
                  isInferenceActive
                    ? 'bg-accent-red/20 text-accent-red hover:bg-accent-red/30'
                    : 'bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30 disabled:opacity-40'
                }`}
                title={isInferenceActive ? 'Stop' : 'Send'}
              >
                {isInferenceActive ? '■' : '↑'}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-4">
          {selectedAgent ? (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-text-primary">Sub-agent</h3>
                <p className="text-xs font-mono text-text-secondary mt-1">
                  {selectedAgent.id}
                </p>
              </div>
              <div>
                <span className="text-xs text-text-secondary">Task</span>
                <p className="text-sm text-text-primary mt-1">{selectedAgent.task}</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <InfoPill label="Status" value={selectedAgent.status} />
                <InfoPill label="Parent" value={selectedAgent.parentId} />
              </div>
              <p className="text-xs text-text-secondary leading-relaxed">
                Placeholder chat for this sub-agent. A transcript will appear here
                once sub-agent message telemetry exists; for now this tab mirrors
                live execution state truthfully.
              </p>
            </div>
          ) : (
            <p className="text-xs text-text-secondary">Sub-agent finished.</p>
          )}
        </div>
      )}
    </aside>
  )
}

function InfoPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-surface p-2">
      <span className="block text-[10px] uppercase tracking-wide text-text-secondary">
        {label}
      </span>
      <span className="block text-xs text-text-primary mt-1 truncate">
        {value}
      </span>
    </div>
  )
}
