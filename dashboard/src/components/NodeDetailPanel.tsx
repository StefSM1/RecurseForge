import { motion, AnimatePresence } from 'framer-motion'
import type { GraphEntity } from '../types/events'

interface NodeDetailPanelProps {
  entity: GraphEntity | null
  onClose: () => void
}

function Field({ label, value, mono = false }: {
  label: string
  value: React.ReactNode
  mono?: boolean
}) {
  return (
    <div className="mb-3">
      <span className="text-xs text-text-secondary">{label}</span>
      <div className={`text-sm text-text-primary ${mono ? 'font-mono' : ''}`}>{value}</div>
    </div>
  )
}

function Preview({ label, value }: { label: string; value?: string }) {
  if (!value) return null
  return (
    <div className="mb-3">
      <span className="text-xs text-text-secondary">{label}</span>
      <pre className="text-xs text-text-primary bg-surface rounded p-2 mt-1 overflow-x-auto whitespace-pre-wrap font-mono">
        {value.slice(0, 2000)}{value.length > 2000 ? '...' : ''}
      </pre>
    </div>
  )
}

export default function NodeDetailPanel({ entity, onClose }: NodeDetailPanelProps) {
  return (
    <AnimatePresence>
      {entity && (
        <motion.div
          initial={{ x: 300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 300, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          className="absolute right-0 top-0 bottom-0 w-80 bg-panel border-l border-border overflow-y-auto z-10"
        >
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-text-primary capitalize">
                {entity.kind} details
              </h3>
              <button
                onClick={onClose}
                className="text-text-secondary hover:text-text-primary w-6 h-6 flex items-center justify-center rounded"
              >x</button>
            </div>

            {entity.kind === 'agent' && (
              <>
                <Field label="Node ID" value={entity.value.id} mono />
                <Field label="Status" value={entity.value.status} />
                <Field label="Task" value={entity.value.task} />
                <Field label="Spawned" value={new Date(entity.value.spawnTime).toLocaleTimeString()} />
                {entity.value.completeTime && (
                  <Field label="Duration" value={`${((entity.value.completeTime - entity.value.spawnTime) / 1000).toFixed(1)}s`} />
                )}
                {entity.value.attempts !== undefined && <Field label="Attempts" value={entity.value.attempts} mono />}
                <Preview label="Result" value={entity.value.result} />
              </>
            )}

            {entity.kind === 'sandbox' && (
              <>
                <Field label="Execution ID" value={entity.value.id} mono />
                <Field label="Owner" value={entity.value.ownerId} mono />
                <Field label="Status" value={entity.value.status} />
                <Field label="Attempt" value={entity.value.attempt} mono />
                <Field label="Trigger" value={entity.value.trigger.replace('_', ' ')} />
                <Field label="Started" value={new Date(entity.value.startTime).toLocaleTimeString()} />
                {entity.value.durationMs !== undefined && <Field label="Duration" value={`${entity.value.durationMs}ms`} mono />}
                {entity.value.exitCode !== undefined && <Field label="Exit code" value={entity.value.exitCode} mono />}
                <Preview label="Code preview" value={entity.value.codePreview} />
                <Preview label="stdout" value={entity.value.stdout} />
                <Preview label="stderr" value={entity.value.stderr} />
              </>
            )}

            {entity.kind === 'correction' && (
              <>
                <Field label="Correction ID" value={entity.value.id} mono />
                <Field label="Owner" value={entity.value.ownerId} mono />
                <Field label="Strategy" value={entity.value.strategy === 'textgrad' ? 'TextGrad' : 'LLM Retry'} />
                <Field label="Status" value={entity.value.status} />
                <Field label="Phase" value={entity.value.phase.replace('_', ' ')} />
                <Field label="Failed execution" value={entity.value.failedExecutionId} mono />
                {entity.value.iterations !== undefined && <Field label="Iteration" value={entity.value.iterations} mono />}
                {entity.value.severity !== undefined && <Field label="Severity" value={entity.value.severity.toFixed(2)} mono />}
                {entity.value.numMutations !== undefined && <Field label="Mutations" value={entity.value.numMutations} mono />}
                {entity.value.mutations.map((mutation, index) => (
                  <Preview
                    key={`${mutation.line}-${index}`}
                    label={`Line ${mutation.line || '?'} correction`}
                    value={`${mutation.cause}\nFix: ${mutation.suggestion}`}
                  />
                ))}
                <Preview label="Correction error" value={entity.value.error} />
              </>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
