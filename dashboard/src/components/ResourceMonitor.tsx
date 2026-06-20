import { useState, useEffect, useRef } from 'react'
import { BACKEND_BASE_URL } from '../config'
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'

interface ResourceData {
  time: string
  vram_mb: number
  cpu_percent: number
  ram_used_gb: number
  ram_total_gb: number
  gpu_percent: number
}

interface ResourceMonitorProps {
  pollIntervalMs?: number
}

export default function ResourceMonitor({ pollIntervalMs = 2000 }: ResourceMonitorProps) {
  const [history, setHistory] = useState<ResourceData[]>([])
  const [current, setCurrent] = useState<ResourceData | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval>>(undefined)

  useEffect(() => {
    const fetchResources = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE_URL}/api/resources`)
        if (!res.ok) return
        const data = await res.json()
        const entry: ResourceData = {
          time: new Date().toLocaleTimeString('en-US', {
            hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
          }),
          vram_mb: data.vram_mb ?? 0,
          cpu_percent: data.cpu_percent ?? 0,
          ram_used_gb: data.ram_used_gb ?? 0,
          ram_total_gb: data.ram_total_gb ?? 0,
          gpu_percent: data.gpu_percent ?? 0,
        }
        setCurrent(entry)
        setHistory(prev => {
          const next = [...prev, entry]
          // Keep last 60 data points (2 minutes at 2s interval)
          return next.slice(-60)
        })
      } catch {
        // Silently fail on fetch errors
      }
    }

    fetchResources()
    timerRef.current = setInterval(fetchResources, pollIntervalMs)

    return () => clearInterval(timerRef.current)
  }, [pollIntervalMs])

  const chartColors = {
    vram: '#eab308',    // yellow
    cpu: '#3b82f6',     // blue
    ram: '#a855f7',     // purple
    gpu: '#22c55e',     // green
  }

  // Warning/critical thresholds for VRAM (from config defaults)
  const vramWarning = 6500
  const vramCritical = 7000

  if (!current) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-sm text-text-secondary">Loading resource data...</p>
      </div>
    )
  }

  return (
    <div className="h-full p-4 overflow-y-auto">
      {/* Current values header */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <StatCard
          label="VRAM"
          value={`${current.vram_mb} MB`}
          color={chartColors.vram}
          percent={Math.min(100, (current.vram_mb / 8192) * 100)}
        />
        <StatCard
          label="CPU"
          value={`${current.cpu_percent.toFixed(1)}%`}
          color={chartColors.cpu}
          percent={current.cpu_percent}
        />
        <StatCard
          label="RAM"
          value={`${current.ram_used_gb} / ${current.ram_total_gb} GB`}
          color={chartColors.ram}
          percent={current.ram_total_gb > 0
            ? (current.ram_used_gb / current.ram_total_gb) * 100
            : 0}
        />
        <StatCard
          label="GPU"
          value={`${current.gpu_percent.toFixed(0)}%`}
          color={chartColors.gpu}
          percent={current.gpu_percent}
        />
      </div>

      {/* Charts grid 2x2 */}
      <div className="grid grid-cols-2 gap-4">
        {/* VRAM Chart */}
        <ChartPanel title="VRAM Usage (MB)" color={chartColors.vram}>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0, 8192]} />
              <Tooltip contentStyle={tooltipStyle} />
              <ReferenceLine y={vramWarning} stroke="#eab308" strokeDasharray="4 4"
                label={{ value: 'Warning', fill: '#eab308', fontSize: 10 }} />
              <ReferenceLine y={vramCritical} stroke="#ef4444" strokeDasharray="4 4"
                label={{ value: 'Critical', fill: '#ef4444', fontSize: 10 }} />
              <Line
                type="monotone"
                dataKey="vram_mb"
                stroke={chartColors.vram}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                name="VRAM"
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        {/* CPU Chart */}
        <ChartPanel title="CPU Usage (%)" color={chartColors.cpu}>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0, 100]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area
                type="monotone"
                dataKey="cpu_percent"
                stroke={chartColors.cpu}
                fill={chartColors.cpu}
                fillOpacity={0.15}
                strokeWidth={2}
                isAnimationActive={false}
                name="CPU"
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartPanel>

        {/* RAM Chart */}
        <ChartPanel title="RAM Usage (GB)" color={chartColors.ram}>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }}
                domain={[0, current.ram_total_gb || 'auto']} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area
                type="monotone"
                dataKey="ram_used_gb"
                stroke={chartColors.ram}
                fill={chartColors.ram}
                fillOpacity={0.15}
                strokeWidth={2}
                isAnimationActive={false}
                name="RAM Used"
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartPanel>

        {/* GPU Chart */}
        <ChartPanel title="GPU Utilization (%)" color={chartColors.gpu}>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0, 100]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line
                type="monotone"
                dataKey="gpu_percent"
                stroke={chartColors.gpu}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                name="GPU"
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helper components
// ---------------------------------------------------------------------------

const tooltipStyle = {
  backgroundColor: '#1e1e2e',
  border: '1px solid #3a3a4a',
  borderRadius: '6px',
  fontSize: '12px',
  color: '#e5e7eb',
}

function StatCard({ label, value, color, percent }: {
  label: string
  value: string
  color: string
  percent: number
}) {
  return (
    <div className="bg-panel rounded-lg p-3 border border-border">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-text-secondary">{label}</span>
        <span className="text-sm font-mono font-medium" style={{ color }}>
          {value}
        </span>
      </div>
      <div className="w-full h-1.5 bg-surface rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.min(100, percent)}%`,
            backgroundColor: color,
          }}
        />
      </div>
    </div>
  )
}

function ChartPanel({ title, color, children }: {
  title: string
  color: string
  children: React.ReactNode
}) {
  return (
    <div className="bg-panel rounded-lg p-3 border border-border">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
        <h3 className="text-xs font-medium text-text-primary">{title}</h3>
      </div>
      {children}
    </div>
  )
}
