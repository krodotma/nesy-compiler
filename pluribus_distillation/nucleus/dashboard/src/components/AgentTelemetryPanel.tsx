/**
 * AgentTelemetryPanel.tsx - Real-time agent debugging telemetry
 *
 * Shows:
 * - Agent activity feed (debugging, analysis, code fixes)
 * - Telemetry error aggregation
 * - Agent-specific event filtering
 * - Code modification timeline
 * - NDJSON raw event viewer
 */

import { component$, useSignal, useComputed$, $ } from '@builder.io/qwik'
import type { BusEvent } from '../lib/state/types'
import { Button } from './ui/Button'
import { Card } from './ui/Card'
import { NeonTitle, NeonBadge } from './ui/NeonTitle'
import { FreshnessBadge } from './ui/FreshnessBadge'

// M3 Components - AgentTelemetryPanel
import '@material/web/ripple/ripple.js'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────
// ... existing types ...
export interface AgentTelemetryPanelProps {
  events: BusEvent[]
  maxHeight?: string
}

export interface AgentActivity {
  actor: string
  activityType: 'debugging' | 'analysis' | 'code_fix' | 'verification' | 'inference' | 'coordination' | 'error'
  summary: string
  timestamp: string
  topic: string
  details?: Record<string, unknown>
  severity: 'ok' | 'info' | 'warn' | 'error'
}

// ... existing classification logic ...
const ACTIVITY_ICONS: Record<string, string> = {
  debugging: '🔧',
  analysis: '🔍',
  code_fix: '✏️',
  verification: '✅',
  inference: '💭',
  coordination: '🤝',
  error: '❌',
}

const ACTOR_ICONS: Record<string, string> = {
  'claude': '🟣',
  'claude-opus': '🟣',
  'codex': '🔵',
  'codex-cli': '🔵',
  'gemini': '🟢',
  'gemini-cli': '🟢',
  'dashboard': '📊',
  'dashboard-telemetry': '📡',
  'plurichat': '💬',
  'strp-worker': '👷',
  'dialogosd': '🗣️',
  'vps-session': '🔄',
  'catalog-daemon': '📚',
  'browser-daemon': '🌐',
  'default': '🤖',
}

function classifyActivity(event: BusEvent): AgentActivity | null {
  const topic = event.topic
  const data = event.data as Record<string, unknown> | undefined
  const actor = event.actor || 'unknown'

  // Telemetry events (client debugging)
  if (topic.startsWith('telemetry.client.')) {
    const errorType = topic.replace('telemetry.client.', '')
    return {
      actor,
      activityType: event.level === 'error' ? 'error' : 'debugging',
      summary: `Client ${errorType}: ${(data?.message as string) || (data?.error as string) || 'telemetry event'}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: event.level === 'error' ? 'error' : 'info',
    }
  }

  // Browser automation / CUA
  if (topic.startsWith('browser.')) {
    return {
      actor,
      activityType: 'debugging',
      summary: `Browser action: ${topic.replace('browser.', '')}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: event.level as AgentActivity['severity'] || 'info',
    }
  }

  // OITERATE operator
  if (topic.startsWith('oiterate.')) {
    return {
      actor,
      activityType: 'coordination',
      summary: `OITERATE: ${(data?.action as string) || topic.replace('oiterate.', '')}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: 'info',
    }
  }

  // Dialogos inference
  if (topic === 'dialogos.submit') {
    const providers = Array.isArray(data?.providers) ? (data.providers as string[]).join(', ') : 'auto'
    return {
      actor,
      activityType: 'inference',
      summary: `Inference request → [${providers}]`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: 'info',
    }
  }

  if (topic === 'dialogos.cell.end') {
    const ok = Boolean(data?.ok)
    const status = String(data?.status || (ok ? 'success' : 'error'))
    return {
      actor,
      activityType: 'inference',
      summary: `Inference ${status}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: ok ? 'ok' : 'error',
    }
  }

  // Lens routing decisions
  if (topic === 'plurichat.lens.decision') {
    const depth = String(data?.depth || '')
    const lane = String(data?.lane || '')
    const topo = String(data?.topology || 'single')
    return {
      actor,
      activityType: 'analysis',
      summary: `LENS route: depth=${depth} lane=${lane} topo=${topo}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: 'info',
    }
  }

  // STRp topology actions
  if (topic.startsWith('strp.')) {
    const action = topic.replace('strp.', '')
    const goal = String(data?.goal || data?.goal_summary || '')
    return {
      actor,
      activityType: action.includes('verify') || action.includes('test') ? 'verification' : 'coordination',
      summary: `STRp ${action}${goal ? `: ${goal.slice(0, 60)}` : ''}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: event.level === 'error' ? 'error' : 'info',
    }
  }

  // Verification events
  if (topic.startsWith('verify.') || topic.startsWith('tests.')) {
    const cmd = String(data?.cmd || data?.summary || topic)
    return {
      actor,
      activityType: 'verification',
      summary: `Verify: ${cmd.slice(0, 80)}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: event.level === 'error' ? 'error' : event.level === 'warn' ? 'warn' : 'ok',
    }
  }

  // A2A negotiation
  if (topic.startsWith('a2a.')) {
    const action = topic.replace('a2a.', '')
    return {
      actor,
      activityType: 'coordination',
      summary: `A2A ${action}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: 'info',
    }
  }

  // MCP tool calls
  if (topic === 'mcp.host.call' || topic === 'mcp.host.response') {
    const tool = String(data?.tool || '')
    const server = String(data?.server || '')
    const ok = topic === 'mcp.host.response' ? Boolean(data?.ok) : true
    return {
      actor,
      activityType: 'analysis',
      summary: `MCP ${server}.${tool}${!ok ? ' (failed)' : ''}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: ok ? 'info' : 'error',
    }
  }

  // Git operations (code fixes)
  if (topic.startsWith('git.')) {
    return {
      actor,
      activityType: 'code_fix',
      summary: `Git: ${topic.replace('git.', '')}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: 'info',
    }
  }

  // Pluribus check / health
  if (topic.startsWith('pluribus.check.')) {
    return {
      actor,
      activityType: 'verification',
      summary: `Health check: ${topic.replace('pluribus.check.', '')}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: event.level as AgentActivity['severity'] || 'ok',
    }
  }

  // InferCell actions
  if (topic.startsWith('infercell.')) {
    return {
      actor,
      activityType: 'analysis',
      summary: `InferCell: ${topic.replace('infercell.', '')}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: 'info',
    }
  }

  // Service lifecycle
  if (topic.startsWith('service.')) {
    return {
      actor,
      activityType: 'coordination',
      summary: `Service: ${topic.replace('service.', '')}`,
      timestamp: event.iso || '',
      topic,
      details: data,
      severity: 'info',
    }
  }

  return null
}

export const AgentTelemetryPanel = component$<AgentTelemetryPanelProps>(({
  events,
  maxHeight = '400px'
}) => {
  const selectedActor = useSignal<string | null>(null)
  const selectedActivityType = useSignal<string | null>(null)
  const showNdjson = useSignal(false)
  const expandedActivity = useSignal<number | null>(null)

  const activities = useComputed$(() => {
    const result: AgentActivity[] = []
    for (const event of events.slice(-500)) {
      const activity = classifyActivity(event)
      if (activity) result.push(activity)
    }
    return result
  })

  const filteredActivities = useComputed$(() => {
    let filtered = activities.value
    if (selectedActor.value) filtered = filtered.filter(a => a.actor === selectedActor.value)
    if (selectedActivityType.value) filtered = filtered.filter(a => a.activityType === selectedActivityType.value)
    return filtered.slice(-100).reverse()
  })

  const actorStats = useComputed$(() => {
    const stats = new Map<string, { count: number; errors: number; lastActivity: string }>()
    for (const activity of activities.value) {
      const existing = stats.get(activity.actor) || { count: 0, errors: 0, lastActivity: '' }
      existing.count++
      if (activity.severity === 'error') existing.errors++
      if (activity.timestamp > existing.lastActivity) existing.lastActivity = activity.timestamp
      stats.set(activity.actor, existing)
    }
    return Array.from(stats.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 8)
  })

  const activityTypeStats = useComputed$(() => {
    const stats = new Map<string, number>()
    for (const activity of activities.value) {
      stats.set(activity.activityType, (stats.get(activity.activityType) || 0) + 1)
    }
    return Array.from(stats.entries()).sort((a, b) => b[1] - a[1])
  })

  const errorRate = useComputed$(() => {
    const recent = activities.value.slice(-100)
    const errors = recent.filter(a => a.severity === 'error').length
    return recent.length > 0 ? Math.round((errors / recent.length) * 100) : 0
  })

  const lastActivityTs = useComputed$(() => {
    if (activities.value.length === 0) return null
    return activities.value[0]?.timestamp || null
  })

  const getActorIcon = (actor: string) => {
    for (const [key, icon] of Object.entries(ACTOR_ICONS)) {
      if (actor.toLowerCase().includes(key)) return icon
    }
    return ACTOR_ICONS.default
  }

  return (
    <Card variant="outlined" padding="p-0" class="overflow-hidden">
      {/* Header */}
      <div class="p-3 border-b border-md-outline/30 flex items-center justify-between bg-md-surface-container-low">
        <div class="flex items-center gap-2">
          <NeonTitle level="span" color="cyan" size="xs" animation="flicker">Agent Telemetry</NeonTitle>
          <NeonBadge color="cyan" glow>{activities.value.length} activities</NeonBadge>
          {errorRate.value > 0 && (
            <NeonBadge
              color={errorRate.value > 20 ? 'rose' : errorRate.value > 10 ? 'amber' : 'emerald'}
              glow
              pulse={errorRate.value > 20}
            >
              {errorRate.value}% errors
            </NeonBadge>
          )}
          <FreshnessBadge
            timestamp={lastActivityTs.value}
            source="bus"
            ttlFresh={30}
            ttlRecent={120}
            ttlStale={600}
          />
        </div>
        <Button
          variant="secondary"
          class="h-7 text-[10px] px-3"
          onClick$={() => showNdjson.value = !showNdjson.value}
        >
          {showNdjson.value ? 'Show Feed' : 'Raw NDJSON'}
        </Button>
      </div>

      {/* Actor Filter Badges */}
      <div class="px-3 py-2 border-b border-md-outline/20 flex flex-wrap gap-1 bg-md-surface-container-lowest">
        <Button
          variant={!selectedActor.value ? 'tonal' : 'text'}
          class="h-6 text-[10px] px-2"
          onClick$={() => selectedActor.value = null}
        >
          All Agents
        </Button>
        {actorStats.value.map(([actor, stats]) => (
          <Button
            key={actor}
            variant={selectedActor.value === actor ? 'primary' : 'secondary'}
            class={`h-6 text-[10px] px-2 ${stats.errors > 0 && selectedActor.value !== actor ? 'text-md-error border-md-error/30' : ''}`}
            onClick$={() => selectedActor.value = selectedActor.value === actor ? null : actor}
          >
            <span class="mr-1">{getActorIcon(actor)}</span>
            <span class="mono">{actor}</span>
            <span class="ml-1 opacity-70">{stats.count}</span>
            {stats.errors > 0 && <span class="ml-0.5">⚠️</span>}
          </Button>
        ))}
      </div>

      {/* Activity Type Filter */}
      <div class="px-3 py-2 border-b border-md-outline/20 flex flex-wrap gap-1 bg-md-surface-container-lowest">
        {activityTypeStats.value.map(([type, count]) => (
          <Button
            key={type}
            variant={selectedActivityType.value === type ? 'tonal' : 'text'}
            class="h-6 text-[10px] px-2"
            onClick$={() => selectedActivityType.value = selectedActivityType.value === type ? null : type}
          >
            <span class="mr-1">{ACTIVITY_ICONS[type] || '📦'}</span>
            <span>{type}</span>
            <span class="ml-1 opacity-60">{count}</span>
          </Button>
        ))}
      </div>

      {/* Activity Feed or NDJSON View */}
      <div class="overflow-auto scrollbar-thin" style={{ maxHeight }}>
        {showNdjson.value ? (
          <div class="p-3 font-mono text-[10px] bg-md-surface-container-highest/30">
            {filteredActivities.value.slice(0, 50).map((activity, i) => (
              <div
                key={i}
                class={`py-1 border-b border-md-outline/10 ${
                  activity.severity === 'error' ? 'text-md-error' :
                  activity.severity === 'warn' ? 'text-md-warning' :
                  'text-md-on-surface-variant'
                }`}
              >
                {JSON.stringify({
                  ts: activity.timestamp,
                  actor: activity.actor,
                  type: activity.activityType,
                  topic: activity.topic,
                  ...activity.details
                })}
              </div>
            ))}
          </div>
        ) : (
          <div class="p-2 space-y-1">
            {filteredActivities.value.length === 0 ? (
              <div class="p-8 text-center text-md-on-surface-variant/60 text-sm italic">
                No agent activities detected yet.
              </div>
            ) : (
              filteredActivities.value.map((activity, i) => {
                const isExpanded = expandedActivity.value === i
                const ts = activity.timestamp ? activity.timestamp.slice(11, 19) : '--:--:--'

                return (
                  <div
                    key={i}
                    class={`p-2 rounded-lg border transition-all cursor-pointer relative overflow-hidden ${
                      isExpanded ? 'bg-md-surface-container' : 'hover:bg-md-surface-container-low'
                    } ${
                      activity.severity === 'error' ? 'border-md-error/30 bg-md-error/5' :
                      activity.severity === 'warn' ? 'border-md-warning/30 bg-md-warning/5' :
                      activity.severity === 'ok' ? 'border-md-primary/20 bg-md-primary/5' :
                      'border-md-outline/10 bg-md-surface/30'
                    }`}
                    onClick$={() => expandedActivity.value = isExpanded ? null : i}
                  >
                    <md-ripple></md-ripple>
                    <div class="flex items-start gap-3">
                      <span class="text-[10px] text-md-on-surface-variant/60 font-mono flex-shrink-0 mt-0.5">{ts}</span>
                      <span class="text-base flex-shrink-0">{ACTIVITY_ICONS[activity.activityType] || '📦'}</span>

                      <div class="flex-1 min-w-0">
                        <div class="flex items-center gap-2 flex-wrap mb-1">
                          <span class="text-[10px] font-bold text-md-primary uppercase tracking-tight flex items-center gap-1">
                            {getActorIcon(activity.actor)} {activity.actor}
                          </span>
                          <span class="text-[9px] px-1.5 py-0.5 rounded bg-md-secondary-container text-md-on-secondary-container font-medium">
                            {activity.activityType}
                          </span>
                        </div>
                        <div class={`text-xs font-medium ${
                          activity.severity === 'error' ? 'text-md-error' :
                          activity.severity === 'warn' ? 'text-md-warning' :
                          'text-md-on-surface'
                        }`}>
                          {activity.summary}
                        </div>

                        {isExpanded && activity.details && (
                          <div class="mt-2 p-2 rounded-md bg-md-surface-container-highest/50 border border-md-outline/10">
                            <div class="text-[9px] text-md-on-surface-variant mb-1 font-mono uppercase tracking-tighter">Topic: {activity.topic}</div>
                            <pre class="text-[10px] text-md-on-surface-variant/80 font-mono whitespace-pre-wrap break-all max-h-48 overflow-auto">
                              {JSON.stringify(activity.details, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>

                      <md-icon class="text-md-on-surface-variant/40 text-xs">
                        {isExpanded ? 'expand_less' : 'expand_more'}
                      </md-icon>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        )}
      </div>
    </Card>
  )
})

export interface TelemetryBadgeProps {
  events: BusEvent[]
}

export const TelemetryBadge = component$<TelemetryBadgeProps>(({ events }) => {
  const stats = useComputed$(() => {
    let errors = 0
    let activities = 0
    const actors = new Set<string>()

    for (const event of events.slice(-200)) {
      const activity = classifyActivity(event)
      if (activity) {
        activities++
        actors.add(activity.actor)
        if (activity.severity === 'error') errors++
      }
    }
    return { errors, activities, actorCount: actors.size }
  })

  return (
    <div class="flex items-center gap-1.5 text-[10px] font-medium">
      <span class="px-2 py-0.5 rounded-full bg-md-primary/10 text-md-primary border border-md-primary/20">
        🤖 {stats.value.actorCount}
      </span>
      <span class="px-2 py-0.5 rounded-full bg-md-secondary/10 text-md-secondary border border-md-secondary/20">
        ⚡ {stats.value.activities}
      </span>
      {stats.value.errors > 0 && (
        <span class="px-2 py-0.5 rounded-full bg-md-error/10 text-md-error border border-md-error/20">
          ❌ {stats.value.errors}
        </span>
      )}
    </div>
  )
})
