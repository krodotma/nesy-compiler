/**
 * EventVisualization.tsx - Rich event visualization components
 *
 * Provides:
 * - EventSearchBox: Symbolic pattern matching with regex, glob, LTL hints
 * - TimelineSparkline: Event frequency visualization over time buckets
 * - EventFlowmap: Directed graph of actor→topic relationships
 * - EnrichedEventCard: Card with LTL annotations, vector indicators, KG links
 * - EventStatsBadges: Semantic enrichment badges (impact, domain, temporal)
 */

import { component$, useSignal, useComputed$, $ } from '@builder.io/qwik'
import { useTracking } from '../lib/telemetry/use-tracking'
import type { BusEvent } from '../lib/state/types'
import { Button } from './ui/Button'
import { Input } from './ui/Input'

// M3 Components - EventVisualization
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/button/text-button.js';
import { Card } from './ui/Card'
import { NeonTitle, NeonBadge, NeonSectionHeader } from './ui/NeonTitle'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface EventSearchProps {
  onSearch$: (pattern: string, mode: SearchMode) => void
  placeholder?: string
}

export type SearchMode = 'glob' | 'regex' | 'ltl' | 'actor' | 'topic'

export interface TimelineSparklineProps {
  events: BusEvent[]
  buckets?: number
  height?: number
  width?: number
}

export interface EventFlowmapProps {
  events: BusEvent[]
  maxNodes?: number
  height?: number
}

export interface EnrichedEventCardProps {
  event: BusEvent
  index: number
  onExpand$?: (event: BusEvent) => void
  showLTL?: boolean
  showVectors?: boolean
  showKG?: boolean
}

// ─────────────────────────────────────────────────────────────────────────────
// LTL Pattern Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Infer LTL-style temporal annotations from event sequences
 * ◇ = eventually (diamond), □ = always (box), ○ = next, U = until
 */
function inferLTLPattern(event: BusEvent, recentTopics: string[]): string | null {
  const topic = event.topic

  // Pattern: request → response (◇ response after request)
  if (topic.endsWith('.request')) {
    const base = topic.replace('.request', '')
    const hasResponse = recentTopics.some(t => t === `${base}.response` || t === `${base}.ack`)
    return hasResponse ? `◇ ${base}.response` : `⏳ ${base}.*`
  }

  // Pattern: spawn → complete cycle
  if (topic.includes('.spawn') || topic.includes('.started')) {
    const base = topic.split('.')[0]
    const hasEnd = recentTopics.some(t => t.startsWith(base) && (t.includes('.end') || t.includes('.done')))
    return hasEnd ? `□ ${base}.*` : `○ ${base}.end`
  }

  // Error patterns
  if (event.level === 'error') {
    return '⚠ error state'
  }

  return null
}

/**
 * Extract semantic domain from topic
 */
function extractDomain(topic: string): { domain: string; subdomain: string; action: string } {
  const parts = topic.split('.')
  return {
    domain: parts[0] || 'unknown',
    subdomain: parts[1] || '',
    action: parts.slice(2).join('.') || parts[1] || ''
  }
}

/**
 * Compute vector embedding indicator (mock - real would use actual embeddings)
 */
function computeVectorIndicator(event: BusEvent): { hasEmbedding: boolean; similarity?: number; cluster?: string } {
  // In production, this would check against actual RAG vector store
  const semantic = (event as any).semantic
  const data = event.data as Record<string, unknown> | undefined

  const hasEmbedding = !!(
    semantic?.embedding ||
    data?.vector ||
    data?.embedding ||
    event.kind === 'artifact' ||
    event.topic.includes('rag.') ||
    event.topic.includes('sota.')
  )

  // Mock cluster assignment based on domain
  const { domain } = extractDomain(event.topic)
  const clusterMap: Record<string, string> = {
    dialogos: 'inference',
    strp: 'topology',
    lens: 'routing',
    verify: 'evidence',
    mcp: 'tooling',
    a2a: 'negotiation',
    git: 'versioning',
    rag: 'retrieval',
    sota: 'knowledge',
  }

  return {
    hasEmbedding,
    similarity: hasEmbedding ? Math.random() * 0.3 + 0.7 : undefined,
    cluster: clusterMap[domain] || 'general'
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// EventSearchBox Component
// ─────────────────────────────────────────────────────────────────────────────

export const EventSearchBox = component$<EventSearchProps>(({ onSearch$, placeholder }) => {
  const searchText = useSignal('')
  const searchMode = useSignal<SearchMode>('glob')
  const showHelp = useSignal(false)

  const handleSearch = $(() => {
    onSearch$(searchText.value, searchMode.value)
  })

  const modeIcons: Record<SearchMode, string> = {
    glob: '🔍',
    regex: '⚙️',
    ltl: '◇',
    actor: '👤',
    topic: '📌'
  }

  return (
    <div class="space-y-2">
      <div class="flex gap-2">
        {/* Search Input */}
        <div class="flex-1 relative flex gap-2">
          <Input
            label="Search Events"
            value={searchText.value}
            onInput$={(_, el) => searchText.value = el.value}
            onKeyDown$={(e) => {
              if (e.key === 'Enter') handleSearch()
            }}
            placeholder={placeholder || 'strp.*, /error/i, ◇response'}
            class="flex-1"
          />
          <Button
            variant="text"
            onClick$={() => showHelp.value = !showHelp.value}
            icon={showHelp.value ? 'close' : 'help'}
          />
          <Button
            variant="tonal"
            onClick$={handleSearch}
          >
            Search
          </Button>
        </div>

        {/* Mode Selector */}
        <div class="flex gap-1">
          {(Object.keys(modeIcons) as SearchMode[]).map(mode => (
            <Button
              key={mode}
              variant={searchMode.value === mode ? 'tonal' : 'secondary'}
              onClick$={() => searchMode.value = mode}
              class="px-3 min-w-[40px]"
              title={mode}
            >
              {modeIcons[mode]}
            </Button>
          ))}
        </div>
      </div>

      {/* Help Panel */}
      {showHelp.value && (
        <Card variant="outlined" padding="p-3" class="text-xs space-y-2 bg-muted/20">
          <NeonTitle level="div" color="amber" size="xs">Search Modes:</NeonTitle>
          <div class="grid grid-cols-2 gap-2">
            <div><span class="text-primary">🔍 Glob:</span> strp.*, dialogos.cell.* </div>
            <div><span class="text-primary">⚙️ Regex:</span> /error|warn/i, /strp\.\w+/ </div>
            <div><span class="text-primary">◇ LTL:</span> ◇response, □error, ○spawn </div>
            <div><span class="text-primary">👤 Actor:</span> claude, codex, dashboard </div>
            <div><span class="text-primary">📌 Topic:</span> exact topic prefix match </div>
          </div>
        </Card>
      )}
    </div>
  )
})

// ─────────────────────────────────────────────────────────────────────────────
// TimelineSparkline Component
// ─────────────────────────────────────────────────────────────────────────────

export const TimelineSparkline = component$<TimelineSparklineProps>(({
  events,
  buckets = 60,
  height = 40,
  width = 300
}) => {
  const bucketData = useComputed$(() => {
    if (events.length === 0) return Array(buckets).fill(0)

    // Get time range
    const times = events
      .map(e => new Date(e.iso || 0).getTime())
      .filter(t => t > 0)
      .sort((a, b) => a - b)

    if (times.length === 0) return Array(buckets).fill(0)

    const minTime = times[0]
    const maxTime = times[times.length - 1]
    const range = maxTime - minTime || 1
    const bucketSize = range / buckets

    // Count events per bucket
    const counts = Array(buckets).fill(0)
    for (const time of times) {
      const bucket = Math.min(buckets - 1, Math.floor((time - minTime) / bucketSize))
      counts[bucket]++
    }

    return counts
  })

  const maxCount = useComputed$(() => Math.max(1, ...bucketData.value))

  const errorBuckets = useComputed$(() => {
    if (events.length === 0) return Array(buckets).fill(0)

    const times = events
      .filter(e => e.level === 'error')
      .map(e => new Date(e.iso || 0).getTime())
      .filter(t => t > 0)
      .sort((a, b) => a - b)

    if (times.length === 0) return Array(buckets).fill(0)

    const allTimes = events
      .map(e => new Date(e.iso || 0).getTime())
      .filter(t => t > 0)
      .sort((a, b) => a - b)

    const minTime = allTimes[0]
    const maxTime = allTimes[allTimes.length - 1]
    const range = maxTime - minTime || 1
    const bucketSize = range / buckets

    const counts = Array(buckets).fill(0)
    for (const time of times) {
      const bucket = Math.min(buckets - 1, Math.floor((time - minTime) / bucketSize))
      counts[bucket]++
    }

    return counts
  })

  const barWidth = width / buckets - 1

  return (
    <Card padding="p-3">
      <div class="flex items-center justify-between mb-2">
        <NeonTitle level="span" color="cyan" size="xs">EVENT TIMELINE</NeonTitle>
        <div class="flex items-center gap-2 text-[10px]">
          <span class="flex items-center gap-1">
            <span class="w-2 h-2 rounded-sm bg-primary/60"></span>
            events
          </span>
          <span class="flex items-center gap-1">
            <span class="w-2 h-2 rounded-sm bg-red-500/60"></span>
            errors
          </span>
        </div>
      </div>

      <svg width={width} height={height} class="overflow-visible">
        {/* Background grid */}
        <defs>
          <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="0.5"/>
          </pattern>
        </defs>
        <rect width={width} height={height} fill="url(#grid)" rx="4" />

        {/* Event bars */}
        {bucketData.value.map((count, i) => {
          const barHeight = (count / maxCount.value) * (height - 4)
          const errorHeight = (errorBuckets.value[i] / maxCount.value) * (height - 4)
          const x = i * (barWidth + 1)

          return (
            <g key={i}>
              {/* Main event bar */}
              <rect
                x={x}
                y={height - barHeight - 2}
                width={barWidth}
                height={barHeight}
                fill="rgba(var(--primary-rgb), 0.4)"
                rx="1"
              />
              {/* Error overlay */}
              {errorHeight > 0 && (
                <rect
                  x={x}
                  y={height - errorHeight - 2}
                  width={barWidth}
                  height={errorHeight}
                  fill="rgba(239, 68, 68, 0.6)"
                  rx="1"
                />
              )}
            </g>
          )
        })}

        {/* Baseline */}
        <line x1="0" y1={height - 1} x2={width} y2={height - 1} stroke="rgba(255,255,255,0.1)" />
      </svg>

      <div class="flex justify-between text-[10px] text-muted-foreground mt-1">
        <span>oldest</span>
        <span>{events.length} events</span>
        <span>newest</span>
      </div>
    </Card>
  )
})

// ─────────────────────────────────────────────────────────────────────────────
// EventFlowmap Component - Knowledge Graph Visualization
// ─────────────────────────────────────────────────────────────────────────────

export const EventFlowmap = component$<EventFlowmapProps>(({
  events,
  maxNodes = 20,
  height = 200
}) => {
  const graphData = useComputed$(() => {
    // Build actor → topic edges with counts
    const edges = new Map<string, { from: string; to: string; count: number; kind: string }>()
    const actorCounts = new Map<string, number>()
    const topicCounts = new Map<string, number>()

    for (const event of events.slice(-500)) {
      const actor = event.actor || 'unknown'
      const { domain, subdomain } = extractDomain(event.topic)
      const topicNode = subdomain ? `${domain}.${subdomain}` : domain

      actorCounts.set(actor, (actorCounts.get(actor) || 0) + 1)
      topicCounts.set(topicNode, (topicCounts.get(topicNode) || 0) + 1)

      const edgeKey = `${actor}→${topicNode}`
      const existing = edges.get(edgeKey)
      if (existing) {
        existing.count++
      } else {
        edges.set(edgeKey, { from: actor, to: topicNode, count: 1, kind: event.kind || 'event' })
      }
    }

    // Get top actors and topics
    const topActors = Array.from(actorCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, Math.floor(maxNodes / 2))
      .map(([name, count]) => ({ name, count, type: 'actor' as const }))

    const topTopics = Array.from(topicCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, Math.floor(maxNodes / 2))
      .map(([name, count]) => ({ name, count, type: 'topic' as const }))

    // Filter edges to only include top nodes
    const topActorNames = new Set(topActors.map(a => a.name))
    const topTopicNames = new Set(topTopics.map(t => t.name))

    const filteredEdges = Array.from(edges.values())
      .filter(e => topActorNames.has(e.from) && topTopicNames.has(e.to))
      .sort((a, b) => b.count - a.count)
      .slice(0, 30)

    return { actors: topActors, topics: topTopics, edges: filteredEdges }
  })

  // Layout: actors on left, topics on right
  const actorY = (i: number) => 30 + i * (height - 40) / Math.max(1, graphData.value.actors.length - 1)
  const topicY = (i: number) => 30 + i * (height - 40) / Math.max(1, graphData.value.topics.length - 1)

  return (
    <Card padding="p-3">
      <div class="flex items-center justify-between mb-2">
        <NeonTitle level="span" color="purple" size="xs" animation="flicker">EVENT FLOWMAP (KG)</NeonTitle>
        <div class="flex items-center gap-2 text-[10px]">
          <span class="flex items-center gap-1">
            <span class="w-2 h-2 rounded-full bg-cyan-500"></span>
            actors
          </span>
          <span class="flex items-center gap-1">
            <span class="w-2 h-2 rounded-full bg-purple-500"></span>
            topics
          </span>
        </div>
      </div>

      <svg width="100%" height={height} viewBox={`0 0 400 ${height}`} class="overflow-visible">
        {/* Edges */}
        {graphData.value.edges.map((edge, i) => {
          const fromIdx = graphData.value.actors.findIndex(a => a.name === edge.from)
          const toIdx = graphData.value.topics.findIndex(t => t.name === edge.to)
          if (fromIdx === -1 || toIdx === -1) return null

          const x1 = 100
          const y1 = actorY(fromIdx)
          const x2 = 300
          const y2 = topicY(toIdx)

          const opacity = Math.min(0.8, 0.1 + edge.count * 0.1)
          const strokeWidth = Math.min(3, 0.5 + edge.count * 0.2)

          return (
            <path
              key={i}
              d={`M ${x1} ${y1} C ${200} ${y1}, ${200} ${y2}, ${x2} ${y2}`}
              fill="none"
              stroke={edge.kind === 'error' ? 'rgba(239,68,68,0.5)' : 'rgba(168,85,247,0.3)'}
              stroke-width={strokeWidth}
              opacity={opacity}
            />
          )
        })}

        {/* Actor nodes (left) */}
        {graphData.value.actors.map((actor, i) => (
          <g key={`actor-${actor.name}`} transform={`translate(10, ${actorY(i)})`}>
            <circle r="6" fill="rgba(6,182,212,0.8)" stroke="rgba(6,182,212,1)" stroke-width="1" />
            <text x="14" y="4" class="text-[9px] fill-cyan-300 font-mono">{actor.name}</text>
            <text x="14" y="12" class="text-[8px] fill-muted-foreground">{actor.count}</text>
          </g>
        ))}

        {/* Topic nodes (right) */}
        {graphData.value.topics.map((topic, i) => (
          <g key={`topic-${topic.name}`} transform={`translate(390, ${topicY(i)})`}>
            <circle r="6" fill="rgba(168,85,247,0.8)" stroke="rgba(168,85,247,1)" stroke-width="1" />
            <text x="-84" y="4" class="text-[9px] fill-purple-300 font-mono text-end" text-anchor="end">{topic.name}</text>
            <text x="-84" y="12" class="text-[8px] fill-muted-foreground text-end" text-anchor="end">{topic.count}</text>
          </g>
        ))}
      </svg>
    </Card>
  )
})

// ─────────────────────────────────────────────────────────────────────────────
// EnrichedEventCard Component
// ─────────────────────────────────────────────────────────────────────────────

export const EnrichedEventCard = component$<EnrichedEventCardProps>(({
  event,
  index,
  onExpand$,
  showLTL = true,
  showVectors = true,
  showKG = true
}) => {
  const expanded = useSignal(false)

  const semantic = (event as any).semantic
  const data = event.data as Record<string, unknown> | undefined
  const dataPreview = data ? JSON.stringify(data, null, 2).slice(0, 300) : null

  // Extract enrichments
  const { domain, subdomain, action } = extractDomain(event.topic)
  const vectorInfo = computeVectorIndicator(event)

  // Mock LTL pattern (in production, would analyze actual event sequences)
  const ltlPattern = showLTL ? inferLTLPattern(event, []) : null

  // Determine card border color
  const borderClass = event.level === 'error' ? 'border-red-500/50 shadow-[0_0_15px_-5px_rgba(239,68,68,0.3)]' :
    event.kind === 'request' ? 'border-blue-500/30 shadow-[0_0_15px_-5px_rgba(59,130,246,0.2)]' :
    event.kind === 'artifact' ? 'border-purple-500/30 shadow-[0_0_15px_-5px_rgba(168,85,247,0.2)]' :
    'border-[var(--glass-border)] hover:border-[var(--glass-border-hover)]'

  return (
    <Card
      class={`flex flex-col gap-3 ${borderClass}`}
      variant="outlined"
    >
      {/* Header with domain badge */}
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class={`w-2 h-2 rounded-full ${
            event.level === 'error' ? 'bg-red-500 animate-pulse' :
            event.level === 'warn' ? 'bg-yellow-500' : 'bg-green-500'
          }`} />
          <span class="text-xs font-mono text-muted-foreground">{event.iso?.slice(11, 19)}</span>

          {/* Domain badge */}
          <span class="text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 uppercase tracking-wider">
            {domain}
          </span>
        </div>

        <div class="flex items-center gap-1">
          {/* Vector indicator */}
          {showVectors && vectorInfo.hasEmbedding && (
            <span
              class="text-[9px] px-1.5 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20"
              title={`Cluster: ${vectorInfo.cluster}, Similarity: ${vectorInfo.similarity?.toFixed(2)}`}
            >
              📊 vec
            </span>
          )}

          {/* v26 Provenance Badge */}
          {data?.provenance_id && (
            <button
              onClick$={(e) => {
                e.stopPropagation();
                window.dispatchEvent(new CustomEvent('pluribus:navigate', {
                  detail: { view: 'events', searchPattern: String(data.provenance_id), searchMode: 'regex' }
                }));
              }}
              class="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500/40"
            >
              🔗 PROV
            </button>
          )}

          {/* Kind badge */}
          <span class={`text-[10px] px-2 py-0.5 rounded uppercase font-bold tracking-wider ${
            event.kind === 'metric' ? 'bg-green-500/10 text-green-400' :
            event.kind === 'request' ? 'bg-blue-500/10 text-blue-400' :
            event.kind === 'artifact' ? 'bg-purple-500/10 text-purple-400' :
            'bg-muted/50 text-muted-foreground'
          }`}>
            {event.kind}
          </span>
        </div>
      </div>

      {/* Topic with subdomain highlight */}
      <div>
        <div class="text-sm font-bold text-foreground break-all">
          <span class="text-primary">{domain}</span>
          {subdomain && <span class="text-muted-foreground">.{subdomain}</span>}
          {action && <span class="text-foreground/70">.{action}</span>}
        </div>
        <div class="text-xs text-muted-foreground mt-0.5 flex items-center gap-2">
          <span>@{event.actor}</span>
          {showKG && (
            <span class="text-[9px] px-1 rounded bg-purple-500/10 text-purple-400">
              KG: {domain}→{event.actor}
            </span>
          )}
        </div>
      </div>

      {/* LTL Pattern */}
      {showLTL && ltlPattern && (
        <div class="text-[10px] px-2 py-1 rounded bg-yellow-500/10 text-yellow-300 border border-yellow-500/20 font-mono">
          LTL: {ltlPattern}
        </div>
      )}

      {/* Semantic/Data Preview */}
      <div
        class="flex-1 bg-black/40 rounded p-2 overflow-hidden border border-[var(--glass-border-subtle)] relative group cursor-pointer"
        onClick$={() => {
          expanded.value = !expanded.value
          if (onExpand$) onExpand$(event)
        }}
      >
        {/* Compliance Warning */}
        {data?._auom_compliance && (data._auom_compliance as any).compliant === false && (
          <div class="text-[9px] text-amber-500 font-bold mb-1 flex items-center gap-1">
            ⚠️ NON-COMPLIANT: {(data._auom_compliance as any).checks?.map((c: any) => c.law).join(', ')}
          </div>
        )}
        
        {semantic?.summary ? (
          <div class="text-xs text-cyan-300 italic mb-1">"{semantic.summary}"</div>
        ) : null}

        {dataPreview && (
          <pre class={`font-mono text-[10px] text-gray-400 whitespace-pre-wrap break-all overflow-y-auto custom-scrollbar ${
            expanded.value ? 'max-h-96' : 'h-24'
          }`}>
            {expanded.value ? JSON.stringify(data, null, 2) : dataPreview}
            {!expanded.value && data && JSON.stringify(data).length > 300 ? '...' : ''}
          </pre>
        )}

        {!dataPreview && !semantic && (
          <div class="text-[10px] text-gray-600 italic text-center py-4">No payload data</div>
        )}

        {/* Expand hint */}
        <div class="absolute bottom-1 right-1 text-[9px] text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">
          {expanded.value ? '▲ collapse' : '▼ expand'}
        </div>
      </div>

      {/* Footer with enrichment badges */}
      <div class="flex items-center justify-between text-[10px] text-gray-600 font-mono mt-auto gap-2">
        <span title={event.id}>ID: {event.id?.slice(0, 8)}...</span>

        <div class="flex items-center gap-1">
          {/* Impact badge */}
          {semantic?.impact && (
            <span class={`px-1.5 rounded ${
              semantic.impact === 'critical' ? 'text-red-400 bg-red-900/20' :
              semantic.impact === 'high' ? 'text-orange-400 bg-orange-900/20' :
              'text-blue-400 bg-blue-900/20'
            }`}>
              {semantic.impact}
            </span>
          )}

          {/* Cluster badge */}
          {showVectors && vectorInfo.cluster && (
            <span class="px-1.5 rounded text-purple-400 bg-purple-900/20">
              {vectorInfo.cluster}
            </span>
          )}
        </div>
      </div>
    </Card>
  )
})

// ─────────────────────────────────────────────────────────────────────────────
// EventStatsBadges - Summary badges for event collections
// ─────────────────────────────────────────────────────────────────────────────

export interface EventStatsBadgesProps {
  events: BusEvent[]
}

export const EventStatsBadges = component$<EventStatsBadgesProps>(({ events }) => {
  const stats = useComputed$(() => {
    const domains = new Map<string, number>()
    const actors = new Map<string, number>()
    let errors = 0
    let requests = 0
    let artifacts = 0
    let withVectors = 0

    for (const event of events) {
      const { domain } = extractDomain(event.topic)
      domains.set(domain, (domains.get(domain) || 0) + 1)
      actors.set(event.actor || 'unknown', (actors.get(event.actor || 'unknown') || 0) + 1)

      if (event.level === 'error') errors++
      if (event.kind === 'request') requests++
      if (event.kind === 'artifact') artifacts++
      if (computeVectorIndicator(event).hasEmbedding) withVectors++
    }

    const topDomains = Array.from(domains.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)

    const topActors = Array.from(actors.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)

    return { topDomains, topActors, errors, requests, artifacts, withVectors, total: events.length }
  })

  return (
    <div class="flex flex-wrap gap-2 text-[10px]">
      {/* Domain distribution */}
      {stats.value.topDomains.map(([domain, count]) => (
        <span
          key={domain}
          class="px-2 py-1 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
        >
          {domain}: {count}
        </span>
      ))}

      {/* Metric badges */}
      <span class="px-2 py-1 rounded bg-red-500/10 text-red-400 border border-red-500/20">
        errors: {stats.value.errors}
      </span>
      <span class="px-2 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
        requests: {stats.value.requests}
      </span>
      <span class="px-2 py-1 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
        artifacts: {stats.value.artifacts}
      </span>
      <span class="px-2 py-1 rounded bg-green-500/10 text-green-400 border border-green-500/20">
        📊 vectors: {stats.value.withVectors}
      </span>
    </div>
  )
})
