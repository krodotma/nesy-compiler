import { component$, useComputed$, useSignal, $, type QRL } from '@builder.io/qwik'

import type { BusEvent, VPSSession } from '../lib/state/types'
import { computeSuperMotdRunlevel, selectSuperMotdLines, type SuperMotdLine } from '../lib/supermotd'

export interface SuperMotdProps {
  connected: boolean
  events: BusEvent[]
  session: VPSSession
  emitBus$?: QRL<(topic: string, kind: string, data: Record<string, unknown>) => void>
}

const SEV: Record<string, { badge: string; row: string; icon: string }> = {
  ok: { badge: 'bg-green-500/20 text-green-400 border-green-500/30', row: 'text-green-300', icon: '✓' },
  info: { badge: 'bg-muted/40 text-muted-foreground border-border/50', row: 'text-muted-foreground', icon: 'ℹ' },
  warn: { badge: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30', row: 'text-yellow-200', icon: '⚠' },
  error: { badge: 'bg-red-500/20 text-red-300 border-red-500/30', row: 'text-red-200', icon: '✗' },
}

// Subsystem icons for semantic enrichment
const SUBSYSTEM_ICONS: Record<string, string> = {
  DIALOGOS: '💬',
  LENS: '🔍',
  STRP: '🌐',
  VERIFY: '✅',
  MCP: '🔧',
  A2A: '🤝',
  SERVICES: '⚙️',
  PLURICHAT: '💭',
  CKIN: '📊',
  PBFLUSH: '🚿',
  SLOU: '🖥️',
  SUPERMOTD: '📋',
  STUDIO: '🎬',
}

export const SuperMotd = component$<SuperMotdProps>(({ connected, events, session, emitBus$ }) => {
  const lastReqId = useSignal<string | null>(null)

  const runlevel = useComputed$(() => computeSuperMotdRunlevel({ connected, events, session }))
  const lines = useComputed$(() => selectSuperMotdLines({ events, limit: 48 }))

  const requestCollab = $(async () => {
    const reqId = typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `req-${Date.now()}`
    lastReqId.value = reqId
    if (!emitBus$) return
    await emitBus$('supermotd.proposal.request', 'request', {
      req_id: reqId,
      branch: 'SUPERMOTD',
      intent: 'improve_boot_stream',
      targets: ['dashboard', 'tui'],
      ask: 'Propose idiolect/vocab + event grammar + curated topic filters for a unix-boot-style system stream.',
      constraints: { append_only: true, non_blocking: true, tests_first: true },
    })
  })

  // Compute subsystem stats for semantic enrichment
  const subsystemStats = useComputed$(() => {
    const stats = new Map<string, { count: number; errors: number; lastSeverity: string }>()
    for (const l of lines.value) {
      const existing = stats.get(l.subsystem) || { count: 0, errors: 0, lastSeverity: 'info' }
      existing.count++
      if (l.severity === 'error') existing.errors++
      existing.lastSeverity = l.severity
      stats.set(l.subsystem, existing)
    }
    return Array.from(stats.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 6)
  })

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      <div class="p-3 border-b border-border flex items-center justify-between gap-3">
        <div class="flex items-center gap-3 min-w-0">
          <div class="flex items-center gap-2">
            <span class="text-xs font-mono text-muted-foreground">SUPERMOTD</span>
            <span class={`text-xs font-mono px-2 py-0.5 rounded border ${
              runlevel.value.level === 'L5' ? 'border-purple-500/50 bg-purple-500/20 text-purple-400' :
              runlevel.value.level === 'L4' ? 'border-blue-500/50 bg-blue-500/20 text-blue-400' :
              runlevel.value.level === 'L3' ? 'border-cyan-500/50 bg-cyan-500/20 text-cyan-400' :
              runlevel.value.level === 'L2' ? 'border-green-500/50 bg-green-500/20 text-green-400' :
              runlevel.value.level === 'L1' ? 'border-yellow-500/50 bg-yellow-500/20 text-yellow-400' :
              'border-border/50 bg-muted/30 text-muted-foreground'
            }`}>
              {runlevel.value.level} {runlevel.value.label}
            </span>
          </div>
          <div class="text-xs text-muted-foreground truncate">
            {runlevel.value.hint}
          </div>
        </div>

        <div class="flex items-center gap-2 flex-shrink-0">
          <button
            onClick$={requestCollab}
            class="text-xs px-2 py-1 rounded bg-primary/15 hover:bg-primary/25 text-primary border border-primary/20"
            title="Emit a non-blocking bus request for other agents to propose improvements"
          >
            Request Collab
          </button>
        </div>
      </div>

      {/* Subsystem Activity Badges */}
      <div class="px-3 py-2 border-b border-border/50 flex flex-wrap gap-1">
        {subsystemStats.value.map(([subsystem, stats]) => {
          const icon = SUBSYSTEM_ICONS[subsystem] || '📦'
          const hasErrors = stats.errors > 0
          return (
            <span
              key={subsystem}
              class={`text-[9px] px-1.5 py-0.5 rounded flex items-center gap-1 ${
                hasErrors
                  ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                  : 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
              }`}
              title={`${subsystem}: ${stats.count} events, ${stats.errors} errors`}
            >
              <span>{icon}</span>
              <span class="font-mono">{subsystem}</span>
              <span class="text-[8px] opacity-70">{stats.count}</span>
              {hasErrors && <span class="text-red-400">({stats.errors})</span>}
            </span>
          )
        })}
      </div>

      <div class="p-3">
        <div class="text-xs text-muted-foreground mb-2">
          Boot-stream style meta events (curated). Prefer reading this over raw `dialogos.cell.output` noise.
          {lastReqId.value && (
            <span class="ml-2 font-mono text-primary/80">req_id={lastReqId.value}</span>
          )}
        </div>

        <div class="max-h-[260px] overflow-auto font-mono text-xs space-y-1">
          {lines.value.length === 0 ? (
            <div class="p-2 rounded bg-muted/20 text-muted-foreground">
              No curated meta events yet. Generate activity (e.g. PluriChat `/status`, spawn a worker, run verify).
            </div>
          ) : (
            lines.value.slice().reverse().map((l, i) => {
              const style = SEV[l.severity] || SEV.info
              const icon = SUBSYSTEM_ICONS[l.subsystem] || '📦'
              const ts = l.iso ? l.iso.slice(11, 19) : '--:--:--'
              return (
                <div key={i} class={`flex items-start gap-2 p-1.5 rounded bg-muted/20 border border-border/30 ${style.row} hover:bg-muted/30 transition-colors`}>
                  <span class="text-muted-foreground flex-shrink-0">{ts}</span>
                  <span class="text-[10px] flex-shrink-0" title={l.severity}>{style.icon}</span>
                  <span class={`flex-shrink-0 text-[10px] px-1.5 py-0.5 rounded border flex items-center gap-1 ${style.badge}`}>
                    <span>{icon}</span>
                    <span>{l.subsystem}</span>
                  </span>
                  <span class="flex-1 min-w-0 break-words">
                    {l.message}
                  </span>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
})

