import type { BusEvent, LogLevel, VPSSession } from './state/types'

export type SuperMotdSeverity = 'ok' | 'info' | 'warn' | 'error'

export interface SuperMotdRunlevel {
  level: string
  label: string
  hint: string
}

export interface SuperMotdLine {
  iso: string
  severity: SuperMotdSeverity
  subsystem: string
  message: string
  topic: string
  actor: string
}

function severityFromLevel(level: LogLevel): SuperMotdSeverity {
  if (level === 'error') return 'error'
  if (level === 'warn') return 'warn'
  return 'info'
}

function asRecord(v: unknown): Record<string, unknown> {
  return v && typeof v === 'object' ? (v as Record<string, unknown>) : {}
}

function shortId(v: unknown): string {
  const s = String(v || '').trim()
  if (!s) return ''
  return s.length <= 8 ? s : s.slice(0, 8)
}

function truncate(s: string, max = 220): string {
  const t = (s || '').replace(/\s+/g, ' ').trim()
  if (t.length <= max) return t
  return t.slice(0, max - 1) + '…'
}

export function computeSuperMotdRunlevel(opts: {
  connected: boolean
  events: BusEvent[]
  session?: VPSSession
}): SuperMotdRunlevel {
  const { connected, events, session } = opts
  if (!connected) {
    return { level: 'L0', label: 'DISCONNECTED', hint: 'bridge offline / pin / ws' }
  }

  const topics = events.slice(-250).map((e) => e.topic)
  const hasDialogos = topics.some((t) => t.startsWith('dialogos.'))
  const hasLens = topics.some((t) => t.startsWith('lens.') || t === 'plurichat.lens.decision')
  const hasStrp = topics.some((t) => t.startsWith('strp.'))
  const hasVerify = topics.some((t) => t.startsWith('verify.') || t.startsWith('tests.'))
  const active = session?.activeFallback || null

  if (hasVerify) return { level: 'L5', label: 'GATING', hint: 'verification / evidence / gates' }
  if (hasStrp) return { level: 'L4', label: 'SWARM', hint: 'STRp topology active' }
  if (hasLens) return { level: 'L3', label: 'ROUTING', hint: active ? `active_fallback=${active}` : 'lens/collimator online' }
  if (hasDialogos) return { level: 'L2', label: 'INFERENCE', hint: active ? `active_fallback=${active}` : 'dialogos lane' }
  return { level: 'L1', label: 'MEMBRANE', hint: 'bus online' }
}

export function busEventToSuperMotdLine(event: BusEvent): SuperMotdLine | null {
  if (!event || !event.topic) return null
  const d = asRecord(event.data)
  const topic = event.topic
  const iso = event.iso || ''
  const actor = event.actor || ''

  // Avoid spam: ignore raw streaming chunks unless error.
  if (topic === 'dialogos.cell.output') {
    const t = String(d.type || 'text')
    if (event.level !== 'error' && t !== 'error') return null
    const content = truncate(String(d.content || ''))
    return {
      iso,
      severity: 'error',
      subsystem: 'DIALOGOS',
      message: content || 'cell.output error',
      topic,
      actor,
    }
  }

  if (topic === 'dialogos.cell.end') {
    const ok = Boolean(d.ok)
    const status = String(d.status || (ok ? 'success' : 'error'))
    const providers = Array.isArray(d.providers) ? (d.providers as unknown[]).map(String).filter(Boolean) : []
    const msg = `cell.end status=${status}${providers.length ? ` providers=[${providers.join(',')}]` : ''}`
    return { iso, severity: ok ? 'ok' : 'error', subsystem: 'DIALOGOS', message: msg, topic, actor }
  }

  if (topic === 'dialogos.submit') {
    const providers = Array.isArray(d.providers) ? (d.providers as unknown[]).map(String).filter(Boolean) : []
    return {
      iso,
      severity: severityFromLevel(event.level),
      subsystem: 'DIALOGOS',
      message: `submit → [${providers.join(',') || 'auto'}]`,
      topic,
      actor,
    }
  }

  if (topic === 'plurichat.lens.decision') {
    const depth = String(d.depth || '')
    const lane = String(d.lane || '')
    const topo = String(d.topology || 'single')
    const fanout = Number(d.fanout || 1)
    const provider = String(d.selected_provider || d.provider || '')
    const persona = String(d.persona || '')
    const msg = `LENS depth=${depth} lane=${lane} topo=${topo}×${Number.isFinite(fanout) ? fanout : 1} → ${provider || 'auto'}${persona ? ` persona=${persona}` : ''}`
    return { iso, severity: 'info', subsystem: 'LENS', message: msg, topic, actor }
  }

  if (topic === 'plurichat.web.connected') {
    const remote = String(d.remote || '')
    const session = String(d.session || '')
    return { iso, severity: 'info', subsystem: 'PLURICHAT', message: `web connected ${remote} session=${session}`, topic, actor }
  }

  if (topic.startsWith('service.control/')) {
    const svc = String(d.service_id || d.serviceId || '')
    const req = String(d.request_id || d.requestId || '')
    return { iso, severity: 'info', subsystem: 'SERVICES', message: `${topic.replace('service.control/', '')} ${svc}${req ? ` req=${req}` : ''}`, topic, actor }
  }

  if (topic.startsWith('service.')) {
    return { iso, severity: severityFromLevel(event.level), subsystem: 'SERVICES', message: truncate(JSON.stringify(d)), topic, actor }
  }

  if (topic.startsWith('lens.collimator.')) {
    return { iso, severity: severityFromLevel(event.level), subsystem: 'LENS', message: truncate(String(event.semantic || event.reasoning || topic)), topic, actor }
  }

  if (topic.startsWith('strp.')) {
    const kind = String(d.kind || '')
    const goal = String(d.goal || d.goal_summary || '')
    const msg = truncate(`${kind ? `${kind} ` : ''}${goal ? `— ${goal}` : topic}`, 240)
    return { iso, severity: severityFromLevel(event.level), subsystem: 'STRP', message: msg, topic, actor }
  }

  if (topic.startsWith('verify.') || topic.startsWith('tests.')) {
    const msg = truncate(String(d.summary || d.cmd || event.semantic || topic))
    return { iso, severity: severityFromLevel(event.level), subsystem: 'VERIFY', message: msg, topic, actor }
  }

  if (topic.startsWith('operator.pbflush.')) {
    const req = shortId(d.req_id || d.request_id)
    const intent = String(d.intent || '')
    const msg = truncate(String(d.message || d.reason || ''), 180)
    const base = `${topic.replace('operator.pbflush.', 'pbflush.')} ${req ? `req=${req}` : ''}${intent ? ` intent=${intent}` : ''}`.trim()
    const line = msg ? `${base} — ${msg}` : base
    return {
      iso,
      severity: topic.endsWith('.request') ? 'warn' : severityFromLevel(event.level),
      subsystem: 'PBFLUSH',
      message: line,
      topic,
      actor,
    }
  }

  if (topic === 'ckin.report') {
    const pb = asRecord(d.pbflush)
    const latest = asRecord(pb.latest_request)
    const req = shortId(latest.req_id)
    const acks = Number(pb.acks_window ?? 0)
    const reqs = Number(pb.requests_window ?? 0)
    const v = String(d.protocol_version || '')
    const msg = `CKIN v${v || '?'} pbflush(req=${reqs} acks=${acks}${req ? ` last=${req}` : ''})`
    return { iso, severity: 'info', subsystem: 'CKIN', message: msg, topic, actor }
  }

  if ((topic === 'infer_sync.request' || topic === 'infer_sync.response') && String(d.intent || '') === 'pbflush') {
    const req = shortId(d.req_id || d.request_id)
    const msg = truncate(String(d.message || d.reason || ''), 180)
    const base = `${topic} ${req ? `req=${req}` : ''}`.trim()
    return { iso, severity: severityFromLevel(event.level), subsystem: 'PBFLUSH', message: msg ? `${base} — ${msg}` : base, topic, actor }
  }

  if (topic === 'system.boot.log') {
    const lines = Array.isArray(d.boot_log) ? d.boot_log : []
    const msg = `SLOU Boot Sequence (${lines.length} lines)`
    return { iso, severity: 'info', subsystem: 'SLOU', message: msg, topic, actor }
  }

  if (topic.startsWith('supermotd.')) {
    const msg = truncate(String(d.text || event.semantic || JSON.stringify(d) || topic))
    return { iso, severity: severityFromLevel(event.level), subsystem: 'SUPERMOTD', message: msg, topic, actor }
  }

  if (topic.startsWith('mcp.')) {
    if (topic === 'mcp.host.call') {
      const req = shortId(d.req_id)
      const server = String(d.server || '')
      const tool = String(d.tool || '')
      return {
        iso,
        severity: 'info',
        subsystem: 'MCP',
        message: `host.call ${req ? `req=${req}` : ''} ${server}.${tool}`.trim(),
        topic,
        actor,
      }
    }
    if (topic === 'mcp.host.response') {
      const req = shortId(d.req_id)
      const ok = Boolean(d.ok)
      const server = String(d.server || '')
      const tool = String(d.tool || '')
      return {
        iso,
        severity: ok ? 'ok' : 'error',
        subsystem: 'MCP',
        message: `host.response ${req ? `req=${req}` : ''} ${ok ? 'ok' : 'err'} ${server}.${tool}`.trim(),
        topic,
        actor,
      }
    }
    const msg = truncate(String(d.status || d.server || d.tool || d.text || event.semantic || topic))
    return { iso, severity: severityFromLevel(event.level), subsystem: 'MCP', message: msg, topic, actor }
  }

  if (topic.startsWith('a2a.')) {
    const req = shortId(d.req_id || d.request_id)
    if (topic === 'a2a.negotiate.request') {
      const initiator = String(d.initiator || d.from || actor || '').trim()
      const target = String(d.target || '').trim()
      const constraints = asRecord(d.constraints)
      const caps = Array.isArray(constraints.required_capabilities) ? constraints.required_capabilities : []
      const msg = `negotiate.request ${req ? `req=${req}` : ''}${initiator ? ` from=${initiator}` : ''}${target ? ` to=${target}` : ''}${caps.length ? ` caps=${caps.length}` : ''}`.trim()
      return { iso, severity: 'info', subsystem: 'A2A', message: msg, topic, actor }
    }
    if (topic === 'a2a.negotiate.response') {
      const decision = String(d.decision || '').trim()
      const sev: SuperMotdSeverity = decision === 'agree' ? 'ok' : decision === 'negotiate' ? 'warn' : 'error'
      const msg = `negotiate.response ${req ? `req=${req}` : ''}${decision ? ` decision=${decision}` : ''}`.trim()
      return { iso, severity: sev, subsystem: 'A2A', message: msg, topic, actor }
    }
    if (topic === 'a2a.decline') {
      const reason = truncate(String(d.reason || 'decline'), 180)
      const base = `decline ${req ? `req=${req}` : ''}`.trim()
      return { iso, severity: 'warn', subsystem: 'A2A', message: reason ? `${base} — ${reason}` : base, topic, actor }
    }
    if (topic === 'a2a.redirect') {
      const to = String(d.redirect_to || '').trim()
      const reason = truncate(String(d.reason || 'redirect'), 180)
      const base = `redirect ${req ? `req=${req}` : ''}${to ? ` to=${to}` : ''}`.trim()
      return { iso, severity: 'info', subsystem: 'A2A', message: reason ? `${base} — ${reason}` : base, topic, actor }
    }
    return { iso, severity: severityFromLevel(event.level), subsystem: 'A2A', message: truncate(JSON.stringify(d)), topic, actor }
  }

  if (topic === 'studio.flow.roundtrip') {
    const req = shortId(d.req_id)
    const ok = Boolean(d.ok)
    const inPath = String(d.in_path || '').trim()
    const msg = `flow.roundtrip ${req ? `req=${req}` : ''} ${ok ? 'ok' : 'fail'}${inPath ? ` in=${inPath.split('/').slice(-1)[0]}` : ''}`.trim()
    return { iso, severity: ok ? 'ok' : 'error', subsystem: 'STUDIO', message: msg, topic, actor }
  }

  if (topic.startsWith('crush.')) {
    const sessionId = shortId(d.session_id)
    if (topic === 'crush.session.start') {
      const model = String(d.model || 'default')
      return { iso, severity: 'info', subsystem: 'CRUSH', message: `session.start ${sessionId ? `sid=${sessionId}` : ''} model=${model}`.trim(), topic, actor }
    }
    if (topic === 'crush.prompt.submit') {
      const preview = truncate(String(d.prompt_preview || ''), 60)
      return { iso, severity: 'info', subsystem: 'CRUSH', message: `prompt ${sessionId ? `sid=${sessionId}` : ''} "${preview}"`.trim(), topic, actor }
    }
    if (topic === 'crush.response.end') {
      const ok = Boolean(d.ok)
      const duration = Number(d.duration_ms || 0)
      return { iso, severity: ok ? 'ok' : 'error', subsystem: 'CRUSH', message: `response.end ${sessionId ? `sid=${sessionId}` : ''} ${ok ? 'ok' : 'err'} ${duration}ms`.trim(), topic, actor }
    }
    if (topic === 'crush.error') {
      const err = truncate(String(d.error || d.stderr || 'error'), 100)
      return { iso, severity: 'error', subsystem: 'CRUSH', message: `error ${sessionId ? `sid=${sessionId}` : ''} ${err}`.trim(), topic, actor }
    }
    return { iso, severity: severityFromLevel(event.level), subsystem: 'CRUSH', message: truncate(String(d.intent || topic)), topic, actor }
  }

  if (topic.startsWith('operator.crush.')) {
    const req = shortId(d.req_id)
    const intent = String(d.intent || '')
    if (topic === 'operator.crush.request') {
      return { iso, severity: 'info', subsystem: 'CRUSH', message: `operator.${intent || 'request'} ${req ? `req=${req}` : ''}`.trim(), topic, actor }
    }
    if (topic === 'operator.crush.response') {
      const ok = Boolean(d.ok)
      return { iso, severity: ok ? 'ok' : 'error', subsystem: 'CRUSH', message: `operator.response ${req ? `req=${req}` : ''} ${intent} ${ok ? 'ok' : 'err'}`.trim(), topic, actor }
    }
    if (topic === 'operator.crush.status') {
      const online = Boolean(d.online)
      return { iso, severity: online ? 'ok' : 'warn', subsystem: 'CRUSH', message: `status ${online ? 'online' : 'offline'}`, topic, actor }
    }
  }

  return null
}

export function selectSuperMotdLines(opts: {
  events: BusEvent[]
  limit?: number
}): SuperMotdLine[] {
  const limit = Math.max(1, Math.min(200, opts.limit ?? 40))
  const out: SuperMotdLine[] = []

  for (const e of opts.events.slice(-800)) {
    const line = busEventToSuperMotdLine(e)
    if (!line) continue
    out.push(line)
  }

  // Keep last N, stable order, and lightly de-dupe identical consecutive messages.
  const trimmed = out.slice(-limit)
  const deduped: SuperMotdLine[] = []
  for (const l of trimmed) {
    const last = deduped[deduped.length - 1]
    // Only collapse true duplicates (same rendered line at the same timestamp), e.g. from double-append/sync.
    if (last && last.iso === l.iso && last.topic === l.topic && last.message === l.message && last.subsystem === l.subsystem) continue
    deduped.push(l)
  }
  return deduped
}
