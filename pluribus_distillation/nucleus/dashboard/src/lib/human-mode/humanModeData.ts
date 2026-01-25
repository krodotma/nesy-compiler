import type { AgentStatus, BusEvent, ServiceDef, STRpRequest, TaskLedgerEntry } from '../state/types';

export type HumanModeMode = 'historic' | 'present' | 'future';

export type HumanModeGapKind = 'aleatoric' | 'epistemic';
export type HumanModeGapSeverity = 'low' | 'medium' | 'high';

export type HumanModeTaskStatus = 'planned' | 'in_progress' | 'blocked' | 'completed' | 'abandoned';
export type HumanModeTaskSource = 'task_ledger' | 'strp';

export interface HumanModeTask {
  id: string;
  reqId?: string;
  status: HumanModeTaskStatus;
  rawStatus?: string;
  actor: string;
  label: string;
  source: HumanModeTaskSource;
  topic?: string;
  createdIso?: string;
  timestampMs: number;
}

export interface HumanModeGap {
  id: string;
  kind: HumanModeGapKind;
  severity: HumanModeGapSeverity;
  summary: string;
  evidence?: string;
  source?: string;
}

export interface HumanModeSourceMeta {
  path?: string;
  available?: boolean;
  error?: string;
  [key: string]: unknown;
}

export interface HumanModeLane {
  id?: string;
  name?: string;
  status?: string;
  wip_pct?: number;
}

export interface HumanModeSummary {
  eventCount: number;
  taskCount: number;
  agentCount: number;
  serviceCount: number;
  eventRate: number;
  lastEventAgeMs: number | null;
  window_start_iso: string;
  window_end_iso: string;
  event_count: number;
  task_count: number;
  agent_count: number;
  lane_count: number;
  event_volatility: number;
  status_counts: Record<string, number>;
}

export interface HumanModeWindow {
  startMs: number;
  endMs: number;
  startIso: string;
  endIso: string;
  windowMinutes: number;
}

export interface HumanModeSnapshot {
  mode: HumanModeMode;
  anchor_iso: string;
  window_minutes: number;
  window: HumanModeWindow;
  summary: HumanModeSummary;
  gaps: HumanModeGap[];
  tasks: HumanModeTask[];
  agents: AgentStatus[];
  lanes: HumanModeLane[];
  events: BusEvent[];
  evidence: BusEvent[];
  sources: Record<string, HumanModeSourceMeta>;
}

export interface HumanModeParams {
  mode: HumanModeMode;
  anchorMs?: number | null;
  windowMinutes: number;
  events: BusEvent[];
  requests: STRpRequest[];
  taskLedger?: TaskLedgerEntry[];
  taskLedgerAvailable?: boolean;
  taskLedgerError?: string | null;
  taskLedgerPath?: string | null;
  lanes?: HumanModeLane[];
  agents: AgentStatus[];
  services: ServiceDef[];
  connected: boolean;
}

export function eventTimestampMs(event: BusEvent): number {
  const ts = typeof event.ts === 'number' ? event.ts : 0;
  if (ts > 1_000_000_000_000) return ts;
  if (ts > 1_000_000_000) return ts * 1000;
  if (event.iso) {
    const parsed = Date.parse(event.iso);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return 0;
}

export function computeWindowRange(mode: HumanModeMode, anchorMs: number, windowMinutes: number): HumanModeWindow {
  const windowMs = Math.max(1, windowMinutes) * 60 * 1000;
  if (mode === 'future') {
    return {
      startMs: anchorMs,
      endMs: anchorMs + windowMs,
      startIso: new Date(anchorMs).toISOString(),
      endIso: new Date(anchorMs + windowMs).toISOString(),
      windowMinutes,
    };
  }
  return {
    startMs: anchorMs - windowMs,
    endMs: anchorMs,
    startIso: new Date(anchorMs - windowMs).toISOString(),
    endIso: new Date(anchorMs).toISOString(),
    windowMinutes,
  };
}

function computeVolatility(events: BusEvent[]): number {
  const timestamps = events
    .map(eventTimestampMs)
    .filter((ts) => ts > 0)
    .sort((a, b) => a - b);
  if (timestamps.length < 4) return 0;
  const intervals: number[] = [];
  for (let i = 1; i < timestamps.length; i += 1) {
    const delta = timestamps[i] - timestamps[i - 1];
    if (delta > 0) intervals.push(delta);
  }
  if (intervals.length === 0) return 0;
  const mean = intervals.reduce((sum, v) => sum + v, 0) / intervals.length;
  if (mean <= 0) return 0;
  const variance = intervals.reduce((sum, v) => sum + (v - mean) ** 2, 0) / intervals.length;
  return Math.sqrt(variance) / mean;
}

function filterTasksByMode(mode: HumanModeMode, tasks: HumanModeTask[]): HumanModeTask[] {
  if (mode === 'historic') {
    return tasks.filter((task) => task.status === 'completed' || task.status === 'abandoned');
  }
  if (mode === 'future') {
    return tasks.filter((task) => task.status === 'planned');
  }
  return tasks.filter((task) => (
    task.status === 'planned' || task.status === 'in_progress' || task.status === 'blocked'
  ));
}

function evidenceFromEvents(events: BusEvent[]): BusEvent[] {
  return events.filter((event) => {
    const topic = event.topic || '';
    const kind = event.kind || '';
    if (kind === 'artifact' || kind === 'alert') return true;
    return ['evidence', 'beam', 'task_ledger', 'report'].some((token) => topic.includes(token));
  });
}

function requestTimestampMs(request: STRpRequest): number {
  if (!request.created_iso) return 0;
  const parsed = Date.parse(request.created_iso);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function ledgerTimestampMs(entry: TaskLedgerEntry): number {
  if (typeof entry.ts === 'number') {
    if (entry.ts > 1_000_000_000_000) return entry.ts;
    if (entry.ts > 1_000_000_000) return entry.ts * 1000;
    return entry.ts * 1000;
  }
  if (entry.iso) {
    const parsed = Date.parse(entry.iso);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return 0;
}

const STRP_STATUS_MAP: Record<STRpRequest['status'], HumanModeTaskStatus> = {
  pending: 'planned',
  working: 'in_progress',
  completed: 'completed',
  failed: 'abandoned',
};

function labelFromLedger(entry: TaskLedgerEntry): string {
  if (entry.intent) return entry.intent;
  const meta = entry.meta;
  if (meta && typeof meta === 'object') {
    const desc = (meta as Record<string, unknown>).desc;
    if (typeof desc === 'string' && desc) return desc;
    const intent = (meta as Record<string, unknown>).intent;
    if (typeof intent === 'string' && intent) return intent;
  }
  if (entry.topic) return entry.topic;
  return entry.req_id || entry.id || 'task';
}

function normalizeLedgerTasks(entries: TaskLedgerEntry[]): HumanModeTask[] {
  const byReq = new Map<string, TaskLedgerEntry>();
  entries.forEach((entry) => {
    const key = entry.req_id || entry.id;
    if (!key) return;
    const nextTs = ledgerTimestampMs(entry);
    const prev = byReq.get(key);
    if (!prev) {
      byReq.set(key, entry);
      return;
    }
    const prevTs = ledgerTimestampMs(prev);
    if (nextTs >= prevTs) {
      byReq.set(key, entry);
    }
  });

  return Array.from(byReq.values()).map((entry) => ({
    id: entry.req_id || entry.id || '',
    reqId: entry.req_id,
    status: entry.status,
    rawStatus: entry.status,
    actor: entry.actor,
    label: labelFromLedger(entry),
    source: 'task_ledger',
    topic: entry.topic,
    createdIso: entry.iso,
    timestampMs: ledgerTimestampMs(entry),
  }));
}

function normalizeStrpTasks(requests: STRpRequest[]): HumanModeTask[] {
  return requests.map((req) => ({
    id: req.id,
    reqId: req.id,
    status: STRP_STATUS_MAP[req.status],
    rawStatus: req.status,
    actor: req.actor,
    label: req.goal || req.kind,
    source: 'strp',
    topic: req.kind,
    createdIso: req.created_iso,
    timestampMs: requestTimestampMs(req),
  }));
}

function mergeTasks(ledgerTasks: HumanModeTask[], strpTasks: HumanModeTask[]): HumanModeTask[] {
  const merged = new Map<string, HumanModeTask>();
  ledgerTasks.forEach((task) => merged.set(task.id, task));
  strpTasks.forEach((task) => {
    if (!merged.has(task.id)) {
      merged.set(task.id, task);
    }
  });
  return Array.from(merged.values());
}

export function buildHumanModeSnapshot(params: HumanModeParams): HumanModeSnapshot {
  const now = Date.now();
  const anchorMs = params.anchorMs ?? now;
  const window = computeWindowRange(params.mode, anchorMs, params.windowMinutes);
  const anchorIso = new Date(anchorMs).toISOString();

  const windowedEvents = params.events.filter((event) => {
    const ts = eventTimestampMs(event);
    return ts >= window.startMs && ts <= window.endMs;
  });

  const ledgerTasks = normalizeLedgerTasks(params.taskLedger ?? []);
  const strpTasks = normalizeStrpTasks(params.requests);
  const combinedTasks = mergeTasks(ledgerTasks, strpTasks);
  const tasksByMode = filterTasksByMode(params.mode, combinedTasks).filter((task) => {
    if (!task.timestampMs) return true;
    return task.timestampMs >= window.startMs && task.timestampMs <= window.endMs;
  });

  const lastEventTs = windowedEvents.length
    ? Math.max(...windowedEvents.map(eventTimestampMs))
    : null;
  const statusCounts = tasksByMode.reduce<Record<string, number>>((acc, task) => {
    const key = task.status || 'unknown';
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const volatility = computeVolatility(windowedEvents);
  const summary: HumanModeSummary = {
    eventCount: windowedEvents.length,
    taskCount: tasksByMode.length,
    agentCount: params.agents.length,
    serviceCount: params.services.length,
    eventRate: windowedEvents.length / Math.max(1, params.windowMinutes),
    lastEventAgeMs: lastEventTs ? now - lastEventTs : null,
    window_start_iso: window.startIso,
    window_end_iso: window.endIso,
    event_count: windowedEvents.length,
    task_count: tasksByMode.length,
    agent_count: params.agents.length,
    lane_count: params.lanes?.length ?? 0,
    event_volatility: volatility,
    status_counts: statusCounts,
  };

  const gaps: HumanModeGap[] = [];
  if (!params.connected) {
    gaps.push({
      id: 'bus.disconnected',
      kind: 'epistemic',
      severity: 'high',
      summary: 'Bus disconnected, live updates paused',
      source: 'bus',
    });
  }
  if (windowedEvents.length === 0) {
    gaps.push({
      id: 'events.empty',
      kind: 'epistemic',
      severity: 'high',
      summary: 'No events captured in the selected window',
      source: 'bus',
    });
  }
  if (params.taskLedgerAvailable === false) {
    gaps.push({
      id: 'task_ledger.unavailable',
      kind: 'epistemic',
      severity: 'high',
      summary: 'Task ledger unavailable',
      evidence: params.taskLedgerError || undefined,
      source: 'task_ledger',
    });
  }
  if (tasksByMode.length === 0) {
    gaps.push({
      id: 'tasks.empty',
      kind: 'epistemic',
      severity: 'medium',
      summary: 'No tasks visible for this window',
      source: 'task_ledger',
    });
  }

  if (volatility > 1.2) {
    gaps.push({
      id: 'events.volatile',
      kind: 'aleatoric',
      severity: 'medium',
      summary: 'High event volatility in window',
      evidence: `cv=${volatility.toFixed(2)}`,
      source: 'bus',
    });
  }

  const sources: Record<string, HumanModeSourceMeta> = {
    bus: { available: params.connected },
    task_ledger: {
      available: params.taskLedgerAvailable,
      error: params.taskLedgerError || undefined,
      path: params.taskLedgerPath || undefined,
    },
  };

  return {
    mode: params.mode,
    anchor_iso: anchorIso,
    window_minutes: params.windowMinutes,
    window,
    summary,
    gaps,
    tasks: tasksByMode,
    agents: params.agents,
    lanes: params.lanes ?? [],
    events: windowedEvents,
    evidence: evidenceFromEvents(windowedEvents),
    sources,
  };
}
