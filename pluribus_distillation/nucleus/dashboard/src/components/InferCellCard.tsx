/**
 * InferCellCard - Interactive subsystem card with InferCell session support
 *
 * Transforms static dashboard badges into live InferCell events that can:
 * - Trigger verification, inspection, or fork events
 * - Show verbose module info in expandable panel
 * - Display InferCell session with trace correlation
 * - Emit bus events for all interactions
 */

import { component$, useSignal, useStore, $, type QRL } from '@builder.io/qwik';
import { NeonTitle, NeonBadge } from './ui/NeonTitle';

// M3 Components - InferCellCard
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/text-button.js';

export type InferCellStatus = 'idle' | 'checking' | 'ok' | 'warn' | 'error' | 'forked';

export interface ModuleInfo {
  name: string;
  file: string;
  description?: string;
  icon?: string;
  goldenId?: string; // G1-G10 reference
}

export interface InferCellSession {
  traceId: string;
  cellId: string;
  parentTraceId?: string;
  state: 'pending' | 'active' | 'paused' | 'complete' | 'merged';
  forkReason?: string;
  children: string[];
  workingMemory: Record<string, unknown>;
  lastEvent?: {
    topic: string;
    timestamp: string;
    summary: string;
  };
}

export interface LiveModuleData {
  loaded: boolean;
  lastCheck: string;
  imports?: string[];
  exports?: string[];
  testStatus?: 'pass' | 'fail' | 'skip';
  lineCount?: number;
  errors?: string[];
  metrics?: Record<string, number>;
}

export interface InferCellCardProps {
  module: ModuleInfo;
  status?: InferCellStatus;
  session?: InferCellSession;
  liveData?: LiveModuleData;
  onTrigger$?: QRL<(action: string, module: ModuleInfo) => void>;
  compact?: boolean;
}

const STATUS_STYLES: Record<InferCellStatus, { dot: string; border: string; bg: string }> = {
  idle: { dot: 'bg-gray-400', border: 'border-gray-500/30', bg: 'bg-gray-500/10' },
  checking: { dot: 'bg-yellow-400 animate-pulse', border: 'border-yellow-500/30', bg: 'bg-yellow-500/10' },
  ok: { dot: 'bg-green-400', border: 'border-green-500/30', bg: 'bg-green-500/10' },
  warn: { dot: 'bg-orange-400', border: 'border-orange-500/30', bg: 'bg-orange-500/10' },
  error: { dot: 'bg-red-400', border: 'border-red-500/30', bg: 'bg-red-500/10' },
  forked: { dot: 'bg-purple-400 animate-pulse', border: 'border-purple-500/30', bg: 'bg-purple-500/10' },
};

const STATE_BADGES: Record<string, { color: string; label: string }> = {
  pending: { color: 'bg-gray-500/20 text-gray-400', label: 'PENDING' },
  active: { color: 'bg-green-500/20 text-green-400', label: 'ACTIVE' },
  paused: { color: 'bg-yellow-500/20 text-yellow-400', label: 'PAUSED' },
  complete: { color: 'bg-blue-500/20 text-blue-400', label: 'COMPLETE' },
  merged: { color: 'bg-purple-500/20 text-purple-400', label: 'MERGED' },
};

export const InferCellCard = component$<InferCellCardProps>(({
  module,
  status = 'idle',
  session,
  liveData,
  onTrigger$,
  compact = false,
}) => {
  const expanded = useSignal(false);
  const activeAction = useSignal<string | null>(null);
  const styles = STATUS_STYLES[status];

  const handleAction = $(async (action: string) => {
    activeAction.value = action;
    if (onTrigger$) {
      await onTrigger$(action, module);
    }
    // Auto-clear action indicator after 2s
    setTimeout(() => {
      activeAction.value = null;
    }, 2000);
  });

  if (compact) {
    // Compact badge-style (for grid layouts)
    return (
      <div
        class={`rounded border ${styles.border} ${styles.bg} p-2 flex items-center gap-2 cursor-pointer hover:brightness-110 transition-all`}
        onClick$={() => expanded.value = !expanded.value}
      >
        <span class={`w-2 h-2 rounded-full ${styles.dot}`} />
        <div class="flex-1 min-w-0">
          <div class="text-xs font-medium truncate">{module.name}</div>
          <div class="text-xs text-muted-foreground mono truncate">{module.file}</div>
        </div>
        {module.goldenId && (
          <span class="text-xs font-mono text-primary/60">{module.goldenId}</span>
        )}
      </div>
    );
  }

  return (
    <div class={`rounded-lg border ${styles.border} ${styles.bg} overflow-hidden transition-all`}>
      {/* Header - Always visible */}
      <div
        class="p-3 flex items-center gap-3 cursor-pointer hover:bg-[var(--glass-bg-hover)] transition-colors"
        onClick$={() => expanded.value = !expanded.value}
      >
        {/* Status dot */}
        <span class={`w-3 h-3 rounded-full ${styles.dot} flex-shrink-0`} />

        {/* Icon */}
        {module.icon && <span class="text-xl">{module.icon}</span>}

        {/* Module info */}
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2">
            {module.goldenId && (
              <NeonBadge color="cyan" glow>{module.goldenId}</NeonBadge>
            )}
            <NeonTitle level="span" color="cyan" size="sm">{module.name}</NeonTitle>
          </div>
          <div class="text-xs text-muted-foreground mono">{module.file}</div>
        </div>

        {/* Session state badge */}
        {session && (
          <span class={`text-xs px-2 py-0.5 rounded ${STATE_BADGES[session.state]?.color || ''}`}>
            {STATE_BADGES[session.state]?.label || session.state}
          </span>
        )}

        {/* Expand indicator */}
        <span class={`text-muted-foreground transition-transform ${expanded.value ? 'rotate-180' : ''}`}>
          ▼
        </span>
      </div>

      {/* Expanded content */}
      {expanded.value && (
        <div class="border-t border-[var(--glass-border)]">
          {/* Action buttons */}
          <div class="p-3 border-b border-[var(--glass-border)] flex flex-wrap gap-2">
            <button
              class={`text-xs px-3 py-1 rounded border border-green-500/30 bg-green-500/10 hover:bg-green-500/20 transition-colors ${activeAction.value === 'verify' ? 'animate-pulse' : ''}`}
              onClick$={() => handleAction('verify')}
            >
              ✓ Verify
            </button>
            <button
              class={`text-xs px-3 py-1 rounded border border-blue-500/30 bg-blue-500/10 hover:bg-blue-500/20 transition-colors ${activeAction.value === 'inspect' ? 'animate-pulse' : ''}`}
              onClick$={() => handleAction('inspect')}
            >
              🔍 Inspect
            </button>
            <button
              class={`text-xs px-3 py-1 rounded border border-purple-500/30 bg-purple-500/10 hover:bg-purple-500/20 transition-colors ${activeAction.value === 'fork' ? 'animate-pulse' : ''}`}
              onClick$={() => handleAction('fork')}
            >
              🔀 Fork
            </button>
            <button
              class={`text-xs px-3 py-1 rounded border border-orange-500/30 bg-orange-500/10 hover:bg-orange-500/20 transition-colors ${activeAction.value === 'test' ? 'animate-pulse' : ''}`}
              onClick$={() => handleAction('test')}
            >
              🧪 Test
            </button>
            <button
              class={`text-xs px-3 py-1 rounded border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 transition-colors ${activeAction.value === 'emit' ? 'animate-pulse' : ''}`}
              onClick$={() => handleAction('emit')}
            >
              📡 Emit Event
            </button>
          </div>

          {/* Module description */}
          {module.description && (
            <div class="p-3 border-b border-[var(--glass-border)] text-sm text-muted-foreground">
              {module.description}
            </div>
          )}

          {/* Live module data */}
          {liveData && (
            <div class="p-3 border-b border-[var(--glass-border)] space-y-2">
              <NeonTitle level="div" color="emerald" size="xs">Live Module Data</NeonTitle>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                <div class="flex items-center gap-2">
                  <span class={`w-2 h-2 rounded-full ${liveData.loaded ? 'bg-green-400' : 'bg-red-400'}`} />
                  <span>{liveData.loaded ? 'Loaded' : 'Not Loaded'}</span>
                </div>
                {liveData.lineCount !== undefined && (
                  <div class="text-muted-foreground">
                    {liveData.lineCount} lines
                  </div>
                )}
                {liveData.testStatus && (
                  <div class={`${liveData.testStatus === 'pass' ? 'text-green-400' : liveData.testStatus === 'fail' ? 'text-red-400' : 'text-gray-400'}`}>
                    Test: {liveData.testStatus.toUpperCase()}
                  </div>
                )}
                <div class="text-muted-foreground">
                  {liveData.lastCheck}
                </div>
              </div>
              {liveData.exports && liveData.exports.length > 0 && (
                <div class="text-xs">
                  <span class="text-muted-foreground">Exports: </span>
                  <span class="font-mono text-cyan-400">{liveData.exports.slice(0, 5).join(', ')}</span>
                  {liveData.exports.length > 5 && <span class="text-muted-foreground"> +{liveData.exports.length - 5} more</span>}
                </div>
              )}
              {liveData.errors && liveData.errors.length > 0 && (
                <div class="text-xs text-red-400">
                  Errors: {liveData.errors.join('; ')}
                </div>
              )}
              {liveData.metrics && Object.keys(liveData.metrics).length > 0 && (
                <div class="flex flex-wrap gap-2">
                  {Object.entries(liveData.metrics).map(([k, v]) => (
                    <span key={k} class="text-xs px-2 py-0.5 rounded bg-[var(--glass-bg-card)] border border-[var(--glass-border)]">
                      {k}: <span class="text-primary">{v}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* InferCell Session */}
          {session && (
            <div class="p-3 space-y-2">
              <div class="flex items-center gap-2">
                <span>🧬</span>
                <NeonTitle level="span" color="purple" size="xs">InferCell Session</NeonTitle>
              </div>
              <div class="space-y-1 text-xs font-mono">
                <div class="flex items-center gap-2">
                  <span class="text-muted-foreground w-24">trace_id:</span>
                  <span class="text-cyan-400">{session.traceId.slice(0, 8)}...</span>
                </div>
                <div class="flex items-center gap-2">
                  <span class="text-muted-foreground w-24">cell_id:</span>
                  <span class="text-purple-400">{session.cellId.slice(0, 8)}...</span>
                </div>
                {session.parentTraceId && (
                  <div class="flex items-center gap-2">
                    <span class="text-muted-foreground w-24">parent:</span>
                    <span class="text-orange-400">{session.parentTraceId.slice(0, 8)}...</span>
                  </div>
                )}
                <div class="flex items-center gap-2">
                  <span class="text-muted-foreground w-24">state:</span>
                  <span class={STATE_BADGES[session.state]?.color.replace('bg-', 'text-').replace('/20', '') || ''}>{session.state}</span>
                </div>
                {session.children.length > 0 && (
                  <div class="flex items-center gap-2">
                    <span class="text-muted-foreground w-24">children:</span>
                    <span class="text-blue-400">{session.children.length} forked</span>
                  </div>
                )}
                {session.forkReason && (
                  <div class="flex items-center gap-2">
                    <span class="text-muted-foreground w-24">fork_reason:</span>
                    <span>{session.forkReason}</span>
                  </div>
                )}
              </div>

              {/* Working memory summary */}
              {Object.keys(session.workingMemory).length > 0 && (
                <div class="mt-2 p-2 rounded bg-black/20 border border-[var(--glass-border-subtle)]">
                  <div class="text-xs text-muted-foreground mb-1">Working Memory</div>
                  <pre class="text-xs overflow-x-auto">
                    {JSON.stringify(session.workingMemory, null, 2).slice(0, 200)}
                    {JSON.stringify(session.workingMemory).length > 200 && '...'}
                  </pre>
                </div>
              )}

              {/* Last event */}
              {session.lastEvent && (
                <div class="mt-2 p-2 rounded bg-[var(--glass-bg-card)] border border-[var(--glass-border)]">
                  <div class="text-xs text-muted-foreground">Last Event</div>
                  <div class="text-xs font-mono text-cyan-400">{session.lastEvent.topic}</div>
                  <div class="text-xs text-muted-foreground">{session.lastEvent.timestamp}</div>
                  <div class="text-xs mt-1">{session.lastEvent.summary}</div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
});

// Grid wrapper for multiple InferCell cards
export const InferCellGrid = component$<{
  modules: ModuleInfo[];
  sessions?: Record<string, InferCellSession>;
  statuses?: Record<string, InferCellStatus>;
  liveData?: Record<string, LiveModuleData>;
  onTrigger$?: QRL<(action: string, module: ModuleInfo) => void>;
  compact?: boolean;
  columns?: number;
}>(({ modules, sessions, statuses, liveData, onTrigger$, compact = true, columns = 5 }) => {
  const gridCols = {
    2: 'grid-cols-2',
    3: 'grid-cols-2 md:grid-cols-3',
    4: 'grid-cols-2 md:grid-cols-4',
    5: 'grid-cols-2 md:grid-cols-5',
    6: 'grid-cols-2 md:grid-cols-3 lg:grid-cols-6',
  }[columns] || 'grid-cols-2 md:grid-cols-5';

  return (
    <div class={`grid ${gridCols} gap-2`}>
      {modules.map((mod) => (
        <InferCellCard
          key={mod.name}
          module={mod}
          status={statuses?.[mod.name] || 'idle'}
          session={sessions?.[mod.name]}
          liveData={liveData?.[mod.name]}
          onTrigger$={onTrigger$}
          compact={compact}
        />
      ))}
    </div>
  );
});
