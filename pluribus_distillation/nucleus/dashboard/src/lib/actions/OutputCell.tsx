/**
 * OutputCell - Jupyter/Marimo-style output cell for action results
 *
 * Displays streaming outputs from server actions in a notebook-style cell.
 * Supports multiple output types: text, code, json, table, progress, error.
 */

import { component$, useComputed$, type PropFunction } from '@builder.io/qwik';
import type { ActionCell, ActionOutput, ActionStatus } from './types';

// Status indicator colors
const STATUS_COLORS: Record<ActionStatus, string> = {
  idle: 'bg-gray-500',
  pending: 'bg-yellow-500 animate-pulse',
  streaming: 'bg-blue-500 animate-pulse',
  success: 'bg-green-500',
  error: 'bg-red-500',
};

// Output type icons
const OUTPUT_ICONS: Record<ActionOutput['type'], string> = {
  text: '📝',
  code: '💻',
  json: '📦',
  table: '📊',
  progress: '⏳',
  error: '❌',
};

interface OutputCellProps {
  cell: ActionCell;
  onToggleCollapse$?: PropFunction<() => void>;
  onRerun$?: PropFunction<() => void>;
}

export const OutputCell = component$<OutputCellProps>(({ cell, onToggleCollapse$ }) => {
  const status = useComputed$(() => cell.result?.status || 'idle');
  const outputs = useComputed$(() => cell.result?.outputs || []);
  const events = useComputed$(() => cell.result?.events || []);
  const artifacts = useComputed$(() => 
    events.value.filter(ev => 
      ev.kind === 'artifact' || 
      (ev.data && typeof ev.data === 'object' && 'file_path' in ev.data)
    )
  );
  const duration = useComputed$(() => {
    if (!cell.result?.startedAt) return null;
    const end = cell.result.completedAt || Date.now();
    return ((end - cell.result.startedAt) / 1000).toFixed(2);
  });

  return (
    <div class="output-cell glass-surface glass-border-card rounded-lg mb-4 overflow-hidden">
      {/* Cell Header */}
      <div
        class="cell-header flex items-center justify-between px-4 py-2 bg-[var(--glass-bg-card)] border-b border-[var(--glass-border)] cursor-pointer hover:bg-[var(--glass-bg-hover)] transition-colors"
        onClick$={onToggleCollapse$}
      >
        <div class="flex items-center gap-3">
          {/* Status indicator */}
          <div class={`w-2 h-2 rounded-full ${STATUS_COLORS[status.value]}`} />

          {/* Action type badge */}
          <span class="text-xs font-mono bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded">
            {cell.request.type}
          </span>

          {/* Timestamp */}
          <span class="text-xs text-[var(--glass-text-tertiary)]">
            {new Date(cell.request.timestamp).toLocaleTimeString()}
          </span>
        </div>

        <div class="flex items-center gap-3">
          {/* Tab pill */}
          <span class="text-[11px] px-2 py-0.5 rounded bg-[var(--glass-bg-card)] border border-[var(--glass-border)] text-[var(--glass-text-secondary)]">
            {cell.activeTab}
          </span>

          {/* Duration */}
          {duration.value && (
            <span class="text-xs text-[var(--glass-text-tertiary)]">
              {duration.value}s
            </span>
          )}

          {/* Output count */}
          <span class="text-xs text-[var(--glass-text-tertiary)]">
            {outputs.value.length} output{outputs.value.length !== 1 ? 's' : ''}
          </span>

          {/* Artifact count */}
          {artifacts.value.length > 0 && (
            <span class="text-xs text-yellow-500/60">
              {artifacts.value.length} artifact{artifacts.value.length !== 1 ? 's' : ''}
            </span>
          )}

          {/* Trace count */}
          <span class="text-xs text-[var(--glass-text-tertiary)]">
            {events.value.length} event{events.value.length !== 1 ? 's' : ''}
          </span>

          {/* Collapse indicator */}
          <span class="text-[var(--glass-text-tertiary)]">
            {cell.collapsed ? '▶' : '▼'}
          </span>
        </div>
      </div>

      {/* Cell Body - collapsible */}
      {!cell.collapsed && (
        <div class="cell-body">
          {/* Tabs */}
          <div class="px-4 py-2 border-b border-[var(--glass-border)] bg-[var(--glass-bg-card)] flex items-center justify-between gap-3">
            <div class="flex items-center gap-2">
              <button
                class={`text-xs px-2 py-1 rounded border transition-colors ${
                  cell.activeTab === 'outputs'
                    ? 'bg-cyan-500/20 border-[var(--glass-accent-cyan-subtle)] text-cyan-200'
                    : 'bg-[var(--glass-bg-card)] border-[var(--glass-border)] text-[var(--glass-text-secondary)] hover:bg-[var(--glass-bg-hover)]'
                }`}
                onClick$={(e) => {
                  e.stopPropagation();
                  cell.activeTab = 'outputs';
                }}
              >
                Outputs
              </button>
              <button
                class={`text-xs px-2 py-1 rounded border transition-colors ${
                  cell.activeTab === 'artifacts'
                    ? 'bg-yellow-500/20 border-[var(--glass-accent-amber-subtle)] text-yellow-200'
                    : 'bg-[var(--glass-bg-card)] border-[var(--glass-border)] text-[var(--glass-text-secondary)] hover:bg-[var(--glass-bg-hover)]'
                }`}
                onClick$={(e) => {
                  e.stopPropagation();
                  cell.activeTab = 'artifacts';
                }}
              >
                Artifacts
              </button>
              <button
                class={`text-xs px-2 py-1 rounded border transition-colors ${
                  cell.activeTab === 'trace'
                    ? 'bg-purple-500/20 border-[var(--glass-accent-purple-subtle)] text-purple-200'
                    : 'bg-[var(--glass-bg-card)] border-[var(--glass-border)] text-[var(--glass-text-secondary)] hover:bg-[var(--glass-bg-hover)]'
                }`}
                onClick$={(e) => {
                  e.stopPropagation();
                  cell.activeTab = 'trace';
                }}
              >
                Trace
              </button>
              <button
                class={`text-xs px-2 py-1 rounded border transition-colors ${
                  cell.activeTab === 'request'
                    ? 'bg-[var(--glass-bg-hover)] border-[var(--glass-border-hover)] text-[var(--glass-text-primary)]'
                    : 'bg-[var(--glass-bg-card)] border-[var(--glass-border)] text-[var(--glass-text-secondary)] hover:bg-[var(--glass-bg-hover)]'
                }`}
                onClick$={(e) => {
                  e.stopPropagation();
                  cell.activeTab = 'request';
                }}
              >
                Request
              </button>
            </div>

            <div class="flex items-center gap-2">
              <button
                class="text-xs px-2 py-1 rounded bg-[var(--glass-bg-card)] border border-[var(--glass-border)] text-[var(--glass-text-secondary)] hover:bg-[var(--glass-bg-hover)]"
                onClick$={(e) => {
                  e.stopPropagation();
                  try {
                    navigator.clipboard?.writeText(JSON.stringify(cell.request.payload ?? {}, null, 2));
                  } catch {}
                }}
                title="Copy request JSON"
              >
                Copy
              </button>
            </div>
          </div>

          {cell.activeTab === 'request' && (
            <div class="p-4">
              <pre class="text-xs text-[var(--glass-text-secondary)] font-mono whitespace-pre-wrap">
                {JSON.stringify(cell.request.payload, null, 2)}
              </pre>
            </div>
          )}

          {cell.activeTab === 'artifacts' && (
            <div class="p-4 space-y-2">
              {artifacts.value.length === 0 ? (
                <div class="text-[var(--glass-text-tertiary)] text-sm italic">No artifacts produced.</div>
              ) : (
                artifacts.value.map((ev, idx) => (
                  <div key={idx} class="rounded border border-yellow-500/20 bg-yellow-500/5 px-3 py-2 flex items-start justify-between">
                    <div>
                      <div class="flex items-center gap-2 text-xs mb-1">
                        <span class="text-yellow-400 font-mono">
                          {(ev.data as any)?.file_path || (ev.data as any)?.id || 'Artifact'}
                        </span>
                        <span class="text-[var(--glass-text-tertiary)]">({ev.kind})</span>
                      </div>
                      <div class="text-xs text-[var(--glass-text-secondary)] line-clamp-2">
                         {JSON.stringify(ev.data)}
                      </div>
                    </div>
                    <button 
                      class="text-xs bg-[var(--glass-bg-active)] hover:bg-white/20 px-2 py-1 rounded text-[var(--glass-text-primary)]"
                      onClick$={() => {
                        // In a real app this would download or preview
                        navigator.clipboard?.writeText(JSON.stringify(ev.data, null, 2));
                      }}
                    >
                      Copy
                    </button>
                  </div>
                ))
              )}
            </div>
          )}

          {cell.activeTab === 'trace' && (
            <div class="p-4 space-y-2">
              {events.value.length === 0 ? (
                <div class="text-[var(--glass-text-tertiary)] text-sm italic">No correlated bus events yet.</div>
              ) : (
                events.value.slice(-100).map((ev, idx) => (
                  <div key={idx} class="rounded border border-[var(--glass-border)] bg-[var(--glass-bg-card)] px-3 py-2">
                    <div class="flex items-center gap-2 text-xs">
                      <span class="text-white/50 font-mono">{new Date(ev.ts).toLocaleTimeString()}</span>
                      <span class="text-[var(--glass-text-secondary)]">{ev.actor}</span>
                      <span class="text-cyan-300 font-mono">{ev.topic}</span>
                      {ev.level && <span class="text-[var(--glass-text-tertiary)]">({ev.level})</span>}
                    </div>
                    {(ev.semantic || ev.reasoning) && (
                      <div class="text-xs text-[var(--glass-text-secondary)] mt-1">
                        {ev.semantic || ev.reasoning}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {cell.activeTab === 'outputs' && (
            <div class="outputs p-4 space-y-3">
              {outputs.value.length === 0 && status.value === 'pending' && (
                <div class="text-[var(--glass-text-tertiary)] text-sm italic">Waiting for response...</div>
              )}

              {outputs.value.map((output, idx) => (
                <OutputDisplay key={idx} output={output} />
              ))}

              {/* Error display */}
              {cell.result?.error && (
                <div class="error-output bg-red-500/10 border border-red-500/30 rounded p-3">
                  <div class="text-red-400 font-mono text-sm">{cell.result.error}</div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
});

// Individual output renderer
interface OutputDisplayProps {
  output: ActionOutput;
}

const OutputDisplay = component$<OutputDisplayProps>(({ output }) => {
  const renderContent = useComputed$(() => {
    switch (output.type) {
      case 'text':
        return (
          <div class="text-output text-[var(--glass-text-primary)] text-sm whitespace-pre-wrap">
            {String(output.content)}
          </div>
        );

      case 'code':
        return (
          <div class="code-output bg-black/60 rounded p-3 overflow-x-auto">
            {output.metadata?.title && (
              <div class="text-xs text-[var(--glass-text-tertiary)] mb-2">{output.metadata.title}</div>
            )}
            <pre class="text-sm font-mono text-green-400">
              <code>{String(output.content)}</code>
            </pre>
            {output.metadata?.language && (
              <div class="text-xs text-[var(--glass-text-tertiary)] mt-2 text-right">
                {output.metadata.language}
              </div>
            )}
          </div>
        );

      case 'json':
        return (
          <div class="json-output bg-black/60 rounded p-3 overflow-x-auto">
            <pre class="text-sm font-mono text-yellow-400">
              {typeof output.content === 'string'
                ? output.content
                : JSON.stringify(output.content, null, 2)}
            </pre>
          </div>
        );

      case 'table':
        const data = output.content as Record<string, unknown>[];
        if (!Array.isArray(data) || data.length === 0) {
          return <div class="text-[var(--glass-text-tertiary)] text-sm">No data</div>;
        }
        const headers = Object.keys(data[0]);
        return (
          <div class="table-output overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-[var(--glass-border)]">
                  {headers.map((h) => (
                    <th key={h} class="text-left px-3 py-2 text-[var(--glass-text-secondary)] font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((row, i) => (
                  <tr key={i} class="border-b border-[var(--glass-border-subtle)] hover:bg-[var(--glass-bg-hover)]">
                    {headers.map((h) => (
                      <td key={h} class="px-3 py-2 text-[var(--glass-text-primary)]">
                        {String(row[h] ?? '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'progress':
        const progress = output.metadata?.progress || 0;
        const total = output.metadata?.total || 100;
        const pct = Math.round((progress / total) * 100);
        return (
          <div class="progress-output">
            <div class="flex items-center justify-between text-sm mb-1">
              <span class="text-[var(--glass-text-secondary)]">{String(output.content)}</span>
              <span class="text-[var(--glass-text-tertiary)]">{pct}%</span>
            </div>
            <div class="h-2 bg-[var(--glass-bg-active)] rounded-full overflow-hidden">
              <div
                class="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-300"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );

      case 'error':
        return (
          <div class="error-output bg-red-500/10 border border-red-500/30 rounded p-3">
            <div class="text-red-400 font-mono text-sm">{String(output.content)}</div>
          </div>
        );

      default:
        return (
          <div class="text-[var(--glass-text-secondary)] text-sm">
            Unknown output type: {output.type}
          </div>
        );
    }
  });

  return (
    <div class="output-item">
      <div class="flex items-center gap-2 mb-1">
        <span class="text-xs">{OUTPUT_ICONS[output.type]}</span>
        <span class="text-xs text-[var(--glass-text-tertiary)]">
          {new Date(output.timestamp).toLocaleTimeString()}
        </span>
      </div>
      {renderContent.value}
    </div>
  );
});

// Action button that triggers actions
interface ActionButtonProps {
  actionType: string;
  payload: Record<string, unknown>;
  label: string;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
  onDispatch?: (requestId: string) => void;
}

export const ActionButton = component$<ActionButtonProps>(({
  actionType,
  payload,
  label,
  variant = 'primary',
  disabled = false,
}) => {
  const variantStyles = {
    primary: 'bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border-cyan-500/30',
    secondary: 'bg-[var(--glass-bg-active)] hover:bg-white/20 text-[var(--glass-text-primary)] border-[var(--glass-border-active)]',
    danger: 'bg-red-500/20 hover:bg-red-500/30 text-red-300 border-red-500/30',
  };

  return (
    <button
      class={`px-3 py-1.5 text-sm font-medium rounded border transition-colors ${variantStyles[variant]} ${
        disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
      }`}
      disabled={disabled}
      data-action-type={actionType}
      data-action-payload={JSON.stringify(payload)}
    >
      {label}
    </button>
  );
});

// Output cells container
interface OutputCellsProps {
  cells: ActionCell[];
  onToggleCollapse?: (cellId: string) => void;
  onClear?: () => void;
}

export const OutputCells = component$<OutputCellsProps>(({ cells, onToggleCollapse, onClear }) => {
  return (
    <div class="output-cells">
      {/* Header with clear button */}
      {cells.length > 0 && (
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-sm font-medium text-[var(--glass-text-secondary)]">
            Action Results ({cells.length})
          </h3>
          {onClear && (
            <button
              class="text-xs text-[var(--glass-text-tertiary)] hover:text-[var(--glass-text-secondary)] transition-colors"
              onClick$={onClear}
            >
              Clear All
            </button>
          )}
        </div>
      )}

      {/* Cells */}
      <div class="space-y-4">
        {cells.map((cell) => (
          <OutputCell
            key={cell.id}
            cell={cell}
            onToggleCollapse$={() => onToggleCollapse?.(cell.id)}
          />
        ))}
      </div>

      {/* Empty state */}
      {cells.length === 0 && (
        <div class="text-center py-8 text-[var(--glass-text-tertiary)]">
          No action results yet. Run an action to see output here.
        </div>
      )}
    </div>
  );
});

export default OutputCell;
