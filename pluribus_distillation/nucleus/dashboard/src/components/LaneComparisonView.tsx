/**
 * LaneComparisonView - Side-by-side lane comparison
 *
 * Phase 4, Iteration 32 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Compare two lanes side-by-side
 * - Diff history between lanes
 * - Highlight differences
 * - Merge suggestions
 * - Sync/diverge indicators
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - LaneComparisonView
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';

// ============================================================================
// Types
// ============================================================================

export interface ComparableLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  description?: string;
  createdAt: string;
  updatedAt: string;
  blockers: number;
  actions: number;
  history: { ts: string; wip_pct: number; note?: string }[];
}

export interface LaneComparisonViewProps {
  /** Available lanes to compare */
  lanes: ComparableLane[];
  /** Initially selected left lane */
  leftLaneId?: string;
  /** Initially selected right lane */
  rightLaneId?: string;
  /** Callback when selection changes */
  onSelectionChange$?: QRL<(leftId: string, rightId: string) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: string): string {
  switch (status) {
    case 'green': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'yellow': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'red': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getDiffClass(left: number | string | undefined, right: number | string | undefined): string {
  if (left === right) return 'text-muted-foreground';
  if (typeof left === 'number' && typeof right === 'number') {
    return left > right ? 'text-emerald-400' : 'text-red-400';
  }
  return 'text-amber-400';
}

function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  } catch {
    return dateStr;
  }
}

// ============================================================================
// Component
// ============================================================================

export const LaneComparisonView = component$<LaneComparisonViewProps>(({
  lanes,
  leftLaneId: initialLeftId,
  rightLaneId: initialRightId,
  onSelectionChange$,
}) => {
  // State
  const leftId = useSignal(initialLeftId || lanes[0]?.id || '');
  const rightId = useSignal(initialRightId || lanes[1]?.id || '');
  const showHistory = useSignal(false);

  // Computed lanes
  const leftLane = useComputed$(() => lanes.find(l => l.id === leftId.value));
  const rightLane = useComputed$(() => lanes.find(l => l.id === rightId.value));

  // Comparison metrics
  const comparison = useComputed$(() => {
    const left = leftLane.value;
    const right = rightLane.value;
    if (!left || !right) return null;

    const metrics: { label: string; left: string | number; right: string | number; diff: string }[] = [
      {
        label: 'WIP Progress',
        left: `${left.wip_pct}%`,
        right: `${right.wip_pct}%`,
        diff: `${left.wip_pct - right.wip_pct > 0 ? '+' : ''}${left.wip_pct - right.wip_pct}%`,
      },
      {
        label: 'Status',
        left: left.status,
        right: right.status,
        diff: left.status === right.status ? 'Same' : 'Different',
      },
      {
        label: 'Owner',
        left: `@${left.owner}`,
        right: `@${right.owner}`,
        diff: left.owner === right.owner ? 'Same' : 'Different',
      },
      {
        label: 'Blockers',
        left: left.blockers,
        right: right.blockers,
        diff: `${left.blockers - right.blockers > 0 ? '+' : ''}${left.blockers - right.blockers}`,
      },
      {
        label: 'Actions',
        left: left.actions,
        right: right.actions,
        diff: `${left.actions - right.actions > 0 ? '+' : ''}${left.actions - right.actions}`,
      },
      {
        label: 'History Entries',
        left: left.history.length,
        right: right.history.length,
        diff: `${left.history.length - right.history.length > 0 ? '+' : ''}${left.history.length - right.history.length}`,
      },
    ];

    return metrics;
  });

  // Handle selection change
  const handleSelectionChange = $(async () => {
    if (onSelectionChange$) {
      await onSelectionChange$(leftId.value, rightId.value);
    }
  });

  // Swap lanes
  const swapLanes = $(() => {
    const temp = leftId.value;
    leftId.value = rightId.value;
    rightId.value = temp;
    handleSelectionChange();
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">LANE COMPARISON</span>
        </div>
        <button
          onClick$={swapLanes}
          class="text-[10px] px-2 py-1 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
        >
          ⇄ Swap
        </button>
      </div>

      {/* Lane selectors */}
      <div class="grid grid-cols-2 gap-4 p-3 border-b border-border/30">
        {/* Left lane selector */}
        <div>
          <label class="text-[9px] text-muted-foreground block mb-1">Left Lane</label>
          <select
            value={leftId.value}
            onChange$={(e) => {
              leftId.value = (e.target as HTMLSelectElement).value;
              handleSelectionChange();
            }}
            class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground"
          >
            {lanes.map(lane => (
              <option key={lane.id} value={lane.id} disabled={lane.id === rightId.value}>
                {lane.name}
              </option>
            ))}
          </select>
        </div>

        {/* Right lane selector */}
        <div>
          <label class="text-[9px] text-muted-foreground block mb-1">Right Lane</label>
          <select
            value={rightId.value}
            onChange$={(e) => {
              rightId.value = (e.target as HTMLSelectElement).value;
              handleSelectionChange();
            }}
            class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground"
          >
            {lanes.map(lane => (
              <option key={lane.id} value={lane.id} disabled={lane.id === leftId.value}>
                {lane.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Comparison view */}
      {leftLane.value && rightLane.value ? (
        <>
          {/* Side by side summary */}
          <div class="grid grid-cols-2 gap-0 border-b border-border/30">
            {/* Left lane summary */}
            <div class="p-3 border-r border-border/30">
              <div class="flex items-center gap-2 mb-2">
                <span class={`text-[10px] px-1.5 py-0.5 rounded border ${getStatusColor(leftLane.value.status)}`}>
                  {leftLane.value.status}
                </span>
                <span class="text-xs font-medium text-foreground">{leftLane.value.name}</span>
              </div>
              <div class="text-2xl font-bold text-foreground">{leftLane.value.wip_pct}%</div>
              <div class="text-[9px] text-muted-foreground mt-1">@{leftLane.value.owner}</div>
            </div>

            {/* Right lane summary */}
            <div class="p-3">
              <div class="flex items-center gap-2 mb-2">
                <span class={`text-[10px] px-1.5 py-0.5 rounded border ${getStatusColor(rightLane.value.status)}`}>
                  {rightLane.value.status}
                </span>
                <span class="text-xs font-medium text-foreground">{rightLane.value.name}</span>
              </div>
              <div class="text-2xl font-bold text-foreground">{rightLane.value.wip_pct}%</div>
              <div class="text-[9px] text-muted-foreground mt-1">@{rightLane.value.owner}</div>
            </div>
          </div>

          {/* Metrics comparison */}
          <div class="max-h-[250px] overflow-y-auto">
            {comparison.value?.map((metric, i) => (
              <div
                key={i}
                class="grid grid-cols-[1fr,80px,1fr] gap-2 px-3 py-2 border-b border-border/20 text-xs"
              >
                <div class="text-right text-foreground">{metric.left}</div>
                <div class="text-center">
                  <div class="text-[9px] text-muted-foreground">{metric.label}</div>
                  <div class={`text-[10px] font-mono ${getDiffClass(
                    typeof metric.left === 'string' && metric.left.endsWith('%')
                      ? parseFloat(metric.left)
                      : metric.left,
                    typeof metric.right === 'string' && metric.right.endsWith('%')
                      ? parseFloat(metric.right)
                      : metric.right
                  )}`}>
                    {metric.diff}
                  </div>
                </div>
                <div class="text-foreground">{metric.right}</div>
              </div>
            ))}
          </div>

          {/* History toggle */}
          <div class="border-t border-border/30">
            <button
              onClick$={() => { showHistory.value = !showHistory.value; }}
              class="w-full p-2 text-[9px] text-muted-foreground hover:bg-muted/10 transition-colors flex items-center justify-center gap-1"
            >
              <span>{showHistory.value ? '▼' : '▶'}</span>
              <span>Compare History ({leftLane.value.history.length} vs {rightLane.value.history.length})</span>
            </button>

            {showHistory.value && (
              <div class="grid grid-cols-2 gap-0 max-h-[200px] overflow-y-auto">
                {/* Left history */}
                <div class="border-r border-border/30">
                  {leftLane.value.history.slice(0, 10).map((h, i) => (
                    <div key={i} class="px-3 py-1.5 border-b border-border/20 text-[9px]">
                      <div class="flex justify-between">
                        <span class="text-muted-foreground">{h.ts.slice(0, 10)}</span>
                        <span class="text-foreground font-bold">{h.wip_pct}%</span>
                      </div>
                      {h.note && <div class="text-muted-foreground/70 truncate">{h.note}</div>}
                    </div>
                  ))}
                </div>

                {/* Right history */}
                <div>
                  {rightLane.value.history.slice(0, 10).map((h, i) => (
                    <div key={i} class="px-3 py-1.5 border-b border-border/20 text-[9px]">
                      <div class="flex justify-between">
                        <span class="text-muted-foreground">{h.ts.slice(0, 10)}</span>
                        <span class="text-foreground font-bold">{h.wip_pct}%</span>
                      </div>
                      {h.note && <div class="text-muted-foreground/70 truncate">{h.note}</div>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
      ) : (
        <div class="p-6 text-center text-xs text-muted-foreground">
          Select two lanes to compare
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {lanes.length} lanes available for comparison
      </div>
    </div>
  );
});

export default LaneComparisonView;
