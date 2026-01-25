/**
 * MetaTestWidget - Comprehensive Test Dashboard for METATEST
 *
 * Displays test inventory, coverage, and critical gaps across:
 * - Unit tests (nucleus/tools/tests/)
 * - E2E tests (Playwright specs)
 * - Service tests (systemd services)
 *
 * Protocol: DKIN v29 | PAIP v15 | CITIZEN v1
 */

import { component$, useSignal, useVisibleTask$, useComputed$, $ } from '@builder.io/qwik';
import { Button } from './ui/Button';

// M3 Components - MetaTestWidget
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/button/filled-tonal-button.js';

interface TestItem {
  name: string;
  path: string;
  type: string;
  target?: string;
}

interface CoverageItem {
  tool: string;
  path: string;
  has_test: boolean;
  test_file: string | null;
  tier: 'critical' | 'high' | 'medium' | 'low';
  status: string;
  critical: boolean;
  critical_reason: string | null;
}

interface Category {
  id: string;
  name: string;
  icon: string;
  count: number;
  coverage_percent?: number;
  items: (TestItem | CoverageItem)[];
}

interface MetaTestSummary {
  total_tools: number;
  total_tests: number;
  total_e2e: number;
  total_services: number;
  coverage_percent: number;
  tested_count: number;
  untested_count: number;
  critical_gaps: number;
}

interface MetaTestState {
  version: string;
  generated: string;
  summary: MetaTestSummary;
  categories: Category[];
  critical_gaps: CoverageItem[];
  protocol: string;
}

export interface MetaTestWidgetProps {
  refreshIntervalMs?: number;
  maxItems?: number;
}

/**
 * Generate a text-based coverage meter bar
 */
function coverageMeter(pct: number, width: number = 12): string {
  const clamped = Math.max(0, Math.min(100, pct));
  const filled = Math.round((clamped / 100) * width);
  const empty = width - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
}

type Tone = 'primary' | 'secondary' | 'tertiary' | 'error' | 'neutral';

const token = (name: string, fallback: string) => `var(${name}, ${fallback})`;

const tonePalette: Record<Tone, { text: string; container: string; border: string }> = {
  primary: {
    text: token('--md-sys-color-primary', 'var(--mat-primary)'),
    container: token('--md-sys-color-primary-container', 'oklch(from var(--mat-primary) l c h / 0.18)'),
    border: token('--md-sys-color-primary', 'var(--mat-primary)'),
  },
  secondary: {
    text: token('--md-sys-color-secondary', 'var(--mat-secondary)'),
    container: token('--md-sys-color-secondary-container', 'oklch(from var(--mat-secondary) l c h / 0.18)'),
    border: token('--md-sys-color-secondary', 'var(--mat-secondary)'),
  },
  tertiary: {
    text: token('--md-sys-color-tertiary', 'var(--mat-secondary)'),
    container: token('--md-sys-color-tertiary-container', 'oklch(from var(--mat-secondary) l c h / 0.18)'),
    border: token('--md-sys-color-tertiary', 'var(--mat-secondary)'),
  },
  error: {
    text: token('--md-sys-color-error', 'var(--destructive)'),
    container: token('--md-sys-color-error-container', 'oklch(from var(--destructive) l c h / 0.18)'),
    border: token('--md-sys-color-error', 'var(--destructive)'),
  },
  neutral: {
    text: token('--md-sys-color-on-surface-variant', 'var(--muted-foreground)'),
    container: token('--md-sys-color-surface-variant', 'var(--muted)'),
    border: token('--md-sys-color-outline-variant', 'var(--border)'),
  },
};

const toneTextStyle = (tone: Tone) => ({ color: tonePalette[tone].text });
const toneBadgeStyle = (tone: Tone) => ({
  color: tonePalette[tone].text,
  backgroundColor: tonePalette[tone].container,
  borderColor: tonePalette[tone].border,
});

function coverageTone(pct: number): Tone {
  if (pct >= 75) return 'tertiary';
  if (pct >= 50) return 'primary';
  if (pct >= 25) return 'secondary';
  return 'error';
}

export const MetaTestWidget = component$<MetaTestWidgetProps>(({
  refreshIntervalMs = 60000,
  maxItems = 10,
}) => {
  const state = useSignal<MetaTestState | null>(null);
  const lastGood = useSignal<MetaTestState | null>(null);
  const loading = useSignal(true);
  const error = useSignal<string | null>(null);
  const lastFetch = useSignal<string | null>(null);
  const expandedCategory = useSignal<string | null>(null);
  const stale = useSignal(false);

  // Fetch inventory from metatest_collector API
  const fetchInventory = $(async (forceRefresh = false) => {
    try {
      loading.value = true;
      const url = forceRefresh ? '/api/metatest/inventory?refresh=1' : '/api/metatest/inventory';
      const res = await fetch(url);
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      if (!data) {
        throw new Error('Invalid JSON response');
      }
      if (data.error) {
        throw new Error(String(data.error));
      }
      state.value = data;
      lastGood.value = data;
      lastFetch.value = new Date().toISOString();
      stale.value = Boolean(data.cache?.stale);
      error.value = null;
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch';
      if (lastGood.value) {
        state.value = lastGood.value;
        stale.value = true;
      } else {
        state.value = null;
        stale.value = false;
      }
    } finally {
      loading.value = false;
    }
  });

  // Initial fetch and periodic refresh
  useVisibleTask$(({ cleanup }) => {
    fetchInventory(false);
    const interval = setInterval(() => {
      fetchInventory(false);
    }, refreshIntervalMs);
    cleanup(() => clearInterval(interval));
  });

  // Computed stats
  const stats = useComputed$(() => {
    if (!state.value) {
      return { coverage: 0, tests: 0, gaps: 0, services: 0 };
    }
    const s = state.value.summary;
    return {
      coverage: s.coverage_percent,
      tests: s.total_tests + s.total_e2e,
      gaps: s.critical_gaps,
      services: s.total_services,
    };
  });

  const toggleCategory = $((categoryId: string) => {
    expandedCategory.value = expandedCategory.value === categoryId ? null : categoryId;
  });

  const coverageToneValue = coverageTone(stats.value.coverage);

  return (
    <div class="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <h3 class="text-sm font-semibold text-muted-foreground">METATEST</h3>
          <span class="text-[10px] px-2 py-0.5 rounded border" style={toneBadgeStyle(coverageToneValue)}>
            {stats.value.coverage}% coverage
          </span>
          {stats.value.gaps > 0 && (
            <span class="text-[10px] px-2 py-0.5 rounded border" style={toneBadgeStyle('error')}>
              {stats.value.gaps} critical gaps
            </span>
          )}
        </div>
        <div class="flex items-center gap-2">
          {lastFetch.value && (
            <span class="text-[9px] text-muted-foreground mono">
              {lastFetch.value.slice(11, 19)}
            </span>
          )}
          {stale.value && (
            <span class="text-[9px]" style={toneTextStyle('secondary')}>stale</span>
          )}
          <Button
            onClick$={() => fetchInventory(true)}
            variant="text"
            class="text-[10px]"
            title="Refresh inventory"
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats summary */}
      <div class="flex flex-wrap gap-2 mb-3 text-[10px]">
        <span class="px-2 py-0.5 rounded border" style={toneBadgeStyle('secondary')}>
          {state.value?.summary.total_tools ?? 0} tools
        </span>
        <span class="px-2 py-0.5 rounded border" style={toneBadgeStyle('primary')}>
          {stats.value.tests} tests
        </span>
        <span class="px-2 py-0.5 rounded border" style={toneBadgeStyle('tertiary')}>
          {stats.value.services} services
        </span>
      </div>

      {/* Coverage meter */}
      <div class="mb-3">
        <div class="flex items-center gap-2">
          <code class="text-[11px] mono" style={toneTextStyle(coverageToneValue)}>
            {coverageMeter(stats.value.coverage)}
          </code>
          <span class="text-[10px] font-bold" style={toneTextStyle(coverageToneValue)}>
            {stats.value.coverage}%
          </span>
          <span class="text-[9px] text-muted-foreground">
            ({state.value?.summary.tested_count ?? 0}/{state.value?.summary.total_tools ?? 0} tested)
          </span>
        </div>
      </div>

      {loading.value ? (
        <div class="text-xs text-muted-foreground animate-pulse">Loading inventory...</div>
      ) : error.value && !state.value ? (
        <div class="text-xs" style={toneTextStyle('error')}>{error.value}</div>
      ) : state.value ? (
        <div class="space-y-2">
          {/* Categories */}
          {state.value.categories.map((category) => (
            <div
              key={category.id}
              class="rounded-md border border-border/50 overflow-hidden"
            >
              <Button
                onClick$={() => toggleCategory(category.id)}
                variant="text"
                class="w-full"
              >
                <div class="flex items-center justify-between w-full">
                  <div class="flex items-center gap-2">
                    <span class="text-sm">{category.icon}</span>
                    <span class="text-xs font-medium text-foreground">{category.name}</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="text-[10px] px-1.5 py-0.5 rounded border" style={toneBadgeStyle('neutral')}>
                      {category.count}
                    </span>
                    {category.coverage_percent !== undefined && (
                      <span class="text-[10px]" style={toneTextStyle(coverageTone(category.coverage_percent))}>
                        {category.coverage_percent}%
                      </span>
                    )}
                    <span class="text-xs text-muted-foreground">
                      {expandedCategory.value === category.id ? '▼' : '▶'}
                    </span>
                  </div>
                </div>
              </Button>

              {expandedCategory.value === category.id && (
                <div class="border-t border-border/30 p-2 bg-muted/5">
                  {category.items.length === 0 ? (
                    <div class="text-[10px] text-muted-foreground">No items loaded.</div>
                  ) : (
                    <div class="space-y-1 max-h-40 overflow-y-auto">
                      {category.items.slice(0, maxItems).map((item: TestItem | CoverageItem, idx: number) => (
                        <div key={idx} class="flex items-center gap-2 text-[10px]">
                          {'has_test' in item ? (
                            <>
                              <span style={toneTextStyle(item.has_test ? 'tertiary' : 'error')}>
                                {item.has_test ? '✓' : '✗'}
                              </span>
                              <span class="text-foreground truncate flex-1">{item.tool}</span>
                              {item.critical && (
                                <span class="px-1 py-0.5 rounded border text-[8px]" style={toneBadgeStyle('error')}>
                                  CRITICAL
                                </span>
                              )}
                            </>
                          ) : (
                            <>
                              <span style={toneTextStyle('primary')}>•</span>
                              <span class="text-foreground truncate">{item.name}</span>
                            </>
                          )}
                        </div>
                      ))}
                      {category.items.length > maxItems && (
                        <div class="text-[9px] text-muted-foreground mt-1">
                          +{category.items.length - maxItems} more...
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Critical Gaps Alert */}
          {state.value.critical_gaps.length > 0 && (
            <div class="mt-3 pt-3 border-t border-border/50">
              <div class="flex items-center gap-2 mb-2">
                <span class="text-[10px] font-semibold" style={toneTextStyle('error')}>⚠️ CRITICAL GAPS</span>
              </div>
              <div class="space-y-1">
                {state.value.critical_gaps.slice(0, 5).map((gap, idx) => (
                  <div key={idx} class="flex items-center gap-2 text-[10px]">
                    <span style={toneTextStyle('error')}>✗</span>
                    <span class="text-foreground truncate flex-1">{gap.tool}</span>
                    <span class="text-[9px] text-muted-foreground truncate max-w-[150px]">
                      {gap.critical_reason}
                    </span>
                  </div>
                ))}
                {state.value.critical_gaps.length > 5 && (
                  <div class="text-[9px]" style={toneTextStyle('error')}>
                    +{state.value.critical_gaps.length - 5} more critical gaps
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      ) : null}

      {/* Footer */}
      <div class="mt-3 pt-2 border-t border-border/30 flex items-center justify-between">
        <span class="text-[8px] text-muted-foreground">
          {state.value?.protocol ?? 'DKIN v29 | PAIP v15 | CITIZEN v1'}
        </span>
        <span class="text-[8px] text-muted-foreground">
          v{state.value?.version ?? '1.0.0'}
        </span>
      </div>
    </div>
  );
});

export default MetaTestWidget;
