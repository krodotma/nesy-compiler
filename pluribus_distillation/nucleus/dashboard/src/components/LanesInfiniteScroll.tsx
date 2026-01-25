/**
 * LanesInfiniteScroll - Infinite scroll wrapper for lanes data
 *
 * Phase 2, Iteration 7 of OITERATE lanes-widget-enhancement
 * Uses Metafizzy's infinite-scroll for paginated lane loading
 *
 * Qwik Integration Pattern:
 * - Dynamic import in useVisibleTask$ (browser-only)
 * - Container ref with useSignal
 * - Scroll sentinel element for triggering loads
 * - Proper cleanup on unmount
 *
 * @see https://infinite-scroll.com/
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  useComputed$,
  $,
  type Signal,
  Slot,
} from '@builder.io/qwik';

// M3 Components - LanesInfiniteScroll
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/progress/circular-progress.js';

// ============================================================================
// Types
// ============================================================================

interface LaneHistory {
  ts: string;
  wip_pct: number;
  note: string;
}

interface Lane {
  id: string;
  name: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  owner: string;
  description: string;
  commits: string[];
  blockers: string[];
  next_actions: string[];
  history: LaneHistory[];
  tier?: string;
  host?: string;
  slot?: number;
}

export interface LanesInfiniteScrollProps {
  /** Initial lanes to display */
  initialLanes: Lane[];
  /** Page size for pagination */
  pageSize?: number;
  /** Callback to fetch more lanes */
  onLoadMore$?: (page: number) => Promise<Lane[]>;
  /** Whether there are more lanes to load */
  hasMore?: boolean;
  /** Loading state signal from parent */
  loading?: Signal<boolean>;
  /** Max height for scroll container */
  maxHeight?: string;
  /** Enable infinite scroll (can be disabled for static lists) */
  enableInfiniteScroll?: boolean;
}

// ============================================================================
// Component
// ============================================================================

export const LanesInfiniteScroll = component$<LanesInfiniteScrollProps>(
  ({
    initialLanes,
    pageSize = 20,
    onLoadMore$,
    hasMore = false,
    loading,
    maxHeight = '600px',
    enableInfiniteScroll = true,
  }) => {
    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    const containerRef = useSignal<HTMLDivElement>();
    const sentinelRef = useSignal<HTMLDivElement>();
    const lanes = useSignal<Lane[]>(initialLanes);
    const currentPage = useSignal(1);
    const isLoading = useSignal(false);
    const hasMoreData = useSignal(hasMore);
    const infiniteScrollStatus = useSignal<'idle' | 'ready' | 'unavailable'>('idle');
    const infiniteScrollError = useSignal<string | null>(null);

    // Sync with parent loading signal if provided
    const effectiveLoading = useComputed$(() => {
      return loading?.value || isLoading.value;
    });

    // -------------------------------------------------------------------------
    // Load More Handler
    // -------------------------------------------------------------------------
    const loadMore = $(async () => {
      if (isLoading.value || !hasMoreData.value || !onLoadMore$) return;

      isLoading.value = true;
      try {
        const nextPage = currentPage.value + 1;
        const newLanes = await onLoadMore$(nextPage);

        if (newLanes.length > 0) {
          lanes.value = [...lanes.value, ...newLanes];
          currentPage.value = nextPage;
        }

        if (newLanes.length < pageSize) {
          hasMoreData.value = false;
        }
      } catch (error) {
        console.error('[LanesInfiniteScroll] Load more failed:', error);
        infiniteScrollError.value = String(error);
      } finally {
        isLoading.value = false;
      }
    });

    // -------------------------------------------------------------------------
    // Infinite Scroll Integration
    // -------------------------------------------------------------------------
    useVisibleTask$(async ({ cleanup }) => {
      if (!enableInfiniteScroll) {
        infiniteScrollStatus.value = 'idle';
        return;
      }

      const container = containerRef.value;
      const sentinel = sentinelRef.value;
      if (!container || !sentinel) return;

      // Use IntersectionObserver as a fallback/primary approach
      // This is more reliable in Qwik than the infinite-scroll library
      const observer = new IntersectionObserver(
        (entries) => {
          const entry = entries[0];
          if (entry.isIntersecting && hasMoreData.value && !isLoading.value) {
            loadMore();
          }
        },
        {
          root: container,
          rootMargin: '100px',
          threshold: 0.1,
        }
      );

      observer.observe(sentinel);
      infiniteScrollStatus.value = 'ready';

      cleanup(() => {
        observer.disconnect();
      });
    });

    // -------------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------------
    const stats = useComputed$(() => {
      const total = lanes.value.length;
      const greenCount = lanes.value.filter((l) => l.status === 'green').length;
      const yellowCount = lanes.value.filter((l) => l.status === 'yellow').length;
      const redCount = lanes.value.filter((l) => l.status === 'red').length;
      return { total, greenCount, yellowCount, redCount };
    });

    // -------------------------------------------------------------------------
    // Render
    // -------------------------------------------------------------------------
    return (
      <div class="rounded-lg border border-border bg-card">
        {/* Header */}
        <div class="flex items-center justify-between p-3 border-b border-border/50">
          <div class="flex items-center gap-2">
            <span class="text-sm font-semibold text-muted-foreground">LANES</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
              {stats.value.total} loaded
            </span>
            {enableInfiniteScroll && (
              <span
                class={`text-[10px] px-2 py-0.5 rounded border ${
                  infiniteScrollStatus.value === 'ready'
                    ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10'
                    : 'border-border bg-muted/30 text-muted-foreground'
                }`}
              >
                {infiniteScrollStatus.value === 'ready' ? 'scroll:on' : 'scroll:off'}
              </span>
            )}
          </div>
          <div class="flex items-center gap-2 text-[10px]">
            <span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {stats.value.greenCount}
            </span>
            <span class="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
              {stats.value.yellowCount}
            </span>
            <span class="px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
              {stats.value.redCount}
            </span>
          </div>
        </div>

        {/* Scrollable Container */}
        <div
          ref={containerRef}
          class="overflow-y-auto p-3"
          style={{ maxHeight }}
        >
          {/* Slot for lane items */}
          <Slot />

          {/* Default lane rendering if no slot content */}
          {lanes.value.length === 0 && (
            <div class="text-xs text-muted-foreground text-center py-8">
              No lanes to display
            </div>
          )}

          {/* Loading indicator */}
          {effectiveLoading.value && (
            <div class="flex items-center justify-center py-4 gap-2">
              <div class="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
              <span class="text-xs text-muted-foreground">Loading more lanes...</span>
            </div>
          )}

          {/* Sentinel element for infinite scroll trigger */}
          <div
            ref={sentinelRef}
            class="h-1 w-full"
            aria-hidden="true"
          />

          {/* End of list indicator */}
          {!hasMoreData.value && lanes.value.length > 0 && (
            <div class="text-center py-4">
              <span class="text-[10px] px-3 py-1 rounded-full bg-muted/30 text-muted-foreground border border-border/30">
                All {lanes.value.length} lanes loaded
              </span>
            </div>
          )}

          {/* Error state */}
          {infiniteScrollError.value && (
            <div class="text-xs text-red-400 text-center py-4 border border-red-500/30 bg-red-500/10 rounded-md mx-2 mb-2">
              Failed to load more: {infiniteScrollError.value}
              <button
                onClick$={loadMore}
                class="ml-2 underline hover:no-underline"
              >
                Retry
              </button>
            </div>
          )}
        </div>

        {/* Footer with manual load button */}
        {hasMoreData.value && !enableInfiniteScroll && (
          <div class="p-3 border-t border-border/50">
            <button
              onClick$={loadMore}
              disabled={isLoading.value}
              class="w-full text-xs py-2 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground border border-border/30 transition-colors disabled:opacity-50"
            >
              {isLoading.value ? 'Loading...' : `Load More (Page ${currentPage.value + 1})`}
            </button>
          </div>
        )}
      </div>
    );
  }
);

export default LanesInfiniteScroll;
