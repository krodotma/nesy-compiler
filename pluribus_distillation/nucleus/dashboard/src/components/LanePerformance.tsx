/**
 * LanePerformance - Performance optimization utilities and components
 *
 * Phase 6, Iteration 47 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Virtualized list rendering
 * - Debounced updates
 * - Memoized computations
 * - Lazy loading wrappers
 * - Performance metrics display
 */

import {
  component$,
  useSignal,
  useComputed$,
  useVisibleTask$,
  $,
  type QRL,
  Slot,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface VirtualizedItem {
  id: string;
  height?: number;
}

export interface VirtualizedListProps<T extends VirtualizedItem> {
  /** Items to render */
  items: T[];
  /** Fixed item height (required for virtual scrolling) */
  itemHeight: number;
  /** Container height */
  containerHeight: number;
  /** Overscan count (extra items to render) */
  overscan?: number;
  /** Render function for each item */
  renderItem$: QRL<(item: T, index: number) => unknown>;
}

export interface PerformanceMetrics {
  renderTime: number;
  updateCount: number;
  memoryUsage?: number;
  fps?: number;
}

export interface PerformanceMonitorProps {
  /** Enable monitoring */
  enabled?: boolean;
  /** Update interval in ms */
  interval?: number;
  /** Show detailed metrics */
  showDetails?: boolean;
}

// ============================================================================
// Debounce Hook
// ============================================================================

export function useDebounce<T>(value: T, delay: number): T {
  const debouncedValue = useSignal<T>(value);

  useVisibleTask$(({ track, cleanup }) => {
    track(() => value);

    const timer = setTimeout(() => {
      debouncedValue.value = value;
    }, delay);

    cleanup(() => clearTimeout(timer));
  });

  return debouncedValue.value;
}

// ============================================================================
// Throttle Hook
// ============================================================================

export function useThrottle<T>(value: T, limit: number): T {
  const throttledValue = useSignal<T>(value);
  const lastRan = useSignal(Date.now());

  useVisibleTask$(({ track }) => {
    track(() => value);

    const now = Date.now();
    if (now - lastRan.value >= limit) {
      throttledValue.value = value;
      lastRan.value = now;
    }
  });

  return throttledValue.value;
}

// ============================================================================
// Virtualized List Component
// ============================================================================

export const VirtualizedList = component$(<T extends VirtualizedItem>(props: VirtualizedListProps<T>) => {
  const {
    items,
    itemHeight,
    containerHeight,
    overscan = 3,
    renderItem$,
  } = props;

  // State
  const scrollTop = useSignal(0);
  const containerRef = useSignal<HTMLDivElement>();

  // Calculate visible range
  const visibleRange = useComputed$(() => {
    const startIndex = Math.max(0, Math.floor(scrollTop.value / itemHeight) - overscan);
    const endIndex = Math.min(
      items.length - 1,
      Math.ceil((scrollTop.value + containerHeight) / itemHeight) + overscan
    );
    return { startIndex, endIndex };
  });

  // Total height
  const totalHeight = items.length * itemHeight;

  // Handle scroll
  const handleScroll = $((e: Event) => {
    const target = e.target as HTMLDivElement;
    scrollTop.value = target.scrollTop;
  });

  return (
    <div
      ref={containerRef}
      onScroll$={handleScroll}
      style={{ height: `${containerHeight}px`, overflow: 'auto' }}
      class="relative"
    >
      {/* Spacer for total height */}
      <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
        {/* Render visible items */}
        {items.slice(visibleRange.value.startIndex, visibleRange.value.endIndex + 1).map((item, idx) => {
          const actualIndex = visibleRange.value.startIndex + idx;
          return (
            <div
              key={item.id}
              style={{
                position: 'absolute',
                top: `${actualIndex * itemHeight}px`,
                left: 0,
                right: 0,
                height: `${itemHeight}px`,
              }}
            >
              {renderItem$(item, actualIndex)}
            </div>
          );
        })}
      </div>
    </div>
  );
});

// ============================================================================
// Lazy Load Wrapper
// ============================================================================

export interface LazyLoadProps {
  /** Threshold for intersection observer (0-1) */
  threshold?: number;
  /** Root margin */
  rootMargin?: string;
  /** Placeholder height */
  placeholderHeight?: number;
  /** Loading state render */
  loadingSlot?: boolean;
}

export const LazyLoad = component$<LazyLoadProps>(({
  threshold = 0.1,
  rootMargin = '100px',
  placeholderHeight = 100,
}) => {
  const isVisible = useSignal(false);
  const hasLoaded = useSignal(false);
  const containerRef = useSignal<HTMLDivElement>();

  useVisibleTask$(({ cleanup }) => {
    if (!containerRef.value) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !hasLoaded.value) {
            isVisible.value = true;
            hasLoaded.value = true;
            observer.disconnect();
          }
        });
      },
      { threshold, rootMargin }
    );

    observer.observe(containerRef.value);

    cleanup(() => observer.disconnect());
  });

  return (
    <div ref={containerRef}>
      {isVisible.value ? (
        <Slot />
      ) : (
        <div
          class="animate-pulse bg-muted/20 rounded"
          style={{ height: `${placeholderHeight}px` }}
        />
      )}
    </div>
  );
});

// ============================================================================
// Performance Monitor Component
// ============================================================================

export const PerformanceMonitor = component$<PerformanceMonitorProps>(({
  enabled = true,
  interval = 1000,
  showDetails = false,
}) => {
  const metrics = useSignal<PerformanceMetrics>({
    renderTime: 0,
    updateCount: 0,
    memoryUsage: 0,
    fps: 60,
  });

  const frameCount = useSignal(0);
  const lastFrameTime = useSignal(performance.now());

  useVisibleTask$(({ cleanup }) => {
    if (!enabled) return;

    let animationId: number;
    let updateIntervalId: ReturnType<typeof setInterval>;

    // FPS counter
    const measureFPS = () => {
      frameCount.value++;
      animationId = requestAnimationFrame(measureFPS);
    };
    animationId = requestAnimationFrame(measureFPS);

    // Update metrics periodically
    updateIntervalId = setInterval(() => {
      const now = performance.now();
      const elapsed = now - lastFrameTime.value;
      const fps = Math.round((frameCount.value / elapsed) * 1000);

      metrics.value = {
        ...metrics.value,
        fps,
        updateCount: metrics.value.updateCount + 1,
        memoryUsage: (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize
          ? Math.round((performance as unknown as { memory: { usedJSHeapSize: number } }).memory.usedJSHeapSize / 1024 / 1024)
          : undefined,
      };

      frameCount.value = 0;
      lastFrameTime.value = now;
    }, interval);

    cleanup(() => {
      cancelAnimationFrame(animationId);
      clearInterval(updateIntervalId);
    });
  });

  if (!enabled) return null;

  return (
    <div class="fixed bottom-4 right-4 z-50">
      <div class="rounded-lg border border-border bg-card/95 backdrop-blur-sm shadow-lg p-2">
        <div class="flex items-center gap-3 text-[9px]">
          {/* FPS */}
          <div class="flex items-center gap-1">
            <span class={`font-mono font-bold ${
              (metrics.value.fps || 0) >= 55 ? 'text-emerald-400' :
              (metrics.value.fps || 0) >= 30 ? 'text-amber-400' :
              'text-red-400'
            }`}>
              {metrics.value.fps || '-'}
            </span>
            <span class="text-muted-foreground">FPS</span>
          </div>

          {/* Memory */}
          {metrics.value.memoryUsage !== undefined && (
            <div class="flex items-center gap-1">
              <span class="font-mono text-foreground">{metrics.value.memoryUsage}</span>
              <span class="text-muted-foreground">MB</span>
            </div>
          )}

          {showDetails && (
            <div class="flex items-center gap-1">
              <span class="font-mono text-foreground">{metrics.value.updateCount}</span>
              <span class="text-muted-foreground">updates</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// Optimized Lane List Component
// ============================================================================

export interface OptimizedLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
}

export interface OptimizedLaneListProps {
  /** Lanes to render */
  lanes: OptimizedLane[];
  /** Container height */
  height?: number;
  /** Callback when lane is clicked */
  onLaneClick$?: QRL<(lane: OptimizedLane) => void>;
}

export const OptimizedLaneList = component$<OptimizedLaneListProps>(({
  lanes,
  height = 400,
  onLaneClick$,
}) => {
  const itemHeight = 60;

  const renderLane = $((lane: OptimizedLane, _index: number) => {
    const statusColors: Record<string, string> = {
      green: 'bg-emerald-500',
      yellow: 'bg-amber-500',
      red: 'bg-red-500',
    };

    return (
      <div
        class="p-2 border-b border-border/30 hover:bg-muted/10 cursor-pointer transition-colors"
        onClick$={() => onLaneClick$ && onLaneClick$(lane)}
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <div class={`w-2 h-2 rounded-full ${statusColors[lane.status]}`} />
            <span class="text-xs font-medium text-foreground">{lane.name}</span>
          </div>
          <span class={`text-[10px] font-bold ${
            lane.wip_pct >= 90 ? 'text-emerald-400' :
            lane.wip_pct >= 50 ? 'text-cyan-400' :
            'text-amber-400'
          }`}>
            {lane.wip_pct}%
          </span>
        </div>
        <div class="text-[9px] text-muted-foreground mt-1">@{lane.owner}</div>
      </div>
    );
  });

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      <div class="p-2 border-b border-border/50 bg-muted/5">
        <span class="text-[9px] font-semibold text-muted-foreground">
          {lanes.length} LANES (VIRTUALIZED)
        </span>
      </div>
      <VirtualizedList
        items={lanes}
        itemHeight={itemHeight}
        containerHeight={height}
        overscan={5}
        renderItem$={renderLane}
      />
    </div>
  );
});

export default VirtualizedList;
