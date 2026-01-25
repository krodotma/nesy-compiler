/**
 * LaneAccessibility - Accessibility utilities and components
 *
 * Phase 6, Iteration 48 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - ARIA labels and roles
 * - Keyboard navigation
 * - Screen reader support
 * - Focus management
 * - High contrast mode support
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  $,
  Slot,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - LaneAccessibility
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/focus/md-focus-ring.js';

// ============================================================================
// Types
// ============================================================================

export interface FocusTrapProps {
  /** Whether the trap is active */
  active?: boolean;
  /** Callback when escape is pressed */
  onEscape$?: QRL<() => void>;
  /** Initial focus element selector */
  initialFocus?: string;
  /** Return focus on deactivation */
  returnFocus?: boolean;
}

export interface KeyboardNavProps {
  /** Orientation of navigation */
  orientation?: 'horizontal' | 'vertical' | 'both';
  /** Loop navigation at boundaries */
  loop?: boolean;
  /** Items to navigate (selectors or elements) */
  itemSelector?: string;
  /** Callback when item is selected */
  onSelect$?: QRL<(index: number) => void>;
}

export interface ScreenReaderOnlyProps {
  /** Element tag */
  as?: 'span' | 'div' | 'p';
}

export interface LiveRegionProps {
  /** Politeness level */
  politeness?: 'polite' | 'assertive' | 'off';
  /** Atomic updates */
  atomic?: boolean;
  /** Relevant changes */
  relevant?: 'additions' | 'removals' | 'text' | 'all';
}

export interface SkipLinkProps {
  /** Target element ID */
  targetId: string;
  /** Link text */
  text?: string;
}

// ============================================================================
// Screen Reader Only Component
// ============================================================================

export const ScreenReaderOnly = component$<ScreenReaderOnlyProps>(({
  as = 'span',
}) => {
  const Tag = as;

  return (
    <Tag
      class="absolute w-px h-px p-0 -m-px overflow-hidden whitespace-nowrap border-0"
      style={{ clip: 'rect(0, 0, 0, 0)' }}
    >
      <Slot />
    </Tag>
  );
});

// ============================================================================
// Live Region Component
// ============================================================================

export const LiveRegion = component$<LiveRegionProps>(({
  politeness = 'polite',
  atomic = true,
  relevant = 'additions text',
}) => {
  return (
    <div
      aria-live={politeness}
      aria-atomic={atomic}
      aria-relevant={relevant}
      class="absolute w-px h-px p-0 -m-px overflow-hidden whitespace-nowrap border-0"
      style={{ clip: 'rect(0, 0, 0, 0)' }}
    >
      <Slot />
    </div>
  );
});

// ============================================================================
// Skip Link Component
// ============================================================================

export const SkipLink = component$<SkipLinkProps>(({
  targetId,
  text = 'Skip to main content',
}) => {
  return (
    <a
      href={`#${targetId}`}
      class="absolute -top-10 left-0 z-[100] px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-br focus:top-0 transition-all focus:outline-none focus:ring-2 focus:ring-primary"
    >
      {text}
    </a>
  );
});

// ============================================================================
// Focus Trap Component
// ============================================================================

export const FocusTrap = component$<FocusTrapProps>(({
  active = true,
  onEscape$,
  initialFocus,
  returnFocus = true,
}) => {
  const containerRef = useSignal<HTMLDivElement>();
  const previousFocusRef = useSignal<HTMLElement | null>(null);

  useVisibleTask$(({ cleanup }) => {
    if (!active || !containerRef.value) return;

    // Store current focus
    previousFocusRef.value = document.activeElement as HTMLElement;

    // Get focusable elements
    const getFocusable = () => {
      if (!containerRef.value) return [];
      return Array.from(
        containerRef.value.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )
      ).filter(el => !el.hasAttribute('disabled'));
    };

    // Set initial focus
    if (initialFocus) {
      const initial = containerRef.value.querySelector<HTMLElement>(initialFocus);
      initial?.focus();
    } else {
      const focusable = getFocusable();
      focusable[0]?.focus();
    }

    // Handle keydown
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onEscape$) {
        onEscape$();
        return;
      }

      if (e.key !== 'Tab') return;

      const focusable = getFocusable();
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    cleanup(() => {
      document.removeEventListener('keydown', handleKeyDown);

      // Return focus
      if (returnFocus && previousFocusRef.value) {
        previousFocusRef.value.focus();
      }
    });
  });

  return (
    <div ref={containerRef}>
      <Slot />
    </div>
  );
});

// ============================================================================
// Keyboard Navigation Component
// ============================================================================

export const KeyboardNav = component$<KeyboardNavProps>(({
  orientation = 'vertical',
  loop = true,
  itemSelector = '[role="option"], [role="menuitem"], button',
  onSelect$,
}) => {
  const containerRef = useSignal<HTMLDivElement>();
  const currentIndex = useSignal(0);

  useVisibleTask$(({ cleanup }) => {
    if (!containerRef.value) return;

    const getItems = () => {
      if (!containerRef.value) return [];
      return Array.from(
        containerRef.value.querySelectorAll<HTMLElement>(itemSelector)
      ).filter(el => !el.hasAttribute('disabled'));
    };

    const focusItem = (index: number) => {
      const items = getItems();
      if (items[index]) {
        items[index].focus();
        currentIndex.value = index;
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      const items = getItems();
      if (items.length === 0) return;

      const isVertical = orientation === 'vertical' || orientation === 'both';
      const isHorizontal = orientation === 'horizontal' || orientation === 'both';

      let newIndex = currentIndex.value;
      let handled = false;

      switch (e.key) {
        case 'ArrowDown':
          if (isVertical) {
            newIndex = loop
              ? (currentIndex.value + 1) % items.length
              : Math.min(currentIndex.value + 1, items.length - 1);
            handled = true;
          }
          break;
        case 'ArrowUp':
          if (isVertical) {
            newIndex = loop
              ? (currentIndex.value - 1 + items.length) % items.length
              : Math.max(currentIndex.value - 1, 0);
            handled = true;
          }
          break;
        case 'ArrowRight':
          if (isHorizontal) {
            newIndex = loop
              ? (currentIndex.value + 1) % items.length
              : Math.min(currentIndex.value + 1, items.length - 1);
            handled = true;
          }
          break;
        case 'ArrowLeft':
          if (isHorizontal) {
            newIndex = loop
              ? (currentIndex.value - 1 + items.length) % items.length
              : Math.max(currentIndex.value - 1, 0);
            handled = true;
          }
          break;
        case 'Home':
          newIndex = 0;
          handled = true;
          break;
        case 'End':
          newIndex = items.length - 1;
          handled = true;
          break;
        case 'Enter':
        case ' ':
          if (onSelect$) {
            e.preventDefault();
            onSelect$(currentIndex.value);
          }
          return;
      }

      if (handled) {
        e.preventDefault();
        focusItem(newIndex);
      }
    };

    containerRef.value.addEventListener('keydown', handleKeyDown);

    cleanup(() => {
      containerRef.value?.removeEventListener('keydown', handleKeyDown);
    });
  });

  return (
    <div ref={containerRef} role="listbox">
      <Slot />
    </div>
  );
});

// ============================================================================
// Accessible Lane Card Component
// ============================================================================

export interface AccessibleLaneCardProps {
  lane: {
    id: string;
    name: string;
    owner: string;
    status: 'green' | 'yellow' | 'red';
    wip_pct: number;
    blockers?: number;
    description?: string;
  };
  isSelected?: boolean;
  onClick$?: QRL<() => void>;
}

export const AccessibleLaneCard = component$<AccessibleLaneCardProps>(({
  lane,
  isSelected = false,
  onClick$,
}) => {
  const statusLabels: Record<string, string> = {
    green: 'On track',
    yellow: 'At risk',
    red: 'Blocked',
  };

  const statusColors: Record<string, string> = {
    green: 'bg-emerald-500/20 border-emerald-500/30',
    yellow: 'bg-amber-500/20 border-amber-500/30',
    red: 'bg-red-500/20 border-red-500/30',
  };

  const handleKeyDown = $((e: KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && onClick$) {
      e.preventDefault();
      onClick$();
    }
  });

  return (
    <div
      role="option"
      tabIndex={0}
      aria-selected={isSelected}
      aria-label={`Lane ${lane.name}, owned by ${lane.owner}, status ${statusLabels[lane.status]}, ${lane.wip_pct}% complete${lane.blockers ? `, ${lane.blockers} blockers` : ''}`}
      onClick$={onClick$}
      onKeyDown$={handleKeyDown}
      class={`p-3 rounded-lg border cursor-pointer transition-all focus:outline-none focus:ring-2 focus:ring-primary ${
        isSelected
          ? 'border-primary bg-primary/10'
          : `${statusColors[lane.status]} hover:border-primary/50`
      }`}
    >
      {/* Status indicator with accessible label */}
      <div class="flex items-center gap-2 mb-2">
        <div
          class={`w-2 h-2 rounded-full ${
            lane.status === 'green' ? 'bg-emerald-500' :
            lane.status === 'yellow' ? 'bg-amber-500' :
            'bg-red-500'
          }`}
          role="img"
          aria-label={statusLabels[lane.status]}
        />
        <span class="text-xs font-medium text-foreground">{lane.name}</span>
      </div>

      {/* Progress */}
      <div class="mb-2">
        <div class="flex items-center justify-between text-[9px] mb-1">
          <span class="text-muted-foreground">Progress</span>
          <span class="text-foreground font-bold">{lane.wip_pct}%</span>
        </div>
        <div
          role="progressbar"
          aria-valuenow={lane.wip_pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${lane.wip_pct}% complete`}
          class="h-1.5 rounded-full bg-muted/30 overflow-hidden"
        >
          <div
            class={`h-full rounded-full transition-all ${
              lane.status === 'green' ? 'bg-emerald-500' :
              lane.status === 'yellow' ? 'bg-amber-500' :
              'bg-red-500'
            }`}
            style={{ width: `${lane.wip_pct}%` }}
          />
        </div>
      </div>

      {/* Meta info */}
      <div class="flex items-center justify-between text-[9px] text-muted-foreground">
        <span>@{lane.owner}</span>
        {lane.blockers !== undefined && lane.blockers > 0 && (
          <span class="text-red-400" aria-label={`${lane.blockers} blockers`}>
            {lane.blockers} blockers
          </span>
        )}
      </div>

      {/* Hidden description for screen readers */}
      {lane.description && (
        <ScreenReaderOnly>
          <span>{lane.description}</span>
        </ScreenReaderOnly>
      )}
    </div>
  );
});

// ============================================================================
// High Contrast Mode Hook
// ============================================================================

export function useHighContrastMode() {
  const isHighContrast = useSignal(false);

  useVisibleTask$(() => {
    // Check for forced-colors media query (Windows High Contrast)
    const mediaQuery = window.matchMedia('(forced-colors: active)');
    isHighContrast.value = mediaQuery.matches;

    const handler = (e: MediaQueryListEvent) => {
      isHighContrast.value = e.matches;
    };

    mediaQuery.addEventListener('change', handler);

    return () => mediaQuery.removeEventListener('change', handler);
  });

  return isHighContrast;
}

// ============================================================================
// Reduced Motion Hook
// ============================================================================

export function useReducedMotion() {
  const prefersReducedMotion = useSignal(false);

  useVisibleTask$(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    prefersReducedMotion.value = mediaQuery.matches;

    const handler = (e: MediaQueryListEvent) => {
      prefersReducedMotion.value = e.matches;
    };

    mediaQuery.addEventListener('change', handler);

    return () => mediaQuery.removeEventListener('change', handler);
  });

  return prefersReducedMotion;
}

export default AccessibleLaneCard;
