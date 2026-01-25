/**
 * ExpandableCard - Progressive Disclosure Card Pattern
 * Phase 4, Step 103 - Progressive Disclosure Implementation
 *
 * Card with "More Details" expansion for progressive information disclosure
 */

import { component$, Slot, useSignal, type PropFunction } from '@builder.io/qwik';

// M3 Components
import '@material/web/ripple/ripple.js';
import '@material/web/elevation/elevation.js';

export interface ExpandableCardProps {
  /** Card title */
  title?: string;
  /** Card subtitle */
  subtitle?: string;
  /** Icon */
  icon?: string;
  /** Default expanded state */
  defaultExpanded?: boolean;
  /** Controlled expanded state */
  expanded?: boolean;
  /** Callback when expansion changes */
  onToggle$?: PropFunction<(expanded: boolean) => void>;
  /** Expand button label */
  expandLabel?: string;
  /** Collapse button label */
  collapseLabel?: string;
  /** Show expand button in footer */
  footerExpand?: boolean;
  /** Interactive card (clickable header) */
  interactive?: boolean;
  /** Status indicator */
  status?: 'ok' | 'warning' | 'error' | 'loading';
  /** Additional class names */
  class?: string;
  /** Card padding */
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

const STATUS_STYLES: Record<string, { dot: string; border: string }> = {
  ok: {
    dot: 'bg-[var(--glass-status-success)] shadow-[0_0_8px_var(--glass-status-success)]',
    border: 'border-[var(--glass-status-success)]/30',
  },
  warning: {
    dot: 'bg-[var(--glass-status-warning)] shadow-[0_0_8px_var(--glass-status-warning)]',
    border: 'border-[var(--glass-status-warning)]/30',
  },
  error: {
    dot: 'bg-[var(--glass-status-error)] shadow-[0_0_8px_var(--glass-status-error)]',
    border: 'border-[var(--glass-status-error)]/30',
  },
  loading: {
    dot: 'bg-[var(--glass-accent-cyan)] shadow-[0_0_8px_var(--glass-accent-cyan)] animate-pulse',
    border: 'border-[var(--glass-accent-cyan)]/30',
  },
};

const PADDING_STYLES = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
};

/**
 * Expandable Card with progressive disclosure
 *
 * Usage:
 * ```tsx
 * <ExpandableCard
 *   title="Service Status"
 *   subtitle="All systems operational"
 *   icon="🟢"
 *   status="ok"
 *   defaultExpanded={false}
 * >
 *   <p q:slot="summary">3 services running</p>
 *
 *   <div q:slot="details">
 *     <ul>
 *       <li>API Server: Running</li>
 *       <li>Database: Connected</li>
 *       <li>Cache: Active</li>
 *     </ul>
 *   </div>
 * </ExpandableCard>
 * ```
 */
export const ExpandableCard = component$<ExpandableCardProps>((props) => {
  const internalExpanded = useSignal(props.defaultExpanded ?? false);
  const isExpanded = props.expanded !== undefined ? props.expanded : internalExpanded.value;
  const padding = PADDING_STYLES[props.padding || 'md'];
  const statusStyle = props.status ? STATUS_STYLES[props.status] : null;

  const handleToggle = $(() => {
    const newState = !isExpanded;
    internalExpanded.value = newState;
    props.onToggle$?.(newState);
  });

  return (
    <div
      class={`
        relative
        glass-surface-elevated
        rounded-xl
        overflow-hidden
        glass-transition-all
        ${statusStyle?.border || ''}
        ${props.class || ''}
      `}
    >
      <md-elevation class="rounded-[inherit]" />

      {/* Header */}
      {(props.title || props.icon) && (
        <div
          class={`
            relative
            flex items-center gap-3
            ${padding}
            ${props.interactive ? 'cursor-pointer hover:bg-[var(--glass-state-layer-hover)]' : ''}
          `}
          onClick$={props.interactive ? handleToggle : undefined}
        >
          {props.interactive && <md-ripple />}

          {/* Status dot */}
          {statusStyle && (
            <div class={`w-2 h-2 rounded-full flex-shrink-0 ${statusStyle.dot}`} />
          )}

          {/* Icon */}
          {props.icon && !statusStyle && (
            <span class="text-xl flex-shrink-0">{props.icon}</span>
          )}

          {/* Title/subtitle */}
          <div class="flex-grow min-w-0">
            {props.title && (
              <h3 class="text-sm font-semibold text-[var(--glass-text-primary)] truncate">
                {props.title}
              </h3>
            )}
            {props.subtitle && (
              <p class="text-xs text-[var(--glass-text-tertiary)] truncate mt-0.5">
                {props.subtitle}
              </p>
            )}
          </div>

          {/* Header slot */}
          <Slot name="header" />

          {/* Inline expand toggle */}
          {!props.footerExpand && (
            <button
              class={`
                text-[var(--glass-text-tertiary)]
                glass-transition-transform
                ${isExpanded ? 'rotate-180' : ''}
              `}
              onClick$={(e) => {
                e.stopPropagation();
                handleToggle();
              }}
              aria-expanded={isExpanded}
            >
              ▼
            </button>
          )}
        </div>
      )}

      {/* Summary content (always visible) */}
      <div class={`${padding} ${props.title ? 'pt-0' : ''}`}>
        <Slot name="summary" />
        <Slot />
      </div>

      {/* Expandable details */}
      <div
        class={`
          grid
          glass-transition-all
          ${isExpanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}
        `}
      >
        <div class="overflow-hidden">
          {/* Divider */}
          <div class="h-px bg-gradient-to-r from-transparent via-[var(--glass-border)] to-transparent mx-4" />

          {/* Details content */}
          <div class={`${padding} glass-animate-enter`}>
            <Slot name="details" />
          </div>
        </div>
      </div>

      {/* Footer with expand button */}
      {props.footerExpand && (
        <div class={`${padding} pt-0 flex justify-center`}>
          <button
            class={`
              text-xs
              text-[var(--glass-accent-cyan)]
              hover:text-[var(--glass-accent-cyan)]
              hover:underline
              glass-transition-colors
              flex items-center gap-1
            `}
            onClick$={handleToggle}
          >
            {isExpanded
              ? (props.collapseLabel || 'Show less')
              : (props.expandLabel || 'Show more')
            }
            <span class={`transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
              ▼
            </span>
          </button>
        </div>
      )}
    </div>
  );
});

/**
 * QuickStatsCard - Compact stats card with expand for details
 */
export const QuickStatsCard = component$<{
  label: string;
  value: string | number;
  subValue?: string;
  icon?: string;
  trend?: 'up' | 'down' | 'neutral';
  details?: string;
  class?: string;
}>((props) => {
  const showDetails = useSignal(false);

  const trendColor = props.trend === 'up'
    ? 'text-[var(--glass-status-success)]'
    : props.trend === 'down'
      ? 'text-[var(--glass-status-error)]'
      : 'text-[var(--glass-text-tertiary)]';

  const trendIcon = props.trend === 'up' ? '↑' : props.trend === 'down' ? '↓' : '→';

  return (
    <div
      class={`
        glass-surface
        rounded-xl
        p-4
        ${props.details ? 'cursor-pointer hover:bg-[var(--glass-state-layer-hover)]' : ''}
        glass-transition-all
        ${props.class || ''}
      `}
      onClick$={props.details ? () => showDetails.value = !showDetails.value : undefined}
    >
      <div class="flex items-start justify-between">
        {props.icon && (
          <span class="text-2xl">{props.icon}</span>
        )}
        <div class="text-right flex-grow">
          <div class="text-2xl font-bold text-[var(--glass-text-primary)]">
            {props.value}
          </div>
          {props.subValue && (
            <div class={`text-xs flex items-center justify-end gap-1 ${trendColor}`}>
              <span>{trendIcon}</span>
              <span>{props.subValue}</span>
            </div>
          )}
        </div>
      </div>
      <div class="text-xs text-[var(--glass-text-tertiary)] mt-2 uppercase tracking-wider">
        {props.label}
      </div>

      {/* Expandable details */}
      {props.details && showDetails.value && (
        <div class="mt-3 pt-3 border-t border-[var(--glass-border)] text-xs text-[var(--glass-text-secondary)] glass-animate-enter">
          {props.details}
        </div>
      )}
    </div>
  );
});

/**
 * RevealOnHover - Content that reveals more on hover
 */
export const RevealOnHover = component$<{
  class?: string;
}>((props) => {
  return (
    <div
      class={`
        group
        relative
        overflow-hidden
        ${props.class || ''}
      `}
    >
      {/* Always visible content */}
      <Slot name="visible" />

      {/* Revealed on hover */}
      <div
        class={`
          absolute inset-x-0 bottom-0
          opacity-0 translate-y-full
          group-hover:opacity-100 group-hover:translate-y-0
          glass-transition-all
          bg-gradient-to-t from-[var(--glass-bg-overlay)] to-transparent
          p-4 pt-8
        `}
      >
        <Slot name="hidden" />
      </div>
    </div>
  );
});

export default ExpandableCard;
