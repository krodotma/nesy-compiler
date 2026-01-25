/**
 * CollapsibleSection - Progressive Disclosure Pattern
 * Phase 4, Step 101 - Progressive Disclosure Implementation
 *
 * Collapsible section with glass styling and smooth animations
 */

import { component$, Slot, useSignal, type PropFunction } from '@builder.io/qwik';

// M3 Ripple for interactive header
import '@material/web/ripple/ripple.js';

export interface CollapsibleSectionProps {
  /** Section title */
  title: string;
  /** Optional subtitle */
  subtitle?: string;
  /** Icon to display before title */
  icon?: string;
  /** Whether section is expanded by default */
  defaultExpanded?: boolean;
  /** Controlled expanded state */
  expanded?: boolean;
  /** Callback when expansion state changes */
  onToggle$?: PropFunction<(expanded: boolean) => void>;
  /** Title style variant */
  titleVariant?: 'neon' | 'default' | 'muted';
  /** Badge content (count, status, etc.) */
  badge?: string | number;
  /** Badge color */
  badgeColor?: 'cyan' | 'magenta' | 'amber' | 'emerald' | 'purple' | 'error';
  /** Whether the section can be collapsed */
  collapsible?: boolean;
  /** Additional header content (right side) */
  headerAction?: string;
  /** Additional class names */
  class?: string;
  /** Border style */
  bordered?: boolean;
}

const BADGE_COLORS: Record<string, string> = {
  cyan: 'bg-[var(--glass-accent-cyan)]/20 text-[var(--glass-accent-cyan)] border-[var(--glass-accent-cyan)]/30',
  magenta: 'bg-[var(--glass-accent-magenta)]/20 text-[var(--glass-accent-magenta)] border-[var(--glass-accent-magenta)]/30',
  amber: 'bg-[var(--glass-accent-amber)]/20 text-[var(--glass-accent-amber)] border-[var(--glass-accent-amber)]/30',
  emerald: 'bg-[var(--glass-accent-emerald)]/20 text-[var(--glass-accent-emerald)] border-[var(--glass-accent-emerald)]/30',
  purple: 'bg-[var(--glass-accent-purple)]/20 text-[var(--glass-accent-purple)] border-[var(--glass-accent-purple)]/30',
  error: 'bg-[var(--glass-status-error)]/20 text-[var(--glass-status-error)] border-[var(--glass-status-error)]/30',
};

const TITLE_VARIANTS = {
  neon: 'text-[var(--glass-accent-cyan)] text-shadow-[0_0_10px_var(--glass-accent-cyan-glow)] font-semibold uppercase tracking-wider text-xs',
  default: 'text-[var(--glass-text-primary)] font-semibold text-sm',
  muted: 'text-[var(--glass-text-secondary)] font-medium text-xs uppercase tracking-wider',
};

/**
 * Collapsible Section with glass styling
 *
 * Usage:
 * ```tsx
 * <CollapsibleSection
 *   title="SYSTEM STATUS"
 *   icon="📊"
 *   titleVariant="neon"
 *   defaultExpanded={false}
 *   badge={3}
 *   badgeColor="cyan"
 * >
 *   <p>Detailed status content here...</p>
 * </CollapsibleSection>
 * ```
 */
export const CollapsibleSection = component$<CollapsibleSectionProps>((props) => {
  const internalExpanded = useSignal(props.defaultExpanded ?? false);

  // Use controlled or internal state
  const isExpanded = props.expanded !== undefined ? props.expanded : internalExpanded.value;
  const collapsible = props.collapsible !== false;

  const handleToggle = $(() => {
    if (!collapsible) return;

    const newState = !isExpanded;
    internalExpanded.value = newState;
    props.onToggle$?.(newState);
  });

  const titleClass = TITLE_VARIANTS[props.titleVariant || 'default'];
  const badgeClass = BADGE_COLORS[props.badgeColor || 'cyan'];

  return (
    <div
      class={`
        glass-surface
        overflow-hidden
        glass-transition-all
        ${props.bordered ? 'border border-[var(--glass-border)]' : 'border-none'}
        ${props.class || ''}
      `}
    >
      {/* Header */}
      <button
        class={`
          relative
          w-full
          flex items-center justify-between
          p-4
          text-left
          ${collapsible ? 'cursor-pointer hover:bg-[var(--glass-state-layer-hover)]' : 'cursor-default'}
          glass-transition-colors
        `}
        onClick$={handleToggle}
        aria-expanded={isExpanded}
        disabled={!collapsible}
      >
        {collapsible && <md-ripple class="rounded-t-[inherit]" />}

        <div class="flex items-center gap-3 flex-grow">
          {/* Icon */}
          {props.icon && (
            <span class="text-lg flex-shrink-0">{props.icon}</span>
          )}

          {/* Title and subtitle */}
          <div class="flex-grow min-w-0">
            <div class="flex items-center gap-2">
              <span class={titleClass}>{props.title}</span>
              {props.badge !== undefined && (
                <span class={`px-1.5 py-0.5 text-[10px] rounded-full border ${badgeClass}`}>
                  {props.badge}
                </span>
              )}
            </div>
            {props.subtitle && (
              <p class="text-[10px] text-[var(--glass-text-tertiary)] mt-0.5 truncate">
                {props.subtitle}
              </p>
            )}
          </div>
        </div>

        {/* Right side action/content */}
        <div class="flex items-center gap-2">
          <Slot name="header-action" />

          {/* Expand/collapse indicator */}
          {collapsible && (
            <span
              class={`
                text-[var(--glass-text-tertiary)]
                glass-transition-transform
                ${isExpanded ? 'rotate-180' : 'rotate-0'}
              `}
            >
              ▼
            </span>
          )}
        </div>
      </button>

      {/* Divider */}
      {isExpanded && (
        <div class="h-px bg-gradient-to-r from-transparent via-[var(--glass-border)] to-transparent" />
      )}

      {/* Content with smooth expand/collapse */}
      <div
        class={`
          grid
          glass-transition-all
          ${isExpanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}
        `}
      >
        <div class="overflow-hidden">
          <div class="p-4 glass-animate-enter">
            <Slot />
          </div>
        </div>
      </div>
    </div>
  );
});

/**
 * Accordion - Multiple collapsible sections where only one can be open
 */
export const Accordion = component$<{
  /** Allow multiple sections to be open */
  multiple?: boolean;
  /** Default open section(s) */
  defaultOpen?: string | string[];
  /** Additional class names */
  class?: string;
}>((props) => {
  const openSections = useSignal<Set<string>>(
    new Set(
      Array.isArray(props.defaultOpen)
        ? props.defaultOpen
        : props.defaultOpen
          ? [props.defaultOpen]
          : []
    )
  );

  return (
    <div
      class={`space-y-2 ${props.class || ''}`}
      data-accordion
      data-multiple={props.multiple}
    >
      <Slot />
    </div>
  );
});

/**
 * Simple collapsible details (native HTML with glass styling)
 */
export const GlassDetails = component$<{
  summary: string;
  icon?: string;
  defaultOpen?: boolean;
  class?: string;
}>((props) => {
  return (
    <details
      class={`
        glass-surface
        rounded-xl
        overflow-hidden
        group
        ${props.class || ''}
      `}
      open={props.defaultOpen}
    >
      <summary
        class={`
          flex items-center gap-3
          p-4
          cursor-pointer
          select-none
          hover:bg-[var(--glass-state-layer-hover)]
          glass-transition-colors
          list-none
          [&::-webkit-details-marker]:hidden
        `}
      >
        {props.icon && <span class="text-lg">{props.icon}</span>}
        <span class="flex-grow text-sm font-medium text-[var(--glass-text-primary)]">
          {props.summary}
        </span>
        <span class="text-[var(--glass-text-tertiary)] transition-transform group-open:rotate-180">
          ▼
        </span>
      </summary>
      <div class="p-4 pt-0 border-t border-[var(--glass-border)]">
        <Slot />
      </div>
    </details>
  );
});

export default CollapsibleSection;
