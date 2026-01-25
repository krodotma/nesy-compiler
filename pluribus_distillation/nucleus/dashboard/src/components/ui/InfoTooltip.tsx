/**
 * InfoTooltip - Hover Information Component
 * Phase 4, Step 109 - Progressive Disclosure
 *
 * Glass-styled tooltip for quick information display
 */

import { component$, $, Slot, useSignal, useVisibleTask$ } from '@builder.io/qwik';

export type TooltipPosition = 'top' | 'bottom' | 'left' | 'right';

export interface InfoTooltipProps {
  /** Tooltip content text */
  content: string;
  /** Tooltip position */
  position?: TooltipPosition;
  /** Delay before showing (ms) */
  delay?: number;
  /** Whether tooltip is disabled */
  disabled?: boolean;
  /** Additional class for the wrapper */
  class?: string;
  /** Additional class for the tooltip */
  tooltipClass?: string;
  /** Max width of tooltip */
  maxWidth?: number;
  /** Show arrow indicator */
  arrow?: boolean;
}

const POSITION_CLASSES: Record<TooltipPosition, { tooltip: string; arrow: string }> = {
  top: {
    tooltip: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    arrow: 'top-full left-1/2 -translate-x-1/2 border-t-[var(--glass-bg-dark)] border-x-transparent border-b-transparent',
  },
  bottom: {
    tooltip: 'top-full left-1/2 -translate-x-1/2 mt-2',
    arrow: 'bottom-full left-1/2 -translate-x-1/2 border-b-[var(--glass-bg-dark)] border-x-transparent border-t-transparent',
  },
  left: {
    tooltip: 'right-full top-1/2 -translate-y-1/2 mr-2',
    arrow: 'left-full top-1/2 -translate-y-1/2 border-l-[var(--glass-bg-dark)] border-y-transparent border-r-transparent',
  },
  right: {
    tooltip: 'left-full top-1/2 -translate-y-1/2 ml-2',
    arrow: 'right-full top-1/2 -translate-y-1/2 border-r-[var(--glass-bg-dark)] border-y-transparent border-l-transparent',
  },
};

/**
 * Info Tooltip Component
 *
 * Usage:
 * ```tsx
 * <InfoTooltip content="Additional information here">
 *   <span>Hover me</span>
 * </InfoTooltip>
 *
 * // With info icon
 * <InfoTooltip content="Explanation of this setting" position="right">
 *   <InfoIcon />
 * </InfoTooltip>
 * ```
 */
export const InfoTooltip = component$<InfoTooltipProps>((props) => {
  const visible = useSignal(false);
  const hoverTimeout = useSignal<number | null>(null);
  const position = props.position || 'top';
  const delay = props.delay ?? 200;
  const positionClass = POSITION_CLASSES[position];
  const maxWidth = props.maxWidth || 250;

  const showTooltip = $(() => {
    if (props.disabled) return;
    hoverTimeout.value = window.setTimeout(() => {
      visible.value = true;
    }, delay);
  });

  const hideTooltip = $(() => {
    if (hoverTimeout.value) {
      clearTimeout(hoverTimeout.value);
      hoverTimeout.value = null;
    }
    visible.value = false;
  });

  return (
    <div
      class={`relative inline-flex ${props.class || ''}`}
      onMouseEnter$={showTooltip}
      onMouseLeave$={hideTooltip}
      onFocus$={showTooltip}
      onBlur$={hideTooltip}
    >
      {/* Trigger element */}
      <Slot />

      {/* Tooltip */}
      {visible.value && (
        <div
          class={`
            absolute
            z-50
            ${positionClass.tooltip}
            pointer-events-none
            glass-animate-enter
            ${props.tooltipClass || ''}
          `}
          role="tooltip"
        >
          <div
            class={`
              px-3 py-2
              text-xs
              font-medium
              text-[var(--glass-text-primary)]
              bg-[var(--glass-bg-dark)]
              backdrop-blur-[var(--glass-blur-lg)]
              border border-[var(--glass-border)]
              rounded-lg
              shadow-lg
              whitespace-normal
            `}
            style={{ maxWidth: `${maxWidth}px` }}
          >
            {props.content}
          </div>

          {/* Arrow */}
          {props.arrow !== false && (
            <div
              class={`
                absolute
                w-0 h-0
                border-4
                ${positionClass.arrow}
              `}
            />
          )}
        </div>
      )}
    </div>
  );
});

/**
 * Info Icon with built-in tooltip
 */
export const InfoIcon = component$<{
  content: string;
  position?: TooltipPosition;
  size?: 'sm' | 'md' | 'lg';
  class?: string;
}>((props) => {
  const sizeClasses = {
    sm: 'w-3 h-3 text-[10px]',
    md: 'w-4 h-4 text-xs',
    lg: 'w-5 h-5 text-sm',
  };
  const size = props.size || 'md';

  return (
    <InfoTooltip content={props.content} position={props.position}>
      <button
        type="button"
        class={`
          inline-flex items-center justify-center
          ${sizeClasses[size]}
          rounded-full
          border border-[var(--glass-text-tertiary)]
          text-[var(--glass-text-tertiary)]
          hover:border-[var(--glass-accent-cyan)]
          hover:text-[var(--glass-accent-cyan)]
          glass-transition-colors
          cursor-help
          ${props.class || ''}
        `}
        aria-label="More information"
      >
        i
      </button>
    </InfoTooltip>
  );
});

/**
 * Help Text - Inline help with optional tooltip
 */
export const HelpText = component$<{
  text: string;
  tooltip?: string;
  class?: string;
}>((props) => {
  if (props.tooltip) {
    return (
      <div class={`flex items-center gap-1 ${props.class || ''}`}>
        <span class="text-xs text-[var(--glass-text-tertiary)]">{props.text}</span>
        <InfoIcon content={props.tooltip} size="sm" />
      </div>
    );
  }

  return (
    <span class={`text-xs text-[var(--glass-text-tertiary)] ${props.class || ''}`}>
      {props.text}
    </span>
  );
});

/**
 * Label with Info - Form label with optional help tooltip
 */
export const LabelWithInfo = component$<{
  label: string;
  htmlFor?: string;
  info?: string;
  required?: boolean;
  class?: string;
}>((props) => {
  return (
    <label
      for={props.htmlFor}
      class={`flex items-center gap-1.5 text-sm font-medium text-[var(--glass-text-secondary)] ${props.class || ''}`}
    >
      {props.label}
      {props.required && (
        <span class="text-[var(--glass-status-error)]">*</span>
      )}
      {props.info && <InfoIcon content={props.info} size="sm" />}
    </label>
  );
});

export default InfoTooltip;
