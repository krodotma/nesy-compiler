/**
 * NeonControls - Glassmorphism styled form controls
 * =================================================
 * Neon-styled sliders, checkboxes, and input components
 * for the Pluribus dashboard control panels.
 *
 * @usage
 * ```tsx
 * <NeonSlider
 *   label="Performance max"
 *   value={config.eventMax}
 *   min={6}
 *   max={60}
 *   color="cyan"
 *   onChange$={(v) => config.eventMax = v}
 * />
 *
 * <NeonCheckbox
 *   label="Show tolerances"
 *   checked={config.showTolerances}
 *   color="cyan"
 *   onChange$={(v) => config.showTolerances = v}
 * />
 * ```
 */

import { component$, type PropFunction, useSignal } from '@builder.io/qwik';

export type NeonColor = 'cyan' | 'magenta' | 'amber' | 'emerald' | 'purple' | 'rose' | 'orange';

// ═══════════════════════════════════════════════════════════════════════════════
// NEON SLIDER
// ═══════════════════════════════════════════════════════════════════════════════

export interface NeonSliderProps {
  /** Current value */
  value: number;
  /** Minimum value */
  min: number;
  /** Maximum value */
  max: number;
  /** Step increment */
  step?: number;
  /** Optional label */
  label?: string;
  /** Show current value */
  showValue?: boolean;
  /** Value format function */
  formatValue?: (value: number) => string;
  /** Neon color accent */
  color?: NeonColor;
  /** Disabled state */
  disabled?: boolean;
  /** Compact mode (smaller) */
  compact?: boolean;
  /** Change handler */
  onChange$?: PropFunction<(value: number) => void>;
  /** Additional class */
  class?: string;
}

const SLIDER_COLORS: Record<NeonColor, { track: string; thumb: string; glow: string; text: string }> = {
  cyan: {
    track: 'bg-gradient-to-r from-cyan-500/30 to-cyan-400/50',
    thumb: 'bg-cyan-400 shadow-[0_0_10px_rgba(0,255,255,0.6),0_0_20px_rgba(0,255,255,0.3)]',
    glow: 'shadow-[0_0_8px_rgba(0,255,255,0.4)]',
    text: 'text-cyan-400',
  },
  magenta: {
    track: 'bg-gradient-to-r from-fuchsia-500/30 to-fuchsia-400/50',
    thumb: 'bg-fuchsia-400 shadow-[0_0_10px_rgba(255,0,255,0.6),0_0_20px_rgba(255,0,255,0.3)]',
    glow: 'shadow-[0_0_8px_rgba(255,0,255,0.4)]',
    text: 'text-fuchsia-400',
  },
  amber: {
    track: 'bg-gradient-to-r from-amber-500/30 to-amber-400/50',
    thumb: 'bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.6),0_0_20px_rgba(251,191,36,0.3)]',
    glow: 'shadow-[0_0_8px_rgba(251,191,36,0.4)]',
    text: 'text-amber-400',
  },
  emerald: {
    track: 'bg-gradient-to-r from-emerald-500/30 to-emerald-400/50',
    thumb: 'bg-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.6),0_0_20px_rgba(16,185,129,0.3)]',
    glow: 'shadow-[0_0_8px_rgba(16,185,129,0.4)]',
    text: 'text-emerald-400',
  },
  purple: {
    track: 'bg-gradient-to-r from-purple-500/30 to-purple-400/50',
    thumb: 'bg-purple-400 shadow-[0_0_10px_rgba(168,85,247,0.6),0_0_20px_rgba(168,85,247,0.3)]',
    glow: 'shadow-[0_0_8px_rgba(168,85,247,0.4)]',
    text: 'text-purple-400',
  },
  rose: {
    track: 'bg-gradient-to-r from-rose-500/30 to-rose-400/50',
    thumb: 'bg-rose-400 shadow-[0_0_10px_rgba(244,63,94,0.6),0_0_20px_rgba(244,63,94,0.3)]',
    glow: 'shadow-[0_0_8px_rgba(244,63,94,0.4)]',
    text: 'text-rose-400',
  },
  orange: {
    track: 'bg-gradient-to-r from-orange-500/30 to-orange-400/50',
    thumb: 'bg-orange-400 shadow-[0_0_10px_rgba(251,146,60,0.6),0_0_20px_rgba(251,146,60,0.3)]',
    glow: 'shadow-[0_0_8px_rgba(251,146,60,0.4)]',
    text: 'text-orange-400',
  },
};

export const NeonSlider = component$<NeonSliderProps>((props) => {
  const colors = SLIDER_COLORS[props.color || 'cyan'];
  const showValue = props.showValue ?? true;
  const formatValue = props.formatValue ?? ((v: number) => v.toString());
  const compact = props.compact ?? false;
  const percentage = ((props.value - props.min) / (props.max - props.min)) * 100;

  return (
    <label
      class={`
        flex flex-col gap-1
        ${props.disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${props.class || ''}
      `}
    >
      {(props.label || showValue) && (
        <div class={`flex items-center justify-between ${compact ? 'text-[9px]' : 'text-[10px]'} font-mono`}>
          {props.label && (
            <span class="text-slate-400 uppercase tracking-wider">{props.label}</span>
          )}
          {showValue && (
            <span class={`${colors.text} font-medium`}>{formatValue(props.value)}</span>
          )}
        </div>
      )}
      <div class="relative w-full h-6 flex items-center">
        {/* Background track */}
        <div class="absolute inset-x-0 h-1.5 rounded-full bg-slate-800/80 border border-slate-700/50" />

        {/* Filled track with gradient and glow */}
        <div
          class={`absolute left-0 h-1.5 rounded-full ${colors.track} ${colors.glow}`}
          style={{ width: `${percentage}%` }}
        />

        {/* Hidden native input for accessibility */}
        <input
          type="range"
          min={props.min}
          max={props.max}
          step={props.step ?? 1}
          value={props.value}
          disabled={props.disabled}
          class="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed z-10"
          onInput$={(ev) => {
            const v = Number((ev.target as HTMLInputElement).value);
            props.onChange$?.(v);
          }}
        />

        {/* Custom thumb */}
        <div
          class={`
            absolute top-1/2 -translate-y-1/2 -translate-x-1/2
            ${compact ? 'w-3 h-3' : 'w-4 h-4'}
            rounded-full
            ${colors.thumb}
            transition-all duration-150
            pointer-events-none
            ${!props.disabled ? 'hover:scale-110' : ''}
          `}
          style={{ left: `${percentage}%` }}
        />
      </div>
    </label>
  );
});

// ═══════════════════════════════════════════════════════════════════════════════
// NEON CHECKBOX
// ═══════════════════════════════════════════════════════════════════════════════

export interface NeonCheckboxProps {
  /** Checked state */
  checked: boolean;
  /** Label text */
  label?: string;
  /** Neon color */
  color?: NeonColor;
  /** Disabled state */
  disabled?: boolean;
  /** Compact mode */
  compact?: boolean;
  /** Change handler */
  onChange$?: PropFunction<(checked: boolean) => void>;
  /** Additional class */
  class?: string;
}

const CHECKBOX_COLORS: Record<NeonColor, { checked: string; border: string; glow: string }> = {
  cyan: {
    checked: 'bg-cyan-500/90',
    border: 'border-cyan-500/60',
    glow: 'shadow-[0_0_8px_rgba(0,255,255,0.5)]',
  },
  magenta: {
    checked: 'bg-fuchsia-500/90',
    border: 'border-fuchsia-500/60',
    glow: 'shadow-[0_0_8px_rgba(255,0,255,0.5)]',
  },
  amber: {
    checked: 'bg-amber-500/90',
    border: 'border-amber-500/60',
    glow: 'shadow-[0_0_8px_rgba(251,191,36,0.5)]',
  },
  emerald: {
    checked: 'bg-emerald-500/90',
    border: 'border-emerald-500/60',
    glow: 'shadow-[0_0_8px_rgba(16,185,129,0.5)]',
  },
  purple: {
    checked: 'bg-purple-500/90',
    border: 'border-purple-500/60',
    glow: 'shadow-[0_0_8px_rgba(168,85,247,0.5)]',
  },
  rose: {
    checked: 'bg-rose-500/90',
    border: 'border-rose-500/60',
    glow: 'shadow-[0_0_8px_rgba(244,63,94,0.5)]',
  },
  orange: {
    checked: 'bg-orange-500/90',
    border: 'border-orange-500/60',
    glow: 'shadow-[0_0_8px_rgba(251,146,60,0.5)]',
  },
};

export const NeonCheckbox = component$<NeonCheckboxProps>((props) => {
  const colors = CHECKBOX_COLORS[props.color || 'cyan'];
  const compact = props.compact ?? false;
  const size = compact ? 'w-3 h-3' : 'w-4 h-4';
  const checkSize = compact ? 'w-2 h-2' : 'w-2.5 h-2.5';

  return (
    <label
      class={`
        inline-flex items-center gap-2 cursor-pointer select-none
        ${compact ? 'text-[9px]' : 'text-[10px]'}
        ${props.disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${props.class || ''}
      `}
    >
      <div class="relative">
        <input
          type="checkbox"
          checked={props.checked}
          disabled={props.disabled}
          class="sr-only"
          onChange$={(ev) => {
            props.onChange$?.((ev.target as HTMLInputElement).checked);
          }}
        />
        <div
          class={`
            ${size} rounded
            border transition-all duration-150
            flex items-center justify-center
            ${props.checked
              ? `${colors.checked} ${colors.border} ${colors.glow}`
              : 'bg-slate-800/80 border-slate-600/50 hover:border-slate-500/70'
            }
          `}
        >
          {props.checked && (
            <svg
              class={`${checkSize} text-white`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="3"
                d="M5 13l4 4L19 7"
              />
            </svg>
          )}
        </div>
      </div>
      {props.label && (
        <span class="text-slate-400 font-mono uppercase tracking-wider">{props.label}</span>
      )}
    </label>
  );
});

// ═══════════════════════════════════════════════════════════════════════════════
// NEON BUTTON
// ═══════════════════════════════════════════════════════════════════════════════

export interface NeonButtonProps {
  /** Button content */
  children?: any;
  /** Neon color */
  color?: NeonColor;
  /** Active/selected state */
  active?: boolean;
  /** Disabled state */
  disabled?: boolean;
  /** Compact mode */
  compact?: boolean;
  /** Click handler */
  onClick$?: PropFunction<() => void>;
  /** Additional class */
  class?: string;
}

const BUTTON_COLORS: Record<NeonColor, { active: string; border: string; glow: string; text: string }> = {
  cyan: {
    active: 'bg-cyan-500/20',
    border: 'border-cyan-400/60',
    glow: 'shadow-[0_0_12px_rgba(0,255,255,0.4)]',
    text: 'text-cyan-300',
  },
  magenta: {
    active: 'bg-fuchsia-500/20',
    border: 'border-fuchsia-400/60',
    glow: 'shadow-[0_0_12px_rgba(255,0,255,0.4)]',
    text: 'text-fuchsia-300',
  },
  amber: {
    active: 'bg-amber-500/20',
    border: 'border-amber-400/60',
    glow: 'shadow-[0_0_12px_rgba(251,191,36,0.4)]',
    text: 'text-amber-300',
  },
  emerald: {
    active: 'bg-emerald-500/20',
    border: 'border-emerald-400/60',
    glow: 'shadow-[0_0_12px_rgba(16,185,129,0.4)]',
    text: 'text-emerald-300',
  },
  purple: {
    active: 'bg-purple-500/20',
    border: 'border-purple-400/60',
    glow: 'shadow-[0_0_12px_rgba(168,85,247,0.4)]',
    text: 'text-purple-300',
  },
  rose: {
    active: 'bg-rose-500/20',
    border: 'border-rose-400/60',
    glow: 'shadow-[0_0_12px_rgba(244,63,94,0.4)]',
    text: 'text-rose-300',
  },
  orange: {
    active: 'bg-orange-500/20',
    border: 'border-orange-400/60',
    glow: 'shadow-[0_0_12px_rgba(251,146,60,0.4)]',
    text: 'text-orange-300',
  },
};

export const NeonButton = component$<NeonButtonProps>((props) => {
  const colors = BUTTON_COLORS[props.color || 'cyan'];
  const compact = props.compact ?? false;
  const padding = compact ? 'px-2 py-0.5' : 'px-3 py-1';
  const text = compact ? 'text-[8px]' : 'text-[9px]';

  return (
    <button
      class={`
        ${padding} ${text}
        font-mono uppercase tracking-wider
        rounded border transition-all duration-150
        ${props.active
          ? `${colors.active} ${colors.border} ${colors.text} ${colors.glow}`
          : 'bg-slate-800/60 border-slate-600/50 text-slate-300 hover:border-slate-500/70 hover:bg-slate-700/60'
        }
        ${props.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        ${props.class || ''}
      `}
      disabled={props.disabled}
      onClick$={props.onClick$}
    >
      {props.children}
    </button>
  );
});

// ═══════════════════════════════════════════════════════════════════════════════
// NEON BADGE (for status indicators)
// ═══════════════════════════════════════════════════════════════════════════════

export interface NeonBadgeProps {
  /** Status level */
  status: 'ok' | 'warn' | 'crit' | 'info';
  /** Badge content */
  children?: any;
  /** Compact mode */
  compact?: boolean;
  /** Pulse animation */
  pulse?: boolean;
  /** Additional class */
  class?: string;
}

const BADGE_COLORS: Record<string, { bg: string; border: string; text: string; glow: string }> = {
  ok: {
    bg: 'bg-emerald-500/20',
    border: 'border-emerald-500/40',
    text: 'text-emerald-300',
    glow: 'shadow-[0_0_8px_rgba(16,185,129,0.3)]',
  },
  warn: {
    bg: 'bg-amber-500/20',
    border: 'border-amber-500/40',
    text: 'text-amber-300',
    glow: 'shadow-[0_0_8px_rgba(251,191,36,0.3)]',
  },
  crit: {
    bg: 'bg-red-500/20',
    border: 'border-red-500/40',
    text: 'text-red-300',
    glow: 'shadow-[0_0_8px_rgba(239,68,68,0.3)]',
  },
  info: {
    bg: 'bg-cyan-500/20',
    border: 'border-cyan-500/40',
    text: 'text-cyan-300',
    glow: 'shadow-[0_0_8px_rgba(0,255,255,0.3)]',
  },
};

export const NeonStatusBadge = component$<NeonBadgeProps>((props) => {
  const colors = BADGE_COLORS[props.status];
  const compact = props.compact ?? false;

  return (
    <span
      class={`
        inline-flex items-center justify-center
        ${compact ? 'px-1.5 py-0.5 text-[7px]' : 'px-2 py-1 text-[8px]'}
        font-mono uppercase tracking-wider
        rounded border
        ${colors.bg} ${colors.border} ${colors.text} ${colors.glow}
        ${props.pulse ? 'animate-pulse' : ''}
        ${props.class || ''}
      `}
    >
      {props.children}
    </span>
  );
});

// ═══════════════════════════════════════════════════════════════════════════════
// NEON CARD (for control panels)
// ═══════════════════════════════════════════════════════════════════════════════

export interface NeonCardProps {
  /** Card title */
  title?: string;
  /** Neon accent color */
  color?: NeonColor;
  /** Card content */
  children?: any;
  /** Additional class */
  class?: string;
}

export const NeonCard = component$<NeonCardProps>((props) => {
  const color = props.color || 'cyan';
  const borderColor = `border-${color}-500/30`;

  return (
    <div
      class={`
        rounded-lg
        bg-slate-950/80 backdrop-blur-sm
        border border-slate-700/50
        ${props.class || ''}
      `}
    >
      {props.title && (
        <div class={`
          px-3 py-2 border-b border-slate-700/50
          text-[10px] font-mono uppercase tracking-[0.2em]
          text-${color}-400/80
        `}>
          {props.title}
        </div>
      )}
      <div class="p-3">
        {props.children}
      </div>
    </div>
  );
});

export default {
  NeonSlider,
  NeonCheckbox,
  NeonButton,
  NeonStatusBadge,
  NeonCard,
};
