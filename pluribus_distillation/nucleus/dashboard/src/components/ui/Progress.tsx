/**
 * Progress - Material Web 3 Progress Indicators
 * Phase 4, Step 96 - MW3 Component Integration
 *
 * Glass-styled progress indicators with M3 semantics
 */

import { component$ } from '@builder.io/qwik';

// M3 Progress components
import '@material/web/progress/linear-progress.js';
import '@material/web/progress/circular-progress.js';

export interface LinearProgressProps {
  /** Current value (0-100 or 0-max) */
  value?: number;
  /** Maximum value */
  max?: number;
  /** Indeterminate (loading) state */
  indeterminate?: boolean;
  /** Buffer value for buffered progress */
  buffer?: number;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Color variant */
  color?: 'cyan' | 'magenta' | 'amber' | 'emerald' | 'purple' | 'gradient';
  /** Additional class names */
  class?: string;
  /** Show percentage label */
  showLabel?: boolean;
}

export interface CircularProgressProps {
  /** Current value (0-100 or 0-max) */
  value?: number;
  /** Maximum value */
  max?: number;
  /** Indeterminate (loading) state */
  indeterminate?: boolean;
  /** Size in pixels */
  size?: number;
  /** Stroke width */
  strokeWidth?: number;
  /** Color variant */
  color?: 'cyan' | 'magenta' | 'amber' | 'emerald' | 'purple' | 'gradient';
  /** Additional class names */
  class?: string;
  /** Show percentage in center */
  showValue?: boolean;
}

const COLOR_MAP: Record<string, string> = {
  cyan: 'var(--glass-accent-cyan)',
  magenta: 'var(--glass-accent-magenta)',
  amber: 'var(--glass-accent-amber)',
  emerald: 'var(--glass-accent-emerald)',
  purple: 'var(--glass-accent-purple)',
  gradient: 'var(--glass-accent-cyan)',
};

const SIZE_MAP = {
  sm: '4px',
  md: '8px',
  lg: '12px',
};

/**
 * Linear Progress Bar with glassmorphism styling
 *
 * Usage:
 * ```tsx
 * <LinearProgress value={75} color="cyan" showLabel />
 * <LinearProgress indeterminate color="gradient" />
 * ```
 */
export const LinearProgress = component$<LinearProgressProps>((props) => {
  const value = props.value ?? 0;
  const max = props.max ?? 100;
  const percentage = Math.round((value / max) * 100);
  const color = COLOR_MAP[props.color || 'cyan'];
  const height = SIZE_MAP[props.size || 'md'];

  const glassStyles = {
    '--md-linear-progress-active-indicator-color': color,
    '--md-linear-progress-track-color': 'var(--glass-bg-card)',
    '--md-linear-progress-track-shape': '9999px',
    '--md-linear-progress-active-indicator-height': height,
    '--md-linear-progress-track-height': height,
  };

  // For gradient effect, we use a custom overlay
  const isGradient = props.color === 'gradient';

  return (
    <div class={`relative ${props.class || ''}`}>
      <md-linear-progress
        class={`
          w-full
          [&::part(bar)]:rounded-full
          ${isGradient ? 'opacity-0' : ''}
        `}
        style={glassStyles as any}
        value={value}
        max={max}
        indeterminate={props.indeterminate}
        buffer={props.buffer}
      />

      {/* Custom gradient progress overlay */}
      {isGradient && !props.indeterminate && (
        <div
          class="absolute inset-0 rounded-full overflow-hidden"
          style={{ height }}
        >
          <div class="absolute inset-0 bg-[var(--glass-bg-card)]" />
          <div
            class="absolute inset-y-0 left-0 rounded-full glass-progress-gradient"
            style={{
              width: `${percentage}%`,
              background: 'linear-gradient(90deg, var(--glass-accent-cyan), var(--glass-accent-magenta))',
              boxShadow: '0 0 10px var(--glass-accent-cyan-glow)',
              transition: 'width 0.3s ease',
            }}
          />
        </div>
      )}

      {/* Indeterminate gradient */}
      {isGradient && props.indeterminate && (
        <div
          class="absolute inset-0 rounded-full overflow-hidden"
          style={{ height }}
        >
          <div class="absolute inset-0 bg-[var(--glass-bg-card)]" />
          <div
            class="absolute inset-y-0 w-1/3 rounded-full animate-[glass-flow_1.5s_ease-in-out_infinite]"
            style={{
              background: 'linear-gradient(90deg, transparent, var(--glass-accent-cyan), var(--glass-accent-magenta), transparent)',
            }}
          />
        </div>
      )}

      {/* Percentage label */}
      {props.showLabel && !props.indeterminate && (
        <div class="absolute right-0 -top-6 text-xs text-[var(--glass-text-secondary)]">
          {percentage}%
        </div>
      )}
    </div>
  );
});

/**
 * Circular Progress Indicator with glassmorphism styling
 *
 * Usage:
 * ```tsx
 * <CircularProgress value={75} size={64} color="cyan" showValue />
 * <CircularProgress indeterminate size={32} />
 * ```
 */
export const CircularProgress = component$<CircularProgressProps>((props) => {
  const value = props.value ?? 0;
  const max = props.max ?? 100;
  const percentage = Math.round((value / max) * 100);
  const size = props.size ?? 48;
  const color = COLOR_MAP[props.color || 'cyan'];

  const glassStyles = {
    '--md-circular-progress-active-indicator-color': color,
    '--md-circular-progress-size': `${size}px`,
    '--md-circular-progress-active-indicator-width': `${props.strokeWidth || Math.max(2, size / 12)}`,
  };

  const isGradient = props.color === 'gradient';

  return (
    <div
      class={`relative inline-flex items-center justify-center ${props.class || ''}`}
      style={{ width: `${size}px`, height: `${size}px` }}
    >
      <md-circular-progress
        class={`
          ${isGradient ? '[&::part(progress)]:stroke-url(#gradient-progress)' : ''}
        `}
        style={glassStyles as any}
        value={value}
        max={max}
        indeterminate={props.indeterminate}
      />

      {/* SVG gradient definition for gradient mode */}
      {isGradient && (
        <svg class="absolute w-0 h-0">
          <defs>
            <linearGradient id="gradient-progress" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="var(--glass-accent-cyan)" />
              <stop offset="100%" stopColor="var(--glass-accent-magenta)" />
            </linearGradient>
          </defs>
        </svg>
      )}

      {/* Center value display */}
      {props.showValue && !props.indeterminate && (
        <div
          class="absolute inset-0 flex items-center justify-center text-xs font-bold"
          style={{ color }}
        >
          {percentage}%
        </div>
      )}

      {/* Glow effect */}
      <div
        class="absolute inset-0 rounded-full pointer-events-none"
        style={{
          boxShadow: `0 0 ${size / 4}px ${color}40`,
          opacity: props.indeterminate ? 0.5 : percentage / 100,
          transition: 'opacity 0.3s ease',
        }}
      />
    </div>
  );
});

/**
 * Loading Spinner - Convenience wrapper for indeterminate circular progress
 */
export const LoadingSpinner = component$<{
  size?: number;
  color?: CircularProgressProps['color'];
  class?: string;
}>((props) => {
  return (
    <CircularProgress
      indeterminate
      size={props.size ?? 24}
      color={props.color ?? 'cyan'}
      class={props.class}
    />
  );
});

/**
 * Progress with Label - Progress bar with inline label
 */
export const ProgressWithLabel = component$<{
  label: string;
  value: number;
  max?: number;
  color?: LinearProgressProps['color'];
  class?: string;
}>((props) => {
  const percentage = Math.round((props.value / (props.max ?? 100)) * 100);

  return (
    <div class={`space-y-1 ${props.class || ''}`}>
      <div class="flex justify-between items-center text-xs">
        <span class="text-[var(--glass-text-secondary)]">{props.label}</span>
        <span class="font-mono" style={{ color: COLOR_MAP[props.color || 'cyan'] }}>
          {percentage}%
        </span>
      </div>
      <LinearProgress
        value={props.value}
        max={props.max}
        color={props.color}
        size="sm"
      />
    </div>
  );
});

export default LinearProgress;
