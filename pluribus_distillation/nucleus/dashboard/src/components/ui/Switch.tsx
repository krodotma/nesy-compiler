/**
 * Switch - Material Web 3 Switch Wrapper
 * Phase 4, Step 97 - MW3 Component Integration
 *
 * Glass-styled toggle switch with M3 semantics
 */

import { component$, type PropFunction } from '@builder.io/qwik';

// M3 Switch component
import '@material/web/switch/switch.js';

export interface SwitchProps {
  /** Whether the switch is checked */
  checked?: boolean;
  /** Callback when switch state changes */
  onChange$?: PropFunction<(checked: boolean) => void>;
  /** Whether the switch is disabled */
  disabled?: boolean;
  /** Show icons for on/off states */
  icons?: boolean;
  /** Switch label (for accessibility and display) */
  label?: string;
  /** Label position */
  labelPosition?: 'left' | 'right';
  /** Color when selected */
  color?: 'cyan' | 'magenta' | 'amber' | 'emerald' | 'purple';
  /** Additional class names */
  class?: string;
}

const COLOR_STYLES: Record<string, Record<string, string>> = {
  cyan: {
    '--md-switch-selected-handle-color': 'var(--glass-accent-cyan)',
    '--md-switch-selected-track-color': 'rgba(0, 255, 255, 0.3)',
    '--md-switch-selected-icon-color': 'var(--glass-bg-dark)',
    '--md-switch-selected-hover-handle-color': 'var(--glass-accent-cyan)',
    '--md-switch-selected-hover-track-color': 'rgba(0, 255, 255, 0.4)',
    '--md-switch-selected-pressed-handle-color': 'var(--glass-accent-cyan)',
    '--md-switch-selected-focus-handle-color': 'var(--glass-accent-cyan)',
  },
  magenta: {
    '--md-switch-selected-handle-color': 'var(--glass-accent-magenta)',
    '--md-switch-selected-track-color': 'rgba(255, 0, 255, 0.3)',
    '--md-switch-selected-icon-color': 'var(--glass-bg-dark)',
    '--md-switch-selected-hover-handle-color': 'var(--glass-accent-magenta)',
    '--md-switch-selected-hover-track-color': 'rgba(255, 0, 255, 0.4)',
    '--md-switch-selected-pressed-handle-color': 'var(--glass-accent-magenta)',
    '--md-switch-selected-focus-handle-color': 'var(--glass-accent-magenta)',
  },
  amber: {
    '--md-switch-selected-handle-color': 'var(--glass-accent-amber)',
    '--md-switch-selected-track-color': 'rgba(255, 191, 0, 0.3)',
    '--md-switch-selected-icon-color': 'var(--glass-bg-dark)',
    '--md-switch-selected-hover-handle-color': 'var(--glass-accent-amber)',
    '--md-switch-selected-hover-track-color': 'rgba(255, 191, 0, 0.4)',
    '--md-switch-selected-pressed-handle-color': 'var(--glass-accent-amber)',
    '--md-switch-selected-focus-handle-color': 'var(--glass-accent-amber)',
  },
  emerald: {
    '--md-switch-selected-handle-color': 'var(--glass-accent-emerald)',
    '--md-switch-selected-track-color': 'rgba(16, 185, 129, 0.3)',
    '--md-switch-selected-icon-color': 'var(--glass-bg-dark)',
    '--md-switch-selected-hover-handle-color': 'var(--glass-accent-emerald)',
    '--md-switch-selected-hover-track-color': 'rgba(16, 185, 129, 0.4)',
    '--md-switch-selected-pressed-handle-color': 'var(--glass-accent-emerald)',
    '--md-switch-selected-focus-handle-color': 'var(--glass-accent-emerald)',
  },
  purple: {
    '--md-switch-selected-handle-color': 'var(--glass-accent-purple)',
    '--md-switch-selected-track-color': 'rgba(138, 43, 226, 0.3)',
    '--md-switch-selected-icon-color': 'var(--glass-bg-dark)',
    '--md-switch-selected-hover-handle-color': 'var(--glass-accent-purple)',
    '--md-switch-selected-hover-track-color': 'rgba(138, 43, 226, 0.4)',
    '--md-switch-selected-pressed-handle-color': 'var(--glass-accent-purple)',
    '--md-switch-selected-focus-handle-color': 'var(--glass-accent-purple)',
  },
};

const BASE_STYLES = {
  '--md-switch-track-color': 'var(--glass-bg-card)',
  '--md-switch-track-outline-color': 'var(--glass-border)',
  '--md-switch-handle-color': 'var(--glass-text-tertiary)',
  '--md-switch-hover-handle-color': 'var(--glass-text-secondary)',
  '--md-switch-disabled-handle-color': 'var(--glass-text-disabled)',
  '--md-switch-disabled-track-color': 'var(--glass-state-layer-disabled)',
};

/**
 * Material Web 3 Switch with glassmorphism styling
 *
 * Usage:
 * ```tsx
 * <Switch
 *   checked={darkMode.value}
 *   onChange$={(checked) => darkMode.value = checked}
 *   label="Dark Mode"
 *   color="cyan"
 *   icons
 * />
 * ```
 */
export const Switch = component$<SwitchProps>((props) => {
  const colorStyles = COLOR_STYLES[props.color || 'cyan'];
  const combinedStyles = { ...BASE_STYLES, ...colorStyles };

  const handleChange = $((e: Event) => {
    const target = e.target as HTMLInputElement;
    props.onChange$?.(target.selected);
  });

  const switchElement = (
    <md-switch
      class={`
        glass-transition-all
        [&::part(track)]:rounded-full
        [&::part(handle)]:rounded-full
        [&:focus-visible::part(track)]:outline
        [&:focus-visible::part(track)]:outline-2
        [&:focus-visible::part(track)]:outline-[var(--glass-accent-cyan)]
        [&:focus-visible::part(track)]:outline-offset-2
      `}
      style={combinedStyles as any}
      selected={props.checked}
      disabled={props.disabled}
      icons={props.icons}
      onChange$={handleChange}
    />
  );

  if (!props.label) {
    return <div class={props.class}>{switchElement}</div>;
  }

  return (
    <label
      class={`
        inline-flex items-center gap-3
        cursor-pointer
        ${props.disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${props.class || ''}
      `}
    >
      {props.labelPosition === 'left' && (
        <span class="text-sm text-[var(--glass-text-secondary)]">{props.label}</span>
      )}
      {switchElement}
      {props.labelPosition !== 'left' && (
        <span class="text-sm text-[var(--glass-text-secondary)]">{props.label}</span>
      )}
    </label>
  );
});

/**
 * Toggle Group - Multiple switches with labels
 */
export const ToggleGroup = component$<{
  options: Array<{
    id: string;
    label: string;
    checked: boolean;
    disabled?: boolean;
  }>;
  onChange$?: PropFunction<(id: string, checked: boolean) => void>;
  color?: SwitchProps['color'];
  class?: string;
}>((props) => {
  return (
    <div class={`space-y-3 ${props.class || ''}`}>
      {props.options.map((option) => (
        <Switch
          key={option.id}
          label={option.label}
          checked={option.checked}
          disabled={option.disabled}
          color={props.color}
          onChange$={(checked) => props.onChange$?.(option.id, checked)}
        />
      ))}
    </div>
  );
});

export default Switch;
