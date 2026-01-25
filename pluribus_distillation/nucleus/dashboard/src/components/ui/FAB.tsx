/**
 * FAB - Floating Action Button
 * Phase 4, Step 116 - Menu & Navigation Refinement
 *
 * Glass-styled floating action button with M3 semantics
 */

import { component$, Slot, type PropFunction } from '@builder.io/qwik';

// M3 FAB component
import '@material/web/fab/fab.js';
import '@material/web/fab/branded-fab.js';

export interface FABProps {
  /** FAB icon */
  icon?: string;
  /** FAB label (for extended FAB) */
  label?: string;
  /** FAB size */
  size?: 'small' | 'medium' | 'large';
  /** Color variant */
  color?: 'cyan' | 'magenta' | 'amber' | 'emerald' | 'purple' | 'primary';
  /** Whether to lower elevation when pressed */
  lowered?: boolean;
  /** FAB position when used as fixed element */
  position?: 'bottom-right' | 'bottom-left' | 'bottom-center' | 'top-right' | 'top-left';
  /** Click handler */
  onClick$?: PropFunction<() => void>;
  /** Whether FAB is disabled */
  disabled?: boolean;
  /** Title/tooltip */
  title?: string;
  /** Additional class names */
  class?: string;
  /** Fixed position on screen */
  fixed?: boolean;
}

const COLOR_STYLES: Record<string, { bg: string; color: string; glow: string }> = {
  cyan: {
    bg: 'var(--glass-accent-cyan)',
    color: 'var(--glass-bg-dark)',
    glow: '0 0 20px var(--glass-accent-cyan-glow)',
  },
  magenta: {
    bg: 'var(--glass-accent-magenta)',
    color: 'var(--glass-bg-dark)',
    glow: '0 0 20px var(--glass-accent-magenta-glow)',
  },
  amber: {
    bg: 'var(--glass-accent-amber)',
    color: 'var(--glass-bg-dark)',
    glow: '0 0 20px var(--glass-accent-amber-subtle)',
  },
  emerald: {
    bg: 'var(--glass-accent-emerald)',
    color: 'var(--glass-bg-dark)',
    glow: '0 0 20px var(--glass-accent-emerald-subtle)',
  },
  purple: {
    bg: 'var(--glass-accent-purple)',
    color: 'white',
    glow: '0 0 20px var(--glass-accent-purple-subtle)',
  },
  primary: {
    bg: 'var(--glass-accent-cyan)',
    color: 'var(--glass-bg-dark)',
    glow: '0 0 20px var(--glass-accent-cyan-glow)',
  },
};

const POSITION_CLASSES: Record<string, string> = {
  'bottom-right': 'bottom-6 right-6',
  'bottom-left': 'bottom-6 left-6',
  'bottom-center': 'bottom-6 left-1/2 -translate-x-1/2',
  'top-right': 'top-6 right-6',
  'top-left': 'top-6 left-6',
};

const SIZE_STYLES: Record<string, { size: string; iconSize: string }> = {
  small: { size: '40px', iconSize: '20px' },
  medium: { size: '56px', iconSize: '24px' },
  large: { size: '96px', iconSize: '36px' },
};

/**
 * Floating Action Button with glass styling
 *
 * Usage:
 * ```tsx
 * <FAB
 *   icon="+"
 *   color="cyan"
 *   position="bottom-right"
 *   fixed
 *   onClick$={handleAdd}
 * />
 *
 * // Extended FAB with label
 * <FAB icon="✏️" label="Compose" color="magenta" />
 * ```
 */
export const FAB = component$<FABProps>((props) => {
  const colorStyle = COLOR_STYLES[props.color || 'primary'];
  const sizeStyle = SIZE_STYLES[props.size || 'medium'];
  const positionClass = props.position ? POSITION_CLASSES[props.position] : '';

  const fabStyles = {
    '--md-fab-container-color': colorStyle.bg,
    '--md-fab-icon-color': colorStyle.color,
    '--md-fab-container-width': sizeStyle.size,
    '--md-fab-container-height': sizeStyle.size,
    '--md-fab-icon-size': sizeStyle.iconSize,
    boxShadow: colorStyle.glow,
  };

  // Extended FAB (with label)
  if (props.label) {
    return (
      <md-fab
        class={`
          glass-transition-all
          hover:scale-105
          active:scale-95
          [&:hover]:shadow-lg
          ${props.fixed ? `fixed ${positionClass} z-50` : ''}
          ${props.class || ''}
        `}
        style={fabStyles as any}
        label={props.label}
        lowered={props.lowered}
        disabled={props.disabled}
        title={props.title}
        onClick$={props.onClick$}
      >
        {props.icon && <span slot="icon" class="text-lg">{props.icon}</span>}
      </md-fab>
    );
  }

  // Standard FAB
  return (
    <md-fab
      class={`
        glass-transition-all
        hover:scale-105
        active:scale-95
        [&:hover]:shadow-lg
        ${props.fixed ? `fixed ${positionClass} z-50` : ''}
        ${props.class || ''}
      `}
      style={fabStyles as any}
      lowered={props.lowered}
      disabled={props.disabled}
      title={props.title}
      onClick$={props.onClick$}
    >
      {props.icon && <span slot="icon" class="text-2xl">{props.icon}</span>}
      <Slot />
    </md-fab>
  );
});

/**
 * Speed Dial FAB - FAB that reveals multiple actions
 */
export const SpeedDialFAB = component$<{
  icon?: string;
  actions: Array<{
    id: string;
    icon: string;
    label: string;
    color?: FABProps['color'];
  }>;
  onAction$?: PropFunction<(actionId: string) => void>;
  position?: FABProps['position'];
  class?: string;
}>((props) => {
  const expanded = useSignal(false);
  const positionClass = props.position ? POSITION_CLASSES[props.position] : POSITION_CLASSES['bottom-right'];

  return (
    <div
      class={`
        fixed ${positionClass} z-50
        flex flex-col-reverse items-center gap-3
        ${props.class || ''}
      `}
    >
      {/* Action buttons */}
      <div
        class={`
          flex flex-col-reverse items-center gap-2
          glass-transition-all
          ${expanded.value ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'}
        `}
      >
        {props.actions.map((action, index) => (
          <div
            key={action.id}
            class="flex items-center gap-2 glass-animate-enter"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            {/* Label tooltip */}
            <span class="px-2 py-1 text-xs rounded-lg bg-[var(--glass-bg-dark)] text-[var(--glass-text-primary)] shadow-lg whitespace-nowrap">
              {action.label}
            </span>

            {/* Mini FAB */}
            <FAB
              icon={action.icon}
              size="small"
              color={action.color || 'cyan'}
              onClick$={() => {
                props.onAction$?.(action.id);
                expanded.value = false;
              }}
            />
          </div>
        ))}
      </div>

      {/* Main FAB */}
      <FAB
        icon={expanded.value ? '✕' : (props.icon || '+')}
        color="primary"
        onClick$={() => expanded.value = !expanded.value}
        class={`
          glass-transition-transform
          ${expanded.value ? 'rotate-45' : 'rotate-0'}
        `}
      />
    </div>
  );
});

// Need useSignal import
import { useSignal } from '@builder.io/qwik';

export default FAB;
