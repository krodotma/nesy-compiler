/**
 * HamburgerMenu - Main Menu Toggle Component
 * Phase 4, Step 111 - Navigation Refinement
 *
 * Glass-styled hamburger menu button with animation
 */

import { component$, $, type PropFunction, useSignal } from '@builder.io/qwik';

// M3 Ripple
import '@material/web/ripple/ripple.js';

export interface HamburgerMenuProps {
  /** Whether the menu is open */
  open?: boolean;
  /** Callback when toggled */
  onToggle$?: PropFunction<(open: boolean) => void>;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Color when closed */
  color?: string;
  /** Color when open */
  openColor?: string;
  /** Animation variant */
  variant?: 'standard' | 'spin' | 'squeeze' | 'elastic';
  /** Aria label */
  label?: string;
  /** Additional class names */
  class?: string;
}

const SIZE_STYLES = {
  sm: { width: 18, height: 14, barHeight: 2, gap: 4 },
  md: { width: 24, height: 18, barHeight: 2, gap: 5 },
  lg: { width: 32, height: 24, barHeight: 3, gap: 7 },
};

/**
 * Hamburger Menu Button with animated transition
 *
 * Usage:
 * ```tsx
 * <HamburgerMenu
 *   open={menuOpen.value}
 *   onToggle$={(open) => menuOpen.value = open}
 *   variant="spin"
 * />
 * ```
 */
export const HamburgerMenu = component$<HamburgerMenuProps>((props) => {
  const internalOpen = useSignal(false);
  const isOpen = props.open !== undefined ? props.open : internalOpen.value;
  const size = props.size || 'md';
  const variant = props.variant || 'standard';
  const sizeStyle = SIZE_STYLES[size];

  const color = props.color || 'var(--glass-text-primary)';
  const openColor = props.openColor || 'var(--glass-accent-cyan)';
  const activeColor = isOpen ? openColor : color;

  const handleClick = $(() => {
    const newState = !isOpen;
    internalOpen.value = newState;
    props.onToggle$?.(newState);
  });

  // Calculate bar positions
  const barStyle = {
    width: `${sizeStyle.width}px`,
    height: `${sizeStyle.barHeight}px`,
    backgroundColor: activeColor,
    transition: 'all 0.3s ease',
  };

  const containerStyle = {
    width: `${sizeStyle.width}px`,
    height: `${sizeStyle.height}px`,
  };

  // Animation transforms based on variant
  const getBarTransforms = () => {
    const topOffset = sizeStyle.gap + sizeStyle.barHeight;

    switch (variant) {
      case 'spin':
        return {
          top: isOpen
            ? `translateY(${topOffset}px) rotate(45deg)`
            : 'translateY(0) rotate(0)',
          middle: isOpen ? 'scale(0)' : 'scale(1)',
          bottom: isOpen
            ? `translateY(-${topOffset}px) rotate(-45deg)`
            : 'translateY(0) rotate(0)',
        };
      case 'squeeze':
        return {
          top: isOpen
            ? `translateY(${topOffset}px) rotate(45deg) scaleX(1.1)`
            : 'translateY(0) rotate(0)',
          middle: isOpen ? 'scaleX(0)' : 'scaleX(1)',
          bottom: isOpen
            ? `translateY(-${topOffset}px) rotate(-45deg) scaleX(1.1)`
            : 'translateY(0) rotate(0)',
        };
      case 'elastic':
        return {
          top: isOpen
            ? `translateY(${topOffset}px) rotate(135deg)`
            : 'translateY(0) rotate(0)',
          middle: isOpen ? 'opacity-0 scaleX(0)' : 'opacity-1 scaleX(1)',
          bottom: isOpen
            ? `translateY(-${topOffset}px) rotate(-135deg)`
            : 'translateY(0) rotate(0)',
        };
      default: // standard
        return {
          top: isOpen
            ? `translateY(${topOffset}px) rotate(45deg)`
            : 'translateY(0) rotate(0)',
          middle: isOpen ? 'opacity-0' : 'opacity-1',
          bottom: isOpen
            ? `translateY(-${topOffset}px) rotate(-45deg)`
            : 'translateY(0) rotate(0)',
        };
    }
  };

  const transforms = getBarTransforms();

  return (
    <button
      type="button"
      class={`
        relative
        p-2
        rounded-xl
        glass-interactive
        glass-transition-all
        hover:bg-[var(--glass-state-layer-hover)]
        focus:outline-none
        focus-visible:ring-2
        focus-visible:ring-[var(--glass-accent-cyan)]
        ${props.class || ''}
      `}
      onClick$={handleClick}
      aria-label={props.label || (isOpen ? 'Close menu' : 'Open menu')}
      aria-expanded={isOpen}
    >
      <md-ripple class="rounded-xl" />

      <div
        class="relative flex flex-col justify-between"
        style={containerStyle}
      >
        {/* Top bar */}
        <span
          class="rounded-full origin-center"
          style={{
            ...barStyle,
            transform: transforms.top,
            boxShadow: isOpen ? `0 0 8px ${openColor}` : 'none',
          }}
        />

        {/* Middle bar */}
        <span
          class="rounded-full origin-center"
          style={{
            ...barStyle,
            transform: transforms.middle,
            opacity: isOpen && (variant === 'standard' || variant === 'elastic') ? 0 : 1,
          }}
        />

        {/* Bottom bar */}
        <span
          class="rounded-full origin-center"
          style={{
            ...barStyle,
            transform: transforms.bottom,
            boxShadow: isOpen ? `0 0 8px ${openColor}` : 'none',
          }}
        />
      </div>
    </button>
  );
});

/**
 * Menu Button - Icon button for triggering menus
 */
export const MenuButton = component$<{
  icon?: string;
  label?: string;
  onClick$?: PropFunction<() => void>;
  class?: string;
}>((props) => {
  return (
    <button
      type="button"
      class={`
        relative
        p-2
        rounded-xl
        glass-interactive
        glass-transition-all
        hover:bg-[var(--glass-state-layer-hover)]
        text-[var(--glass-text-secondary)]
        hover:text-[var(--glass-text-primary)]
        ${props.class || ''}
      `}
      onClick$={props.onClick$}
      aria-label={props.label || 'Menu'}
    >
      <md-ripple class="rounded-xl" />
      <span class="text-xl">{props.icon || 'more_vert'}</span>
    </button>
  );
});

/**
 * Kebab Menu - Three dots menu button
 */
export const KebabMenu = component$<{
  onClick$?: PropFunction<() => void>;
  vertical?: boolean;
  class?: string;
}>((props) => {
  const vertical = props.vertical !== false;

  return (
    <button
      type="button"
      class={`
        relative
        p-2
        rounded-lg
        glass-interactive
        glass-transition-all
        hover:bg-[var(--glass-state-layer-hover)]
        ${props.class || ''}
      `}
      onClick$={props.onClick$}
      aria-label="More options"
    >
      <md-ripple class="rounded-lg" />
      <div class={`flex ${vertical ? 'flex-col' : 'flex-row'} items-center gap-1`}>
        <span class="w-1 h-1 rounded-full bg-[var(--glass-text-secondary)]" />
        <span class="w-1 h-1 rounded-full bg-[var(--glass-text-secondary)]" />
        <span class="w-1 h-1 rounded-full bg-[var(--glass-text-secondary)]" />
      </div>
    </button>
  );
});

export default HamburgerMenu;
