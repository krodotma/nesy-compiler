/**
 * NeonTitle - Neon-styled title component for the Pluribus Dashboard
 * ===================================================================
 *
 * Provides consistent neon typography across the dashboard with
 * support for multiple colors, sizes, and animation variants.
 *
 * Uses tokens from glass-tokens.css (Phase 1, Steps 1-10)
 */

import { component$, Slot } from '@builder.io/qwik';

type NeonColor = 'cyan' | 'magenta' | 'amber' | 'purple' | 'emerald' | 'rose';
type NeonSize = 'xs' | 'sm' | 'base' | 'lg' | 'xl' | '2xl' | '3xl';
type NeonAnimation = 'none' | 'flicker' | 'pulse' | 'breathe';
type HeadingLevel = 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' | 'span' | 'div' | 'p';

interface NeonTitleProps {
  /** Heading level / element type (default: span) */
  level?: HeadingLevel;
  /** Neon color variant (default: cyan) */
  color?: NeonColor;
  /** Text size (default: sm) */
  size?: NeonSize;
  /** Animation type (default: none) */
  animation?: NeonAnimation;
  /** Uppercase text (default: true) */
  uppercase?: boolean;
  /** Show underline accent (default: false) */
  underline?: boolean;
  /** Additional CSS classes */
  class?: string;
}

const sizeClasses: Record<NeonSize, string> = {
  xs: 'glass-title-neon-xs',
  sm: 'glass-title-neon-sm',
  base: 'glass-title-neon-base',
  lg: 'glass-title-neon-lg',
  xl: 'glass-title-neon-xl',
  '2xl': 'glass-title-neon-2xl',
  '3xl': 'glass-title-neon-3xl',
};

const colorClasses: Record<NeonColor, string> = {
  cyan: 'glass-title-neon',
  magenta: 'glass-title-neon-magenta',
  amber: 'glass-title-neon-amber',
  purple: 'glass-title-neon-purple',
  emerald: 'glass-title-neon-emerald',
  rose: 'glass-title-neon-rose',
};

const animationClasses: Record<NeonAnimation, string> = {
  none: '',
  flicker: 'glass-title-neon-animated',
  pulse: 'glass-title-neon-pulse',
  breathe: 'glass-title-neon-breathe',
};

export const NeonTitle = component$<NeonTitleProps>((props) => {
  const {
    level = 'span',
    color = 'cyan',
    size = 'sm',
    animation = 'none',
    uppercase = true,
    underline = false,
  } = props;

  const Element = level;

  const classes = [
    colorClasses[color],
    sizeClasses[size],
    animationClasses[animation],
    uppercase && 'uppercase',
    underline && 'glass-title-underline-neon',
    props.class,
  ].filter(Boolean).join(' ');

  return (
    <Element class={classes}>
      <Slot />
    </Element>
  );
});

/**
 * NeonBadge - Small neon badge for status indicators
 */
interface NeonBadgeProps {
  color?: NeonColor;
  glow?: boolean;
  pulse?: boolean;
  class?: string;
}

export const NeonBadge = component$<NeonBadgeProps>((props) => {
  const { color = 'cyan', glow = false, pulse = false } = props;

  const classes = [
    'inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider',
    colorClasses[color],
    glow && 'glass-border-glow',
    pulse && 'glass-title-neon-pulse',
    props.class,
  ].filter(Boolean).join(' ');

  return (
    <span class={classes}>
      <Slot />
    </span>
  );
});

/**
 * NeonSectionHeader - Section header with neon title and gradient line
 */
interface NeonSectionHeaderProps {
  title: string;
  color?: NeonColor;
  size?: NeonSize;
  animation?: NeonAnimation;
  class?: string;
}

export const NeonSectionHeader = component$<NeonSectionHeaderProps>((props) => {
  const { title, color = 'cyan', size = 'sm', animation = 'none' } = props;

  const gradientColor = {
    cyan: 'var(--glass-accent-cyan)',
    magenta: 'var(--glass-accent-magenta)',
    amber: 'var(--glass-accent-amber)',
    purple: 'var(--glass-accent-purple)',
    emerald: 'var(--glass-accent-emerald)',
    rose: 'var(--glass-accent-rose)',
  }[color];

  return (
    <div class={['flex items-center gap-3 mb-3', props.class].filter(Boolean).join(' ')}>
      <NeonTitle level="h3" color={color} size={size} animation={animation}>
        {title}
      </NeonTitle>
      <div
        class="flex-1 h-px"
        style={{
          background: `linear-gradient(90deg, ${gradientColor}, transparent)`,
          boxShadow: `0 0 8px ${gradientColor}`,
        }}
      />
    </div>
  );
});

/**
 * NeonLink - Neon-styled anchor/link
 */
interface NeonLinkProps {
  href?: string;
  color?: NeonColor;
  external?: boolean;
  class?: string;
}

export const NeonLink = component$<NeonLinkProps>((props) => {
  const { href = '#', color = 'cyan', external = false } = props;

  const colorClass = colorClasses[color];

  return (
    <a
      href={href}
      class={[
        colorClass,
        'glass-link-neon transition-all duration-200',
        'hover:underline hover:underline-offset-4',
        props.class,
      ].filter(Boolean).join(' ')}
      target={external ? '_blank' : undefined}
      rel={external ? 'noopener noreferrer' : undefined}
    >
      <Slot />
    </a>
  );
});

/**
 * NeonGradientText - Text with animated gradient background
 */
interface NeonGradientTextProps {
  class?: string;
}

export const NeonGradientText = component$<NeonGradientTextProps>((props) => {
  return (
    <span class={['glass-text-gradient', props.class].filter(Boolean).join(' ')}>
      <Slot />
    </span>
  );
});

export default NeonTitle;
