import { component$, Slot, type PropFunction } from '@builder.io/qwik';
import { cn } from './Button'; // Reuse cn utility

// M3 Components (Step 60 - Card with Ripple)
import '@material/web/ripple/ripple.js';
import '@material/web/elevation/elevation.js';

type CardVariant = 'elevated' | 'filled' | 'outlined';

interface CardProps {
  variant?: CardVariant;
  class?: string;
  style?: any;
  padding?: string;
  interactive?: boolean; // Enable ripple for clickable cards
  onClick$?: PropFunction<() => void>;
}

/**
 * Material Web 3 Card Wrapper
 *
 * Implements the M3 Card spec using Tailwind utilities mapped to M3 tokens.
 * M3 doesn't have a stable <md-card> web component yet, so we style a <div>
 * with our token system. Now includes md-elevation and optional md-ripple.
 */
export const Card = component$((props: CardProps) => {
  const variant = props.variant || 'elevated';
  // Interactive if explicitly set OR if onClick$ is present
  const interactive = props.interactive ?? !!props.onClick$; 

  const baseStyles = "relative rounded-xl overflow-hidden transition-all duration-300";

  const variants = {
    elevated: "bg-md-surface-container-low shadow-md hover:shadow-lg border-none",
    filled: "bg-md-surface-container-highest border-none",
    outlined: "bg-md-surface border border-md-outline hover:border-md-outline-variant bg-transparent",
  };

  const elevationLevel = variant === 'elevated' ? 'glass-card-elevated' :
                         variant === 'filled' ? 'glass-card-filled' :
                         'glass-card-outlined';

  const padding = props.padding || "p-4";

  return (
    <div 
      class={cn(baseStyles, variants[variant], padding, "glass-card", elevationLevel, props.class, interactive ? 'cursor-pointer' : '')}
      style={props.style}
      onClick$={props.onClick$}
    >
      {/* M3 Elevation layer */}
      <md-elevation class="glass-card-elevation"></md-elevation>

      {/* M3 Ripple for interactive cards */}
      {interactive && <md-ripple class="glass-card-ripple"></md-ripple>}

      <Slot />
    </div>
  );
});

export const CardHeader = component$(({ class: className }: { class?: string }) => (
  <div class={cn("flex flex-col space-y-1.5 p-6", className)}>
    <Slot />
  </div>
));

export const CardTitle = component$(({ class: className }: { class?: string }) => (
  <h3 class={cn("font-semibold leading-none tracking-tight text-md-on-surface", className)}>
    <Slot />
  </h3>
));

export const CardDescription = component$(({ class: className }: { class?: string }) => (
  <p class={cn("text-sm text-md-on-surface-variant", className)}>
    <Slot />
  </p>
));

export const CardContent = component$(({ class: className }: { class?: string }) => (
  <div class={cn("p-6 pt-0", className)}>
    <Slot />
  </div>
));

export const CardFooter = component$(({ class: className }: { class?: string }) => (
  <div class={cn("flex items-center p-6 pt-0", className)}>
    <Slot />
  </div>
));