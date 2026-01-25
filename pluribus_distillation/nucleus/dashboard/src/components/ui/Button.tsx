import { component$, Slot, type PropFunction } from '@builder.io/qwik';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// M3 Components - Button variants (ROOT CAUSE FIX)
import '@material/web/button/filled-button.js';
import '@material/web/button/filled-tonal-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/button/text-button.js';
import '@material/web/iconbutton/icon-button.js';
import '@material/web/fab/fab.js';
import '@material/web/icon/icon.js';

// Helper for tailwind classes
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type ButtonVariant = 'primary' | 'secondary' | 'text' | 'icon' | 'tonal' | 'fab';

interface ButtonProps {
  variant?: ButtonVariant;
  class?: string;
  disabled?: boolean;
  icon?: string; // Material Symbol name
  // Allow event argument in handler
  onClick$?: PropFunction<(event: Event, element: HTMLElement) => void>;
  type?: 'button' | 'submit' | 'reset';
  title?: string;
  [key: string]: any; // Allow other HTML attributes
}

export const Button = component$((props: ButtonProps) => {
  const { variant = 'primary', icon, class: className, onClick$, disabled, type, title, ...rest } = props;
  
  // Mapping to Material Web 3 Components
  // Note: We pass props directly. Events like onClick work natively in Qwik on Custom Elements.
  
  if (variant === 'icon') {
    return (
      <md-icon-button
        class={className}
        disabled={disabled}
        onClick$={onClick$}
        title={title}
        {...rest}
      >
        <md-icon>{icon || <Slot />}</md-icon>
      </md-icon-button>
    );
  }

  if (variant === 'fab') {
    return (
      <md-fab
        class={className}
        disabled={disabled}
        onClick$={onClick$}
        title={title}
        label={icon ? undefined : 'Action'} // FAB usually has icon
        {...rest}
      >
        {icon && <md-icon slot="icon">{icon}</md-icon>}
      </md-fab>
    );
  }

  // Text / Ghost
  if (variant === 'text') {
    return (
      <md-text-button
        class={className}
        disabled={disabled}
        onClick$={onClick$}
        type={type}
        title={title}
        {...rest}
      >
        {icon && <md-icon slot="icon">{icon}</md-icon>}
        <Slot />
      </md-text-button>
    );
  }

  // Outlined (Secondary)
  if (variant === 'secondary') {
    return (
      <md-outlined-button
        class={className}
        disabled={disabled}
        onClick$={onClick$}
        type={type}
        title={title}
        {...rest}
      >
        {icon && <md-icon slot="icon">{icon}</md-icon>}
        <Slot />
      </md-outlined-button>
    );
  }

  // Tonal
  if (variant === 'tonal') {
    return (
      <md-filled-tonal-button
        class={className}
        disabled={disabled}
        onClick$={onClick$}
        type={type}
        title={title}
        {...rest}
      >
        {icon && <md-icon slot="icon">{icon}</md-icon>}
        <Slot />
      </md-filled-tonal-button>
    );
  }

  // Default: Primary (Filled)
  return (
    <md-filled-button
      class={className}
      disabled={disabled}
      onClick$={onClick$}
      type={type}
      title={title}
      {...rest}
    >
      {icon && <md-icon slot="icon">{icon}</md-icon>}
      <Slot />
    </md-filled-button>
  );
});
