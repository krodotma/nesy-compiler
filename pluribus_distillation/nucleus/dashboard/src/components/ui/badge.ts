/**
 * Badge Component - Shadcn Style
 *
 * Status badges for service kinds, tags, etc.
 */

import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { cva, type VariantProps } from 'class-variance-authority';

// ============================================================================
// Class Variants
// ============================================================================

export const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-primary text-primary-foreground hover:bg-primary/80',
        secondary: 'border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80',
        destructive: 'border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80',
        outline: 'text-foreground',
        success: 'border-transparent bg-green-500 text-white',
        warning: 'border-transparent bg-yellow-500 text-black',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export type BadgeVariants = VariantProps<typeof badgeVariants>;

// ============================================================================
// Lit Element
// ============================================================================

@customElement('ui-badge')
export class UIBadge extends LitElement {
  static styles = css`
    :host {
      display: inline-flex;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 9999px;
      padding: 0.125rem 0.625rem;
      font-size: 0.75rem;
      font-weight: 600;
      transition: all 150ms;
    }

    .badge.default {
      background: hsl(var(--primary, 221.2 83.2% 53.3%));
      color: hsl(var(--primary-foreground, 210 40% 98%));
    }

    .badge.secondary {
      background: hsl(var(--secondary, 210 40% 96%));
      color: hsl(var(--secondary-foreground, 222.2 47.4% 11.2%));
    }

    .badge.success {
      background: hsl(142.1 76.2% 36.3%);
      color: white;
    }

    .badge.warning {
      background: hsl(45.4 93.4% 47.5%);
      color: black;
    }

    .badge.destructive {
      background: hsl(var(--destructive, 0 84.2% 60.2%));
      color: hsl(var(--destructive-foreground, 210 40% 98%));
    }

    .badge.outline {
      background: transparent;
      border: 1px solid currentColor;
    }

    /* Service kind badges */
    .badge.port { background: #e3f2fd; color: #1565c0; }
    .badge.process { background: #f3e5f5; color: #7b1fa2; }
    .badge.composition { background: #fff3e0; color: #ef6c00; }
  `;

  @property({ type: String }) variant: 'default' | 'secondary' | 'success' | 'warning' | 'destructive' | 'outline' | 'port' | 'process' | 'composition' = 'default';

  render() {
    return html`
      <span class="badge ${this.variant}">
        <slot></slot>
      </span>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'ui-badge': UIBadge;
  }
}
