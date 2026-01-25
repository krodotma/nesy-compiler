/**
 * Status Indicator Component
 *
 * Unified status display for:
 * - Provider availability (VPS Session)
 * - Service health
 * - Connection state
 *
 * Maps to TUI color scheme:
 *   Available/Healthy  = Green [+]
 *   Unavailable/Error  = Red [-]
 *   Unknown/Pending    = Gray [?]
 */

import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';

export type StatusType = 'available' | 'unavailable' | 'unknown' | 'pending' | 'healthy' | 'unhealthy';

// ============================================================================
// Class helpers for Qwik/React
// ============================================================================

export function getStatusClass(status: StatusType): string {
  switch (status) {
    case 'available':
    case 'healthy':
      return 'bg-green-500';
    case 'unavailable':
    case 'unhealthy':
      return 'bg-red-500';
    case 'pending':
      return 'bg-yellow-500';
    default:
      return 'bg-gray-400';
  }
}

export function getStatusIcon(status: StatusType): string {
  switch (status) {
    case 'available':
    case 'healthy':
      return '[+]';
    case 'unavailable':
    case 'unhealthy':
      return '[-]';
    case 'pending':
      return '[~]';
    default:
      return '[?]';
  }
}

// ============================================================================
// Lit Element
// ============================================================================

@customElement('status-dot')
export class StatusDot extends LitElement {
  static styles = css`
    :host {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }

    .dot {
      width: 0.5rem;
      height: 0.5rem;
      border-radius: 50%;
    }

    .dot.available,
    .dot.healthy {
      background: hsl(var(--status-success, 142.1 76.2% 36.3%));
    }

    .dot.unavailable,
    .dot.unhealthy {
      background: hsl(var(--status-error, 0 84.2% 60.2%));
    }

    .dot.pending {
      background: hsl(var(--status-warning, 45.4 93.4% 47.5%));
      animation: pulse 1.5s ease-in-out infinite;
    }

    .dot.unknown {
      background: hsl(var(--muted-foreground, 215.4 16.3% 46.9%));
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .label {
      font-size: 0.875rem;
    }
  `;

  @property({ type: String }) status: StatusType = 'unknown';
  @property({ type: Boolean }) showLabel = false;

  render() {
    return html`
      <span class="dot ${this.status}"></span>
      ${this.showLabel ? html`<span class="label"><slot></slot></span>` : ''}
    `;
  }
}

@customElement('status-indicator')
export class StatusIndicator extends LitElement {
  static styles = css`
    :host {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.5rem 0;
    }

    .icon {
      font-family: ui-monospace, monospace;
      font-weight: 600;
      width: 1.5rem;
    }

    .icon.available,
    .icon.healthy {
      color: hsl(var(--status-success, 142.1 76.2% 36.3%));
    }

    .icon.unavailable,
    .icon.unhealthy {
      color: hsl(var(--status-error, 0 84.2% 60.2%));
    }

    .icon.pending {
      color: hsl(var(--status-warning, 45.4 93.4% 47.5%));
    }

    .icon.unknown {
      color: hsl(var(--muted-foreground, 215.4 16.3% 46.9%));
    }

    .content {
      flex: 1;
    }

    .name {
      font-weight: 500;
      font-family: ui-monospace, monospace;
    }

    .detail {
      font-size: 0.75rem;
      color: hsl(var(--muted-foreground, 215.4 16.3% 46.9%));
    }
  `;

  @property({ type: String }) status: StatusType = 'unknown';
  @property({ type: String }) name = '';
  @property({ type: String }) detail = '';

  render() {
    const icon = getStatusIcon(this.status);
    return html`
      <span class="icon ${this.status}">${icon}</span>
      <div class="content">
        <div class="name">${this.name}</div>
        ${this.detail ? html`<div class="detail">${this.detail}</div>` : ''}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'status-dot': StatusDot;
    'status-indicator': StatusIndicator;
  }
}
