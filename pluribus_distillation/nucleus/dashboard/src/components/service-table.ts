/**
 * Service Table Web Component
 *
 * Lit Element component for displaying services.
 * Works in any web framework (Qwik, React, Vue, etc.)
 */

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { ServiceDef, ServiceInstance } from '../lib/state/types';

@customElement('pluribus-service-table')
export class ServiceTable extends LitElement {
  static override styles = css`
    :host {
      display: block;
      font-family: system-ui, -apple-system, sans-serif;
    }

    .table-container {
      overflow-x: auto;
      border: 1px solid var(--border-color, #e0e0e0);
      border-radius: 8px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }

    th, td {
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid var(--border-color, #e0e0e0);
    }

    th {
      background: var(--header-bg, #f5f5f5);
      font-weight: 600;
      color: var(--header-color, #333);
    }

    tr:hover {
      background: var(--hover-bg, #fafafa);
    }

    tr.selected {
      background: var(--selected-bg, #e3f2fd);
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .status-dot.running { background: #4caf50; }
    .status-dot.stopped { background: #9e9e9e; }
    .status-dot.error { background: #f44336; }
    .status-dot.starting { background: #ff9800; }

    .health {
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 4px;
    }

    .health.healthy { background: #e8f5e9; color: #2e7d32; }
    .health.unhealthy { background: #ffebee; color: #c62828; }
    .health.unknown { background: #f5f5f5; color: #757575; }

    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }

    .tag {
      font-size: 11px;
      padding: 2px 6px;
      background: var(--tag-bg, #e0e0e0);
      border-radius: 4px;
      color: var(--tag-color, #555);
    }

    .kind {
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 4px;
    }

    .kind.port { background: #e3f2fd; color: #1565c0; }
    .kind.process { background: #f3e5f5; color: #7b1fa2; }
    .kind.composition { background: #fff3e0; color: #e65100; }

    .port {
      font-family: monospace;
      color: var(--port-color, #666);
    }

    .empty {
      text-align: center;
      padding: 40px;
      color: var(--muted-color, #999);
    }
  `;

  @property({ type: Array })
  services: ServiceDef[] = [];

  @property({ type: Array })
  instances: ServiceInstance[] = [];

  @property({ type: String })
  selectedId: string | null = null;

  @state()
  private sortColumn = 'name';

  @state()
  private sortDirection: 'asc' | 'desc' = 'asc';

  override render() {
    if (this.services.length === 0) {
      return html`<div class="empty">No services registered</div>`;
    }

    const sorted = this.sortedServices;

    return html`
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th @click=${() => this.toggleSort('id')}>ID</th>
              <th @click=${() => this.toggleSort('name')}>Name</th>
              <th @click=${() => this.toggleSort('kind')}>Kind</th>
              <th>Port</th>
              <th>Status</th>
              <th>Health</th>
              <th>Tags</th>
            </tr>
          </thead>
          <tbody>
            ${sorted.map(svc => this.renderRow(svc))}
          </tbody>
        </table>
      </div>
    `;
  }

  private renderRow(svc: ServiceDef) {
    const instance = this.instances.find(i => i.service_id === svc.id && i.status === 'running');
    const status = instance?.status || 'stopped';
    const health = instance?.health || 'unknown';
    const selected = this.selectedId === svc.id;

    return html`
      <tr
        class=${selected ? 'selected' : ''}
        @click=${() => this.selectService(svc.id)}
      >
        <td><code>${svc.id}</code></td>
        <td>${svc.name}</td>
        <td><span class="kind ${svc.kind}">${svc.kind}</span></td>
        <td class="port">${svc.port || '-'}</td>
        <td>
          <span class="status">
            <span class="status-dot ${status}"></span>
            ${status}
          </span>
        </td>
        <td><span class="health ${health}">${health}</span></td>
        <td>
          <div class="tags">
            ${svc.tags.slice(0, 3).map(tag => html`<span class="tag">${tag}</span>`)}
            ${svc.tags.length > 3 ? html`<span class="tag">+${svc.tags.length - 3}</span>` : ''}
          </div>
        </td>
      </tr>
    `;
  }

  private get sortedServices() {
    return [...this.services].sort((a, b) => {
      const aVal = String(a[this.sortColumn as keyof ServiceDef] || '');
      const bVal = String(b[this.sortColumn as keyof ServiceDef] || '');
      const cmp = aVal.localeCompare(bVal);
      return this.sortDirection === 'asc' ? cmp : -cmp;
    });
  }

  private toggleSort(column: string) {
    if (this.sortColumn === column) {
      this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortColumn = column;
      this.sortDirection = 'asc';
    }
  }

  private selectService(id: string) {
    this.dispatchEvent(new CustomEvent('service-selected', {
      detail: { id },
      bubbles: true,
      composed: true,
    }));
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'pluribus-service-table': ServiceTable;
  }
}
