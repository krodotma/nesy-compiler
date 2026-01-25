/**
 * LitBusMonitor.ts - Web Components Event Monitor Widget
 *
 * Standards-based Lit web component for monitoring Pluribus bus events.
 * Can be embedded anywhere (dashboards, external sites, browser extensions).
 *
 * Features:
 * - Small bundle (~5kb gzipped with Lit)
 * - Shadow DOM isolation
 * - Reactive updates via WebSocket
 * - Configurable filters and display modes
 *
 * Specification from SOTA catalog:
 * - standards_based: native web components
 * - small_bundle: ~5kb core
 * - reactive: efficient updates
 *
 * Usage:
 * ```html
 * <lit-bus-monitor
 *   ws-url="ws://localhost:9200"
 *   topic-filter="strp.*"
 *   max-events="50"
 *   theme="dark"
 * ></lit-bus-monitor>
 * ```
 */

import { LitElement, html, css, type PropertyValues } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

import { matchesTopicFilter } from '../lib/bus/topicFilter';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BusEvent {
  id?: string;
  topic: string;
  kind: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  actor: string;
  ts: number;
  iso: string;
  data: Record<string, unknown>;
}

type DisplayMode = 'compact' | 'detailed' | 'json';

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const sharedStyles = css`
  :host {
    display: block;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --bg-color: var(--lit-bus-bg, #1a1a2e);
    --text-color: var(--lit-bus-text, #e0e0e0);
    --border-color: var(--lit-bus-border, #2a2a4e);
    --accent-color: var(--lit-bus-accent, #6366f1);
    --error-color: var(--lit-bus-error, #ef4444);
    --warn-color: var(--lit-bus-warn, #f59e0b);
    --info-color: var(--lit-bus-info, #3b82f6);
    --debug-color: var(--lit-bus-debug, #6b7280);
  }

  :host([theme="light"]) {
    --bg-color: #ffffff;
    --text-color: #1a1a1a;
    --border-color: #e0e0e0;
  }

  .container {
    background: var(--bg-color);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: rgba(255, 255, 255, 0.03);
    border-bottom: 1px solid var(--border-color);
  }

  .title {
    font-size: 14px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--debug-color);
  }

  .status-dot.connected {
    background: #22c55e;
    animation: pulse 2s infinite;
  }

  .status-dot.error {
    background: var(--error-color);
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .btn {
    padding: 4px 8px;
    font-size: 11px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-color);
    cursor: pointer;
    transition: background 0.2s;
  }

  .btn:hover {
    background: rgba(255, 255, 255, 0.15);
  }

  .btn.active {
    background: var(--accent-color);
    border-color: var(--accent-color);
  }

  .filter-input {
    padding: 4px 8px;
    font-size: 11px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-color);
    width: 120px;
  }

  .filter-input::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }

  .events-container {
    max-height: var(--lit-bus-height, 400px);
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border-color) transparent;
  }

  .events-container::-webkit-scrollbar {
    width: 6px;
  }

  .events-container::-webkit-scrollbar-track {
    background: transparent;
  }

  .events-container::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 3px;
  }

  .event {
    padding: 8px 16px;
    border-bottom: 1px solid var(--border-color);
    font-size: 12px;
    transition: background 0.2s;
  }

  .event:hover {
    background: rgba(255, 255, 255, 0.03);
  }

  .event:last-child {
    border-bottom: none;
  }

  .event.level-error {
    border-left: 3px solid var(--error-color);
  }

  .event.level-warn {
    border-left: 3px solid var(--warn-color);
  }

  .event.level-info {
    border-left: 3px solid var(--info-color);
  }

  .event.level-debug {
    border-left: 3px solid var(--debug-color);
  }

  .event-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
  }

  .event-topic {
    font-weight: 600;
    color: var(--accent-color);
    font-family: monospace;
  }

  .event-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
  }

  .event-actor {
    background: rgba(255, 255, 255, 0.1);
    padding: 2px 6px;
    border-radius: 3px;
  }

  .event-level {
    text-transform: uppercase;
    font-weight: 600;
  }

  .event-level.error { color: var(--error-color); }
  .event-level.warn { color: var(--warn-color); }
  .event-level.info { color: var(--info-color); }
  .event-level.debug { color: var(--debug-color); }

  .event-body {
    color: rgba(255, 255, 255, 0.7);
    word-break: break-word;
  }

  .event-body.json {
    font-family: monospace;
    font-size: 10px;
    background: rgba(0, 0, 0, 0.2);
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    white-space: pre-wrap;
  }

  .empty-state {
    padding: 32px;
    text-align: center;
    color: rgba(255, 255, 255, 0.4);
  }

  .stats-bar {
    display: flex;
    gap: 16px;
    padding: 8px 16px;
    background: rgba(0, 0, 0, 0.2);
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
  }

  .stat {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .stat-value {
    font-weight: 600;
    color: var(--text-color);
  }
`;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

@customElement('lit-bus-monitor')
export class LitBusMonitor extends LitElement {
  static override styles = sharedStyles;

  // ---------------------------------------------------------------------------
  // Properties
  // ---------------------------------------------------------------------------

  /** WebSocket URL for bus bridge */
  @property({ type: String, attribute: 'ws-url' })
  wsUrl = 'ws://localhost:9200';

  /** Topic filter pattern (supports wildcards: strp.*) */
  @property({ type: String, attribute: 'topic-filter' })
  topicFilter = '';

  /** Maximum events to display */
  @property({ type: Number, attribute: 'max-events' })
  maxEvents = 50;

  /** Display theme: dark | light */
  @property({ type: String, reflect: true })
  theme: 'dark' | 'light' = 'dark';

  /** Display mode: compact | detailed | json */
  @property({ type: String, attribute: 'display-mode' })
  displayMode: DisplayMode = 'compact';

  /** Auto-scroll to new events */
  @property({ type: Boolean, attribute: 'auto-scroll' })
  autoScroll = true;

  /** Show stats bar */
  @property({ type: Boolean, attribute: 'show-stats' })
  showStats = true;

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  @state()
  private events: BusEvent[] = [];

  @state()
  private connected = false;

  @state()
  private error: string | null = null;

  @state()
  private localFilter = '';

  @state()
  private paused = false;

  @state()
  private stats = {
    total: 0,
    errors: 0,
    warns: 0,
    rate: 0,
  };

  private ws: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private rateBuffer: number[] = [];

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  override connectedCallback(): void {
    super.connectedCallback();
    this.connect();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.disconnect();
  }

  override updated(changedProps: PropertyValues): void {
    if (changedProps.has('wsUrl')) {
      this.reconnect();
    }

    // Auto-scroll
    if (changedProps.has('events') && this.autoScroll && !this.paused) {
      const container = this.shadowRoot?.querySelector('.events-container');
      if (container) {
        container.scrollTop = 0; // Scroll to top (newest events)
      }
    }
  }

  // ---------------------------------------------------------------------------
  // WebSocket Management
  // ---------------------------------------------------------------------------

  private connect(): void {
    if (this.ws) {
      this.disconnect();
    }

    try {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        this.connected = true;
        this.error = null;

        // Request sync
        this.ws?.send(JSON.stringify({ type: 'sync' }));

        // Subscribe to filtered topics
        if (this.topicFilter) {
          this.ws?.send(JSON.stringify({ type: 'subscribe', topic: this.topicFilter }));
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === 'sync') {
            this.handleSync(msg.events || []);
          } else if (msg.type === 'event') {
            this.handleEvent(msg.event);
          }
        } catch (err) {
          console.error('[lit-bus-monitor] Parse error:', err);
        }
      };

      this.ws.onerror = () => {
        this.error = 'Connection error';
        this.connected = false;
      };

      this.ws.onclose = () => {
        this.connected = false;
        this.scheduleReconnect();
      };

    } catch (err) {
      this.error = String(err);
      this.connected = false;
    }
  }

  private disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private reconnect(): void {
    this.disconnect();
    this.connect();
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;

    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 3000);
  }

  // ---------------------------------------------------------------------------
  // Event Handling
  // ---------------------------------------------------------------------------

  private handleSync(events: BusEvent[]): void {
    const filtered = this.filterEvents(events);
    this.events = filtered.slice(-this.maxEvents);
    this.updateStats();
  }

  private handleEvent(event: BusEvent): void {
    if (this.paused) return;

    if (!this.matchesFilter(event)) return;

    // Prepend (newest first)
    this.events = [event, ...this.events].slice(0, this.maxEvents);

    // Update stats
    this.stats.total++;
    if (event.level === 'error') this.stats.errors++;
    if (event.level === 'warn') this.stats.warns++;

    // Track rate
    this.rateBuffer.push(Date.now());
    this.rateBuffer = this.rateBuffer.filter(t => Date.now() - t < 60000);
    this.stats.rate = this.rateBuffer.length;
  }

  private filterEvents(events: BusEvent[]): BusEvent[] {
    return events.filter(e => this.matchesFilter(e));
  }

  private matchesFilter(event: BusEvent): boolean {
    // Topic filter
    if (this.topicFilter) {
      if (!matchesTopicFilter(event.topic, this.topicFilter)) return false;
    }

    // Local filter (text search)
    if (this.localFilter) {
      const search = this.localFilter.toLowerCase();
      const text = `${event.topic} ${event.actor} ${event.kind} ${JSON.stringify(event.data)}`.toLowerCase();
      if (!text.includes(search)) return false;
    }

    return true;
  }

  private updateStats(): void {
    this.stats = {
      total: this.events.length,
      errors: this.events.filter(e => e.level === 'error').length,
      warns: this.events.filter(e => e.level === 'warn').length,
      rate: this.rateBuffer.filter(t => Date.now() - t < 60000).length,
    };
  }

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  private handleFilterInput(e: Event): void {
    this.localFilter = (e.target as HTMLInputElement).value;
    this.updateStats();
  }

  private togglePause(): void {
    this.paused = !this.paused;
  }

  private clearEvents(): void {
    this.events = [];
    this.stats = { total: 0, errors: 0, warns: 0, rate: 0 };
  }

  private setDisplayMode(mode: DisplayMode): void {
    this.displayMode = mode;
  }

  private publishTestEvent(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    const event: BusEvent = {
      id: `test-${Date.now()}`,
      topic: 'lit-bus-monitor.test',
      kind: 'test',
      level: 'info',
      actor: 'lit-bus-monitor',
      ts: Date.now(),
      iso: new Date().toISOString(),
      data: { message: 'Test event from LitBusMonitor' },
    };

    this.ws.send(JSON.stringify({ type: 'publish', event }));
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  private renderEvent(event: BusEvent): unknown {
    const time = new Date(event.ts).toLocaleTimeString();

    if (this.displayMode === 'json') {
      return html`
        <div class="event level-${event.level}">
          <div class="event-body json">${JSON.stringify(event, null, 2)}</div>
        </div>
      `;
    }

    const summary = this.getEventSummary(event);

    return html`
      <div class="event level-${event.level}">
        <div class="event-header">
          <span class="event-topic">${event.topic}</span>
          <div class="event-meta">
            <span class="event-actor">${event.actor}</span>
            <span class="event-level ${event.level}">${event.level}</span>
            <span>${time}</span>
          </div>
        </div>
        ${this.displayMode === 'detailed' || summary ? html`
          <div class="event-body">${summary || JSON.stringify(event.data)}</div>
        ` : ''}
      </div>
    `;
  }

  private getEventSummary(event: BusEvent): string {
    const data = event.data;
    if (data.message) return String(data.message);
    if (data.goal) return String(data.goal);
    if (data.status) return String(data.status);
    if (data.error) return String(data.error);
    return '';
  }

  override render(): unknown {
    return html`
      <div class="container">
        <!-- Header -->
        <div class="header">
          <div class="title">
            <div class="status-dot ${this.connected ? 'connected' : this.error ? 'error' : ''}"></div>
            <span>Bus Monitor</span>
            ${this.paused ? html`<span style="color: var(--warn-color)">[PAUSED]</span>` : ''}
          </div>
          <div class="controls">
            <input
              type="text"
              class="filter-input"
              placeholder="Filter..."
              .value=${this.localFilter}
              @input=${this.handleFilterInput}
            />
            <button class="btn ${this.displayMode === 'compact' ? 'active' : ''}" @click=${() => this.setDisplayMode('compact')}>
              Compact
            </button>
            <button class="btn ${this.displayMode === 'detailed' ? 'active' : ''}" @click=${() => this.setDisplayMode('detailed')}>
              Detailed
            </button>
            <button class="btn ${this.displayMode === 'json' ? 'active' : ''}" @click=${() => this.setDisplayMode('json')}>
              JSON
            </button>
            <button class="btn ${this.paused ? 'active' : ''}" @click=${this.togglePause}>
              ${this.paused ? 'Resume' : 'Pause'}
            </button>
            <button class="btn" @click=${this.clearEvents}>Clear</button>
          </div>
        </div>

        <!-- Stats Bar -->
        ${this.showStats ? html`
          <div class="stats-bar">
            <div class="stat">
              <span>Total:</span>
              <span class="stat-value">${this.stats.total}</span>
            </div>
            <div class="stat">
              <span>Errors:</span>
              <span class="stat-value" style="color: var(--error-color)">${this.stats.errors}</span>
            </div>
            <div class="stat">
              <span>Warnings:</span>
              <span class="stat-value" style="color: var(--warn-color)">${this.stats.warns}</span>
            </div>
            <div class="stat">
              <span>Rate:</span>
              <span class="stat-value">${this.stats.rate}/min</span>
            </div>
          </div>
        ` : ''}

        <!-- Events -->
        <div class="events-container">
          ${this.events.length > 0
            ? this.events.map(e => this.renderEvent(e))
            : html`
              <div class="empty-state">
                ${this.connected
                  ? 'Waiting for events...'
                  : this.error
                    ? `Connection error: ${this.error}`
                    : 'Connecting...'
                }
              </div>
            `
          }
        </div>
      </div>
    `;
  }
}

// ---------------------------------------------------------------------------
// Type Declaration for Custom Element
// ---------------------------------------------------------------------------

declare global {
  interface HTMLElementTagNameMap {
    'lit-bus-monitor': LitBusMonitor;
  }
}

export default LitBusMonitor;
