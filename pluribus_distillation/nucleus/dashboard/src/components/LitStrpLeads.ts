/**
 * LitStrpLeads.ts - Web Components STRp Leads Widget
 *
 * Standards-based Lit web component for displaying content curation leads.
 * Can be embedded anywhere (dashboards, external sites, browser extensions).
 *
 * Features:
 * - Small bundle (~5kb gzipped with Lit)
 * - Shadow DOM isolation
 * - Reactive updates via WebSocket
 * - Configurable filters and display modes
 * - Compact card layout for leads
 *
 * Usage:
 * ```html
 * <pluribus-strp-leads
 *   ws-url="ws://localhost:9200"
 *   topic-filter="strp.lead.*"
 *   max-leads="50"
 *   theme="dark"
 * ></pluribus-strp-leads>
 * ```
 */

import { LitElement, html, css, type PropertyValues } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

import { matchesTopicFilter } from '../lib/bus/topicFilter';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Curation decision for a lead */
type LeadDecision = 'promote' | 'defer' | 'reject';

/** Topic category for content classification */
type LeadTopic =
  | 'FILM'
  | 'AI/ML'
  | 'MUSIC'
  | 'ART'
  | 'TECH'
  | 'SCIENCE'
  | 'CULTURE'
  | 'GAMING'
  | 'DESIGN'
  | 'OTHER';

/** Artifact types available for a lead */
interface LeadArtifacts {
  thumb?: string;
  gif?: string;
  clip?: string;
  mp3?: string;
}

/** Main STRpLead interface */
interface STRpLead {
  lead_id: string;
  ts: string;
  actor: string;
  decision: LeadDecision;
  topic: LeadTopic | string;
  title: string;
  url: string;
  keywords: string[];
  next_actions: string[];
  artifacts: LeadArtifacts;
  notes?: string;
  priority?: number;
  ingested?: boolean;
  portal_target?: string;
  archived?: boolean;
}

/** Tab navigation type */
type LeadTab = 'all' | LeadDecision;

/** Actions that can be performed on a lead */
type LeadAction = 'watch' | 'ingest' | 'archive' | 'promote' | 'defer' | 'reject';

/** Bus event envelope */
interface BusEvent {
  type: string;
  topic?: string;
  event?: {
    topic: string;
    data: Record<string, unknown>;
  };
  events?: BusEvent[];
  leads?: STRpLead[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getDecisionColor(decision: LeadDecision): string {
  switch (decision) {
    case 'promote':
      return 'decision-promote';
    case 'defer':
      return 'decision-defer';
    case 'reject':
      return 'decision-reject';
  }
}

function getTopicIcon(topic: string): string {
  const icons: Record<string, string> = {
    FILM: '🎬',
    'AI/ML': '🤖',
    MUSIC: '🎵',
    ART: '🎨',
    TECH: '💻',
    SCIENCE: '🔬',
    CULTURE: '🌍',
    GAMING: '🎮',
    DESIGN: '✏',
    OTHER: '📌',
  };
  return icons[topic] || '📌';
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const sharedStyles = css`
  :host {
    display: block;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --bg-color: var(--lit-strp-bg, #1a1a2e);
    --text-color: var(--lit-strp-text, #e0e0e0);
    --border-color: var(--lit-strp-border, #2a2a4e);
    --accent-color: var(--lit-strp-accent, #6366f1);
    --muted-color: var(--lit-strp-muted, #6b7280);
    --promote-color: var(--lit-strp-promote, #22c55e);
    --defer-color: var(--lit-strp-defer, #f59e0b);
    --reject-color: var(--lit-strp-reject, #ef4444);
    --card-bg: var(--lit-strp-card-bg, rgba(255, 255, 255, 0.03));
  }

  :host([theme='light']) {
    --bg-color: #ffffff;
    --text-color: #1a1a1a;
    --border-color: #e0e0e0;
    --card-bg: rgba(0, 0, 0, 0.02);
    --muted-color: #6b7280;
  }

  * {
    box-sizing: border-box;
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
    background: var(--card-bg);
    border-bottom: 1px solid var(--border-color);
    gap: 12px;
    flex-wrap: wrap;
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
    background: var(--muted-color);
  }

  .status-dot.connected {
    background: var(--promote-color);
    animation: pulse 2s infinite;
  }

  .status-dot.error {
    background: var(--reject-color);
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .filter-input {
    padding: 4px 8px;
    font-size: 11px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-color);
    width: 140px;
  }

  .filter-input::placeholder {
    color: var(--muted-color);
  }

  .filter-select {
    padding: 4px 8px;
    font-size: 11px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-color);
  }

  .count-badge {
    font-size: 11px;
    color: var(--muted-color);
  }

  /* Tabs */
  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border-color);
    background: var(--card-bg);
  }

  .tab {
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 500;
    border: none;
    background: transparent;
    color: var(--muted-color);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }

  .tab:hover {
    color: var(--text-color);
  }

  .tab.active {
    color: var(--text-color);
    border-bottom-color: var(--accent-color);
  }

  .tab.active.promote {
    color: var(--promote-color);
    border-bottom-color: var(--promote-color);
  }

  .tab.active.defer {
    color: var(--defer-color);
    border-bottom-color: var(--defer-color);
  }

  .tab.active.reject {
    color: var(--reject-color);
    border-bottom-color: var(--reject-color);
  }

  .tab-count {
    margin-left: 4px;
    font-size: 10px;
    opacity: 0.7;
  }

  /* Leads Container */
  .leads-container {
    max-height: var(--lit-strp-height, 500px);
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border-color) transparent;
  }

  .leads-container::-webkit-scrollbar {
    width: 6px;
  }

  .leads-container::-webkit-scrollbar-track {
    background: transparent;
  }

  .leads-container::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 3px;
  }

  /* Lead Card */
  .lead-card {
    display: flex;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    transition: background 0.2s;
  }

  .lead-card:hover {
    background: var(--card-bg);
  }

  .lead-card:last-child {
    border-bottom: none;
  }

  /* Thumbnail */
  .lead-thumb {
    width: 64px;
    height: 64px;
    border-radius: 6px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.05);
    flex-shrink: 0;
    position: relative;
  }

  .lead-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .lead-thumb-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: var(--muted-color);
  }

  .media-badges {
    position: absolute;
    bottom: 2px;
    right: 2px;
    display: flex;
    gap: 2px;
  }

  .media-badge {
    font-size: 8px;
    padding: 1px 3px;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 2px;
  }

  /* Lead Content */
  .lead-content {
    flex: 1;
    min-width: 0;
  }

  .lead-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
  }

  .lead-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--accent-color);
    text-decoration: none;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
  }

  .lead-title:hover {
    text-decoration: underline;
  }

  .decision-badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
    text-transform: uppercase;
    font-weight: 600;
    flex-shrink: 0;
  }

  .decision-promote {
    background: rgba(34, 197, 94, 0.2);
    color: var(--promote-color);
    border: 1px solid rgba(34, 197, 94, 0.3);
  }

  .decision-defer {
    background: rgba(245, 158, 11, 0.2);
    color: var(--defer-color);
    border: 1px solid rgba(245, 158, 11, 0.3);
  }

  .decision-reject {
    background: rgba(239, 68, 68, 0.2);
    color: var(--reject-color);
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .lead-meta {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    color: var(--muted-color);
    margin-bottom: 6px;
    flex-wrap: wrap;
  }

  .meta-sep {
    opacity: 0.4;
  }

  .priority-high {
    color: var(--reject-color);
    font-weight: 600;
  }

  /* Keywords */
  .lead-keywords {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 6px;
  }

  .keyword-tag {
    font-size: 9px;
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    color: var(--muted-color);
  }

  .keyword-more {
    background: rgba(255, 255, 255, 0.04);
  }

  /* Notes */
  .lead-notes {
    font-size: 11px;
    color: var(--muted-color);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-bottom: 6px;
  }

  /* Actions */
  .lead-actions {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .action-btn {
    font-size: 10px;
    padding: 3px 8px;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 3px;
    transition: all 0.2s;
  }

  .action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .action-watch {
    background: rgba(99, 102, 241, 0.15);
    color: var(--accent-color);
  }

  .action-watch:hover:not(:disabled) {
    background: rgba(99, 102, 241, 0.25);
  }

  .action-ingest {
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
  }

  .action-ingest:hover:not(:disabled) {
    background: rgba(59, 130, 246, 0.25);
  }

  .action-archive {
    background: rgba(107, 114, 128, 0.15);
    color: var(--muted-color);
  }

  .action-archive:hover:not(:disabled) {
    background: rgba(107, 114, 128, 0.25);
  }

  .decision-actions {
    margin-left: auto;
    display: flex;
    gap: 4px;
  }

  .decision-btn {
    font-size: 10px;
    padding: 3px 8px;
    border: none;
    border-radius: 3px;
    cursor: pointer;
  }

  .decision-btn.promote {
    background: rgba(34, 197, 94, 0.15);
    color: var(--promote-color);
  }

  .decision-btn.promote:hover {
    background: rgba(34, 197, 94, 0.25);
  }

  .decision-btn.defer {
    background: rgba(245, 158, 11, 0.15);
    color: var(--defer-color);
  }

  .decision-btn.defer:hover {
    background: rgba(245, 158, 11, 0.25);
  }

  .decision-btn.reject {
    background: rgba(239, 68, 68, 0.15);
    color: var(--reject-color);
  }

  .decision-btn.reject:hover {
    background: rgba(239, 68, 68, 0.25);
  }

  /* Empty State */
  .empty-state {
    padding: 48px 24px;
    text-align: center;
    color: var(--muted-color);
  }

  .empty-icon {
    font-size: 48px;
    margin-bottom: 16px;
  }

  .empty-text {
    font-size: 13px;
  }

  /* Stats Bar */
  .stats-bar {
    display: flex;
    gap: 8px;
    padding: 10px 16px;
    background: var(--card-bg);
    border-top: 1px solid var(--border-color);
    flex-wrap: wrap;
  }

  .stat-box {
    flex: 1;
    min-width: 60px;
    padding: 8px;
    text-align: center;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-color);
  }

  .stat-box.promote {
    border-color: rgba(34, 197, 94, 0.3);
    background: rgba(34, 197, 94, 0.1);
  }

  .stat-box.defer {
    border-color: rgba(245, 158, 11, 0.3);
    background: rgba(245, 158, 11, 0.1);
  }

  .stat-box.reject {
    border-color: rgba(239, 68, 68, 0.3);
    background: rgba(239, 68, 68, 0.1);
  }

  .stat-box.ingested {
    border-color: rgba(59, 130, 246, 0.3);
    background: rgba(59, 130, 246, 0.1);
  }

  .stat-value {
    font-size: 18px;
    font-weight: 700;
  }

  .stat-value.promote {
    color: var(--promote-color);
  }

  .stat-value.defer {
    color: var(--defer-color);
  }

  .stat-value.reject {
    color: var(--reject-color);
  }

  .stat-value.ingested {
    color: #3b82f6;
  }

  .stat-label {
    font-size: 9px;
    text-transform: uppercase;
    color: var(--muted-color);
    margin-top: 2px;
  }
`;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

@customElement('pluribus-strp-leads')
export class LitStrpLeads extends LitElement {
  static override styles = sharedStyles;

  // ---------------------------------------------------------------------------
  // Properties
  // ---------------------------------------------------------------------------

  /** WebSocket URL for bus bridge */
  @property({ type: String, attribute: 'ws-url' })
  wsUrl = 'ws://localhost:9200';

  /** Topic filter pattern (supports wildcards: strp.*) */
  @property({ type: String, attribute: 'topic-filter' })
  topicFilter = 'strp.lead.*';

  /** Maximum leads to display */
  @property({ type: Number, attribute: 'max-leads' })
  maxLeads = 50;

  /** Display theme: dark | light */
  @property({ type: String, reflect: true })
  theme: 'dark' | 'light' = 'dark';

  /** Show stats bar */
  @property({ type: Boolean, attribute: 'show-stats' })
  showStats = true;

  /** Enable actions (watch, ingest, etc.) */
  @property({ type: Boolean, attribute: 'enable-actions' })
  enableActions = true;

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  @state()
  private leads: STRpLead[] = [];

  @state()
  private connected = false;

  @state()
  private error: string | null = null;

  @state()
  private activeTab: LeadTab = 'all';

  @state()
  private topicFilterLocal: string = 'ALL';

  @state()
  private searchQuery = '';

  private ws: WebSocket | null = null;
  private reconnectTimer: number | null = null;

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

        // Subscribe to leads topics
        if (this.topicFilter) {
          this.ws?.send(JSON.stringify({ type: 'subscribe', topic: this.topicFilter }));
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as BusEvent;
          this.handleMessage(msg);
        } catch (err) {
          console.error('[pluribus-strp-leads] Parse error:', err);
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
  // Message Handling
  // ---------------------------------------------------------------------------

  private handleMessage(msg: BusEvent): void {
    // Handle sync response with leads
    if (msg.type === 'sync') {
      if (msg.leads) {
        this.leads = msg.leads.slice(-this.maxLeads);
      }
      return;
    }

    // Handle individual events
    if (msg.type === 'event' && msg.event) {
      const { topic, data } = msg.event;

      // Filter by topic
      if (this.topicFilter && !matchesTopicFilter(topic, this.topicFilter)) {
        return;
      }

      // Handle different lead events
      if (topic === 'strp.leads.sync' && data.leads) {
        this.leads = (data.leads as STRpLead[]).slice(-this.maxLeads);
      } else if (topic === 'strp.lead.created' && data.lead) {
        this.addLead(data.lead as STRpLead);
      } else if (topic === 'strp.lead.updated' && data.lead) {
        this.updateLead(data.lead as STRpLead);
      } else if (topic === 'strp.lead.deleted' && data.lead_id) {
        this.removeLead(data.lead_id as string);
      }
    }
  }

  private addLead(lead: STRpLead): void {
    // Check if lead already exists
    const existing = this.leads.find((l) => l.lead_id === lead.lead_id);
    if (existing) {
      this.updateLead(lead);
      return;
    }

    this.leads = [lead, ...this.leads].slice(0, this.maxLeads);
  }

  private updateLead(lead: STRpLead): void {
    this.leads = this.leads.map((l) => (l.lead_id === lead.lead_id ? lead : l));
  }

  private removeLead(leadId: string): void {
    this.leads = this.leads.filter((l) => l.lead_id !== leadId);
  }

  // ---------------------------------------------------------------------------
  // Computed Values
  // ---------------------------------------------------------------------------

  private get filteredLeads(): STRpLead[] {
    let result = [...this.leads];

    // Filter by tab
    if (this.activeTab !== 'all') {
      result = result.filter((l) => l.decision === this.activeTab);
    }

    // Filter by topic
    if (this.topicFilterLocal !== 'ALL') {
      result = result.filter((l) => l.topic === this.topicFilterLocal);
    }

    // Filter by search
    if (this.searchQuery.trim()) {
      const q = this.searchQuery.toLowerCase();
      result = result.filter(
        (l) =>
          l.title.toLowerCase().includes(q) ||
          l.keywords.some((k) => k.toLowerCase().includes(q)) ||
          l.notes?.toLowerCase().includes(q)
      );
    }

    // Sort by timestamp (newest first)
    return result.sort((a, b) => b.ts.localeCompare(a.ts));
  }

  private get counts(): Record<LeadTab, number> {
    return {
      all: this.leads.length,
      promote: this.leads.filter((l) => l.decision === 'promote').length,
      defer: this.leads.filter((l) => l.decision === 'defer').length,
      reject: this.leads.filter((l) => l.decision === 'reject').length,
    };
  }

  private get availableTopics(): string[] {
    const topics = new Set<string>();
    for (const lead of this.leads) {
      topics.add(lead.topic);
    }
    return Array.from(topics).sort();
  }

  private get ingestedCount(): number {
    return this.leads.filter((l) => l.ingested).length;
  }

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  private handleSearch(e: Event): void {
    this.searchQuery = (e.target as HTMLInputElement).value;
  }

  private handleTopicFilterChange(e: Event): void {
    this.topicFilterLocal = (e.target as HTMLSelectElement).value;
  }

  private setActiveTab(tab: LeadTab): void {
    this.activeTab = tab;
  }

  private handleAction(action: LeadAction, lead: STRpLead): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    const topicMap: Record<LeadAction, string> = {
      watch: 'strp.lead.action.watch',
      ingest: 'strp.lead.action.ingest',
      archive: 'strp.lead.action.archive',
      promote: 'strp.lead.action.decision',
      defer: 'strp.lead.action.decision',
      reject: 'strp.lead.action.decision',
    };

    const topic = topicMap[action];

    this.ws.send(
      JSON.stringify({
        type: 'publish',
        event: {
          topic,
          kind: 'action',
          level: 'info',
          actor: 'pluribus-strp-leads',
          ts: Date.now(),
          iso: new Date().toISOString(),
          data: {
            lead_id: lead.lead_id,
            action,
            lead,
          },
        },
      })
    );

    // Optimistic update for decision changes
    if (['promote', 'defer', 'reject'].includes(action)) {
      this.updateLead({ ...lead, decision: action as LeadDecision });
    }

    // Dispatch custom event for external listeners
    this.dispatchEvent(
      new CustomEvent('lead-action', {
        detail: { action, lead },
        bubbles: true,
        composed: true,
      })
    );
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  private renderThumbnail(lead: STRpLead): unknown {
    const { artifacts, title } = lead;
    const hasThumb = !!artifacts.thumb;
    const hasGif = !!artifacts.gif;

    return html`
      <div class="lead-thumb">
        ${hasThumb || hasGif
          ? html`<img src="${hasGif ? artifacts.gif : artifacts.thumb}" alt="${title}" loading="lazy" />`
          : html`<div class="lead-thumb-placeholder">📄</div>`}
        <div class="media-badges">
          ${artifacts.clip ? html`<span class="media-badge">🎬</span>` : ''}
          ${artifacts.mp3 ? html`<span class="media-badge">🎵</span>` : ''}
        </div>
      </div>
    `;
  }

  private renderKeywords(keywords: string[]): unknown {
    if (!keywords.length) return '';

    const display = keywords.slice(0, 4);
    const remaining = keywords.length - 4;

    return html`
      <div class="lead-keywords">
        ${display.map((kw) => html`<span class="keyword-tag">${kw}</span>`)}
        ${remaining > 0 ? html`<span class="keyword-tag keyword-more">+${remaining}</span>` : ''}
      </div>
    `;
  }

  private renderLeadCard(lead: STRpLead): unknown {
    const decisionColor = getDecisionColor(lead.decision);
    const topicIcon = getTopicIcon(lead.topic);

    return html`
      <div class="lead-card">
        ${this.renderThumbnail(lead)}
        <div class="lead-content">
          <div class="lead-header">
            <a href="${lead.url}" target="_blank" rel="noopener noreferrer" class="lead-title" title="${lead.title}">
              ${lead.title}
            </a>
            <span class="decision-badge ${decisionColor}">${lead.decision}</span>
          </div>

          <div class="lead-meta">
            <span>${topicIcon} ${lead.topic}</span>
            <span class="meta-sep">|</span>
            <span>${lead.ts.slice(0, 10)}</span>
            ${lead.priority
              ? html`
                  <span class="meta-sep">|</span>
                  <span class="${lead.priority === 1 ? 'priority-high' : ''}">P${lead.priority}</span>
                `
              : ''}
            <span class="meta-sep">|</span>
            <span>${lead.actor}</span>
          </div>

          ${this.renderKeywords(lead.keywords)}
          ${lead.notes ? html`<div class="lead-notes" title="${lead.notes}">${lead.notes}</div>` : ''}
          ${this.enableActions
            ? html`
                <div class="lead-actions">
                  <button class="action-btn action-watch" @click=${() => this.handleAction('watch', lead)}>
                    👁 Watch
                  </button>
                  <button
                    class="action-btn action-ingest"
                    ?disabled=${lead.ingested}
                    @click=${() => this.handleAction('ingest', lead)}
                  >
                    📥 ${lead.ingested ? 'Ingested' : 'Ingest'}
                  </button>
                  <button
                    class="action-btn action-archive"
                    ?disabled=${lead.archived}
                    @click=${() => this.handleAction('archive', lead)}
                  >
                    📦 ${lead.archived ? 'Archived' : 'Archive'}
                  </button>

                  <div class="decision-actions">
                    ${lead.decision !== 'promote'
                      ? html`<button class="decision-btn promote" title="Promote" @click=${() => this.handleAction('promote', lead)}>
                          Promote
                        </button>`
                      : ''}
                    ${lead.decision !== 'defer'
                      ? html`<button class="decision-btn defer" title="Defer" @click=${() => this.handleAction('defer', lead)}>
                          Defer
                        </button>`
                      : ''}
                    ${lead.decision !== 'reject'
                      ? html`<button class="decision-btn reject" title="Reject" @click=${() => this.handleAction('reject', lead)}>
                          Reject
                        </button>`
                      : ''}
                  </div>
                </div>
              `
            : ''}
        </div>
      </div>
    `;
  }

  override render(): unknown {
    const tabs: { id: LeadTab; label: string }[] = [
      { id: 'all', label: 'All' },
      { id: 'promote', label: 'Promote' },
      { id: 'defer', label: 'Defer' },
      { id: 'reject', label: 'Reject' },
    ];

    return html`
      <div class="container">
        <!-- Header -->
        <div class="header">
          <div class="title">
            <div class="status-dot ${this.connected ? 'connected' : this.error ? 'error' : ''}"></div>
            <span>📋 STRp Leads</span>
          </div>
          <div class="controls">
            <select class="filter-select" .value=${this.topicFilterLocal} @change=${this.handleTopicFilterChange}>
              <option value="ALL">All Topics</option>
              ${this.availableTopics.map(
                (t) => html`<option value="${t}">${getTopicIcon(t)} ${t}</option>`
              )}
            </select>
            <input
              type="text"
              class="filter-input"
              placeholder="Search leads..."
              .value=${this.searchQuery}
              @input=${this.handleSearch}
            />
            <span class="count-badge">${this.filteredLeads.length} / ${this.leads.length}</span>
          </div>
        </div>

        <!-- Tabs -->
        <div class="tabs">
          ${tabs.map(
            (tab) => html`
              <button
                class="tab ${this.activeTab === tab.id ? `active ${tab.id}` : ''}"
                @click=${() => this.setActiveTab(tab.id)}
              >
                ${tab.label}
                <span class="tab-count">(${this.counts[tab.id]})</span>
              </button>
            `
          )}
        </div>

        <!-- Leads List -->
        <div class="leads-container">
          ${this.filteredLeads.length > 0
            ? this.filteredLeads.map((lead) => this.renderLeadCard(lead))
            : html`
                <div class="empty-state">
                  <div class="empty-icon">📭</div>
                  <div class="empty-text">
                    ${this.leads.length === 0
                      ? this.connected
                        ? 'No leads available. Waiting for MOTD pipeline...'
                        : 'Connecting...'
                      : 'No leads match the current filters.'}
                  </div>
                </div>
              `}
        </div>

        <!-- Stats Bar -->
        ${this.showStats
          ? html`
              <div class="stats-bar">
                <div class="stat-box">
                  <div class="stat-value">${this.counts.all}</div>
                  <div class="stat-label">Total</div>
                </div>
                <div class="stat-box promote">
                  <div class="stat-value promote">${this.counts.promote}</div>
                  <div class="stat-label">Promote</div>
                </div>
                <div class="stat-box defer">
                  <div class="stat-value defer">${this.counts.defer}</div>
                  <div class="stat-label">Defer</div>
                </div>
                <div class="stat-box reject">
                  <div class="stat-value reject">${this.counts.reject}</div>
                  <div class="stat-label">Reject</div>
                </div>
                <div class="stat-box ingested">
                  <div class="stat-value ingested">${this.ingestedCount}</div>
                  <div class="stat-label">Ingested</div>
                </div>
              </div>
            `
          : ''}
      </div>
    `;
  }
}

// ---------------------------------------------------------------------------
// Type Declaration for Custom Element
// ---------------------------------------------------------------------------

declare global {
  interface HTMLElementTagNameMap {
    'pluribus-strp-leads': LitStrpLeads;
  }
}

export default LitStrpLeads;
