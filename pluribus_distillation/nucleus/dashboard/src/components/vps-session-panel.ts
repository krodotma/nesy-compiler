/**
 * VPS Session Panel Web Component
 *
 * Lit Element component for displaying VPS session status,
 * provider availability, and fallback mode controls.
 */

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { VPSSession, FlowMode, ProviderStatus } from '../lib/state/types';

@customElement('pluribus-vps-session')
export class VPSSessionPanel extends LitElement {
  @property({ type: Object })
  session: VPSSession = {
    flowMode: 'm',
    providers: {
      'chatgpt-web': { available: false, lastCheck: '' },
      'claude-web': { available: false, lastCheck: '' },
      'gemini-web': { available: false, lastCheck: '' },
    },
    fallbackOrder: [],
    activeFallback: null,
    pbpair: { activeRequests: [], pendingProposals: [] },
    auth: { claudeLoggedIn: false, geminiCliLoggedIn: false },
  };

  @state() private chatTarget: string | null = null;
  @state() private chatInput: string = '';
  @state() private chatHistory: Record<string, Array<{role: string, content: string}>> = {};

  static override styles = css`
    :host {
      display: block;
      font-family: system-ui, -apple-system, sans-serif;
      --primary-color: #06b6d4; /* Electric Cyan */
      --bg-dark: #09090b;
      --glass-border: rgba(255, 255, 255, 0.1);
    }

    .panel {
      background: var(--panel-bg, #fff);
      border: 1px solid var(--border-color, #e0e0e0);
      border-radius: 8px;
      padding: 16px;
      position: relative;
    }

    /* ... existing styles ... */
    
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }

    h2 {
      margin: 0;
      font-size: 18px;
      color: var(--heading-color, #333);
    }

    .flow-mode {
      display: flex;
      gap: 8px;
    }

    .mode-btn {
      padding: 8px 16px;
      border: 2px solid var(--border-color, #e0e0e0);
      border-radius: 6px;
      background: var(--btn-bg, #fff);
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      transition: all 0.2s;
    }

    .mode-btn:hover {
      border-color: var(--primary-color, #1976d2);
    }

    .mode-btn.active {
      background: var(--primary-color, #1976d2);
      border-color: var(--primary-color, #1976d2);
      color: white;
    }

    .mode-btn.monitor { --active-color: #ff9800; }
    .mode-btn.auto { --active-color: #4caf50; }

    .mode-btn.active.monitor {
      background: #ff9800;
      border-color: #ff9800;
    }

    .mode-btn.active.auto {
      background: #4caf50;
      border-color: #4caf50;
    }

    .section {
      margin-bottom: 20px;
    }

    .section-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--muted-color, #666);
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .providers {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }

    .provider {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px;
      background: var(--card-bg, #f9f9f9);
      border-radius: 8px;
      transition: all 0.2s;
      position: relative;
    }

    .provider:hover {
      background: var(--card-hover-bg, #f0f0f0);
    }

    .provider.active {
      border: 2px solid var(--primary-color, #1976d2);
      background: var(--active-bg, #e3f2fd);
    }

    .status-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .status-indicator.available { background: #4caf50; }
    .status-indicator.unavailable { background: #f44336; }
    .status-indicator.unknown { background: #9e9e9e; }

    .provider-info {
      flex: 1;
      min-width: 0;
    }

    .provider-name {
      font-weight: 500;
      font-size: 14px;
      color: var(--text-color, #333);
    }

    .provider-status {
      font-size: 12px;
      color: var(--muted-color, #666);
    }

    .chat-btn {
      background: transparent;
      border: 1px solid var(--border-color, #ccc);
      color: var(--text-color, #333);
      cursor: pointer;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 10px;
      opacity: 0.6;
      transition: all 0.2s;
    }
    .chat-btn:hover {
      opacity: 1;
      background: var(--primary-color);
      color: white;
      border-color: var(--primary-color);
    }

    /* Micro-Chat Overlay */
    .micro-chat-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.5);
      backdrop-filter: blur(4px);
      z-index: 100;
      display: flex;
      justify-content: center;
      align-items: center;
      border-radius: 8px;
    }

    .micro-chat-box {
      width: 90%;
      max-width: 400px;
      height: 80%;
      background: var(--bg-dark, #111);
      border: 1px solid var(--primary-color);
      border-radius: 8px;
      box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .chat-header {
      padding: 8px 12px;
      background: rgba(6, 182, 212, 0.1);
      border-bottom: 1px solid var(--glass-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--primary-color);
      font-weight: bold;
      font-size: 13px;
    }

    .close-chat {
      cursor: pointer;
      font-size: 16px;
      opacity: 0.7;
    }
    .close-chat:hover { opacity: 1; }

    .chat-history {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-family: monospace;
      font-size: 12px;
    }

    .msg {
      padding: 6px 10px;
      border-radius: 4px;
      max-width: 85%;
      word-wrap: break-word;
    }
    .msg.user {
      align-self: flex-end;
      background: rgba(6, 182, 212, 0.2);
      color: #fff;
      border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .msg.assistant {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.05);
      color: #ccc;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .chat-input-area {
      padding: 8px;
      border-top: 1px solid var(--glass-border);
      display: flex;
      gap: 4px;
    }

    .chat-input {
      flex: 1;
      background: rgba(0,0,0,0.3);
      border: 1px solid var(--glass-border);
      color: white;
      padding: 6px;
      border-radius: 4px;
      font-family: monospace;
    }
    .chat-input:focus {
      outline: none;
      border-color: var(--primary-color);
    }

    /* Fallback chain styles */
    .fallback-chain {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }

    .fallback-item {
      padding: 6px 12px;
      background: var(--card-bg, #f0f0f0);
      border-radius: 16px;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .fallback-item.active {
      background: var(--primary-color, #1976d2);
      color: white;
    }

    .fallback-arrow {
      color: var(--muted-color, #999);
      font-size: 12px;
    }

    .auth-section {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }

    .auth-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: var(--card-bg, #f9f9f9);
      border-radius: 6px;
      font-size: 13px;
    }

    .auth-status {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .auth-status.logged-in { background: #4caf50; }
    .auth-status.logged-out { background: #f44336; }

    .pbpair-section {
      margin-top: 16px;
    }

    .pbpair-requests {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .pbpair-request {
      padding: 12px;
      background: var(--card-bg, #f9f9f9);
      border-radius: 8px;
      border-left: 4px solid var(--primary-color, #1976d2);
    }

    .pbpair-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }

    .pbpair-provider {
      font-weight: 500;
      font-size: 14px;
    }

    .pbpair-status {
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 4px;
    }

    .pbpair-status.pending { background: #fff3e0; color: #e65100; }
    .pbpair-status.proposed { background: #e3f2fd; color: #1565c0; }
    .pbpair-status.completed { background: #e8f5e9; color: #2e7d32; }

    .pbpair-prompt {
      font-size: 13px;
      color: var(--text-color, #555);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .actions {
      display: flex;
      gap: 8px;
      margin-top: 16px;
    }

    .action-btn {
      padding: 10px 20px;
      border: none;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }

    .action-btn.primary {
      background: var(--primary-color, #1976d2);
      color: white;
    }

    .action-btn.primary:hover {
      background: var(--primary-hover, #1565c0);
    }

    .action-btn.secondary {
      background: var(--secondary-bg, #f0f0f0);
      color: var(--text-color, #333);
    }

    .action-btn.secondary:hover {
      background: var(--secondary-hover, #e0e0e0);
    }
  `;

  private orderedProviderIds(): string[] {
    return [
      'chatgpt-web',
      'claude-web',
      'gemini-web',
    ];
  }

  private providerLabel(id: string): string {
    const labels: Record<string, string> = {
      'chatgpt-web': 'ChatGPT Web',
      'claude-web': 'Claude Web',
      'gemini-web': 'Gemini Web',
    };
    return labels[id] || id;
  }

  private visibleProviderIds(): string[] {
    const providers = this.session?.providers || {};
    const keys = Object.keys(providers);
    const hasNonMock = keys.length > 0;
    return this.orderedProviderIds().filter((id) => {
      if (id === 'mock' && hasNonMock) return false;
      return id in providers;
    });
  }

  override render() {
    const providers = this.session?.providers || {};
    const providerIds = this.visibleProviderIds();
    const fallbackOrder = (this.session?.fallbackOrder || []).filter((p) => p !== 'mock');

    return html`
      <div class="panel">
        <div class="header">
          <h2>VPS Session</h2>
          <div class="flow-mode">
            <button
              class="mode-btn monitor ${this.session.flowMode === 'm' ? 'active' : ''}"
              @click=${() => this.setFlowMode('m')}
              title="Monitor/Approve mode - requires approval for tool chains"
            >
              M (Monitor)
            </button>
            <button
              class="mode-btn auto ${this.session.flowMode === 'A' ? 'active' : ''}"
              @click=${() => this.setFlowMode('A')}
              title="Automatic mode - runs tool chains end-to-end"
            >
              A (Auto)
            </button>
          </div>
        </div>

        <div class="section">
          <div class="section-title">Providers</div>
          <div class="providers">
            ${providerIds.map((id) => this.renderProvider(id, this.providerLabel(id), providers[id]))}
          </div>
        </div>

        ${this.chatTarget ? this.renderChatOverlay() : ''}

        <div class="section">
          <div class="section-title">Fallback Chain</div>
          <div class="fallback-chain">
            ${fallbackOrder.map((provider, i) => html`
              ${i > 0 ? html`<span class="fallback-arrow">→</span>` : ''}
              <span class="fallback-item ${provider === this.session.activeFallback ? 'active' : ''}">
                ${provider}
              </span>
            `)}
          </div>
        </div>

        <div class="section">
          <div class="section-title">Authentication</div>
          <div class="auth-section">
            <div class="auth-item">
              <span class="auth-status ${this.session.auth.geminiCliLoggedIn ? 'logged-in' : 'logged-out'}"></span>
              Gemini CLI
            </div>
            <div class="auth-item">
              <span class="auth-status ${this.session.auth.claudeLoggedIn ? 'logged-in' : 'logged-out'}"></span>
              Claude Code
            </div>
            ${this.session.auth.gcpProject ? html`
              <div class="auth-item">
                <span class="auth-status logged-in"></span>
                GCP: ${this.session.auth.gcpProject}
              </div>
            ` : ''}
          </div>
        </div>

        ${this.session.pbpair.activeRequests.length > 0 ? html`
          <div class="section pbpair-section">
            <div class="section-title">PBPAIR Requests</div>
            <div class="pbpair-requests">
              ${this.session.pbpair.activeRequests.map(req => html`
                <div class="pbpair-request">
                  <div class="pbpair-header">
                    <span class="pbpair-provider">${req.provider} (${req.role})</span>
                    <span class="pbpair-status ${req.status}">${req.status}</span>
                  </div>
                  <div class="pbpair-prompt">${req.prompt}</div>
                </div>
              `)}
            </div>
          </div>
        ` : ''}

        <div class="actions">
          <button class="action-btn primary" @click=${this.refreshProviders}>
            Refresh Providers
          </button>
          <button class="action-btn secondary" @click=${this.openPBPAIR}>
            New PBPAIR
          </button>
        </div>
      </div>
    `;
  }

  private renderProvider(id: string, name: string, status: ProviderStatus) {
    const isActive = this.session.activeFallback === id;
    const statusClass = status.available ? 'available' : 'unavailable';

    return html`
      <div class="provider ${isActive ? 'active' : ''}">
        <span class="status-indicator ${statusClass}"></span>
        <div class="provider-info">
          <div class="provider-name">${name}</div>
          <div class="provider-status">
            ${status.available ? 'Available' : status.error || 'Unavailable'}
          </div>
        </div>
        <button class="chat-btn" @click=${() => this.openChat(id)} title="Micro-Chat with ${name}">
          CHAT
        </button>
      </div>
    `;
  }

  private renderChatOverlay() {
    if (!this.chatTarget) return null;
    const history = this.chatHistory[this.chatTarget] || [];

    return html`
      <div class="micro-chat-overlay" @click=${this.closeChat}>
        <div class="micro-chat-box" @click=${(e: Event) => e.stopPropagation()}>
          <div class="chat-header">
            <span>Talking to: ${this.chatTarget.toUpperCase()}</span>
            <span class="close-chat" @click=${this.closeChat}>×</span>
          </div>
          <div class="chat-history">
            ${history.length === 0 ? html`<div style="color: #666; text-align: center; margin-top: 20px;">No history. Start typing...</div>` : ''}
            ${history.map(msg => html`
              <div class="msg ${msg.role}">${msg.content}</div>
            `)}
          </div>
          <div class="chat-input-area">
            <input 
              class="chat-input" 
              type="text" 
              .value=${this.chatInput}
              @input=${(e: any) => this.chatInput = e.target.value}
              @keydown=${this.handleChatKey}
              placeholder="Message ${this.chatTarget}..."
              autofocus
            />
          </div>
        </div>
      </div>
    `;
  }

  private openChat(providerId: string) {
    this.chatTarget = providerId;
  }

  private closeChat() {
    this.chatTarget = null;
  }

  private handleChatKey(e: KeyboardEvent) {
    if (e.key === 'Enter' && this.chatInput.trim()) {
      this.sendChat();
    }
  }

  private sendChat() {
    if (!this.chatTarget || !this.chatInput.trim()) return;

    const provider = this.chatTarget;
    const text = this.chatInput;
    
    // Optimistic UI update
    if (!this.chatHistory[provider]) this.chatHistory[provider] = [];
    this.chatHistory[provider] = [...this.chatHistory[provider], { role: 'user', content: text }];
    this.chatInput = '';
    this.requestUpdate();

    // Emit event for parent to handle (Gymnist bridge)
    this.dispatchEvent(new CustomEvent('gymnist-chat', {
      detail: { provider, text },
      bubbles: true,
      composed: true,
    }));
  }

  private setFlowMode(mode: FlowMode) {
    this.dispatchEvent(new CustomEvent('flow-mode-change', {
      detail: { mode },
      bubbles: true,
      composed: true,
    }));
  }

  private refreshProviders() {
    this.dispatchEvent(new CustomEvent('refresh-providers', {
      bubbles: true,
      composed: true,
    }));
  }

  private openPBPAIR() {
    this.dispatchEvent(new CustomEvent('open-pbpair', {
      bubbles: true,
      composed: true,
    }));
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'pluribus-vps-session': VPSSessionPanel;
  }
}
