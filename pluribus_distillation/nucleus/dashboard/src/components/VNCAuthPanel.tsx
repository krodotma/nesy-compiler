/**
 * VNCAuthPanel.tsx - Browser Authentication Interface
 *
 * Displays VNC connection status, browser tab health, and provides
 * instructions for manual OAuth login via VNC when browser sessions
 * need authentication.
 *
 * Features:
 * - Real-time browser tab status display (ready, needs_login, blocked_bot, etc.)
 * - VNC connection information for manual authentication
 * - Optional embedded noVNC viewer (when available)
 * - Provider-specific auth status and instructions
 */

import { component$, useSignal, useStore, useVisibleTask$, $ } from '@builder.io/qwik';
import '@material/web/switch/switch.js';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { ProviderPulse } from './ProviderPulse';
import { NeonTitle, NeonSectionHeader } from './ui/NeonTitle';

// ============================================================================
// Types
// ============================================================================

export type TabStatus =
  | 'initializing'
  | 'ready'
  | 'busy'
  | 'error'
  | 'closed'
  | 'needs_login'
  | 'needs_code'
  | 'needs_onboarding'
  | 'blocked_bot'
  | 'expired'
  | 'unknown';

export interface BrowserTab {
  provider_id: string;
  tab_id: string;
  url: string;
  status: TabStatus;
  current_url?: string;
  title?: string;
  error?: string;
  last_health_check?: string;
  last_activity?: string;
  session_start?: string;
  chat_count?: number;
}

export interface VNCConnectionInfo {
  display: string | null;
  vnc_port: number;
  hostname: string;
  connection_string: string;
  instructions: string;
}

export interface BrowserDaemonStatus {
  running: boolean;
  pid: number | null;
  browser_pid?: number | null;
  started_at: string | null;
  tabs: Record<string, BrowserTab>;
  vnc_mode: boolean;
  vnc_info?: VNCConnectionInfo;
}

// ============================================================================
// Constants
// ============================================================================

const STATUS_CONFIG: Record<TabStatus, { color: string; icon: string; label: string; urgency: 'low' | 'medium' | 'high' }> = {
  ready: { color: 'green', icon: '✓', label: 'Ready', urgency: 'low' },
  busy: { color: 'yellow', icon: '⏳', label: 'Busy', urgency: 'low' },
  initializing: { color: 'blue', icon: '⚙️', label: 'Starting', urgency: 'low' },
  needs_login: { color: 'amber', icon: '🔐', label: 'Auth Required', urgency: 'high' },
  needs_code: { color: 'amber', icon: '📲', label: '2FA Code', urgency: 'high' },
  needs_onboarding: { color: 'orange', icon: '🎯', label: 'Onboarding', urgency: 'medium' },
  blocked_bot: { color: 'red', icon: '🤖', label: 'Bot Blocked', urgency: 'high' },
  expired: { color: 'gray', icon: '⌛', label: 'Expired', urgency: 'medium' },
  error: { color: 'red', icon: '❌', label: 'Error', urgency: 'high' },
  closed: { color: 'gray', icon: '○', label: 'Closed', urgency: 'low' },
  unknown: { color: 'gray', icon: '?', label: 'Unknown', urgency: 'low' },
};

const PROVIDER_DISPLAY: Record<string, { name: string; icon: string; authUrl: string; instructions: string }> = {
  'chatgpt-web': {
    name: 'ChatGPT',
    icon: '💬',
    authUrl: 'https://chatgpt.com/',
    instructions: 'Sign in with your OpenAI account. Complete any Cloudflare challenges.',
  },
  'claude-web': {
    name: 'Claude',
    icon: '🟣',
    authUrl: 'https://claude.ai/',
    instructions: 'Sign in with your Anthropic account. May require email verification.',
  },
  'gemini-web': {
    name: 'Gemini',
    icon: '✨',
    authUrl: 'https://aistudio.google.com/',
    instructions: 'Sign in with Google. Complete AI Studio onboarding if prompted.',
  },
};

// ============================================================================
// Component
// ============================================================================

export interface VNCAuthPanelProps {
  /** External provider status from VPS session (optional for enrichment) */
  providerStatus?: Record<string, { available: boolean; error?: string }>;
  /** Compact mode for embedding in other views */
  compact?: boolean;
  /** Full screen mode for main view embedding */
  fullScreen?: boolean;
}

export const VNCAuthPanel = component$<VNCAuthPanelProps>(({
  providerStatus = {},
  compact = false,
  fullScreen = false,
}) => {
  const daemonStatus = useStore<BrowserDaemonStatus>({
    running: false,
    pid: null,
    browser_pid: null,
    started_at: null,
    tabs: {},
    vnc_mode: false,
  });

  const isLoading = useSignal(true);
  const lastRefresh = useSignal<string | null>(null);
  const showNoVNC = useSignal(false);
  const noVNCUrl = useSignal<string | null>(fullScreen ? '/vnc/vnc.html?resize=scale&path=vnc/websockify' : null);
  const autoRefresh = useSignal(true);
  const expandedProvider = useSignal<string | null>(null);
  type VNCAction = 'focus_tab' | 'navigate_login' | 'check_login';
  const vncActionBusy = useSignal<{ provider: string; action: VNCAction } | null>(null);
  const vncActionResults = useStore<Record<string, { action: VNCAction; at: string; data: any }>>({});

  const fetchStatus = $(async () => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    try {
      const res = await fetch('/api/browser/status', { signal: controller.signal });
      clearTimeout(timeoutId);
      if (res.ok) {
        const data = await res.json();
        daemonStatus.running = data.running ?? false;
        daemonStatus.pid = data.pid ?? null;
        daemonStatus.browser_pid = data.browser_pid ?? null;
        daemonStatus.started_at = data.started_at ?? null;
        daemonStatus.tabs = data.tabs ?? {};
        daemonStatus.vnc_mode = data.vnc_mode ?? false;
        daemonStatus.vnc_info = data.vnc_info;

        if (data.vnc_info?.vnc_port) {
          noVNCUrl.value = '/vnc/vnc.html?resize=scale&path=vnc/websockify';
        }
      }
    } catch (e) {
      clearTimeout(timeoutId);
      console.error('Failed to fetch browser daemon status:', e);
      const webProviders = ['chatgpt-web', 'claude-web', 'gemini-web'];
      webProviders.forEach(providerId => {
        const ps = providerStatus[providerId];
        if (ps) {
          const error = ps.error?.toLowerCase() || '';
          let status: TabStatus = ps.available ? 'ready' : 'unknown';
          if (error.includes('login') || error.includes('auth')) status = 'needs_login';
          else if (error.includes('2fa') || error.includes('otp') || error.includes('verification code') || error.includes('needs_code')) status = 'needs_code';
          else if (error.includes('bot') || error.includes('challenge')) status = 'blocked_bot';
          else if (error.includes('onboarding') || error.includes('welcome')) status = 'needs_onboarding';
          else if (!ps.available && ps.error) status = 'error';

          daemonStatus.tabs[providerId] = {
            provider_id: providerId,
            tab_id: 'inferred',
            url: PROVIDER_DISPLAY[providerId]?.authUrl || '',
            status,
            error: ps.error,
          };
        }
      });
    }
    if (fullScreen && !noVNCUrl.value) {
      noVNCUrl.value = '/vnc/vnc.html?resize=scale&path=vnc/websockify';
    }
    lastRefresh.value = new Date().toISOString();
    isLoading.value = false;
  });

  const runVncAction = $(async (providerId: string, action: VNCAction) => {
    const endpoint = action === 'focus_tab'
      ? '/api/browser/vnc/focus_tab'
      : action === 'navigate_login'
        ? '/api/browser/vnc/navigate_login'
        : '/api/browser/vnc/check_login';
    vncActionBusy.value = { provider: providerId, action };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: providerId,
          actor: 'dashboard',
          wait: true,
          timeout_s: 5,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const data = await res.json();
      vncActionResults[providerId] = { action, at: new Date().toISOString(), data };
    } catch (e) {
      clearTimeout(timeoutId);
      vncActionResults[providerId] = {
        action,
        at: new Date().toISOString(),
        data: { success: false, error: `Failed to call ${endpoint}: ${String(e)}` },
      };
    } finally {
      vncActionBusy.value = null;
      await fetchStatus();
    }
  });

  useVisibleTask$(({ cleanup }) => {
    fetchStatus();
    const interval = setInterval(() => {
      if (autoRefresh.value) {
        fetchStatus();
      }
    }, 30000);
    cleanup(() => clearInterval(interval));
  });

  const tabsNeedingAuth = Object.values(daemonStatus.tabs).filter(
    tab => ['needs_login', 'needs_code', 'needs_onboarding', 'blocked_bot'].includes(tab.status)
  );
  const hasAuthIssues = tabsNeedingAuth.length > 0;
  const allReady = Object.values(daemonStatus.tabs).every(tab => tab.status === 'ready');

  const getStatusColorClass = (status: TabStatus) => {
    const config = STATUS_CONFIG[status] || STATUS_CONFIG.unknown;
    return {
      'green': 'bg-green-500/20 text-green-400 border-green-500/30',
      'yellow': 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      'amber': 'bg-amber-500/20 text-amber-400 border-amber-500/30',
      'orange': 'bg-orange-500/20 text-orange-400 border-orange-500/30',
      'red': 'bg-red-500/20 text-red-400 border-red-500/30',
      'blue': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      'gray': 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    }[config.color] || 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  };

  const getDotClass = (status: TabStatus) => {
    const config = STATUS_CONFIG[status] || STATUS_CONFIG.unknown;
    const pulse = config.urgency === 'high' ? 'animate-pulse' : '';
    return {
      'green': `bg-green-400 ${pulse}`,
      'yellow': `bg-yellow-400 ${pulse}`,
      'amber': `bg-amber-400 ${pulse}`,
      'orange': `bg-orange-400 ${pulse}`,
      'red': `bg-red-400 ${pulse}`,
      'blue': `bg-blue-400 ${pulse}`,
      'gray': `bg-gray-400 ${pulse}`,
    }[config.color] || 'bg-gray-400';
  };

  if (compact) {
    return (
      <div class="flex items-center gap-2">
        {hasAuthIssues ? (
          <span class="text-[10px] px-2 py-1 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center gap-1">
            <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            {tabsNeedingAuth.length} Browser{tabsNeedingAuth.length > 1 ? 's' : ''} Need Auth
          </span>
        ) : allReady ? (
          <span class="text-[10px] px-2 py-1 rounded bg-green-500/20 text-green-400 border border-green-500/30 flex items-center gap-1">
            <span class="w-1.5 h-1.5 rounded-full bg-green-400" />
            Browsers Ready
          </span>
        ) : (
          <span class="text-[10px] px-2 py-1 rounded bg-gray-500/20 text-gray-400 border border-gray-500/30">
            Browser Status
          </span>
        )}
      </div>
    );
  }

  return (
    <Card class={`flex flex-col ${fullScreen ? 'h-full' : ''}`}>
      <div class="p-4 border-b border-border flex items-center justify-between flex-shrink-0">
        <div class="flex items-center gap-3">
          <span class="text-xl">🌐</span>
          <div>
            <h2 class="text-sm font-semibold">Browser Authentication</h2>
            <p class="text-[10px] text-muted-foreground">
              VNC access for manual OAuth login
            </p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <label class="flex items-center gap-2 cursor-pointer" title="Auto-refresh status every 30s">
            <span class="text-[10px] text-muted-foreground font-mono uppercase">Auto</span>
            <md-switch
              selected={autoRefresh.value}
              onClick$={() => autoRefresh.value = !autoRefresh.value}
              aria-label="Toggle auto-refresh"
              icons
            ></md-switch>
          </label>
          <Button
            variant="tonal"
            onClick$={fetchStatus}
            disabled={isLoading.value}
            class="h-8 text-[10px]"
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* Daemon Status Banner */}
      <div class={`px-4 py-2 text-xs flex items-center justify-between flex-shrink-0 ${
        daemonStatus.running
          ? 'bg-green-500/10 border-b border-green-500/20'
          : 'bg-red-500/10 border-b border-red-500/20'
      }`}>
        <div class="flex items-center gap-2">
          <span class={`w-2 h-2 rounded-full ${daemonStatus.running ? 'bg-green-400' : 'bg-red-400'}`} />
          <span class={daemonStatus.running ? 'text-green-400' : 'text-red-400'}>
            Browser Daemon: {daemonStatus.running ? 'Running' : 'Stopped'}
          </span>
          {(daemonStatus.pid || daemonStatus.browser_pid) && (
            <span class="text-muted-foreground font-mono">
              (
              {daemonStatus.pid ? `Daemon PID: ${daemonStatus.pid}` : ''}
              {daemonStatus.browser_pid ? `${daemonStatus.pid ? ', ' : ''}Browser PID: ${daemonStatus.browser_pid}` : ''}
              )
            </span>
          )}
        </div>
        {daemonStatus.vnc_mode && (
          <span class="px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30 text-[10px]">
            VNC Mode Active
          </span>
        )}
      </div>

      {/* VNC Connection Info */}
      {daemonStatus.vnc_info && !fullScreen && (
        <div class="px-4 py-3 bg-purple-500/5 border-b border-purple-500/20 flex-shrink-0">
          <div class="flex items-start justify-between">
            <div>
              <NeonTitle level="h3" color="purple" size="xs" class="mb-1">VNC Connection</NeonTitle>
              <div class="space-y-1">
                <div class="text-[11px] font-mono text-muted-foreground">
                  <span class="text-purple-400">Host:</span> {daemonStatus.vnc_info.connection_string}
                </div>
                <div class="text-[11px] font-mono text-muted-foreground">
                  <span class="text-purple-400">Display:</span> {daemonStatus.vnc_info.display || ':1'}
                </div>
                <div class="text-[10px] text-muted-foreground italic mt-1">
                  Password: see <code>/pluribus/.pluribus/vnc_password.txt</code> on the VPS (short PINs are OK).
                </div>
              </div>
            </div>
            {noVNCUrl.value && (
              <Button
                variant="secondary"
                onClick$={() => showNoVNC.value = !showNoVNC.value}
                class="h-6 text-[10px]"
              >
                {showNoVNC.value ? 'Hide noVNC' : 'Open noVNC'}
              </Button>
            )}
          </div>
        </div>
      )}

      {/* noVNC Embed */}
      {((fullScreen && noVNCUrl.value) || (showNoVNC.value && noVNCUrl.value)) && (
        <div class={`border-b border-border ${fullScreen ? 'flex-1 min-h-0' : ''}`}>
          {fullScreen && (
            <div class="px-4 py-2 text-[10px] bg-purple-500/5 border-b border-purple-500/20 flex items-center justify-between">
              <div class="text-muted-foreground">
                Click <span class="text-purple-400 font-semibold">Connect</span> in noVNC • Password: <code>/pluribus/.pluribus/vnc_password.txt</code>
              </div>
              <a href={noVNCUrl.value} target="_blank" rel="noreferrer" class="text-cyan-400 hover:underline">
                Open noVNC in new tab
              </a>
            </div>
          )}
          <iframe
            src={noVNCUrl.value}
            class={`w-full bg-black ${fullScreen ? 'h-full' : 'h-[400px]'}`}
            title="noVNC Browser Access"
          />
        </div>
      )}

      {/* Tab Status List */}
      <div class={`p-4 space-y-3 ${fullScreen ? 'h-64 overflow-auto border-t border-border flex-shrink-0' : ''}`}>
        <NeonSectionHeader title="BROWSER SESSIONS" color="cyan" size="xs" />

        {Object.entries(daemonStatus.tabs).length === 0 ? (
          <div class="text-sm text-muted-foreground text-center py-4">
            No browser tabs configured. Start the browser daemon to enable web provider sessions.
          </div>
        ) : (
          <div class="space-y-2">
            {Object.entries(daemonStatus.tabs).map(([providerId, tab]) => {
              const config = STATUS_CONFIG[tab.status] || STATUS_CONFIG.unknown;
              const providerInfo = PROVIDER_DISPLAY[providerId];
              const isExpanded = expandedProvider.value === providerId;

              return (
                <div key={providerId} class={`rounded-lg border transition-all ${getStatusColorClass(tab.status)}`}>
                  <div
                    class="p-3 cursor-pointer"
                    onClick$={() => expandedProvider.value = isExpanded ? null : providerId}
                  >
                    <div class="flex items-center justify-between">
                      <div class="flex items-center gap-2">
                        <span class={`w-2 h-2 rounded-full ${getDotClass(tab.status)}`} />
                        <span class="text-lg">{providerInfo?.icon || '🌐'}</span>
                        <span class="font-medium text-sm">{providerInfo?.name || providerId}</span>
                      </div>
                      <div class="flex items-center gap-2">
                        <span class={`text-[10px] px-2 py-0.5 rounded border ${getStatusColorClass(tab.status)}`}>
                          {config.icon} {config.label}
                        </span>
                        <span class="text-xs text-muted-foreground">{isExpanded ? '▲' : '▼'}</span>
                      </div>
                    </div>
                    <div class="mt-1 ml-6 mb-1 opacity-80">
                      <ProviderPulse available={['ready', 'busy'].includes(tab.status)} latency={120} />
                    </div>
                    {tab.error && !isExpanded && (
                      <div class="text-[10px] text-muted-foreground mt-1 ml-6 truncate">{tab.error}</div>
                    )}
                  </div>

                  {isExpanded && (
                    <div class="px-3 pb-3 border-t border-[var(--glass-border)] mt-0">
                      <div class="pt-3 space-y-2">
                        {tab.error && (
                          <div class="text-xs text-red-400/80 p-2 rounded bg-red-500/10 border border-red-500/20">
                            {tab.error}
                          </div>
                        )}
                        {['needs_login', 'needs_onboarding', 'blocked_bot'].includes(tab.status) && (
                          <div class="text-xs p-2 rounded bg-amber-500/10 border border-amber-500/20">
                            <div class="font-semibold text-amber-400 mb-1">Action Required:</div>
                            <ol class="list-decimal list-inside space-y-1 text-amber-400/80">
                              <li>Connect to VNC: {daemonStatus.vnc_info?.connection_string || 'Check server config'}</li>
                              <li>Open browser tab for {providerInfo?.name || providerId}</li>
                              <li>{providerInfo?.instructions || 'Complete the login/authentication flow'}</li>
                              <li>Click Refresh to update status</li>
                            </ol>
                          </div>
                        )}
	                        <div class="grid grid-cols-2 gap-2 text-[10px]">
	                          {tab.current_url && (
	                            <div>
	                              <span class="text-muted-foreground">URL:</span>
	                              <button
	                                onClick$={() => runVncAction(providerId, 'focus_tab')}
	                                disabled={!daemonStatus.running || (vncActionBusy.value?.provider === providerId)}
	                                class="text-cyan-400 hover:underline ml-1 truncate block text-left disabled:opacity-50"
	                              >
	                                {tab.current_url}
	                              </button>
	                            </div>
	                          )}
                          {tab.last_health_check && <div><span class="text-muted-foreground">Last Check:</span><span class="ml-1">{new Date(tab.last_health_check).toLocaleTimeString()}</span></div>}
                          {tab.session_start && <div><span class="text-muted-foreground">Session Start:</span><span class="ml-1">{new Date(tab.session_start).toLocaleTimeString()}</span></div>}
                          {tab.chat_count !== undefined && tab.chat_count > 0 && <div><span class="text-muted-foreground">Chats:</span><span class="ml-1">{tab.chat_count}</span></div>}
	                        </div>

	                        <div class="flex flex-wrap gap-2">
	                          <Button
                              variant="secondary"
	                            onClick$={() => runVncAction(providerId, 'focus_tab')}
	                            disabled={!daemonStatus.running || (vncActionBusy.value?.provider === providerId)}
                              class="h-6 text-[10px]"
	                          >
	                            Focus Tab
	                          </Button>
	                          <Button
                              variant="tonal"
	                            onClick$={() => runVncAction(providerId, 'navigate_login')}
	                            disabled={!daemonStatus.running || (vncActionBusy.value?.provider === providerId)}
                              class="h-6 text-[10px]"
	                          >
	                            Login via VNC
	                          </Button>
	                          <Button
                              variant="tonal"
	                            onClick$={() => runVncAction(providerId, 'check_login')}
	                            disabled={!daemonStatus.running || (vncActionBusy.value?.provider === providerId)}
                              class="h-6 text-[10px]"
	                          >
	                            Check Login
	                          </Button>
	                        </div>

	                        {vncActionResults[providerId] && (
	                          <div class={`text-[10px] p-2 rounded border ${
	                            vncActionResults[providerId].data?.success
	                              ? 'bg-green-500/10 border-green-500/20 text-green-400/80'
	                              : 'bg-red-500/10 border-red-500/20 text-red-400/80'
	                          }`}>
	                            <div class="font-semibold">
	                              Last VNC Action: {vncActionResults[providerId].action}
	                              <span class="text-muted-foreground font-normal ml-1">
	                                ({new Date(vncActionResults[providerId].at).toLocaleTimeString()})
	                              </span>
	                            </div>
	                            <div class="mt-1">
	                              {vncActionResults[providerId].data?.message || vncActionResults[providerId].data?.error || 'No message'}
	                            </div>
	                          </div>
	                        )}
	                      </div>
	                    </div>
	                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div class="px-4 py-2 border-t border-border bg-muted/20 flex items-center justify-between text-[10px] text-muted-foreground">
        <span>{lastRefresh.value && `Last updated: ${new Date(lastRefresh.value).toLocaleTimeString()}`}</span>
        <div class="flex items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-green-400" />{Object.values(daemonStatus.tabs).filter(t => t.status === 'ready').length} Ready</span>
          {tabsNeedingAuth.length > 0 && <span class="flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />{tabsNeedingAuth.length} Need Auth</span>}
        </div>
      </div>
    </Card>
  );
});

export const VNCAuthBadge = component$<{ providerStatus?: Record<string, { available: boolean; error?: string }> }>(({
  providerStatus = {},
}) => {
  return <VNCAuthPanel providerStatus={providerStatus} compact={true} />;
});

export default VNCAuthPanel;