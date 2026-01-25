import { component$, useStore, useVisibleTask$, $, type QRL } from '@builder.io/qwik';
import type { BrowserDaemonStatus, TabStatus } from './VNCAuthPanel';

export interface VNCAuthFabProps {
  providerStatus?: Record<string, { available: boolean; error?: string }>;
  onOpen$?: QRL<() => void>;
  /** When true, renders inline for header placement instead of fixed position */
  inline?: boolean;
}

const ACTIONABLE: TabStatus[] = ['needs_login', 'needs_code', 'needs_onboarding', 'blocked_bot'];

/**
 * Animated floating cloud button for browser auth.
 *
 * Features:
 * - Pulsing glow effect when issues detected
 * - Smooth hover scale transitions
 * - Drop shadow with color-coded states
 * - Inline mode for header placement
 *
 * Inspired by:
 * - https://codepen.io/seyedi/pen/YXEqwB (circular menu)
 * - https://www.florin-pop.com/blog/2019/03/css-pulse-effect/
 */
export const VNCAuthFab = component$<VNCAuthFabProps>(({ providerStatus = {}, onOpen$, inline = false }) => {
  const state = useStore<{
    issueCount: number;
    vncMode: boolean;
    running: boolean;
    lastRefresh: string | null;
    geminiCleanOk: boolean | null;
    geminiCleanLastRefresh: string | null;
    hovered: boolean;
  }>({
    issueCount: 0,
    vncMode: false,
    running: false,
    lastRefresh: null,
    geminiCleanOk: null,
    geminiCleanLastRefresh: null,
    hovered: false,
  });

  const noop = $(() => {});
  const handleOpen = onOpen$ ?? noop;

  const fetchStatus = $(async () => {
    try {
      const res = await fetch('/api/browser/status');
      if (res.ok) {
        const data = (await res.json()) as BrowserDaemonStatus;
        state.running = !!data.running;
        state.vncMode = !!data.vnc_mode;
        state.issueCount = Object
          .values(data.tabs || {})
          .filter((t) => ACTIONABLE.includes((t.status as TabStatus) || 'unknown'))
          .length;
        state.lastRefresh = new Date().toISOString();
        return;
      }
    } catch {
      // fall through to inference below
    }

    // Fallback: infer from VPS session providerStatus error strings.
    const inferredTabs: Record<string, { status: TabStatus }> = {};
    for (const providerId of ['chatgpt-web', 'claude-web', 'gemini-web']) {
      const ps = providerStatus[providerId];
      if (!ps) continue;
      const error = (ps.error || '').toLowerCase();
      let status: TabStatus = ps.available ? 'ready' : 'unknown';
      if (error.includes('2fa') || error.includes('otp') || error.includes('verification code') || error.includes('needs_code')) status = 'needs_code';
      else if (error.includes('login') || error.includes('auth')) status = 'needs_login';
      else if (error.includes('bot') || error.includes('challenge')) status = 'blocked_bot';
      else if (error.includes('onboarding') || error.includes('welcome')) status = 'needs_onboarding';
      else if (!ps.available && ps.error) status = 'error';
      inferredTabs[providerId] = { status };
    }
    state.running = false;
    state.vncMode = false;
    state.issueCount = Object
      .values(inferredTabs || {})
      .filter((t) => ACTIONABLE.includes((t.status as TabStatus) || 'unknown'))
      .length;
    state.lastRefresh = new Date().toISOString();
  });

  const refreshGeminiClean = $(async () => {
    try {
      const res = await fetch('/api/browser/gemini_clean/status');
      if (res.ok) {
        const data = (await res.json()) as { ok?: boolean };
        state.geminiCleanOk = typeof data?.ok === 'boolean' ? data.ok : null;
        state.geminiCleanLastRefresh = new Date().toISOString();
      }
    } catch {
      state.geminiCleanOk = null;
    }
  });

  useVisibleTask$(({ cleanup }) => {
    if (__E2E__) return;
    
    // Defer initial fetch to avoid blocking critical paint
    const timer = setTimeout(() => {
        performance.mark('vnc-auth:fetch:start');
        fetchStatus().then(() => {
            performance.mark('vnc-auth:fetch:end');
            performance.measure('vnc-auth:initial-fetch', 'vnc-auth:fetch:start', 'vnc-auth:fetch:end');
        });
        refreshGeminiClean();
    }, 1500);

    const interval = setInterval(fetchStatus, 30_000);
    const intervalGemini = setInterval(refreshGeminiClean, 5 * 60_000);

    // Subscribe to Omega for real-time updates (SOTA Optimization)
    const channel = new BroadcastChannel('pluribus-omega');
    channel.onmessage = (ev) => {
        if (ev.data.type === 'BUS_EVENT' && ev.data.event?.topic === 'browser.daemon.status') {
             const data = ev.data.event.data || {};
             state.running = !!data.running;
             state.vncMode = !!data.vnc_mode;
            const tabs = (data.tabs || {}) as Record<string, { status?: TabStatus }>;
            state.issueCount = Object.values(tabs).filter((tab) => tab.status && ACTIONABLE.includes(tab.status)).length;
             state.lastRefresh = new Date().toISOString();
        }
    };

    cleanup(() => {
        clearTimeout(timer);
        clearInterval(interval);
        clearInterval(intervalGemini);
        channel.close();
    });
  });

  const hasIssues = state.issueCount > 0;

  // Dynamic CSS classes for different states
  const basePosition = inline
    ? 'relative'
    : 'fixed bottom-5 right-5 z-50';

  // Color schemes based on state
  const colorScheme = hasIssues
    ? {
        bg: 'bg-gradient-to-br from-amber-500/30 to-orange-600/20',
        border: 'border-amber-400/60',
        text: 'text-amber-100',
        shadow: '0 4px 20px rgba(245, 158, 11, 0.4), 0 0 40px rgba(245, 158, 11, 0.2)',
        shadowHover: '0 6px 30px rgba(245, 158, 11, 0.6), 0 0 60px rgba(245, 158, 11, 0.3)',
        glow: 'rgba(245, 158, 11, 0.7)',
      }
    : {
        bg: 'bg-gradient-to-br from-cyan-500/20 to-blue-600/10',
        border: 'border-cyan-400/40',
        text: 'text-cyan-200',
        shadow: '0 4px 15px rgba(34, 211, 238, 0.2), 0 0 30px rgba(34, 211, 238, 0.1)',
        shadowHover: '0 6px 25px rgba(34, 211, 238, 0.4), 0 0 50px rgba(34, 211, 238, 0.2)',
        glow: 'rgba(34, 211, 238, 0.6)',
      };

  return (
    <>
      {/* Keyframe animations injected via style tag */}
      <style dangerouslySetInnerHTML={`
        @keyframes fab-pulse-glow {
          0% {
            transform: scale(1);
            box-shadow: ${colorScheme.shadow}, 0 0 0 0 ${colorScheme.glow};
          }
          50% {
            transform: scale(1.02);
            box-shadow: ${colorScheme.shadowHover}, 0 0 0 8px rgba(245, 158, 11, 0);
          }
          100% {
            transform: scale(1);
            box-shadow: ${colorScheme.shadow}, 0 0 0 0 ${colorScheme.glow};
          }
        }
        @keyframes fab-float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-3px); }
        }
        @keyframes fab-shine {
          0% { background-position: -200% center; }
          100% { background-position: 200% center; }
        }
        @keyframes cloud-bounce {
          0%, 100% { transform: scale(1) rotate(0deg); }
          25% { transform: scale(1.1) rotate(-5deg); }
          75% { transform: scale(1.1) rotate(5deg); }
        }
        .fab-cloud-auth {
          transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
          will-change: transform, box-shadow;
        }
        .fab-cloud-auth:hover {
          transform: scale(1.08) translateY(-2px);
        }
        .fab-cloud-auth:active {
          transform: scale(0.95);
        }
        .fab-cloud-auth.has-issues {
          animation: fab-pulse-glow 2s ease-in-out infinite;
        }
        .fab-cloud-auth:not(.has-issues) {
          animation: fab-float 3s ease-in-out infinite;
        }
        .fab-cloud-icon {
          display: inline-block;
          transition: transform 0.3s ease;
          filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }
        .fab-cloud-auth:hover .fab-cloud-icon {
          animation: cloud-bounce 0.6s ease-in-out;
        }
        .fab-badge {
          background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.05));
          backdrop-filter: blur(4px);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.2), 0 2px 4px rgba(0,0,0,0.2);
        }
        .fab-shine-effect {
          background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.1),
            transparent
          );
          background-size: 200% 100%;
        }
        .fab-cloud-auth:hover .fab-shine-effect {
          animation: fab-shine 0.8s ease;
        }
      `} />

      <button
        type="button"
        data-testid="auth-fab"
        title={hasIssues ? `${state.issueCount} provider(s) need auth` : 'Browser Auth'}
        onClick$={handleOpen}
        onMouseEnter$={() => { state.hovered = true; }}
        onMouseLeave$={() => { state.hovered = false; }}
        class={[
          basePosition,
          'fab-cloud-auth',
          hasIssues ? 'has-issues' : '',
          'rounded-full border-2 backdrop-blur-md',
          'px-4 py-2.5',
          'flex items-center gap-2.5 text-sm font-medium',
          'cursor-pointer select-none',
          colorScheme.bg,
          colorScheme.border,
          colorScheme.text,
        ].join(' ')}
        style={{
          boxShadow: state.hovered ? colorScheme.shadowHover : colorScheme.shadow,
        }}
      >
        {/* Shine overlay effect */}
        <div class="fab-shine-effect absolute inset-0 rounded-full pointer-events-none" />

        {/* Cloud icon with enhanced styling */}
        <span class="fab-cloud-icon text-lg relative">
          ☁️
          {hasIssues && (
            <span class="absolute -top-1 -right-1 w-2 h-2 bg-amber-400 rounded-full animate-ping" />
          )}
        </span>

        {/* Content based on state */}
        {hasIssues ? (
          <span class="fab-badge px-2 py-0.5 rounded-full text-xs font-bold">
            {state.issueCount}
          </span>
        ) : (
          <span class="hidden sm:inline font-medium tracking-wide">Auth</span>
        )}

        {/* Status indicators */}
        <div class="flex items-center gap-1">
          {state.geminiCleanOk === false && (
            <span class="fab-badge px-1.5 py-0.5 rounded text-[10px] text-amber-200">GC</span>
          )}
          {state.vncMode && (
            <span class="fab-badge px-1.5 py-0.5 rounded text-[10px] opacity-90">VNC</span>
          )}
          {state.running && (
            <span class="w-1.5 h-1.5 bg-green-400 rounded-full shadow-lg shadow-green-400/50" />
          )}
        </div>
      </button>
    </>
  );
});

export default VNCAuthFab;
