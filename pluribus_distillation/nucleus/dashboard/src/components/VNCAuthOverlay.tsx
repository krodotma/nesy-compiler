import { component$, useStore, useVisibleTask$, $, type QRL } from '@builder.io/qwik';
import { VNCAuthPanel } from './VNCAuthPanel';

// M3 Components - VNCAuthOverlay
import '@material/web/elevation/elevation.js';
import '@material/web/dialog/dialog.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';

export interface VNCAuthOverlayProps {
  open: boolean;
  providerStatus?: Record<string, { available: boolean; error?: string }>;
  onClose$?: QRL<() => void>;
}

export const VNCAuthOverlay = component$<VNCAuthOverlayProps>(({ open, providerStatus = {}, onClose$ }) => {
  const state = useStore<{
    bootstrapping: boolean;
    bootstrapError: string | null;
    bootstrapResult: any | null;
    lastBootstrap: string | null;
    geminiCleanStatus: { ok: boolean; stdout?: string; stderr?: string; exit_code?: number } | null;
    geminiCleanLoading: boolean;
  }>({
    bootstrapping: false,
    bootstrapError: null,
    bootstrapResult: null,
    lastBootstrap: null,
    geminiCleanStatus: null,
    geminiCleanLoading: false,
  });

  const handleClose = $(() => {
    onClose$?.();
  });

  const emitEvent = $(async (topic: string, data: any) => {
    try {
      await fetch('/api/emit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, kind: 'metric', level: 'info', actor: 'dashboard', data }),
      });
    } catch {
      // ignore bus bridge failures
    }
  });

  const bootstrap = $(async () => {
    state.bootstrapping = true;
    state.bootstrapError = null;
    try {
      await emitEvent('dashboard.browser.bootstrap.requested', { source: 'auth-overlay', at: new Date().toISOString() });
      const res = await fetch('/api/browser/bootstrap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor: 'dashboard', wait: false }),
      });
      const data = await res.json().catch(() => ({}));
      state.bootstrapResult = data;
      state.lastBootstrap = new Date().toISOString();
      await emitEvent('dashboard.browser.bootstrap.result', { ok: !!data?.success, started: !!data?.started, queued: (data?.queued || []).length });
    } catch (e) {
      state.bootstrapError = String(e);
      await emitEvent('dashboard.browser.bootstrap.result', { ok: false, error: String(e) });
    } finally {
      state.bootstrapping = false;
    }
  });

  const refreshGeminiCleanStatus = $(async () => {
    state.geminiCleanLoading = true;
    try {
      const res = await fetch('/api/browser/gemini_clean/status');
      const data = await res.json().catch(() => ({}));
      state.geminiCleanStatus = data;
    } catch (e) {
      state.geminiCleanStatus = { ok: false, stderr: String(e) };
    } finally {
      state.geminiCleanLoading = false;
    }
  });

  useVisibleTask$(({ track }) => {
    track(() => open);
    if (!open) return;
    emitEvent('dashboard.auth.overlay.opened', { at: new Date().toISOString() });
    bootstrap();
    refreshGeminiCleanStatus();
  });

  if (!open) return null;

  const inferredIssues = Object.values(providerStatus || {}).filter((p) => !p.available && (p.error || '').length > 0).length;

  return (
    <div class="fixed inset-0 z-[60]" data-testid="auth-overlay">
      <div class="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick$={handleClose} />
      <div class="absolute inset-4 md:inset-10 rounded-xl border border-border bg-card shadow-2xl overflow-hidden flex flex-col">
        <div class="p-3 border-b border-border flex items-center gap-3">
          <div class="flex items-center gap-2">
            <span class="text-lg">☁️</span>
            <div>
              <div class="text-sm font-semibold">Auth / Providers</div>
              <div class="text-[11px] text-muted-foreground">
                Autonomy: env creds + VNC only for OTP/CAPTCHA
              </div>
            </div>
          </div>
          <div class="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick$={bootstrap}
              disabled={state.bootstrapping}
              class="text-xs px-3 py-1.5 rounded border border-border bg-muted/30 hover:bg-muted/50 disabled:opacity-50"
              title="Re-run bootstrap (start daemon + enqueue login checks)"
            >
              {state.bootstrapping ? 'Bootstrapping…' : 'Bootstrap'}
            </button>
            <button
              type="button"
              onClick$={handleClose}
              class="text-xs px-3 py-1.5 rounded bg-primary text-primary-foreground hover:bg-primary/90"
            >
              Close
            </button>
          </div>
        </div>

        {(state.bootstrapError || state.lastBootstrap) && (
          <div class="px-3 py-2 border-b border-border bg-muted/20 text-[11px] text-muted-foreground flex items-center gap-3">
            {state.lastBootstrap && <span>Last bootstrap: {state.lastBootstrap}</span>}
            {state.bootstrapError && <span class="text-red-300">Error: {state.bootstrapError}</span>}
            {inferredIssues > 0 && <span class="ml-auto">Provider issues detected: {inferredIssues}</span>}
          </div>
        )}

        <div class="flex-1 overflow-auto">
          <VNCAuthPanel providerStatus={providerStatus} fullScreen={true} />
        </div>

        <div class="p-3 border-t border-border bg-muted/10 text-[11px] text-muted-foreground space-y-2">
          <div>
            Tip: if you see <span class="text-amber-200">2FA Code</span>, enter the code via VNC and then click “Check Login”.
          </div>
          <details class="rounded border border-border bg-muted/20 p-2">
            <summary class="cursor-pointer select-none flex items-center justify-between gap-2">
              <span>gemini_clean status (server-side)</span>
              <span class="text-[10px] opacity-80">{state.geminiCleanLoading ? 'loading…' : 'refreshable'}</span>
            </summary>
            <div class="mt-2 flex items-center gap-2">
              <button
                type="button"
                onClick$={refreshGeminiCleanStatus}
                disabled={state.geminiCleanLoading}
                class="text-[10px] px-2 py-1 rounded border border-border bg-muted/30 hover:bg-muted/50 disabled:opacity-50"
              >
                Refresh
              </button>
              {state.geminiCleanStatus && (
                <span class={`text-[10px] ${state.geminiCleanStatus.ok ? 'text-green-300' : 'text-amber-200'}`}>
                  {state.geminiCleanStatus.ok ? 'ok' : 'needs attention'}
                </span>
              )}
            </div>
            <pre class="mt-2 text-[10px] whitespace-pre-wrap text-muted-foreground max-h-56 overflow-auto">
              {(state.geminiCleanStatus?.stdout || state.geminiCleanStatus?.stderr || '').trim() || 'no output'}
            </pre>
          </details>
        </div>
      </div>
    </div>
  );
});

export default VNCAuthOverlay;
