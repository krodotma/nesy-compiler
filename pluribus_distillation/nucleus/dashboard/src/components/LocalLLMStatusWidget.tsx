import { component$, useStore, useVisibleTask$, $ } from '@builder.io/qwik';
import { Button } from './ui/Button';
import { Card } from './ui/Card';

// M3 Components - LocalLLMStatusWidget
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/circular-progress.js';

type ProviderId = 'vllm-local' | 'ollama-local';

interface ProviderStatus {
  available: boolean;
  model?: string;
  error?: string;
  lastCheck?: string;
}

interface State {
  loading: boolean;
  error: string | null;
  sessionProviders: Record<string, ProviderStatus>;
  models: string[];
  warmedProvider: ProviderId | null;
  warmedAtIso: string | null;
  warmError: string | null;
}

interface WarmCandidate {
  provider: ProviderId;
  model: string;
}

function nowIso(): string {
  return new Date().toISOString();
}

function resolveWarmModel(
  provider: ProviderId,
  models: string[],
  providers: Record<string, ProviderStatus>
): string | null {
  const providerModel = providers[provider]?.model?.trim();
  if (providerModel) return providerModel;
  return models.includes(provider) ? provider : null;
}

function pickWarmCandidates(models: string[], providers: Record<string, ProviderStatus>): WarmCandidate[] {
  const candidates: ProviderId[] = ['vllm-local', 'ollama-local'];
  return candidates
    .filter((id) => providers[id]?.available)
    .map((provider) => ({
      provider,
      model: resolveWarmModel(provider, models, providers),
    }))
    .filter((candidate): candidate is WarmCandidate => Boolean(candidate.model));
}

export const LocalLLMStatusWidget = component$(() => {
  const state = useStore<State>({
    loading: false,
    error: null,
    sessionProviders: {},
    models: [],
    warmedProvider: null,
    warmedAtIso: null,
    warmError: null,
  });

  const refresh = $(async () => {
    state.loading = true;
    state.error = null;
    try {
      const [sessionRes, modelsRes] = await Promise.all([
        fetch('/api/session', { cache: 'no-store' }),
        fetch('/v1/models', { cache: 'no-store' }),
      ]);

      if (sessionRes.ok) {
        const s = await sessionRes.json().catch(() => ({}));
        const providers = (s && typeof s === 'object' ? (s as any).providers : null) || {};
        state.sessionProviders = providers;
      }

      if (!modelsRes.ok) throw new Error(`models endpoint returned ${modelsRes.status}`);
      const m = await modelsRes.json().catch(() => ({}));
      if ((m as any)?.ok === false) {
        throw new Error(String((m as any)?.error || 'models backend unreachable'));
      }
      const data = Array.isArray((m as any)?.data) ? (m as any).data : [];
      state.models = data
        .map((x: any) => String(x?.id || '').trim())
        .filter((x: string) => x.length > 0);
    } catch (err) {
      state.error = err instanceof Error ? err.message : String(err);
    } finally {
      state.loading = false;
    }
  });

  const warmOnce = $(async () => {
    state.warmError = null;
    state.warmedProvider = null;
    state.warmedAtIso = null;

    const availableProviders = Object.entries(state.sessionProviders)
      .filter(([, status]) => status?.available)
      .map(([id]) => id);
    const candidates = pickWarmCandidates(state.models, state.sessionProviders);
    if (candidates.length === 0) {
      if (availableProviders.length === 0) {
        return;
      }
      state.warmError = `No warmable model found for ${availableProviders.join(', ')} (missing model name).`;
      return;
    }

    for (const candidate of candidates) {
      try {
        const timeout = new Promise<null>((resolve) => {
          setTimeout(() => resolve(null), 2500);
        });
        const res = await Promise.race([
          fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: candidate.model,
              messages: [{ role: 'user', content: 'ping' }],
              temperature: 0,
              max_tokens: 1,
              stream: false,
            }),
          }),
          timeout,
        ]);
        if (!res) {
          throw new Error(`${candidate.provider} warm timed out for ${candidate.model}`);
        }
        const bodyText = await res.text().catch(() => '');
        if (!res.ok) {
          throw new Error(`${candidate.provider} warm failed for ${candidate.model} (${res.status}): ${bodyText.slice(0, 200)}`);
        }
        let parsed: any = null;
        try {
          parsed = JSON.parse(bodyText);
        } catch {
          parsed = null;
        }
        if (parsed && typeof parsed === 'object' && (parsed as any).ok === false) {
          throw new Error(String((parsed as any).error || 'backend_unreachable'));
        }
        state.warmedProvider = candidate.provider;
        state.warmedAtIso = nowIso();
        return;
      } catch (err) {
        state.warmError = err instanceof Error ? err.message : String(err);
      }
    }
  });

  useVisibleTask$(() => {
    void refresh().then(() => warmOnce());
  }, { strategy: 'document-idle' });

  const vllm = state.sessionProviders['vllm-local'];
  const ollama = state.sessionProviders['ollama-local'];

  return (
    <Card>
      <div class="flex items-center gap-2 mb-2">
        <div class="text-sm font-semibold text-muted-foreground">LOCAL LLM (SERVER)</div>
        {state.loading && <div class="text-[10px] text-muted-foreground">checking…</div>}
        <div class="ml-auto flex items-center gap-2">
          {state.warmedProvider && (
            <span class="text-[10px] px-2 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/30">
              warmed:{state.warmedProvider}
            </span>
          )}
          <Button
            variant="text"
            class="h-6 text-[10px]"
            onClick$={() => refresh().then(() => warmOnce())}
          >
            refresh
          </Button>
        </div>
      </div>

      {state.error && (
        <div class="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2 mb-2">
          {state.error}
        </div>
      )}

      {state.warmError && (
        <div class="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded p-2 mb-2">
          {state.warmError}
        </div>
      )}

      <div class="grid grid-cols-2 gap-2 text-xs">
        <Card
          variant={vllm?.available ? 'filled' : 'outlined'}
          padding="p-2"
          interactive={true}
          class={`glass-surface ${vllm?.available ? 'glass-status-ok' : ''}`}
        >
          <div class="flex items-center gap-2">
            <span class={`w-2 h-2 rounded-full ${vllm?.available ? 'bg-green-400' : 'bg-gray-400'}`} />
            <span class="mono">vllm-local</span>
          </div>
          <div class="text-[10px] text-muted-foreground mt-1">
            {vllm?.available ? `model: ${vllm.model || 'unknown'}` : (vllm?.error || 'not available')}
          </div>
        </Card>

        <Card
          variant={ollama?.available ? 'filled' : 'outlined'}
          padding="p-2"
          interactive={true}
          class={`glass-surface ${ollama?.available ? 'glass-status-ok' : ''}`}
        >
          <div class="flex items-center gap-2">
            <span class={`w-2 h-2 rounded-full ${ollama?.available ? 'bg-green-400' : 'bg-gray-400'}`} />
            <span class="mono">ollama-local</span>
          </div>
          <div class="text-[10px] text-muted-foreground mt-1">
            {ollama?.available ? `model: ${ollama.model || 'unknown'}` : (ollama?.error || 'not available')}
          </div>
        </Card>
      </div>

      <div class="text-[10px] text-muted-foreground mt-2">
        Uses `/v1/models` + `/api/session`, warms via `/v1/chat/completions` with explicit `vllm-local`/`ollama-local` (no cloud fallback).
      </div>
    </Card>
  );
});

export default LocalLLMStatusWidget;