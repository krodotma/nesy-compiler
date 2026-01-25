/**
 * SemopsEditor.tsx - Semantic Operators CRUD + Mapping Panel
 *
 * Goals:
 * - Show existing SemOps grouped by domain/category
 * - CRUD user-defined operators (persisted via git_server.py)
 * - Capture mappings: keywords/aliases → tool/bus/UI/agent/app bindings
 * - Provide dynamic suggestions (tools, recent bus actors/topics, SOTA seeds)
 */

import { component$, useSignal, useStore, useVisibleTask$, $ } from '@builder.io/qwik';
import { SemopsFlowWizard } from './SemopsFlowWizard';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Card } from './ui/Card';

// M3 Components - SemopsEditor
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';

export type SemopsBusKind = 'log' | 'request' | 'response' | 'artifact' | 'metric';

export type SemopsActionKind = 'primary' | 'secondary' | 'danger';

export type SemopsEffects = 'none' | 'file' | 'network' | 'system' | 'unknown';

export interface SemopsUIAction {
  id: string;
  label: string;
  kind?: SemopsActionKind;
  payload?: Record<string, unknown>;
}

export interface SemopsTarget {
  type: 'tool' | 'bus' | 'ui' | 'agent' | 'app' | string;
  ref?: string;
  kind?: SemopsBusKind | string;
  route?: string;
  component?: string;
  [k: string]: unknown;
}

export interface SemopsOperator {
  id: string;
  name: string;
  domain: string;
  category: string;
  description: string;
  aliases: string[];
  effects?: SemopsEffects | string;
  tool?: string | null;
  bus_topic?: string | null;
  bus_kind?: string | null;
  secondary_topic?: string | null;
  options?: Record<string, string>;
  invocation?: Record<string, unknown>;
  guarantees?: string[];
  targets?: SemopsTarget[];
  ui?: Record<string, unknown>;
  ui_actions?: SemopsUIAction[];
  flow_hints?: string[];
  agents?: string[];
  apps?: string[];
  user_defined?: boolean;
  [k: string]: unknown;
}

export interface SemopsSchemaResponse {
  operators: Record<string, SemopsOperator>;
  commands: string[];
  alias_map: Record<string, string>;
  tool_map?: Record<string, string>;
  bus_topics?: Record<string, unknown>;
  user_ops_path?: string;
  error?: string;
}

export interface SemopsSuggestionsResponse {
  tool_paths: string[];
  ui_components: string[];
  recent_actors: Array<[string, number]>;
  recent_topics: Array<[string, number]>;
  sota_candidates: string[];
  bus?: {
    active_dir?: string | null;
    primary_dir?: string | null;
    events_path?: string | null;
  };
}

const SEMOPS_CACHE_KEY = 'pluribus.semops.cache.v1';

const loadSemopsCache = () => {
  if (typeof localStorage === 'undefined') return null;
  try {
    const raw = localStorage.getItem(SEMOPS_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { schema?: SemopsSchemaResponse; suggestions?: SemopsSuggestionsResponse };
    if (!parsed?.schema || !parsed?.suggestions) return null;
    return parsed;
  } catch {
    return null;
  }
};

const saveSemopsCache = (schema: SemopsSchemaResponse, suggestions: SemopsSuggestionsResponse) => {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(SEMOPS_CACHE_KEY, JSON.stringify({ schema, suggestions }));
  } catch {
    // ignore cache write failures
  }
};

function parseCommaList(value: string): string[] {
  return value
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function deriveKeyFromToolPath(toolPath: string): string {
  const base = toolPath.split('/').pop() || toolPath;
  const stem = base.replace(/\.[^.]+$/, '');
  const normalized = stem.replace(/_operator$/, '').replace(/_server$/, '').replace(/_daemon$/, '');
  return normalized.toUpperCase();
}

export const SemopsEditor = component$(() => {
  const schema = useStore<SemopsSchemaResponse>({
    operators: {},
    commands: [],
    alias_map: {},
    tool_map: {},
    bus_topics: {},
  });

  const suggestions = useStore<SemopsSuggestionsResponse>({
    tool_paths: [],
    ui_components: [],
    recent_actors: [],
    recent_topics: [],
    sota_candidates: [],
    bus: {},
  });

  const isLoading = useSignal(true);
  const isSaving = useSignal(false);
  const error = useSignal<string | null>(null);
  const status = useSignal<string | null>(null);
  const filter = useSignal('');
  const showSotaSeeds = useSignal(false);
  const selectedKey = useSignal<string | null>(null);
  const invokeMode = useSignal('auto');
  const invokePayload = useSignal('{"input":"..."}');
  const invokeWait = useSignal(false);
  const invokeTimeoutS = useSignal(2.0);
  const invokeLatestResponse = useSignal<string | null>(null);
  const showFlowWizard = useSignal(false);

  const form = useStore({
    key: '',
    id: '',
    name: '',
    domain: 'user',
    category: 'custom',
    effects: 'none' as SemopsEffects,
    description: '',
    aliases: '',
    tool: '',
    bus_topic: '',
    bus_kind: '' as SemopsBusKind | '',
    secondary_topic: '',
    ui_route: '',
    ui_component: '',
    agents: '',
    apps: '',
  });

  const resetForm = $(() => {
    form.key = '';
    form.id = '';
    form.name = '';
    form.domain = 'user';
    form.category = 'custom';
    form.effects = 'none';
    form.description = '';
    form.aliases = '';
    form.tool = '';
    form.bus_topic = '';
    form.bus_kind = '';
    form.secondary_topic = '';
    form.ui_route = '';
    form.ui_component = '';
    form.agents = '';
    form.apps = '';
  });

  const applyTemplate = $((kind: 'tool' | 'policy' | 'evolution' | 'ui') => {
    if (kind === 'tool') {
      form.domain = 'execution';
      form.category = 'tool';
      if (form.effects === 'none') form.effects = 'file';
      if (!form.description) form.description = 'Tool-backed semantic operator.';
      return;
    }
    if (kind === 'policy') {
      form.domain = 'safety';
      form.category = 'policy';
      if (form.effects === 'none') form.effects = 'none';
      if (!form.description) form.description = 'Policy / safety semantic operator (non-blocking; emits bus requests).';
      return;
    }
    if (kind === 'evolution') {
      form.domain = 'evolution';
      form.category = 'git';
      if (form.effects === 'none') form.effects = 'file';
      if (!form.description) form.description = 'Evolutionary operator (bounded transforms; evidence-first).';
      return;
    }
    form.domain = 'ui';
    form.category = 'panel';
    if (form.effects === 'none') form.effects = 'none';
    if (!form.description) form.description = 'UI projection operator (bus-driven navigation/panels).';
  });

  const loadIntoForm = $((operatorKey: string, op: SemopsOperator, mode: 'edit' | 'clone') => {
    const baseKey = (operatorKey || op?.name || op?.id || '').trim().toUpperCase();
    if (mode === 'clone') {
      form.key = `${baseKey}_CUSTOM`;
      form.id = `${String(op.id || baseKey.toLowerCase())}_custom`;
      form.name = `${String(op.name || baseKey)} (CUSTOM)`;
      form.description = `Cloned from ${baseKey}. ${String(op.description || '').trim()}`.trim();
      const existingAliases = (op.aliases || []).filter(Boolean);
      form.aliases = existingAliases.length > 0 ? existingAliases.join(', ') : `${form.id}, ${form.key}`;
    } else {
      form.key = baseKey;
      form.id = String(op.id || baseKey.toLowerCase());
      form.name = String(op.name || baseKey);
      form.description = String(op.description || '');
      form.aliases = (op.aliases || []).join(', ');
    }

    form.domain = String(op.domain || 'user');
    form.category = String(op.category || 'custom');
    form.effects = (String((op as any).effects || 'none').trim().toLowerCase() as SemopsEffects) || 'none';
    form.tool = op.tool ? String(op.tool) : '';
    form.bus_topic = op.bus_topic ? String(op.bus_topic) : '';
    form.bus_kind = (op.bus_kind as any) || '';
    form.secondary_topic = op.secondary_topic ? String(op.secondary_topic) : '';

    const ui = (op.ui || {}) as any;
    form.ui_route = ui?.route ? String(ui.route) : '';
    form.ui_component = ui?.component ? String(ui.component) : '';

    form.agents = (op.agents || []).join(', ');
    form.apps = (op.apps || []).join(', ');

    status.value = mode === 'clone' ? `Cloned ${baseKey} into form (edit key/id then Save)` : `Loaded ${baseKey} for editing`;
  });

  const copyText = $(async (value: string) => {
    const txt = (value || '').trim();
    if (!txt) return;
    try {
      await navigator.clipboard.writeText(txt);
      status.value = 'Copied to clipboard';
    } catch (e) {
      status.value = `Copy failed: ${String(e)}`;
    }
  });

  const navigate = $((view: string, detail: Record<string, unknown>) => {
    try {
      window.dispatchEvent(new CustomEvent('pluribus:navigate', { detail: { view, ...detail } }));
    } catch {
      // ignore
    }
  });

  const parseJsonResponse = async (res: Response, label: string) => {
    const contentType = res.headers.get('content-type') || '';
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`${label} HTTP ${res.status}`);
    }
    try {
      return JSON.parse(text);
    } catch {
      const prefix = text.trim().slice(0, 120);
      throw new Error(`${label} invalid JSON (${contentType || 'unknown'}): ${prefix}`);
    }
  };

  const refresh = $(async () => {
    error.value = null;
    status.value = null;
    isLoading.value = true;
    try {
      if (Object.keys(schema.operators || {}).length === 0) {
        const cached = loadSemopsCache();
        if (cached && cached.schema && cached.suggestions) {
          schema.operators = cached.schema.operators || {};
          schema.commands = cached.schema.commands || [];
          schema.alias_map = cached.schema.alias_map || {};
          schema.tool_map = cached.schema.tool_map || {};
          schema.bus_topics = cached.schema.bus_topics || {};
          schema.user_ops_path = cached.schema.user_ops_path;
          suggestions.tool_paths = cached.suggestions.tool_paths || [];
          suggestions.ui_components = cached.suggestions.ui_components || [];
          suggestions.recent_actors = cached.suggestions.recent_actors || [];
          suggestions.recent_topics = cached.suggestions.recent_topics || [];
          suggestions.sota_candidates = cached.suggestions.sota_candidates || [];
          suggestions.bus = cached.suggestions.bus || {};
          status.value = 'Loaded cached SemOps';
        }
      }
      const bust = `?ts=${Date.now()}`;
      const [schemaRes, suggRes] = await Promise.allSettled([
        fetch(`/api/semops${bust}`, { cache: 'no-store' }),
        fetch(`/api/semops/suggestions${bust}`, { cache: 'no-store' }),
      ]);
      const errors: string[] = [];
      const describeError = (err: unknown) => (err instanceof Error ? err.message : String(err));
      let schemaJson: SemopsSchemaResponse | null = null;
      let suggJson: SemopsSuggestionsResponse | null = null;

      if (schemaRes.status === 'fulfilled') {
        try {
          schemaJson = (await parseJsonResponse(schemaRes.value, 'SemOps schema')) as SemopsSchemaResponse;
        } catch (err) {
          errors.push(`schema: ${describeError(err)}`);
        }
      } else {
        errors.push(`schema: ${describeError(schemaRes.reason)}`);
      }

      if (suggRes.status === 'fulfilled') {
        try {
          suggJson = (await parseJsonResponse(suggRes.value, 'SemOps suggestions')) as SemopsSuggestionsResponse;
        } catch (err) {
          errors.push(`suggestions: ${describeError(err)}`);
        }
      } else {
        errors.push(`suggestions: ${describeError(suggRes.reason)}`);
      }

      if (schemaJson) {
        if ((schemaJson as any)?.error) {
          errors.push(String((schemaJson as any).error));
        }
        schema.operators = schemaJson.operators || {};
        schema.commands = schemaJson.commands || [];
        schema.alias_map = schemaJson.alias_map || {};
        schema.tool_map = schemaJson.tool_map || {};
        schema.bus_topics = schemaJson.bus_topics || {};
        schema.user_ops_path = schemaJson.user_ops_path;
      }

      if (suggJson) {
        suggestions.tool_paths = suggJson.tool_paths || [];
        suggestions.ui_components = suggJson.ui_components || [];
        suggestions.recent_actors = suggJson.recent_actors || [];
        suggestions.recent_topics = suggJson.recent_topics || [];
        suggestions.sota_candidates = suggJson.sota_candidates || [];
        suggestions.bus = suggJson.bus || {};
      }

      if (schemaJson && suggJson) {
        saveSemopsCache(schemaJson, suggJson);
      }

      if (errors.length > 0) {
        if (!schemaJson) {
          error.value = `Failed to load SemOps: ${errors.join(' | ')}`;
        } else {
          status.value = `Loaded with warnings: ${errors.join(' | ')}`;
        }
      }
    } catch (e) {
      error.value = `Failed to load SemOps: ${String(e)}`;
    } finally {
      isLoading.value = false;
    }
  });

  const defineOp = $(async () => {
    isSaving.value = true;
    status.value = null;
    error.value = null;
    try {
      const opKey = (form.key || form.name || form.id).trim();
      if (!opKey) {
        error.value = 'operator key is required';
        return;
      }
      const normalizedKey = opKey.toUpperCase();
      const opId = (form.id || normalizedKey.toLowerCase()).trim();
      const aliases = parseCommaList(form.aliases);
      const agentIds = parseCommaList(form.agents);
      const appIds = parseCommaList(form.apps);

      const ui: Record<string, unknown> = {};
      if (form.ui_route.trim()) ui.route = form.ui_route.trim();
      if (form.ui_component.trim()) ui.component = form.ui_component.trim();

      const targets: SemopsTarget[] = [];
      if (form.tool.trim()) targets.push({ type: 'tool', ref: form.tool.trim() });
      if (form.bus_topic.trim()) targets.push({ type: 'bus', ref: form.bus_topic.trim(), kind: form.bus_kind || undefined });
      if (ui.route || ui.component) targets.push({ type: 'ui', route: String(ui.route || ''), component: String(ui.component || '') });
      for (const a of agentIds) targets.push({ type: 'agent', ref: a });
      for (const a of appIds) targets.push({ type: 'app', ref: a });

      const operator: Record<string, unknown> = {
        key: normalizedKey,
        id: opId,
        name: (form.name || normalizedKey).trim(),
        domain: form.domain.trim() || 'user',
        category: form.category.trim() || 'custom',
        effects: form.effects,
        description: form.description.trim(),
        aliases: aliases.length > 0 ? aliases : [opId, normalizedKey],
        tool: form.tool.trim() || undefined,
        bus_topic: form.bus_topic.trim() || undefined,
        bus_kind: form.bus_kind || undefined,
        secondary_topic: form.secondary_topic.trim() || undefined,
        ui,
        agents: agentIds,
        apps: appIds,
        targets,
      };

      const req_id = crypto.randomUUID();
      const res = await fetch('/api/semops/user_ops/define', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor: 'dashboard', req_id, operator }),
      });
      const out = await res.json();
      if (!out.success) {
        error.value = out.error || 'Define failed';
        return;
      }

      status.value = `Defined ${out.operator_key || normalizedKey}`;
      await refresh();
    } catch (e) {
      error.value = `Define failed: ${String(e)}`;
    } finally {
      isSaving.value = false;
    }
  });

  const handleWizardPublish = $(async (operator: Record<string, unknown>) => {
    isSaving.value = true;
    status.value = null;
    error.value = null;
    try {
      const req_id = crypto.randomUUID();
      const res = await fetch('/api/semops/user_ops/define', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor: 'dashboard', req_id, operator }),
      });
      const out = await res.json();
      if (!out.success) {
        error.value = out.error || 'Define failed';
        return;
      }
      status.value = `Defined ${out.operator_key || operator.key} via Flow Wizard`;
      showFlowWizard.value = false;
      await refresh();
    } catch (e) {
      error.value = `Define failed: ${String(e)}`;
    } finally {
      isSaving.value = false;
    }
  });

  const removeOp = $(async (operatorKey: string) => {
    if (!operatorKey) return;
    if (!confirm(`Remove user-defined operator ${operatorKey}?`)) return;
    isSaving.value = true;
    status.value = null;
    error.value = null;
    try {
      const req_id = crypto.randomUUID();
      const res = await fetch('/api/semops/user_ops/undefine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor: 'dashboard', req_id, operator_key: operatorKey }),
      });
      const out = await res.json();
      if (!out.success) {
        error.value = out.error || 'Undefine failed';
        return;
      }
      status.value = `Removed ${operatorKey}`;
      await refresh();
    } catch (e) {
      error.value = `Undefine failed: ${String(e)}`;
    } finally {
      isSaving.value = false;
    }
  });

  const quickPickTool = $((toolPath: string) => {
    form.tool = toolPath;
    const key = deriveKeyFromToolPath(toolPath);
    if (!form.key) form.key = key;
    if (!form.id) form.id = key.toLowerCase();
    if (!form.name) form.name = key;
    if (!form.aliases) form.aliases = `${form.id}, ${key}`;
  });

  useVisibleTask$(() => {
    refresh();
  });

  const selectedOp = selectedKey.value ? schema.operators?.[selectedKey.value] : null;
  const selectedFlows = (selectedOp?.flow_hints || []) as string[];

  const invokeSelected = $(async () => {
    if (!selectedKey.value) return;
    isSaving.value = true;
    status.value = null;
    error.value = null;
    try {
      const req_id = crypto.randomUUID();
      let payload: Record<string, unknown> = {};
      const raw = invokePayload.value.trim();
      invokeLatestResponse.value = null;
      if (raw) {
        try {
          payload = JSON.parse(raw);
        } catch {
          payload = { input: raw };
        }
      }
      const eff = (selectedOp as any)?.effects;
      const effects = typeof eff === 'string' && eff.trim() ? eff.trim().toLowerCase() : undefined;
      const res = await fetch('/api/semops/invoke', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          actor: 'dashboard',
          req_id,
          operator_key: selectedKey.value,
          mode: invokeMode.value,
          effects,
          payload,
          wait: invokeWait.value,
          timeout_s: invokeTimeoutS.value,
        }),
      });
      const out = await res.json();
      if (!out.success) {
        error.value = out.error || 'Invoke failed';
        return;
      }
      invokeLatestResponse.value = out.response ? JSON.stringify(out.response, null, 2) : null;
      status.value = invokeWait.value
        ? `Invoked ${selectedKey.value} (response received)`
        : `Invoked ${selectedKey.value} (bus request emitted)`;
    } catch (e) {
      error.value = `Invoke failed: ${String(e)}`;
    } finally {
      isSaving.value = false;
    }
  });

  const opEntries = Object.entries(schema.operators || {}).filter(([k, op]) => {
    const q = filter.value.trim().toLowerCase();
    if (!q) return true;
    const hay = [
      k,
      op.id,
      op.name,
      op.domain,
      op.category,
      op.description,
      ...(op.aliases || []),
      String(op.tool || ''),
      String(op.bus_topic || ''),
    ]
      .join(' ')
      .toLowerCase();
    return hay.includes(q);
  });

  // Group by domain -> category
  const grouped: Record<string, Record<string, Array<[string, SemopsOperator]>>> = {};
  for (const [key, op] of opEntries) {
    const domain = op.domain || 'unknown';
    const cat = op.category || 'unknown';
    if (!grouped[domain]) grouped[domain] = {};
    if (!grouped[domain][cat]) grouped[domain][cat] = [];
    grouped[domain][cat].push([key, op]);
  }

  const domainsSorted = Object.keys(grouped).sort((a, b) => a.localeCompare(b));

  const busTopicOptions = Array.from(
    new Set([
      ...Object.keys(schema.bus_topics || {}),
      ...(suggestions.recent_topics || []).map(([t]) => t),
    ])
  ).slice(0, 200);

  const toolPathOptions = Array.from(
    new Set([...(Object.values(schema.tool_map || {}) as string[]), ...(suggestions.tool_paths || [])])
  ).slice(0, 400);

  const uiComponentOptions = Array.from(new Set([...(suggestions.ui_components || [])])).slice(0, 400);

  const agentOptions = Array.from(new Set([...(suggestions.recent_actors || []).map(([a]) => a)])).slice(0, 200);

  return (
    <div class="space-y-6">
      <div class="flex items-start justify-between gap-4">
        <div>
          <div class="text-lg font-semibold">🧠 Semantic Operators (SemOps)</div>
          <div class="text-xs text-muted-foreground mono">
            {schema.user_ops_path ? `user_ops=${schema.user_ops_path}` : 'user_ops=unresolved'} • bus={suggestions.bus?.active_dir || 'unknown'}
          </div>
        </div>
        <div class="flex gap-2">
          <Button
            variant="tonal"
            onClick$={() => (showFlowWizard.value = true)}
          >
            ✨ Flow Wizard
          </Button>
          <Button
            variant="secondary"
            onClick$={refresh}
            disabled={isLoading.value}
          >
            Refresh
          </Button>
          <Button
            variant="secondary"
            onClick$={() => (showSotaSeeds.value = !showSotaSeeds.value)}
          >
            {showSotaSeeds.value ? 'Hide SOTA' : 'Show SOTA'}
          </Button>
        </div>
      </div>

      {error.value && (
        <div class="rounded border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300 font-mono">
          {error.value}
        </div>
      )}
      {status.value && (
        <div class="rounded border border-green-500/30 bg-green-500/10 p-3 text-sm text-green-300 font-mono">
          {status.value}
        </div>
      )}

        <div class="grid grid-cols-12 gap-6">
        {/* List */}
        <div class="col-span-8 rounded-lg border border-border bg-card p-4 space-y-4">
          <div class="flex items-center justify-between gap-4">
            <div class="text-sm font-semibold text-muted-foreground">Registry</div>
            <div class="flex items-center gap-2">
              <Input
                value={filter.value}
                onInput$={(_, el) => (filter.value = el.value)}
                placeholder="Filter (id/domain/topic/alias)…"
                class="w-[320px]"
              />
              <div class="text-xs text-muted-foreground mono">
                {opEntries.length} ops
              </div>
            </div>
          </div>

          {isLoading.value ? (
            <div class="text-sm text-muted-foreground">Loading…</div>
          ) : (
            <div class="space-y-4 max-h-[calc(100vh-260px)] overflow-auto pr-2">
              {domainsSorted.map((domain) => (
                <div key={domain} class="space-y-2">
                  <div class="text-xs font-semibold uppercase tracking-widest text-cyan-300/80">
                    {domain}
                  </div>
                  <div class="space-y-3">
                    {Object.keys(grouped[domain] || {})
                      .sort((a, b) => a.localeCompare(b))
                      .map((cat) => (
                        <Card key={`${domain}.${cat}`} variant="outlined" padding="p-0">
                          <div class="px-3 py-2 border-b border-border/50 flex items-center justify-between">
                            <div class="text-xs font-mono text-muted-foreground">
                              {cat} • {grouped[domain][cat].length}
                            </div>
                          </div>
                          <div class="divide-y divide-border/30">
                            {grouped[domain][cat]
                              .sort(([a], [b]) => a.localeCompare(b))
                              .map(([key, op]) => (
                                <div
                                  key={key}
                                  onClick$={() => {
                                    selectedKey.value = key;
                                    if (op.flow_hints && op.flow_hints.length > 0) invokeMode.value = op.flow_hints[0] as any;
                                  }}
                                  class={`px-3 py-2 flex items-start justify-between gap-3 cursor-pointer hover:bg-muted/20 ${
                                    selectedKey.value === key ? 'ring-2 ring-primary/60 bg-muted/10' : ''
                                  }`}
                                >
                                  <div class="min-w-0">
                                    <div class="flex items-center gap-2">
                                      <span class="font-mono text-sm text-foreground">{key}</span>
                                      {op.user_defined && (
                                        <span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                                          USER
                                        </span>
                                      )}
                                      {op.tool && (
                                        <span class="text-[10px] px-1.5 py-0.5 rounded bg-muted/50 text-muted-foreground border border-border/60 font-mono truncate max-w-[260px]">
                                          {String(op.tool)}
                                        </span>
                                      )}
                                      {op.bus_topic && (
                                        <span class="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-200 border border-cyan-500/20 font-mono truncate max-w-[260px]">
                                          {String(op.bus_topic)}
                                        </span>
                                      )}
                                    </div>
                                    <div class="text-xs text-muted-foreground mt-1 line-clamp-2">
                                      {op.description || ''}
                                    </div>
                                    <div class="text-[11px] text-muted-foreground font-mono mt-1 truncate">
                                      aliases: {(op.aliases || []).join(', ')}
                                    </div>
                                    {(op.agents && op.agents.length > 0) && (
                                      <div class="text-[11px] text-muted-foreground font-mono mt-1 truncate">
                                        agents: {op.agents.join(', ')}
                                      </div>
                                    )}
                                    {(op.apps && op.apps.length > 0) && (
                                      <div class="text-[11px] text-muted-foreground font-mono mt-1 truncate">
                                        apps: {op.apps.join(', ')}
                                      </div>
                                    )}
                                  </div>
                                  <div class="flex items-center gap-2 flex-shrink-0">
                                    <Button
                                      variant="secondary"
                                      onClick$={(ev) => {
                                        ev.stopPropagation();
                                        selectedKey.value = key;
                                        loadIntoForm(key, op, op.user_defined ? 'edit' : 'clone');
                                      }}
                                      disabled={isSaving.value}
                                      class="h-6 text-xs"
                                    >
                                      {op.user_defined ? 'Edit' : 'Clone'}
                                    </Button>
                                    {op.user_defined && (
                                      <Button
                                        variant="tonal"
                                        onClick$={(ev) => {
                                          ev.stopPropagation();
                                          removeOp(key);
                                        }}
                                        disabled={isSaving.value}
                                        class="h-6 text-xs text-red-300 border-red-500/30"
                                      >
                                        Remove
                                      </Button>
                                    )}
                                  </div>
                                </div>
                              ))}
                          </div>
                        </Card>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* CRUD */}
        <div class="col-span-4 rounded-lg border border-border bg-card p-4 space-y-4">
          {selectedKey.value && selectedOp && (
            <div class="rounded border border-border/60 bg-background/40 p-3 space-y-2">
              <div class="flex items-start justify-between gap-2">
                <div class="min-w-0">
                  <div class="font-mono text-sm text-foreground truncate">{selectedKey.value}</div>
                  <div class="text-[11px] text-muted-foreground truncate">{selectedOp.domain} / {selectedOp.category}</div>
                </div>
                <div class="flex gap-2">
                  <Button
                    variant="secondary"
                    onClick$={() => loadIntoForm(selectedKey.value!, selectedOp, selectedOp.user_defined ? 'edit' : 'clone')}
                    class="h-6 text-xs"
                    disabled={isSaving.value}
                  >
                    {selectedOp.user_defined ? 'Edit' : 'Clone'}
                  </Button>
                  {selectedOp.user_defined && (
                    <Button
                      variant="tonal"
                      onClick$={() => removeOp(selectedKey.value!)}
                      class="h-6 text-xs text-red-300 border-red-500/30"
                      disabled={isSaving.value}
                    >
                      Delete
                    </Button>
                  )}
                </div>
              </div>

              {selectedOp.description && (
                <div class="text-xs text-muted-foreground line-clamp-3">{selectedOp.description}</div>
              )}

              {selectedFlows.length > 0 && (
                <div class="flex flex-wrap gap-1">
                  {selectedFlows.slice(0, 8).map((f) => (
                    <span key={f} class="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-200 border border-cyan-500/20 font-mono">
                      {f}
                    </span>
                  ))}
                </div>
              )}

              <div class="space-y-1 text-[11px] font-mono text-muted-foreground">
                {selectedOp.tool && (
                  <div class="truncate">tool: {String(selectedOp.tool)}</div>
                )}
                {selectedOp.bus_topic && (
                  <div class="truncate">bus: {String(selectedOp.bus_topic)} {selectedOp.bus_kind ? `(${selectedOp.bus_kind})` : ''}</div>
                )}
                {(selectedOp.ui as any)?.route && (
                  <div class="truncate">ui.route: {String((selectedOp.ui as any).route)}</div>
                )}
                {(selectedOp.ui as any)?.component && (
                  <div class="truncate">ui.component: {String((selectedOp.ui as any).component)}</div>
                )}
                {(selectedOp as any)?.effects && (
                  <div class={`truncate ${
                    (selectedOp as any).effects === 'system' ? 'text-red-400' :
                    (selectedOp as any).effects === 'network' ? 'text-orange-400' :
                    (selectedOp as any).effects === 'file' ? 'text-yellow-400' :
                    ''
                  }`}>
                    effects: {String((selectedOp as any).effects)}
                    {(selectedOp as any).effects === 'system' && ' ⚠️ elevated'}
                    {(selectedOp as any).effects === 'network' && ' ⚠️ external'}
                    {(selectedOp as any).effects === 'file' && ' ⚠️ filesystem'}
                  </div>
                )}
              </div>

              <div class="flex flex-wrap gap-2">
                <Button variant="secondary" onClick$={() => copyText(selectedKey.value || '')} class="h-6 text-xs">Copy Key</Button>
                {selectedOp.tool && (
                  <Button variant="secondary" onClick$={() => copyText(String(selectedOp.tool))} class="h-6 text-xs">Copy Tool</Button>
                )}
                {selectedOp.bus_topic && (
                  <Button variant="secondary" onClick$={() => navigate('events', { searchPattern: String(selectedOp.bus_topic), searchMode: 'glob' })} class="h-6 text-xs">Open Events</Button>
                )}
              </div>

              <div class="pt-2 border-t border-border/50 space-y-2">
                <div class="text-xs font-semibold text-muted-foreground uppercase tracking-widest">Invoke (Bus Request)</div>
                <div class="grid grid-cols-2 gap-2">
                  <select
                    value={invokeMode.value}
                    onChange$={(e) => (invokeMode.value = (e.target as HTMLSelectElement).value)}
                    class="px-2 py-1 rounded bg-background border border-border text-xs font-mono"
                  >
                    <option value="auto">auto</option>
                    <option value="tool">tool</option>
                    <option value="bus">bus</option>
                    <option value="policy">policy</option>
                    <option value="evolution">evolution</option>
                    <option value="ui">ui</option>
                    <option value="agent">agent</option>
                    <option value="app">app</option>
                  </select>
                  <Button
                    variant="primary"
                    onClick$={invokeSelected}
                    disabled={isSaving.value}
                    class="h-8 text-xs"
                  >
                    Emit
                  </Button>
                </div>
                <div class="flex items-center gap-3 text-xs text-muted-foreground">
                  <label class="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={invokeWait.value}
                      onChange$={(e) => (invokeWait.value = (e.target as HTMLInputElement).checked)}
                    />
                    wait for response
                  </label>
                  <label class="flex items-center gap-2">
                    <span class="text-[11px]">timeout_s</span>
                    <input
                      type="number"
                      min="0.1"
                      step="0.1"
                      value={invokeTimeoutS.value}
                      onInput$={(e) => {
                        const v = parseFloat((e.target as HTMLInputElement).value || '');
                        invokeTimeoutS.value = Number.isFinite(v) ? v : 2.0;
                      }}
                      class="w-24 px-2 py-1 rounded bg-background border border-border text-xs font-mono"
                    />
                  </label>
                </div>
                <Input
                  type="textarea"
                  value={invokePayload.value}
                  onInput$={(_, el) => (invokePayload.value = el.value)}
                  placeholder='{"input":"..."}'
                />
                {invokeLatestResponse.value && (
                  <pre class="px-2 py-2 rounded bg-black/30 border border-border text-[11px] font-mono max-h-48 overflow-auto whitespace-pre-wrap">
                    {invokeLatestResponse.value}
                  </pre>
                )}
                <div class="text-[11px] text-muted-foreground">
                  Emits `semops.invoke.request` to the bus; execution is handled asynchronously by agents/daemons.
                </div>
              </div>
            </div>
          )}

          <div class="text-sm font-semibold text-muted-foreground">Create / Update (User Ops)</div>

          <div class="flex flex-wrap gap-2">
            <Button variant="secondary" onClick$={() => applyTemplate('tool')} class="h-6 text-xs">Tool</Button>
            <Button variant="secondary" onClick$={() => applyTemplate('policy')} class="h-6 text-xs">Policy</Button>
            <Button variant="secondary" onClick$={() => applyTemplate('evolution')} class="h-6 text-xs">Evolution</Button>
            <Button variant="secondary" onClick$={() => applyTemplate('ui')} class="h-6 text-xs">UI</Button>
            <Button variant="secondary" onClick$={resetForm} class="h-6 text-xs">Reset</Button>
          </div>

          <div class="space-y-2">
            <div class="grid grid-cols-2 gap-2">
              <Input
                label="KEY"
                value={form.key}
                onInput$={(_, el) => (form.key = el.value)}
                placeholder="MYOP"
              />
              <Input
                label="id"
                value={form.id}
                onInput$={(_, el) => (form.id = el.value)}
                placeholder="myop"
              />
            </div>
            <Input
              label="Name"
              value={form.name}
              onInput$={(_, el) => (form.name = el.value)}
              placeholder="Display Name"
            />
            <div class="grid grid-cols-2 gap-2">
              <Input
                label="Domain"
                value={form.domain}
                onInput$={(_, el) => (form.domain = el.value)}
                placeholder="user"
              />
              <div class="grid grid-cols-2 gap-2">
                <Input
                  label="Category"
                  value={form.category}
                  onInput$={(_, el) => (form.category = el.value)}
                  placeholder="custom"
                />
                <select
                  value={form.effects}
                  onChange$={(e) => (form.effects = (e.target as HTMLSelectElement).value as SemopsEffects)}
                  class="px-3 py-2 rounded bg-background border border-border text-sm font-mono h-[56px]"
                >
                  <option value="none">effects: none</option>
                  <option value="file">effects: file</option>
                  <option value="network">effects: network</option>
                  <option value="system">effects: system</option>
                  <option value="unknown">effects: unknown</option>
                </select>
              </div>
            </div>
            <Input
              type="textarea"
              label="Description"
              value={form.description}
              onInput$={(_, el) => (form.description = el.value)}
              placeholder="Description..."
            />
            <Input
              label="Aliases"
              value={form.aliases}
              onInput$={(_, el) => (form.aliases = el.value)}
              placeholder="comma-separated"
            />
          </div>

          <div class="space-y-2">
            <div class="text-xs font-semibold text-muted-foreground uppercase tracking-widest">Bindings</div>
            <select
              class="px-3 py-2 rounded bg-background border border-border text-sm font-mono w-full"
              onChange$={(e) => quickPickTool((e.target as HTMLSelectElement).value)}
            >
              <option value="">Quick-pick tool…</option>
              {toolPathOptions.slice(0, 120).map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            <Input
              label="Tool Path"
              value={form.tool}
              onInput$={(_, el) => (form.tool = el.value)}
              placeholder="nucleus/tools/..."
            />

            <Input
              label="Bus Topic"
              value={form.bus_topic}
              onInput$={(_, el) => (form.bus_topic = el.value)}
              placeholder="domain.action"
            />

            <div class="grid grid-cols-2 gap-2">
              <select
                value={form.bus_kind}
                onChange$={(e) => (form.bus_kind = (e.target as HTMLSelectElement).value as SemopsBusKind)}
                class="px-3 py-2 rounded bg-background border border-border text-sm font-mono h-[56px]"
              >
                <option value="">bus_kind</option>
                <option value="request">request</option>
                <option value="response">response</option>
                <option value="artifact">artifact</option>
                <option value="metric">metric</option>
                <option value="log">log</option>
              </select>
              <Input
                label="Secondary Topic"
                value={form.secondary_topic}
                onInput$={(_, el) => (form.secondary_topic = el.value)}
              />
            </div>

            <div class="grid grid-cols-2 gap-2">
              <Input
                label="UI Route"
                value={form.ui_route}
                onInput$={(_, el) => (form.ui_route = el.value)}
              />
              <Input
                label="UI Component"
                value={form.ui_component}
                onInput$={(_, el) => (form.ui_component = el.value)}
              />
            </div>

            <Input
              label="Agents"
              value={form.agents}
              onInput$={(_, el) => (form.agents = el.value)}
              placeholder="comma-separated"
            />

            <Input
              label="Apps"
              value={form.apps}
              onInput$={(_, el) => (form.apps = el.value)}
              placeholder="comma-separated"
            />
          </div>

          <div class="flex items-center gap-2">
            <Button
              variant="primary"
              onClick$={defineOp}
              disabled={isSaving.value}
            >
              {isSaving.value ? 'Saving…' : 'Save'}
            </Button>
            <Button
              variant="secondary"
              onClick$={resetForm}
            >
              Clear
            </Button>
          </div>

          {showSotaSeeds.value && (
            <Card variant="outlined" padding="p-3" class="bg-muted/20">
              <div class="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                SOTA Seeds (from catalog)
              </div>
              <div class="text-xs text-muted-foreground">
                These are suggestions (not wired). Use as prompts for new SemOps, services, or integration tasks.
              </div>
              <div class="max-h-[200px] overflow-auto pr-1 space-y-1">
                {suggestions.sota_candidates.slice(0, 80).map((s) => (
                  <div key={s} class="text-[11px] font-mono text-muted-foreground truncate">
                    {s}
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Flow Wizard Modal */}
      <SemopsFlowWizard
        isOpen={showFlowWizard.value}
        onClose$={() => (showFlowWizard.value = false)}
        onPublish$={handleWizardPublish}
        suggestions={suggestions}
      />
    </div>
  );
});