/**
 * SemopsFlowWizard.tsx - Guided Operator Creation Stepper
 *
 * Flow: Intent → Targets → Effects → Evidence → Publish
 *
 * Integrates with SemopsEditor and emits the final operator via callback.
 */

import { component$, useSignal, useStore, $, type QRL } from '@builder.io/qwik';
import type { SemopsEffects, SemopsTarget, SemopsSuggestionsResponse } from './SemopsEditor';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Card } from './ui/Card';

export interface FlowWizardState {
  // Step 1: Intent
  key: string;
  id: string;
  name: string;
  description: string;
  aliases: string;
  // Step 2: Targets
  targetType: 'tool' | 'bus' | 'ui' | 'agent' | 'app' | 'multi';
  tool: string;
  bus_topic: string;
  bus_kind: string;
  ui_route: string;
  ui_component: string;
  agents: string;
  apps: string;
  // Step 3: Effects
  effects: SemopsEffects;
  requiresGrant: boolean;
  // Step 4: Evidence
  domain: string;
  category: string;
  guarantees: string;
  produces_artifact: boolean;
  // Step 5: Publish (derived from above)
}

interface FlowWizardProps {
  isOpen: boolean;
  onClose$: QRL<() => void>;
  onPublish$: QRL<(operator: Record<string, unknown>) => void>;
  suggestions: SemopsSuggestionsResponse;
  initialState?: Partial<FlowWizardState>;
}

const STEPS = [
  { id: 'intent', label: 'Intent', icon: '💡' },
  { id: 'targets', label: 'Targets', icon: '🎯' },
  { id: 'effects', label: 'Effects', icon: '⚡' },
  { id: 'evidence', label: 'Evidence', icon: '📋' },
  { id: 'publish', label: 'Publish', icon: '🚀' },
] as const;

const EFFECTS_INFO: Record<SemopsEffects, { color: string; description: string; warning?: string }> = {
  none: { color: 'green', description: 'Pure computation; no external side effects.' },
  file: { color: 'yellow', description: 'May read/write files on the local filesystem.', warning: 'Requires filesystem access grant.' },
  network: { color: 'orange', description: 'May make network requests (HTTP, WebSocket, etc.).', warning: 'Requires network access grant.' },
  system: { color: 'red', description: 'May execute system commands or modify OS state.', warning: 'Requires elevated system grant. Use with caution.' },
  unknown: { color: 'gray', description: 'Effects are not yet classified.' },
};

export const SemopsFlowWizard = component$<FlowWizardProps>(({ isOpen, onClose$, onPublish$, suggestions, initialState }) => {
  const currentStep = useSignal(0);
  const validationError = useSignal<string | null>(null);

  const state = useStore<FlowWizardState>({
    key: initialState?.key || '',
    id: initialState?.id || '',
    name: initialState?.name || '',
    description: initialState?.description || '',
    aliases: initialState?.aliases || '',
    targetType: initialState?.targetType || 'tool',
    tool: initialState?.tool || '',
    bus_topic: initialState?.bus_topic || '',
    bus_kind: initialState?.bus_kind || 'request',
    ui_route: initialState?.ui_route || '',
    ui_component: initialState?.ui_component || '',
    agents: initialState?.agents || '',
    apps: initialState?.apps || '',
    effects: initialState?.effects || 'none',
    requiresGrant: initialState?.requiresGrant || false,
    domain: initialState?.domain || 'user',
    category: initialState?.category || 'custom',
    guarantees: initialState?.guarantees || '',
    produces_artifact: initialState?.produces_artifact || false,
  });

  const validateStep = $((step: number): boolean => {
    validationError.value = null;
    switch (step) {
      case 0: // Intent
        if (!state.key.trim() && !state.name.trim()) {
          validationError.value = 'Operator key or name is required.';
          return false;
        }
        return true;
      case 1: // Targets
        if (state.targetType === 'tool' && !state.tool.trim()) {
          validationError.value = 'Tool path is required for tool-type operators.';
          return false;
        }
        if (state.targetType === 'bus' && !state.bus_topic.trim()) {
          validationError.value = 'Bus topic is required for bus-type operators.';
          return false;
        }
        return true;
      case 2: // Effects
        return true; // Effects selection is optional
      case 3: // Evidence
        return true; // Evidence fields are optional
      case 4: // Publish
        return true;
      default:
        return true;
    }
  });

  const nextStep = $(async () => {
    const isValid = await validateStep(currentStep.value);
    if (isValid && currentStep.value < STEPS.length - 1) {
      currentStep.value++;
    }
  });

  const prevStep = $(() => {
    if (currentStep.value > 0) {
      currentStep.value--;
      validationError.value = null;
    }
  });

  const goToStep = $(async (step: number) => {
    // Only allow going back or to completed steps
    if (step < currentStep.value) {
      currentStep.value = step;
      validationError.value = null;
    }
  });

  const buildOperator = $(() => {
    const key = (state.key || state.name || state.id).trim().toUpperCase();
    const id = (state.id || key.toLowerCase()).trim();
    const aliases = state.aliases
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    const targets: SemopsTarget[] = [];
    if (state.tool.trim()) targets.push({ type: 'tool', ref: state.tool.trim() });
    if (state.bus_topic.trim()) targets.push({ type: 'bus', ref: state.bus_topic.trim(), kind: state.bus_kind || undefined });
    if (state.ui_route.trim() || state.ui_component.trim()) {
      targets.push({ type: 'ui', route: state.ui_route.trim(), component: state.ui_component.trim() });
    }
    for (const a of state.agents.split(',').map((s) => s.trim()).filter(Boolean)) {
      targets.push({ type: 'agent', ref: a });
    }
    for (const a of state.apps.split(',').map((s) => s.trim()).filter(Boolean)) {
      targets.push({ type: 'app', ref: a });
    }

    const guarantees = state.guarantees
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    const operator: Record<string, unknown> = {
      key,
      id,
      name: (state.name || key).trim(),
      domain: state.domain.trim() || 'user',
      category: state.category.trim() || 'custom',
      effects: state.effects,
      description: state.description.trim(),
      aliases: aliases.length > 0 ? aliases : [id, key],
      tool: state.tool.trim() || undefined,
      bus_topic: state.bus_topic.trim() || undefined,
      bus_kind: state.bus_kind || undefined,
      ui: {
        route: state.ui_route.trim() || undefined,
        component: state.ui_component.trim() || undefined,
      },
      agents: state.agents.split(',').map((s) => s.trim()).filter(Boolean),
      apps: state.apps.split(',').map((s) => s.trim()).filter(Boolean),
      targets,
      guarantees: guarantees.length > 0 ? guarantees : undefined,
      requires_grant: state.requiresGrant || undefined,
      produces_artifact: state.produces_artifact || undefined,
    };

    return operator;
  });

  const publish = $(async () => {
    const isValid = await validateStep(currentStep.value);
    if (!isValid) return;

    const operator = await buildOperator();
    await onPublish$(operator);
  });

  const reset = $(() => {
    currentStep.value = 0;
    validationError.value = null;
    state.key = '';
    state.id = '';
    state.name = '';
    state.description = '';
    state.aliases = '';
    state.targetType = 'tool';
    state.tool = '';
    state.bus_topic = '';
    state.bus_kind = 'request';
    state.ui_route = '';
    state.ui_component = '';
    state.agents = '';
    state.apps = '';
    state.effects = 'none';
    state.requiresGrant = false;
    state.domain = 'user';
    state.category = 'custom';
    state.guarantees = '';
    state.produces_artifact = false;
  });

  if (!isOpen) return null;

  const effectInfo = EFFECTS_INFO[state.effects];
  const step = STEPS[currentStep.value];

  return (
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <Card class="w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col">
        {/* Header */}
        <div class="flex items-center justify-between border-b border-border px-6 py-4">
          <div class="flex items-center gap-3">
            <span class="text-2xl">{step.icon}</span>
            <div>
              <div class="text-lg font-semibold">Flow Wizard</div>
              <div class="text-xs text-muted-foreground">
                Step {currentStep.value + 1} of {STEPS.length}: {step.label}
              </div>
            </div>
          </div>
          <Button variant="icon" icon="close" onClick$={onClose$} class="text-muted-foreground" />
        </div>

        {/* Stepper Progress */}
        <div class="flex items-center justify-between px-6 py-3 border-b border-border/50 bg-muted/10">
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              onClick$={() => goToStep(i)}
              class={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                i === currentStep.value
                  ? 'bg-primary/20 text-primary border border-primary/30'
                  : i < currentStep.value
                  ? 'bg-green-500/10 text-green-400 border border-green-500/20 cursor-pointer hover:bg-green-500/20'
                  : 'bg-muted/20 text-muted-foreground border border-border/50'
              }`}
            >
              <span>{s.icon}</span>
              <span class="hidden sm:inline">{s.label}</span>
            </button>
          ))}
        </div>

        {/* Content */}
        <div class="px-6 py-6 space-y-4 min-h-[320px]">
          {validationError.value && (
            <div class="rounded border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
              {validationError.value}
            </div>
          )}

          {/* Step 0: Intent */}
          {currentStep.value === 0 && (
            <div class="space-y-4">
              <div class="text-sm text-muted-foreground">
                Define <strong>what</strong> this operator does and how it's identified.
              </div>
              <div class="grid grid-cols-2 gap-3">
                <Input
                  label="Key *"
                  value={state.key}
                  onInput$={(_, el) => (state.key = el.value.toUpperCase())}
                  placeholder="MY_OPERATOR"
                />
                <Input
                  label="ID"
                  value={state.id}
                  onInput$={(_, el) => (state.id = el.value)}
                  placeholder="my_operator"
                />
              </div>
              <Input
                label="Name (Display)"
                value={state.name}
                onInput$={(_, el) => (state.name = el.value)}
                placeholder="My Custom Operator"
              />
              <Input
                type="textarea"
                label="Description"
                value={state.description}
                onInput$={(_, el) => (state.description = el.value)}
                placeholder="What does this operator do? What problem does it solve?"
              />
              <Input
                label="Aliases (comma-separated)"
                value={state.aliases}
                onInput$={(_, el) => (state.aliases = el.value)}
                placeholder="myop, custom_op, my operator"
              />
            </div>
          )}

          {/* Step 1: Targets */}
          {currentStep.value === 1 && (
            <div class="space-y-4">
              <div class="text-sm text-muted-foreground">
                Define <strong>where</strong> this operator routes requests.
              </div>
              <div class="space-y-2">
                <label class="text-xs text-muted-foreground">Primary Target Type</label>
                <div class="flex flex-wrap gap-2">
                  {(['tool', 'bus', 'ui', 'agent', 'app', 'multi'] as const).map((t) => (
                    <Button
                      key={t}
                      variant={state.targetType === t ? 'tonal' : 'secondary'}
                      onClick$={() => (state.targetType = t)}
                      class="h-8 text-xs"
                    >
                      {t}
                    </Button>
                  ))}
                </div>
              </div>

              {(state.targetType === 'tool' || state.targetType === 'multi') && (
                <div class="space-y-1">
                  <label class="text-xs text-muted-foreground">Tool Path</label>
                  <select
                    value={state.tool}
                    onChange$={(e) => (state.tool = (e.target as HTMLSelectElement).value)}
                    class="w-full px-3 py-2 rounded bg-background border border-border text-sm font-mono"
                  >
                    <option value="">Select tool…</option>
                    {(suggestions.tool_paths || []).slice(0, 100).map((p) => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                  <Input
                    value={state.tool}
                    onInput$={(_, el) => (state.tool = el.value)}
                    placeholder="Or type path: nucleus/tools/..."
                    class="mt-1"
                  />
                </div>
              )}

              {(state.targetType === 'bus' || state.targetType === 'multi') && (
                <div class="grid grid-cols-2 gap-3">
                  <Input
                    label="Bus Topic"
                    value={state.bus_topic}
                    onInput$={(_, el) => (state.bus_topic = el.value)}
                    placeholder="my.custom.topic"
                  />
                  <div class="space-y-1">
                    <label class="text-xs text-muted-foreground">Bus Kind</label>
                    <select
                      value={state.bus_kind}
                      onChange$={(e) => (state.bus_kind = (e.target as HTMLSelectElement).value)}
                      class="w-full px-3 py-2 rounded bg-background border border-border text-sm font-mono h-[56px]"
                    >
                      <option value="request">request</option>
                      <option value="response">response</option>
                      <option value="artifact">artifact</option>
                      <option value="metric">metric</option>
                      <option value="log">log</option>
                    </select>
                  </div>
                </div>
              )}

              {(state.targetType === 'ui' || state.targetType === 'multi') && (
                <div class="grid grid-cols-2 gap-3">
                  <Input
                    label="UI Route"
                    value={state.ui_route}
                    onInput$={(_, el) => (state.ui_route = el.value)}
                    placeholder="/services"
                  />
                  <div class="space-y-1">
                    <label class="text-xs text-muted-foreground">UI Component</label>
                    <select
                      value={state.ui_component}
                      onChange$={(e) => (state.ui_component = (e.target as HTMLSelectElement).value)}
                      class="w-full px-3 py-2 rounded bg-background border border-border text-sm font-mono h-[56px]"
                    >
                      <option value="">Select component…</option>
                      {(suggestions.ui_components || []).slice(0, 50).map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              {(state.targetType === 'agent' || state.targetType === 'multi') && (
                <Input
                  label="Agents (comma-separated)"
                  value={state.agents}
                  onInput$={(_, el) => (state.agents = el.value)}
                  placeholder="claude, gemini, codex"
                />
              )}

              {(state.targetType === 'app' || state.targetType === 'multi') && (
                <Input
                  label="Apps (comma-separated)"
                  value={state.apps}
                  onInput$={(_, el) => (state.apps = el.value)}
                  placeholder="firefox, vscode"
                />
              )}
            </div>
          )}

          {/* Step 2: Effects */}
          {currentStep.value === 2 && (
            <div class="space-y-4">
              <div class="text-sm text-muted-foreground">
                Declare <strong>what capabilities</strong> this operator requires.
              </div>
              <div class="grid grid-cols-5 gap-2">
                {(Object.keys(EFFECTS_INFO) as SemopsEffects[]).map((eff) => {
                  const info = EFFECTS_INFO[eff];
                  return (
                    <button
                      key={eff}
                      onClick$={() => {
                        state.effects = eff;
                        state.requiresGrant = !!info.warning;
                      }}
                      class={`px-3 py-2 rounded border text-xs font-mono ${
                        state.effects === eff
                          ? `bg-${info.color}-500/20 text-${info.color}-300 border-${info.color}-500/40`
                          : 'border-border/60 hover:bg-muted/20'
                      }`}
                    >
                      {eff}
                    </button>
                  );
                })}
              </div>
              <div class={`rounded border p-4 space-y-2 ${
                effectInfo.warning
                  ? 'border-yellow-500/30 bg-yellow-500/5'
                  : 'border-border/50 bg-muted/10'
              }`}>
                <div class="text-sm">{effectInfo.description}</div>
                {effectInfo.warning && (
                  <div class="text-xs text-yellow-300 flex items-center gap-2">
                    <span>⚠️</span>
                    {effectInfo.warning}
                  </div>
                )}
              </div>
              <div class="flex items-center gap-3">
                <label class="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={state.requiresGrant}
                    onChange$={(e) => (state.requiresGrant = (e.target as HTMLInputElement).checked)}
                    class="rounded"
                  />
                  Requires explicit grant before execution
                </label>
              </div>
            </div>
          )}

          {/* Step 3: Evidence */}
          {currentStep.value === 3 && (
            <div class="space-y-4">
              <div class="text-sm text-muted-foreground">
                Define <strong>classification</strong> and <strong>guarantees</strong>.
              </div>
              <div class="grid grid-cols-2 gap-3">
                <div class="space-y-1">
                  <label class="text-xs text-muted-foreground">Domain</label>
                  <select
                    value={state.domain}
                    onChange$={(e) => (state.domain = (e.target as HTMLSelectElement).value)}
                    class="w-full px-3 py-2 rounded bg-background border border-border text-sm font-mono h-[56px]"
                  >
                    <option value="user">user</option>
                    <option value="execution">execution</option>
                    <option value="safety">safety</option>
                    <option value="evolution">evolution</option>
                    <option value="ui">ui</option>
                    <option value="navigation">navigation</option>
                    <option value="query">query</option>
                  </select>
                </div>
                <Input
                  label="Category"
                  value={state.category}
                  onInput$={(_, el) => (state.category = el.value)}
                  placeholder="custom, tool, policy, git"
                />
              </div>
              <Input
                label="Guarantees (comma-separated)"
                value={state.guarantees}
                onInput$={(_, el) => (state.guarantees = el.value)}
                placeholder="idempotent, deterministic, bounded, atomic"
              />
              <div class="flex items-center gap-3">
                <label class="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={state.produces_artifact}
                    onChange$={(e) => (state.produces_artifact = (e.target as HTMLInputElement).checked)}
                    class="rounded"
                  />
                  Produces bus artifact on completion
                </label>
              </div>
            </div>
          )}

          {/* Step 4: Publish */}
          {currentStep.value === 4 && (
            <div class="space-y-4">
              <div class="text-sm text-muted-foreground">
                Review and <strong>publish</strong> your operator.
              </div>
              <div class="rounded border border-border/50 bg-muted/10 p-4 space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-xs text-muted-foreground">Key</span>
                  <span class="font-mono text-sm">{state.key || state.name.toUpperCase() || '(unnamed)'}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-xs text-muted-foreground">Domain / Category</span>
                  <span class="text-sm">{state.domain} / {state.category}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-xs text-muted-foreground">Effects</span>
                  <span class={`text-xs px-2 py-0.5 rounded font-mono ${
                    state.effects === 'none' ? 'bg-green-500/20 text-green-300' :
                    state.effects === 'file' ? 'bg-yellow-500/20 text-yellow-300' :
                    state.effects === 'network' ? 'bg-orange-500/20 text-orange-300' :
                    state.effects === 'system' ? 'bg-red-500/20 text-red-300' :
                    'bg-muted/50 text-muted-foreground'
                  }`}>{state.effects}</span>
                </div>
                {state.tool && (
                  <div class="flex items-center justify-between">
                    <span class="text-xs text-muted-foreground">Tool</span>
                    <span class="text-xs font-mono truncate max-w-[200px]">{state.tool}</span>
                  </div>
                )}
                {state.bus_topic && (
                  <div class="flex items-center justify-between">
                    <span class="text-xs text-muted-foreground">Bus Topic</span>
                    <span class="text-xs font-mono truncate max-w-[200px]">{state.bus_topic}</span>
                  </div>
                )}
                {state.requiresGrant && (
                  <div class="text-xs text-yellow-300 flex items-center gap-2">
                    ⚠️ Requires explicit grant
                  </div>
                )}
              </div>
              {state.description && (
                <div class="text-sm text-muted-foreground border-l-2 border-border pl-3 italic">
                  "{state.description}"
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div class="flex items-center justify-between border-t border-border px-6 py-4">
          <Button variant="secondary" onClick$={reset} class="h-8 text-xs">
            Reset
          </Button>
          <div class="flex items-center gap-3">
            {currentStep.value > 0 && (
              <Button variant="secondary" onClick$={prevStep} class="h-9 text-sm">
                ← Back
              </Button>
            )}
            {currentStep.value < STEPS.length - 1 ? (
              <Button variant="primary" onClick$={nextStep} class="h-9 text-sm">
                Next →
              </Button>
            ) : (
              <Button variant="primary" onClick$={publish} class="h-9 text-sm bg-green-500/20 text-green-300 border-green-500/30 hover:bg-green-500/30">
                🚀 Publish
              </Button>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
});