/**
 * SOTAManager - State-of-the-art pattern management and mutation workflow
 *
 * Features:
 * - Pattern list with metadata cards
 * - Generate mutation modal
 * - Validation result display
 * - Pattern filtering and search
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  useContext,
  $,
} from '@builder.io/qwik';
import { MetaIngestContext } from '../../lib/metaingest/store';
import type { SOTAPattern, MutationCandidate, ValidationResult, APIResponse } from '../../lib/metaingest/types';
import { NeonTitle, NeonBadge } from '../ui/NeonTitle';
import { Card, CardContent, CardHeader } from '../ui/Card';

interface SOTAState {
  patterns: SOTAPattern[];
  selectedPattern: SOTAPattern | null;
  generatedMutation: MutationCandidate | null;
  validationResult: ValidationResult | null;
  showGenerateModal: boolean;
  showValidationModal: boolean;
}

export const SOTAManager = component$(() => {
  const state = useContext(MetaIngestContext);
  const loading = useSignal(false);
  const error = useSignal<string | null>(null);
  const searchQuery = useSignal('');
  const filterType = useSignal<'all' | SOTAPattern['techniqueType']>('all');

  const sotaState = useStore<SOTAState>({
    patterns: [],
    selectedPattern: null,
    generatedMutation: null,
    validationResult: null,
    showGenerateModal: false,
    showValidationModal: false,
  });

  // Fetch SOTA patterns
  useVisibleTask$(async ({ cleanup }) => {
    const controller = new AbortController();
    cleanup(() => controller.abort());

    loading.value = true;
    state.loading.sota = true;

    try {
      const res = await fetch('/api/metaingest/sota/patterns', {
        signal: controller.signal,
      });
      const data: APIResponse<SOTAPattern[]> = await res.json();

      if (data.success && data.data) {
        sotaState.patterns = data.data;
        state.sotaPatterns = data.data;
      } else {
        error.value = data.error?.message ?? 'Failed to load SOTA patterns';
      }
    } catch (e) {
      if (e instanceof Error && e.name !== 'AbortError') {
        error.value = e.message;
        state.errors.sota = e.message;
      }
    } finally {
      loading.value = false;
      state.loading.sota = false;
    }
  });

  // Generate mutation from pattern
  const generateMutation = $(async (patternId: string) => {
    loading.value = true;

    try {
      const res = await fetch('/api/metaingest/sota/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patternId,
          targetFile: 'nucleus/tools/example.py', // Default target
        }),
      });

      const data: APIResponse<MutationCandidate> = await res.json();

      if (data.success && data.data) {
        sotaState.generatedMutation = data.data;
        sotaState.showGenerateModal = true;
      } else {
        error.value = data.error?.message ?? 'Failed to generate mutation';
      }
    } catch (e) {
      if (e instanceof Error) {
        error.value = e.message;
      }
    } finally {
      loading.value = false;
    }
  });

  // Validate mutation
  const validateMutation = $(async (mutationId: string) => {
    loading.value = true;

    try {
      const res = await fetch('/api/metaingest/sota/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mutationId }),
      });

      const data: APIResponse<ValidationResult> = await res.json();

      if (data.success && data.data) {
        sotaState.validationResult = data.data;
        sotaState.showValidationModal = true;
      } else {
        error.value = data.error?.message ?? 'Failed to validate mutation';
      }
    } catch (e) {
      if (e instanceof Error) {
        error.value = e.message;
      }
    } finally {
      loading.value = false;
    }
  });

  // Filter patterns
  const filteredPatterns = sotaState.patterns.filter((p) => {
    const matchesType = filterType.value === 'all' || p.techniqueType === filterType.value;
    const matchesSearch = !searchQuery.value ||
      p.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      p.description.toLowerCase().includes(searchQuery.value.toLowerCase());

    return matchesType && matchesSearch;
  });

  if (loading.value && sotaState.patterns.length === 0) {
    return (
      <div class="flex items-center justify-center h-full">
        <div class="text-slate-400 text-sm">Loading SOTA patterns...</div>
      </div>
    );
  }

  if (error.value && sotaState.patterns.length === 0) {
    return (
      <div class="flex items-center justify-center h-full">
        <div class="text-red-400 text-sm">{error.value}</div>
      </div>
    );
  }

  return (
    <div class="flex flex-col h-full gap-4">
      {/* Header */}
      <div class="flex items-center justify-between">
        <NeonTitle level="h2" color="amber" size="lg">
          SOTA Mutation Engine
        </NeonTitle>
        <div class="flex items-center gap-2">
          <NeonBadge color="amber">
            {filteredPatterns.length} Patterns
          </NeonBadge>
        </div>
      </div>

      {/* Search and Filter */}
      <div class="flex items-center gap-2">
        <input
          type="text"
          placeholder="Search patterns..."
          value={searchQuery.value}
          onInput$={(e) => {
            searchQuery.value = (e.target as HTMLInputElement).value;
          }}
          class="flex-1 px-3 py-2 bg-slate-900/50 border border-slate-700 rounded-md text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-amber-500"
        />
        <select
          value={filterType.value}
          onChange$={(e) => {
            filterType.value = (e.target as HTMLSelectElement).value as typeof filterType.value;
          }}
          class="px-3 py-2 bg-slate-900/50 border border-slate-700 rounded-md text-sm text-slate-200 focus:outline-none focus:border-amber-500"
        >
          <option value="all">All Types</option>
          <option value="algorithm">Algorithm</option>
          <option value="architecture">Architecture</option>
          <option value="optimization">Optimization</option>
          <option value="technique">Technique</option>
        </select>
      </div>

      {/* Pattern Grid */}
      <div class="flex-1 overflow-y-auto">
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredPatterns.map((pattern) => (
            <Card key={pattern.patternId} variant="elevated">
              <CardHeader class="pb-2">
                <div class="flex items-start justify-between">
                  <div class="flex-1">
                    <h3 class="text-sm font-semibold text-slate-200 mb-1">
                      {pattern.name}
                    </h3>
                    <div class="flex items-center gap-2">
                      <span class="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">
                        {pattern.techniqueType}
                      </span>
                      <span class="text-xs text-slate-500">
                        {(pattern.confidence * 100).toFixed(0)}% conf
                      </span>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent class="space-y-2">
                {/* Description */}
                <p class="text-xs text-slate-400 line-clamp-3">
                  {pattern.description}
                </p>

                {/* Key Insights */}
                {pattern.keyInsights.length > 0 && (
                  <div>
                    <div class="text-xs font-medium text-slate-300 mb-1">Key Insights</div>
                    <ul class="text-xs text-slate-400 space-y-0.5 list-disc list-inside">
                      {pattern.keyInsights.slice(0, 2).map((insight, i) => (
                        <li key={i} class="line-clamp-1">{insight}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Source */}
                <div class="text-xs text-slate-500">
                  Source: {pattern.source}
                </div>

                {/* Actions */}
                <div class="flex items-center gap-2 pt-2">
                  <button
                    onClick$={() => {
                      sotaState.selectedPattern = pattern;
                      generateMutation(pattern.patternId);
                    }}
                    class="flex-1 px-3 py-1.5 bg-amber-500/20 text-amber-300 rounded text-xs font-medium hover:bg-amber-500/30"
                  >
                    Generate Mutation
                  </button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {filteredPatterns.length === 0 && (
          <div class="flex items-center justify-center h-64 text-slate-500 text-sm">
            No patterns match your search
          </div>
        )}
      </div>

      {/* Generate Mutation Modal */}
      {sotaState.showGenerateModal && sotaState.generatedMutation && (
        <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <Card variant="elevated" class="w-full max-w-3xl max-h-[80vh] overflow-hidden">
            <CardHeader>
              <div class="flex items-center justify-between">
                <NeonTitle level="h3" color="amber" size="base">
                  Generated Mutation
                </NeonTitle>
                <button
                  onClick$={() => {
                    sotaState.showGenerateModal = false;
                  }}
                  class="text-slate-400 hover:text-slate-200"
                >
                  ✕
                </button>
              </div>
            </CardHeader>
            <CardContent class="overflow-y-auto space-y-4">
              {/* Mutation details */}
              <div class="space-y-2">
                <div class="text-sm">
                  <span class="text-slate-400">Target:</span>{' '}
                  <span class="text-slate-200">{sotaState.generatedMutation.targetFile}</span>
                </div>
                <div class="text-sm">
                  <span class="text-slate-400">Type:</span>{' '}
                  <span class="text-amber-300">{sotaState.generatedMutation.mutationType}</span>
                </div>
                <div class="text-sm text-slate-300">
                  {sotaState.generatedMutation.description}
                </div>
              </div>

              {/* Code diff */}
              <div>
                <div class="text-xs font-medium text-slate-300 mb-2">Original Code</div>
                <pre class="p-3 bg-slate-900 rounded text-xs text-slate-300 overflow-x-auto">
                  {sotaState.generatedMutation.originalCode}
                </pre>
              </div>

              <div>
                <div class="text-xs font-medium text-slate-300 mb-2">Proposed Code</div>
                <pre class="p-3 bg-slate-900 rounded text-xs text-emerald-300 overflow-x-auto">
                  {sotaState.generatedMutation.proposedCode}
                </pre>
              </div>

              {/* Rationale */}
              <div>
                <div class="text-xs font-medium text-slate-300 mb-2">Rationale</div>
                <p class="text-sm text-slate-400">
                  {sotaState.generatedMutation.rationale}
                </p>
              </div>

              {/* Actions */}
              <div class="flex items-center gap-2 pt-4">
                <button
                  onClick$={() => validateMutation(sotaState.generatedMutation!.mutationId)}
                  class="px-4 py-2 bg-emerald-500/20 text-emerald-300 rounded text-sm font-medium hover:bg-emerald-500/30"
                >
                  Validate Mutation
                </button>
                <button
                  onClick$={() => {
                    sotaState.showGenerateModal = false;
                  }}
                  class="px-4 py-2 bg-slate-700 text-slate-300 rounded text-sm font-medium hover:bg-slate-600"
                >
                  Close
                </button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Validation Result Modal */}
      {sotaState.showValidationModal && sotaState.validationResult && (
        <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <Card variant="elevated" class="w-full max-w-2xl max-h-[80vh] overflow-hidden">
            <CardHeader>
              <div class="flex items-center justify-between">
                <NeonTitle level="h3" color={sotaState.validationResult.overallPass ? 'emerald' : 'rose'} size="base">
                  Validation Result
                </NeonTitle>
                <button
                  onClick$={() => {
                    sotaState.showValidationModal = false;
                  }}
                  class="text-slate-400 hover:text-slate-200"
                >
                  ✕
                </button>
              </div>
            </CardHeader>
            <CardContent class="overflow-y-auto space-y-4">
              {/* Overall status */}
              <div class="flex items-center gap-4">
                <div class={`px-4 py-2 rounded ${
                  sotaState.validationResult.overallPass
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-red-500/20 text-red-300'
                }`}>
                  {sotaState.validationResult.overallPass ? 'PASSED' : 'FAILED'}
                </div>
                <div class="text-sm text-slate-300">
                  Score: {(sotaState.validationResult.overallScore * 100).toFixed(0)}%
                </div>
              </div>

              {/* Gate results */}
              <div>
                <div class="text-sm font-medium text-slate-300 mb-2">Gate Results</div>
                <div class="space-y-2">
                  {sotaState.validationResult.gates.map((gate, i) => (
                    <div
                      key={i}
                      class={`p-3 rounded border ${
                        gate.passed
                          ? 'bg-emerald-500/10 border-emerald-500/30'
                          : 'bg-red-500/10 border-red-500/30'
                      }`}
                    >
                      <div class="flex items-center justify-between mb-1">
                        <span class="text-sm font-medium text-slate-200">
                          {gate.gateName}
                        </span>
                        <span class="text-xs">
                          {(gate.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div class="text-xs text-slate-400">
                        {gate.details}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendation */}
              <div>
                <div class="text-sm font-medium text-slate-300 mb-2">Recommendation</div>
                <p class="text-sm text-slate-400">
                  {sotaState.validationResult.recommendation}
                </p>
              </div>

              {/* Close */}
              <div class="flex items-center justify-end pt-4">
                <button
                  onClick$={() => {
                    sotaState.showValidationModal = false;
                  }}
                  class="px-4 py-2 bg-slate-700 text-slate-300 rounded text-sm font-medium hover:bg-slate-600"
                >
                  Close
                </button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
});
