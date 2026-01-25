/**
 * SophosBoardView - Sophos (Wisdom) Integration Dashboard
 *
 * Displays the ASL-2025 Etymon Sophos interface for:
 * - Wisdom queries and responses
 * - Epistemic state visualization
 * - Integration with SOTA research pipeline
 * - Cross-agent knowledge synthesis
 *
 * Bus topics:
 * - sophos.query.request
 * - sophos.query.response
 * - sophos.wisdom.synthesized
 */

import { component$, useSignal, type Signal, type QRL } from '@builder.io/qwik';

// M3 Components - SophosBoardView
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/progress/circular-progress.js';

// ============================================================================
// Types
// ============================================================================

/** Sophos query status */
export type SophosQueryStatus = 'pending' | 'processing' | 'completed' | 'failed';

/** Wisdom source attribution */
export interface WisdomSource {
  id: string;
  type: 'sota' | 'epistemic' | 'empirical' | 'inferred';
  reference: string;
  confidence: number;
}

/** Sophos query structure */
export interface SophosQuery {
  id: string;
  question: string;
  context?: string;
  timestamp: string;
  status: SophosQueryStatus;
  sources?: WisdomSource[];
  response?: string;
  processingTimeMs?: number;
}

/** Props for SophosBoardView */
export interface SophosBoardViewProps {
  /** Active queries */
  queries: Signal<SophosQuery[]>;
  /** Callback for submitting new queries */
  onSubmitQuery$?: QRL<(question: string, context?: string) => void>;
  /** Callback for clearing history */
  onClearHistory$?: QRL<() => void>;
  /** Connection status */
  connected?: boolean;
}

// ============================================================================
// Main Component
// ============================================================================

export const SophosBoardView = component$<SophosBoardViewProps>((props) => {
  const { queries, onSubmitQuery$, connected = true } = props;
  const inputValue = useSignal('');

  const handleSubmit = $(() => {
    if (inputValue.value.trim() && onSubmitQuery$) {
      onSubmitQuery$(inputValue.value.trim());
      inputValue.value = '';
    }
  });

  return (
    <div class="space-y-4">
      {/* Header */}
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span>SOPHOS</span>
            {!connected && (
              <span class="text-xs px-2 py-0.5 rounded bg-red-500/20 text-red-400">
                Disconnected
              </span>
            )}
          </h2>
          <p class="text-sm text-muted-foreground">
            ASL-2025 Etymon Wisdom Interface
          </p>
        </div>
      </div>

      {/* Query Input */}
      <div class="flex gap-2">
        <input
          type="text"
          placeholder="Ask a wisdom query..."
          class="flex-1 px-3 py-2 rounded border border-border bg-background text-sm"
          value={inputValue.value}
          onInput$={(e) => {
            inputValue.value = (e.target as HTMLInputElement).value;
          }}
          onKeyDown$={(e) => {
            if (e.key === 'Enter') {
              handleSubmit();
            }
          }}
        />
        <button
          onClick$={handleSubmit}
          class="px-4 py-2 rounded bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
        >
          Query
        </button>
      </div>

      {/* Queries List */}
      <div class="rounded-lg border border-border bg-card">
        {queries.value.length === 0 ? (
          <div class="p-12 text-center text-muted-foreground">
            <div class="text-4xl mb-4">[S]</div>
            <div class="text-sm">No wisdom queries yet</div>
            <div class="text-xs mt-2">
              Submit a query to engage the Sophos interface
            </div>
          </div>
        ) : (
          <div class="divide-y divide-border/30 max-h-[400px] overflow-y-auto">
            {queries.value.map((query) => (
              <div key={query.id} class="p-4">
                <div class="flex items-center justify-between mb-2">
                  <span class="font-medium text-sm">{query.question}</span>
                  <span
                    class={`text-xs px-2 py-0.5 rounded ${
                      query.status === 'completed'
                        ? 'bg-green-500/20 text-green-400'
                        : query.status === 'processing'
                          ? 'bg-blue-500/20 text-blue-400'
                          : query.status === 'failed'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    {query.status}
                  </span>
                </div>
                {query.response && (
                  <p class="text-sm text-muted-foreground">{query.response}</p>
                )}
                {query.sources && query.sources.length > 0 && (
                  <div class="mt-2 flex flex-wrap gap-1">
                    {query.sources.map((src) => (
                      <span
                        key={src.id}
                        class="text-xs px-1.5 py-0.5 rounded bg-muted/50 text-muted-foreground"
                      >
                        {src.type}: {src.confidence.toFixed(0)}%
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
});

export default SophosBoardView;
