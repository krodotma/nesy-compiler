/**
 * SophosBoardView - Socratic Dialogue Interface for SOPHOS Intent Resolution
 *
 * Implements the exoteric interface layer described in ANATOMY_OF_A_HOLON.md:
 * - Question-driven exploration (Socratic dialectic)
 * - Thesis/Antithesis/Synthesis tracking
 * - Dialectical reasoning visualization
 * - Intent resolution display with Pentad/Sextet integration
 * - Connection to STRp request pipeline
 *
 * The board visualizes the journey from natural language to Holon instantiation,
 * showing how questions are resolved through the SOPHOS interface into
 * structured STRp requests with proper Etymon grounding.
 */

import { component$, useSignal, useStore, useVisibleTask$, $, type Signal, noSerialize, type NoSerialize } from '@builder.io/qwik';
import type { STRpRequest, BusEvent } from '../../lib/state/types';
import { createBusClient, type BusClient } from '../../lib/bus/bus-client';

// =============================================================================
// SOPHOS Types
// =============================================================================

/** Resolution status for a dialectical question */
export type ResolutionStatus =
  | 'pending'      // Question posed, not yet processed
  | 'analyzing'    // SOPHOS is analyzing intent
  | 'synthesizing' // Thesis/antithesis being reconciled
  | 'resolved'     // Synthesis achieved, STRp created
  | 'rejected'     // Question rejected (violates Sextet)
  | 'deferred';    // Needs human input or escalation

/** Dialectical position in thesis-antithesis-synthesis triad */
export interface DialecticalPosition {
  text: string;
  confidence: number;    // 0-1
  sources?: string[];    // Etymon references
  timestamp: number;
}

/** A question card in the Socratic dialogue */
export interface SophosQuestion {
  id: string;
  text: string;
  parentId?: string;              // Threading support
  depth: number;                  // Nesting level
  createdAt: number;
  status: ResolutionStatus;

  // Dialectical triad
  thesis?: DialecticalPosition;
  antithesis?: DialecticalPosition;
  synthesis?: DialecticalPosition;

  // Resolution output
  strpRequest?: STRpRequest;
  etymonRef?: string;             // e.g., "ASL-2025"
  pentadSummary?: {
    why: string;
    where: string;
    who: string;
    when: string;
    what: string;
  };
  sextetVerdict?: {
    passed: boolean;
    gates: { [key in 'P' | 'E' | 'L' | 'R' | 'Q' | 'O']?: 'pass' | 'fail' | 'warn' };
  };
}

/** Board state for the Sophos interface */
export interface SophosBoardState {
  questions: SophosQuestion[];
  activeQuestionId: string | null;
  threadExpanded: Record<string, boolean>;
  showResolved: boolean;
  filterStatus: ResolutionStatus | 'all';
}

// =============================================================================
// Subcomponents
// =============================================================================

interface QuestionCardProps {
  question: SophosQuestion;
  isActive: boolean;
  isExpanded: boolean;
  onSelect$: () => void;
  onToggleThread$: () => void;
  onAskFollowUp$: (text: string) => void;
  childCount: number;
}

/** Status indicator with appropriate styling */
const StatusBadge = component$<{ status: ResolutionStatus }>(({ status }) => {
  const styles: Record<ResolutionStatus, string> = {
    pending: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    analyzing: 'bg-blue-500/20 text-blue-400 border-blue-500/30 animate-pulse',
    synthesizing: 'bg-purple-500/20 text-purple-400 border-purple-500/30 animate-pulse',
    resolved: 'bg-green-500/20 text-green-400 border-green-500/30',
    rejected: 'bg-red-500/20 text-red-400 border-red-500/30',
    deferred: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  };

  const icons: Record<ResolutionStatus, string> = {
    pending: '?',
    analyzing: '...',
    synthesizing: '<>',
    resolved: '+',
    rejected: 'x',
    deferred: '!',
  };

  return (
    <span class={`px-2 py-0.5 rounded text-xs font-mono border ${styles[status]}`}>
      [{icons[status]}] {status}
    </span>
  );
});

/** Dialectical triad visualization */
const DialecticalTriad = component$<{
  thesis?: DialecticalPosition;
  antithesis?: DialecticalPosition;
  synthesis?: DialecticalPosition;
}>(({ thesis, antithesis, synthesis }) => {
  if (!thesis && !antithesis && !synthesis) return null;

  return (
    <div class="mt-3 space-y-2">
      {/* Thesis */}
      {thesis && (
        <div class="pl-3 border-l-2 border-blue-500/50">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-semibold text-blue-400">THESIS</span>
            <span class="text-xs text-muted-foreground">
              ({Math.round(thesis.confidence * 100)}% conf)
            </span>
          </div>
          <p class="text-sm text-foreground/80">{thesis.text}</p>
        </div>
      )}

      {/* Antithesis */}
      {antithesis && (
        <div class="pl-3 border-l-2 border-red-500/50">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-semibold text-red-400">ANTITHESIS</span>
            <span class="text-xs text-muted-foreground">
              ({Math.round(antithesis.confidence * 100)}% conf)
            </span>
          </div>
          <p class="text-sm text-foreground/80">{antithesis.text}</p>
        </div>
      )}

      {/* Synthesis - the resolution */}
      {synthesis && (
        <div class="pl-3 border-l-2 border-green-500/50 bg-green-500/5 rounded-r">
          <div class="flex items-center gap-2 mb-1 pt-1">
            <span class="text-xs font-semibold text-green-400">SYNTHESIS</span>
            <span class="text-xs text-muted-foreground">
              ({Math.round(synthesis.confidence * 100)}% conf)
            </span>
          </div>
          <p class="text-sm text-foreground/90 pb-1">{synthesis.text}</p>
          {synthesis.sources && synthesis.sources.length > 0 && (
            <div class="flex gap-1 pb-1">
              {synthesis.sources.map((src) => (
                <span key={src} class="text-xs px-1 rounded bg-cyan-500/20 text-cyan-400">
                  {src}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
});

/** Pentad summary display (The 5 Ws) */
const PentadDisplay = component$<{
  pentad?: SophosQuestion['pentadSummary'];
  sextet?: SophosQuestion['sextetVerdict'];
}>(({ pentad, sextet }) => {
  if (!pentad) return null;

  const gateColors: Record<string, string> = {
    pass: 'text-green-400',
    fail: 'text-red-400',
    warn: 'text-yellow-400',
  };

  return (
    <div class="mt-3 p-3 rounded bg-muted/20 border border-border/50">
      <div class="flex items-center justify-between mb-2">
        <h4 class="text-xs font-semibold text-primary">PENTAD (The Key)</h4>
        {sextet && (
          <span class={`text-xs font-mono ${sextet.passed ? 'text-green-400' : 'text-red-400'}`}>
            SEXTET: {sextet.passed ? 'PASS' : 'FAIL'}
          </span>
        )}
      </div>

      <div class="grid grid-cols-5 gap-2 text-xs">
        <div>
          <div class="text-muted-foreground">WHY</div>
          <div class="truncate" title={pentad.why}>{pentad.why}</div>
        </div>
        <div>
          <div class="text-muted-foreground">WHERE</div>
          <div class="truncate font-mono" title={pentad.where}>{pentad.where}</div>
        </div>
        <div>
          <div class="text-muted-foreground">WHO</div>
          <div class="truncate" title={pentad.who}>{pentad.who}</div>
        </div>
        <div>
          <div class="text-muted-foreground">WHEN</div>
          <div class="truncate" title={pentad.when}>{pentad.when}</div>
        </div>
        <div>
          <div class="text-muted-foreground">WHAT</div>
          <div class="truncate" title={pentad.what}>{pentad.what}</div>
        </div>
      </div>

      {/* Sextet gates visualization */}
      {sextet && sextet.gates && (
        <div class="flex gap-2 mt-2 pt-2 border-t border-border/30">
          {(['P', 'E', 'L', 'R', 'Q', 'O'] as const).map((gate) => {
            const verdict = sextet.gates?.[gate];
            return (
              <span
                key={gate}
                class={`text-xs font-mono ${verdict ? gateColors[verdict] : 'text-muted-foreground'}`}
                title={`${gate}-Gate: ${verdict || 'pending'}`}
              >
                [{gate}:{verdict?.[0]?.toUpperCase() || '?'}]
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
});

/** Individual question card */
const QuestionCard = component$<QuestionCardProps>(({
  question,
  isActive,
  isExpanded,
  onSelect$,
  onToggleThread$,
  childCount,
}) => {
  const followUpInput = useSignal('');

  return (
    <div
      class={`
        rounded-lg border transition-all cursor-pointer
        ${isActive
          ? 'border-primary bg-primary/5 shadow-lg shadow-primary/10'
          : 'border-border bg-card hover:border-primary/50 hover:bg-muted/20'
        }
        ${question.depth > 0 ? `ml-${Math.min(question.depth * 4, 16)}` : ''}
      `}
      style={{ marginLeft: question.depth > 0 ? `${question.depth * 16}px` : '0' }}
      onClick$={onSelect$}
    >
      <div class="p-4">
        {/* Header */}
        <div class="flex items-start justify-between gap-3 mb-2">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-mono text-muted-foreground">
              Q:{question.id.slice(0, 8)}
            </span>
            {question.depth > 0 && (
              <span class="text-xs text-purple-400">
                (follow-up d:{question.depth})
              </span>
            )}
            <StatusBadge status={question.status} />
          </div>
          <span class="text-xs text-muted-foreground whitespace-nowrap">
            {new Date(question.createdAt).toLocaleTimeString()}
          </span>
        </div>

        {/* Question text */}
        <p class="text-sm font-medium text-foreground mb-2">{question.text}</p>

        {/* Dialectical triad */}
        <DialecticalTriad
          thesis={question.thesis}
          antithesis={question.antithesis}
          synthesis={question.synthesis}
        />

        {/* Pentad/Sextet display for resolved questions */}
        {question.status === 'resolved' && (
          <PentadDisplay pentad={question.pentadSummary} sextet={question.sextetVerdict} />
        )}

        {/* STRp request link */}
        {question.strpRequest && (
          <div class="mt-2 p-2 rounded bg-blue-500/10 border border-blue-500/30">
            <div class="flex items-center justify-between">
              <span class="text-xs font-mono text-blue-400">
                STRp: {question.strpRequest.id.slice(0, 12)}
              </span>
              <span class={`text-xs px-1.5 py-0.5 rounded ${
                question.strpRequest.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                question.strpRequest.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                'bg-yellow-500/20 text-yellow-400'
              }`}>
                {question.strpRequest.status}
              </span>
            </div>
            {question.etymonRef && (
              <span class="text-xs text-cyan-400 mt-1 block">
                Etymon: {question.etymonRef}
              </span>
            )}
          </div>
        )}

        {/* Thread controls */}
        <div class="flex items-center justify-between mt-3 pt-3 border-t border-border/30">
          {childCount > 0 ? (
            <button
              onClick$={(e) => { e.stopPropagation(); onToggleThread$(); }}
              class="text-xs text-primary hover:underline"
            >
              {isExpanded ? 'Collapse' : 'Expand'} {childCount} follow-up{childCount > 1 ? 's' : ''}
            </button>
          ) : (
            <span class="text-xs text-muted-foreground">No follow-ups</span>
          )}

          {isActive && (
            <div class="flex items-center gap-2">
              <input
                type="text"
                placeholder="Ask follow-up..."
                class="text-xs px-2 py-1 rounded bg-muted/50 border border-border w-40 focus:outline-none focus:ring-1 focus:ring-primary"
                value={followUpInput.value}
                onInput$={(e) => followUpInput.value = (e.target as HTMLInputElement).value}
                onClick$={(e) => e.stopPropagation()}
              />
              <button
                onClick$={(e) => {
                  e.stopPropagation();
                  if (followUpInput.value.trim()) {
                    // The parent will handle this via onAskFollowUp$
                    // For now we emit a custom event
                    const event = new CustomEvent('sophos:followup', {
                      detail: { parentId: question.id, text: followUpInput.value },
                      bubbles: true,
                    });
                    (e.target as HTMLElement).dispatchEvent(event);
                    followUpInput.value = '';
                  }
                }}
                class="text-xs px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30"
              >
                Ask
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

// =============================================================================
// Main Component
// =============================================================================

interface SophosBoardViewProps {
  requests?: Signal<STRpRequest[]>;
  events?: Signal<BusEvent[]>;
}

export const SophosBoardView = component$<SophosBoardViewProps>(({ requests, events }) => {
  // Board state
  const state = useStore<SophosBoardState>({
    questions: [],
    activeQuestionId: null,
    threadExpanded: {},
    showResolved: true,
    filterStatus: 'all',
  });

  // New question input
  const newQuestionInput = useSignal('');
  const busClientRef = useSignal<NoSerialize<BusClient> | null>(null);
  const connected = useSignal(false);

  // Connect to bus for SOPHOS events
  useVisibleTask$(({ cleanup }) => {
    let disposed = false;
    const client = createBusClient({ platform: 'browser' });

    const run = async () => {
      try {
        await client.connect();
        if (disposed) return;
        connected.value = true;
        busClientRef.value = noSerialize(client);

        // Subscribe to SOPHOS resolution events
        client.subscribe('sophos.resolution.*', (ev: BusEvent) => {
          const data = ev.data as any;
          if (!data?.questionId) return;

          // Update question state based on event
          const idx = state.questions.findIndex(q => q.id === data.questionId);
          if (idx >= 0) {
            const q = state.questions[idx];

            if (ev.topic.includes('analyzing')) {
              state.questions[idx] = { ...q, status: 'analyzing' };
            } else if (ev.topic.includes('thesis')) {
              state.questions[idx] = {
                ...q,
                thesis: {
                  text: data.text,
                  confidence: data.confidence || 0.7,
                  sources: data.sources,
                  timestamp: Date.now(),
                },
              };
            } else if (ev.topic.includes('antithesis')) {
              state.questions[idx] = {
                ...q,
                antithesis: {
                  text: data.text,
                  confidence: data.confidence || 0.6,
                  sources: data.sources,
                  timestamp: Date.now(),
                },
                status: 'synthesizing',
              };
            } else if (ev.topic.includes('synthesis')) {
              state.questions[idx] = {
                ...q,
                synthesis: {
                  text: data.text,
                  confidence: data.confidence || 0.8,
                  sources: data.sources,
                  timestamp: Date.now(),
                },
                status: 'resolved',
                strpRequest: data.strpRequest,
                etymonRef: data.etymonRef,
                pentadSummary: data.pentad,
                sextetVerdict: data.sextet,
              };
            } else if (ev.topic.includes('rejected')) {
              state.questions[idx] = {
                ...q,
                status: 'rejected',
                sextetVerdict: { passed: false, gates: data.gates },
              };
            }
          }
        });

        // Subscribe to STRp updates if we have request tracking
        client.subscribe('strp.request.*', (ev: BusEvent) => {
          const data = ev.data as any;
          if (!data?.id) return;

          // Update any questions linked to this STRp request
          state.questions.forEach((q, i) => {
            if (q.strpRequest?.id === data.id) {
              state.questions[i] = {
                ...q,
                strpRequest: { ...q.strpRequest!, status: data.status },
              };
            }
          });
        });

      } catch (err) {
        console.warn('[SophosBoard] Bus connection failed:', err);
        connected.value = false;
      }
    };

    run();

    cleanup(() => {
      disposed = true;
      try { client.disconnect(); } catch { /* ignore */ }
      connected.value = false;
      busClientRef.value = null;
    });
  });

  // Ask a new question
  const askQuestion = $((text: string, parentId?: string) => {
    if (!text.trim()) return;

    const parent = parentId ? state.questions.find(q => q.id === parentId) : null;
    const question: SophosQuestion = {
      id: `sophos-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      text: text.trim(),
      parentId,
      depth: parent ? parent.depth + 1 : 0,
      createdAt: Date.now(),
      status: 'pending',
    };

    state.questions = [...state.questions, question];
    state.activeQuestionId = question.id;

    // Emit to bus for SOPHOS processing
    const client = busClientRef.value;
    if (client) {
      client.publish({
        topic: 'sophos.question.posed',
        kind: 'request',
        level: 'info',
        actor: 'sophos-board',
        data: {
          questionId: question.id,
          text: question.text,
          parentId: question.parentId,
          depth: question.depth,
        },
      });
    }

    // Simulate dialectical progression for demo purposes
    // In production, this would come from the SOPHOS backend
    setTimeout(() => {
      const idx = state.questions.findIndex(q => q.id === question.id);
      if (idx >= 0) {
        state.questions[idx] = { ...state.questions[idx], status: 'analyzing' };
      }
    }, 500);

    newQuestionInput.value = '';
  });

  // Filter questions based on current filters
  const filteredQuestions = () => {
    let qs = state.questions;

    if (!state.showResolved) {
      qs = qs.filter(q => q.status !== 'resolved');
    }

    if (state.filterStatus !== 'all') {
      qs = qs.filter(q => q.status === state.filterStatus);
    }

    return qs;
  };

  // Get root-level questions (no parent)
  const rootQuestions = () => filteredQuestions().filter(q => !q.parentId);

  // Get children of a question
  const getChildren = (parentId: string) =>
    filteredQuestions().filter(q => q.parentId === parentId);

  // Render questions recursively
  const renderQuestionTree = (questions: SophosQuestion[]): any[] => {
    const result: any[] = [];

    for (const q of questions) {
      const children = getChildren(q.id);
      const isExpanded = state.threadExpanded[q.id] ?? true;

      result.push(
        <QuestionCard
          key={q.id}
          question={q}
          isActive={state.activeQuestionId === q.id}
          isExpanded={isExpanded}
          onSelect$={() => state.activeQuestionId = q.id}
          onToggleThread$={() => {
            state.threadExpanded = {
              ...state.threadExpanded,
              [q.id]: !isExpanded,
            };
          }}
          onAskFollowUp$={(text: string) => askQuestion(text, q.id)}
          childCount={children.length}
        />
      );

      if (isExpanded && children.length > 0) {
        result.push(...renderQuestionTree(children));
      }
    }

    return result;
  };

  // Stats computation
  const stats = () => {
    const qs = state.questions;
    return {
      total: qs.length,
      pending: qs.filter(q => q.status === 'pending').length,
      analyzing: qs.filter(q => q.status === 'analyzing' || q.status === 'synthesizing').length,
      resolved: qs.filter(q => q.status === 'resolved').length,
      rejected: qs.filter(q => q.status === 'rejected').length,
    };
  };

  return (
    <div class="h-full flex flex-col">
      {/* Header */}
      <div class="flex-none p-4 border-b border-border bg-card/50">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-3">
            <h2 class="text-lg font-semibold">Sophos Board</h2>
            <span class={`w-2 h-2 rounded-full ${connected.value ? 'bg-green-400' : 'bg-gray-400'}`} />
            <span class="text-xs text-muted-foreground">Socratic Dialogue Interface</span>
          </div>
          <div class="flex items-center gap-2">
            <label class="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={state.showResolved}
                onChange$={(e) => state.showResolved = (e.target as HTMLInputElement).checked}
                class="rounded"
              />
              Show Resolved
            </label>
            <select
              value={state.filterStatus}
              onChange$={(e) => state.filterStatus = (e.target as HTMLSelectElement).value as any}
              class="text-xs px-2 py-1 rounded bg-muted/50 border border-border"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="analyzing">Analyzing</option>
              <option value="synthesizing">Synthesizing</option>
              <option value="resolved">Resolved</option>
              <option value="rejected">Rejected</option>
              <option value="deferred">Deferred</option>
            </select>
          </div>
        </div>

        {/* Stats bar */}
        <div class="grid grid-cols-5 gap-2 mb-4">
          <div class="rounded bg-muted/30 p-2 text-center">
            <div class="text-xl font-bold text-primary">{stats().total}</div>
            <div class="text-xs text-muted-foreground">Total</div>
          </div>
          <div class="rounded bg-gray-500/10 p-2 text-center">
            <div class="text-xl font-bold text-gray-400">{stats().pending}</div>
            <div class="text-xs text-muted-foreground">Pending</div>
          </div>
          <div class="rounded bg-blue-500/10 p-2 text-center">
            <div class="text-xl font-bold text-blue-400">{stats().analyzing}</div>
            <div class="text-xs text-muted-foreground">In Progress</div>
          </div>
          <div class="rounded bg-green-500/10 p-2 text-center">
            <div class="text-xl font-bold text-green-400">{stats().resolved}</div>
            <div class="text-xs text-muted-foreground">Resolved</div>
          </div>
          <div class="rounded bg-red-500/10 p-2 text-center">
            <div class="text-xl font-bold text-red-400">{stats().rejected}</div>
            <div class="text-xs text-muted-foreground">Rejected</div>
          </div>
        </div>

        {/* New question input */}
        <div class="flex gap-2">
          <input
            type="text"
            placeholder="Pose a Socratic question... (e.g., 'Why should swarms be avoided for implementation tasks?')"
            class="flex-1 px-4 py-2 rounded-lg bg-muted/50 border border-border focus:outline-none focus:ring-2 focus:ring-primary"
            value={newQuestionInput.value}
            onInput$={(e) => newQuestionInput.value = (e.target as HTMLInputElement).value}
            onKeyDown$={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion(newQuestionInput.value);
              }
            }}
          />
          <button
            onClick$={() => askQuestion(newQuestionInput.value)}
            class="px-6 py-2 rounded-lg bg-primary/20 text-primary hover:bg-primary/30 transition-colors font-medium"
            disabled={!newQuestionInput.value.trim()}
          >
            Ask
          </button>
        </div>
      </div>

      {/* Question cards */}
      <div class="flex-1 overflow-auto p-4 space-y-3">
        {state.questions.length === 0 ? (
          <div class="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
            <div class="text-6xl mb-4">?</div>
            <h3 class="text-lg font-medium mb-2">No Questions Yet</h3>
            <p class="text-sm max-w-md">
              The Sophos Board facilitates Socratic dialogue for intent resolution.
              Pose a question to begin the dialectical process of thesis, antithesis, and synthesis.
            </p>
            <div class="mt-6 space-y-2 text-xs text-left">
              <p class="text-muted-foreground/70">Example questions:</p>
              <ul class="list-disc list-inside space-y-1">
                <li>"What topology is best for code implementation?"</li>
                <li>"Why does ASL-2025 mandate single-agent for complex tasks?"</li>
                <li>"How should the Sextet validate file write operations?"</li>
              </ul>
            </div>
          </div>
        ) : (
          renderQuestionTree(rootQuestions())
        )}
      </div>

      {/* Footer with legend */}
      <div class="flex-none p-3 border-t border-border bg-card/30">
        <div class="flex items-center justify-between text-xs text-muted-foreground">
          <div class="flex items-center gap-4">
            <span>Legend:</span>
            <span class="flex items-center gap-1">
              <span class="w-2 h-2 rounded-full bg-blue-400"></span> Thesis
            </span>
            <span class="flex items-center gap-1">
              <span class="w-2 h-2 rounded-full bg-red-400"></span> Antithesis
            </span>
            <span class="flex items-center gap-1">
              <span class="w-2 h-2 rounded-full bg-green-400"></span> Synthesis
            </span>
          </div>
          <div class="flex items-center gap-2">
            <span>Sextet Gates:</span>
            <span class="font-mono">[P]rovenance [E]ffects [L]iveness [R]ecovery [Q]uality [O]mega</span>
          </div>
        </div>
      </div>
    </div>
  );
});

export default SophosBoardView;
