/**
 * NotificationDetailOverlay.tsx
 * =============================
 * Full overlay for viewing complete event details.
 */

import {
  component$,
  $,
  useComputed$,
  useSignal,
  useVisibleTask$,
  noSerialize,
  type NoSerialize,
  type QRL,
} from '@builder.io/qwik';
import type { BusEvent } from '../lib/state/types';
import { createBusClient, type BusClient } from '../lib/bus/bus-client';
import { useTracking } from '../lib/telemetry/use-tracking';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { Input } from './ui/Input';

// M3 Components - NotificationDetailOverlay
import '@material/web/elevation/elevation.js';
import '@material/web/dialog/dialog.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/iconbutton/icon-button.js';

export interface NotificationDetailOverlayProps {
  open: boolean;
  event: BusEvent | null;
  onClose$?: QRL<() => void>;
  onNavigateToActor$?: QRL<(actor: string) => void>;
  onNavigateToTopic$?: QRL<(topic: string) => void>;
}

interface ActionPlanItem {
  id?: string;
  label?: string;
  summary?: string;
  command?: string;
  risk?: string;
  requires_root?: boolean;
}

interface ActionPlan {
  id?: string;
  summary?: string;
  actions?: ActionPlanItem[];
  requires_approval?: boolean;
}

/**
 * Format timestamp for display
 */
function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch {
    return iso;
  }
}

/**
 * Get level badge style
 */
function getLevelStyle(level: string): string {
  switch (level) {
    case 'error': return 'bg-md-error/10 text-md-error border-md-error/20';
    case 'warn': return 'bg-md-warning/10 text-md-warning border-md-warning/20';
    case 'info': return 'bg-md-primary/10 text-md-primary border-md-primary/20';
    default: return 'bg-md-surface-variant text-md-on-surface-variant border-md-outline/20';
  }
}

export const NotificationDetailOverlay = component$<NotificationDetailOverlayProps>(({
  open,
  event,
  onClose$,
  onNavigateToActor$,
  onNavigateToTopic$,
}) => {
  useTracking('comp:notification-detail-overlay');

  const busClientRef = useSignal<NoSerialize<BusClient> | null>(null);
  const busConnected = useSignal(false);
  const actionNote = useSignal('');
  const actionStatus = useSignal<'idle' | 'sending' | 'approved' | 'rejected' | 'error'>('idle');

  useVisibleTask$(({ track, cleanup }) => {
    track(() => open);
    if (!open) return;
    const client = createBusClient({ platform: 'browser' });
    busClientRef.value = noSerialize(client);
    client.connect().then(() => { busConnected.value = true; }).catch(() => { busConnected.value = false; });
    cleanup(() => {
      client.disconnect();
      busClientRef.value = null;
      busConnected.value = false;
      actionStatus.value = 'idle';
      actionNote.value = '';
    });
  });

  const handleClose = $(() => { onClose$?.(); });
  const handleNavigateToActor = $((actor: string) => { onNavigateToActor$?.(actor); handleClose(); });
  const handleNavigateToTopic = $((topic: string) => { onNavigateToTopic$?.(topic); handleClose(); });

  const actionPlan = useComputed$<ActionPlan | null>(() => {
    const data = event?.data as Record<string, unknown> | undefined;
    if (!data) return null;
    const plan = (data.action_plan || data.actionPlan) as ActionPlan | undefined;
    if (!plan || typeof plan !== 'object') return null;
    return { ...plan, actions: Array.isArray(plan.actions) ? plan.actions : [] };
  });

  const publishDecision = $(async (decision: 'approved' | 'rejected') => {
    const plan = actionPlan.value;
    if (!plan || !event) return;
    const client = busClientRef.value as unknown as BusClient | null;
    if (!client) { actionStatus.value = 'error'; return; }
    actionStatus.value = 'sending';
    try {
      await client.publish({
        topic: `qa.action.review.${decision}`,
        kind: 'request',
        level: decision === 'approved' ? 'info' : 'warn',
        actor: 'dashboard',
        trace_id: event.trace_id,
        parent_id: event.id,
        data: {
          action_plan_id: plan.id || event.id,
          decision,
          note: actionNote.value.trim() || null,
          source_event: { id: event.id, topic: event.topic },
          actions: plan.actions || [],
        },
      });
      actionStatus.value = decision;
    } catch { actionStatus.value = 'error'; }
  });

  const eventDataStr = useComputed$(() => {
    if (!event?.data) return null;
    try { return JSON.stringify(event.data, null, 2); } catch { return String(event.data); }
  });

  if (!open || !event) return null;

  return (
    <div class="fixed inset-0 z-[65] flex items-center justify-center p-4 md:p-8 lg:p-12">
      <div class="absolute inset-0 bg-md-scrim/60 backdrop-blur-sm" onClick$={handleClose} />
      
      <Card variant="elevated" padding="p-0" class="relative w-full h-full max-w-6xl flex flex-col bg-md-surface overflow-hidden shadow-2xl rounded-3xl border border-md-outline/10">
        <md-elevation></md-elevation>
        
        {/* Header */}
        <div class="p-6 border-b border-md-outline/10 flex items-center justify-between bg-md-surface-container-low">
          <div class="flex-1 min-w-0">
            <h2 class="text-xl font-black tracking-tight text-md-on-surface truncate">{event.topic}</h2>
            <div class="flex items-center gap-3 mt-1.5">
              <span class="text-[10px] font-mono font-bold text-md-on-surface-variant/60 uppercase tracking-widest">{formatTimestamp(event.iso)}</span>
              <div class={`px-2 py-0.5 rounded-full text-[9px] font-black uppercase tracking-tighter border ${getLevelStyle(event.level)}`}>
                {event.level}
              </div>
              {event.impact && (
                <div class="px-2 py-0.5 rounded-full text-[9px] font-black uppercase tracking-tighter bg-md-error/10 text-md-error border border-md-error/20">
                  {event.impact} IMPACT
                </div>
              )}
            </div>
          </div>
          <Button variant="tonal" icon="close" onClick$={handleClose} class="h-10 w-10 min-w-0 rounded-full" />
        </div>

        {/* Content */}
        <div class="flex-1 overflow-y-auto p-6 space-y-8 no-scrollbar bg-md-surface-container-lowest">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Source Info */}
            <Card variant="outlined" class="bg-md-surface">
              <div class="text-[10px] font-black uppercase tracking-[0.2em] text-md-primary mb-4">Source Origin</div>
              <div class="flex items-center gap-4">
                <Button variant="tonal" onClick$={() => handleNavigateToActor(event.actor)} class="h-12 rounded-2xl">
                  <span class="text-lg mr-2">🤖</span>
                  <span class="font-black uppercase tracking-tighter">@{event.actor}</span>
                </Button>
                <div class="flex flex-col">
                  <span class="text-[9px] font-bold text-md-on-surface-variant/40 uppercase">Kind</span>
                  <span class="text-xs font-black text-md-on-surface uppercase">{event.kind}</span>
                </div>
              </div>
            </Card>

            {/* Trace Info */}
            <Card variant="outlined" class="bg-md-surface">
              <div class="text-[10px] font-black uppercase tracking-[0.2em] text-md-secondary mb-4">Orchestration Trace</div>
              <div class="space-y-2">
                {event.trace_id && (
                  <div class="flex items-center justify-between bg-md-surface-container px-3 py-2 rounded-xl border border-md-outline/5">
                    <span class="text-[9px] font-bold text-md-on-surface-variant/40 uppercase">Trace ID</span>
                    <span class="text-[10px] font-mono font-bold text-md-secondary truncate ml-4">{event.trace_id}</span>
                  </div>
                )}
                {event.parent_id && (
                  <div class="flex items-center justify-between bg-md-surface-container px-3 py-2 rounded-xl border border-md-outline/5">
                    <span class="text-[9px] font-bold text-md-on-surface-variant/40 uppercase">Parent ID</span>
                    <span class="text-[10px] font-mono font-bold text-md-on-surface-variant/60 truncate ml-4">{event.parent_id}</span>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Core Semantic Data */}
          <div class="space-y-6">
            {event.semantic && (
              <div class="space-y-2">
                <div class="text-[10px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/40 ml-2">Semantic Assertion</div>
                <Card variant="filled" class="bg-md-primary/5 border border-md-primary/10 text-md-on-surface text-lg font-medium leading-relaxed italic p-6">
                  "{event.semantic}"
                </Card>
              </div>
            )}

            {event.reasoning && (
              <div class="space-y-2">
                <div class="text-[10px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/40 ml-2">Internal Reasoning</div>
                <div class="text-sm text-md-on-surface-variant leading-relaxed bg-md-surface-container-low p-4 rounded-2xl border border-md-outline/5">
                  {event.reasoning}
                </div>
              </div>
            )}
          </div>

          {/* Action Review Form */}
          {actionPlan.value && (
            <Card variant="elevated" class="bg-md-surface border-t-4 border-t-md-warning p-6 shadow-lg">
              <div class="flex items-center gap-2 mb-6">
                <span class="text-2xl">⚡</span>
                <div class="text-[11px] font-black uppercase tracking-[0.2em] text-md-warning">Human-in-the-Loop Review Required</div>
              </div>
              
              <div class="space-y-4 mb-8">
                {actionPlan.value.actions?.map((action, idx) => (
                  <div key={idx} class="p-4 rounded-2xl bg-md-surface-container-high border border-md-outline/10">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-sm font-black text-md-primary uppercase tracking-tight">{action.label || 'Action Node'}</span>
                      {action.risk && (
                        <span class="text-[9px] font-black px-2 py-0.5 rounded-full bg-md-error/10 text-md-error border border-md-error/20 uppercase">Risk: {action.risk}</span>
                      )}
                    </div>
                    {action.summary && <div class="text-xs text-md-on-surface font-medium mb-3">{action.summary}</div>}
                    {action.command && <pre class="text-[10px] font-mono bg-black/40 p-3 rounded-xl text-md-secondary overflow-x-auto">{action.command}</pre>}
                  </div>
                ))}
              </div>

              <div class="bg-md-surface-container-lowest p-6 rounded-3xl border border-md-outline/10 space-y-4">
                <Input
                  type="textarea"
                  label="Reviewer Context / Decision Note"
                  placeholder="Provide feedback for the next agent iteration..."
                  value={actionNote.value}
                  onInput$={(_, el) => { actionNote.value = el.value; }}
                />
                <div class="flex items-center gap-3">
                  <Button 
                    variant="primary" 
                    class={`h-12 px-8 rounded-2xl ${actionStatus.value === 'approved' ? 'bg-md-success' : ''}`}
                    disabled={!busConnected.value || actionStatus.value === 'sending'}
                    onClick$={() => publishDecision('approved')}
                  >
                    Authorize Action
                  </Button>
                  <Button 
                    variant="tonal" 
                    class="h-12 px-8 rounded-2xl"
                    disabled={!busConnected.value || actionStatus.value === 'sending'}
                    onClick$={() => publishDecision('rejected')}
                  >
                    Reject Plan
                  </Button>
                  <div class="ml-auto flex items-center gap-2">
                    <div class={`w-2 h-2 rounded-full ${busConnected.value ? 'bg-md-success' : 'bg-md-error'}`} />
                    <span class="text-[10px] font-black uppercase tracking-widest opacity-40">{busConnected.value ? 'Live Link' : 'Bus Offline'}</span>
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* Raw Data Trace */}
          <div class="space-y-2">
            <div class="text-[10px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/40 ml-2">Raw NDJSON Object</div>
            <pre class="text-[10px] font-mono bg-md-surface-container-highest/50 p-6 rounded-3xl border border-md-outline/10 overflow-auto max-h-96 no-scrollbar text-md-on-surface-variant/80">
              {eventDataStr.value}
            </pre>
          </div>
        </div>

        {/* Footer */}
        <div class="p-4 border-t border-md-outline/10 bg-md-surface-container-low flex items-center justify-between">
          <span class="text-[10px] font-black uppercase tracking-widest text-md-on-surface-variant/40 ml-2">Event UID: {event.id || 'anonymous'}</span>
          <div class="flex items-center gap-4 text-[10px] font-bold text-md-on-surface-variant/30 uppercase tracking-tighter">
            <span>Press [Esc] to close</span>
            <div class="w-1 h-1 rounded-full bg-md-outline/20" />
            <span>PBMINT System V4</span>
          </div>
        </div>
      </Card>
    </div>
  );
});

export default NotificationDetailOverlay;