/**
 * NotificationCard.tsx
 * ====================
 * Rich notification card component for displaying bus events.
 * Features:
 * - Compressed summary (1-2 lines)
 * - Relative timestamp
 * - Source actor badge
 * - Event kind indicator
 * - Impact level indicator
 * - Expand on hover (fisheye effect support)
 * - View Details button
 */

import { component$, $, type QRL } from '@builder.io/qwik';
import type { BusEvent, ImpactLevel } from '../lib/state/types';
import { Button } from './ui/Button';
import { Card } from './ui/Card';

// M3 Components - NotificationCard
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/text-button.js';

export interface NotificationCardProps {
  event: BusEvent;
  scale?: number; // 0-1 for fisheye effect (1 = center/largest)
  onViewDetails$?: QRL<(event: BusEvent) => void>;
}

function formatRelativeTime(iso: string): string {
  const now = Date.now();
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return 'unknown';
  const diffMs = now - then;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);
  if (diffSec < 5) return 'just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${diffDay}d ago`;
}

function getActorIcon(actor: string | undefined): string {
  const lower = (actor || 'unknown').toLowerCase();
  if (lower.includes('gemini')) return 'G';
  if (lower.includes('claude')) return 'C';
  if (lower.includes('gpt')) return 'O';
  if (lower.includes('webllm')) return 'L';
  if (lower.includes('dashboard')) return 'D';
  if (lower.includes('codex')) return 'X';
  return (actor || '?').charAt(0).toUpperCase();
}

function getKindColor(kind: string): string {
  switch (kind) {
    case 'error': return 'text-md-error';
    case 'metric': return 'text-md-tertiary';
    case 'artifact': return 'text-md-secondary';
    case 'response': return 'text-md-success';
    case 'request': return 'text-md-primary';
    default: return 'text-md-on-surface-variant';
  }
}

function summarizeEvent(event: BusEvent): string {
  if (event.semantic) return event.semantic.length > 80 ? event.semantic.slice(0, 77) + '...' : event.semantic;
  const data = event.data as Record<string, unknown> | undefined;
  if (data) {
    if (typeof data.message === 'string') return data.message.length > 80 ? data.message.slice(0, 77) + '...' : data.message;
    if (typeof data.summary === 'string') return data.summary.length > 80 ? data.summary.slice(0, 77) + '...' : data.summary;
  }
  return event.topic.replace(/\./g, ' > ');
}

export const NotificationCard = component$<NotificationCardProps>(({
  event,
  scale = 0.8,
  onViewDetails$,
}) => {
  const summary = summarizeEvent(event);
  const relTime = formatRelativeTime(event.iso);
  const actorIcon = getActorIcon(event.actor);
  const kindColor = getKindColor(event.kind);

  const isExpanded = scale > 0.9;
  const paddingClass = isExpanded ? 'p-3' : 'p-2';
  const textSizeClass = isExpanded ? 'text-xs' : 'text-[11px]';

  const handleViewDetails = $(() => {
    onViewDetails$?.(event);
  });

  return (
    <Card
      variant={isExpanded ? 'elevated' : 'outlined'}
      padding={paddingClass}
      interactive
      onClick$={handleViewDetails}
      class={`transition-all duration-200 ease-out border-l-4 ${
        event.level === 'error' ? 'border-l-md-error' :
        event.level === 'warn' ? 'border-l-md-warning' :
        'border-l-md-outline/20'
      }`}
      style={{
        transform: `scale(${0.85 + scale * 0.15})`,
        opacity: 0.7 + scale * 0.3,
      }}
    >
      <div class="flex items-center justify-between gap-2 mb-1.5">
        <div class="flex items-center gap-2 min-w-0">
          <div class={`w-5 h-5 rounded-full flex items-center justify-center bg-md-surface-container-highest text-[10px] font-bold border border-md-outline/20`}>
            {actorIcon}
          </div>
          <span class="text-[10px] font-bold text-md-on-surface-variant truncate uppercase tracking-tighter">@{event.actor}</span>
        </div>
        <span class="text-[9px] font-mono text-md-on-surface-variant/50">{relTime}</span>
      </div>

      <div class={`${textSizeClass} font-bold tracking-tight mb-0.5 ${kindColor}`}>
        {event.topic}
      </div>

      <div class={`${textSizeClass} text-md-on-surface/80 leading-relaxed line-clamp-2`}>
        {summary}
      </div>

      <div class="flex items-center justify-between mt-2 pt-2 border-t border-md-outline/5">
        <div class="flex items-center gap-1.5">
          <span class={`text-[9px] font-black uppercase px-1.5 py-0.5 rounded-full bg-md-surface-container border border-md-outline/10 ${kindColor}`}>
            {event.kind}
          </span>
          {event.impact && event.impact !== 'low' && (
            <span class="text-[9px] font-black text-md-error uppercase">Impact: {event.impact}</span>
          )}
        </div>
        <Button
          variant="tonal"
          class={`h-6 text-[9px] px-2 min-w-0 opacity-0 group-hover:opacity-100 ${isExpanded ? 'opacity-100' : ''}`}
          onClick$={handleViewDetails}
        >
          Details
        </Button>
      </div>
    </Card>
  );
});

export default NotificationCard;