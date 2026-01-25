import { component$, useComputed$ } from '@builder.io/qwik';
import type { BusEvent } from '../../lib/state/types';
import { NeonTitle, NeonBadge } from '../ui/NeonTitle';

interface PortalInceptionPanelProps {
  events: BusEvent[];
  limit?: number;
}

const INCEPTION_PREFIXES = ['portal.inception'];

function pickString(data: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = data[key];
    if (typeof value === 'string' && value.trim().length > 0) {
      return value.trim();
    }
  }
  return undefined;
}

function summarizeEvent(event: BusEvent): string {
  const data = (event.data || {}) as Record<string, unknown>;
  return (
    pickString(data, ['goal', 'plan', 'intent', 'summary', 'message']) ||
    event.topic
  );
}

function eventLabel(event: BusEvent): string {
  const data = (event.data || {}) as Record<string, unknown>;
  return (
    pickString(data, ['req_id', 'inception_id', 'ingress_id', 'fragment_id']) ||
    event.id ||
    event.topic
  );
}

export const PortalInceptionPanel = component$<PortalInceptionPanelProps>(({ events, limit = 6 }) => {
  const inceptionEvents = useComputed$(() => {
    const matches = events.filter((event) =>
      INCEPTION_PREFIXES.some((prefix) => event.topic.startsWith(prefix))
    );
    return matches.slice(-limit).reverse();
  });

  return (
    <div class="glass-surface-elevated p-4 glass-hover-lift">
      <md-elevation></md-elevation>
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <NeonTitle level="h3" color="purple" size="sm" animation="flicker">
            Inception
          </NeonTitle>
          <NeonBadge color="purple" glow>
            {inceptionEvents.value.length}
          </NeonBadge>
        </div>
        <span class="text-[10px] text-muted-foreground">portal.inception.*</span>
      </div>

      {inceptionEvents.value.length === 0 ? (
        <div class="text-xs text-muted-foreground">No inception events yet.</div>
      ) : (
        <div class="space-y-2">
          {inceptionEvents.value.map((event, idx) => (
            <div
              key={`${event.id || event.ts}-${idx}`}
              class="rounded-md border border-border/50 bg-muted/20 p-2"
            >
              <div class="flex items-center justify-between text-[10px]">
                <span class="font-mono text-foreground/80">{eventLabel(event)}</span>
                <span class="text-muted-foreground">
                  {event.iso?.slice(11, 19) || ''}
                </span>
              </div>
              <div class="text-[10px] text-muted-foreground">{event.actor}</div>
              <div class="text-xs text-foreground/80 line-clamp-2">
                {summarizeEvent(event)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

export default PortalInceptionPanel;
