/**
 * PortalDestinationSelector - Content Ingestion Destination Selector
 *
 * Allows users to select ingestion targets for curated content:
 * - Knowledge base destinations
 * - SOTA catalog targets
 * - Memory bank assignments
 * - Cross-system routing
 *
 * Bus topics:
 * - portal.ingress.select
 * - portal.ingress.route
 * - portal.ingress.complete
 */

import { component$, useSignal, $, type QRL } from '@builder.io/qwik';

// M3 Components - PortalDestinationSelector
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/button/filled-tonal-button.js';

// ============================================================================
// Types
// ============================================================================

/** Ingress destination type */
export type IngressDestinationType =
  | 'knowledge_base'
  | 'sota_catalog'
  | 'memory_bank'
  | 'archive'
  | 'custom';

/** Ingress destination */
export interface IngressDestination {
  id: string;
  name: string;
  type: IngressDestinationType;
  description?: string;
  icon?: string;
  available: boolean;
  metadata?: Record<string, unknown>;
}

/** Ingress selection result */
export interface IngressSelection {
  destinationId: string;
  contentId: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

/** Props for PortalDestinationSelector */
export interface PortalDestinationSelectorProps {
  /** Content ID being ingested */
  contentId: string;
  /** Content title for display */
  contentTitle?: string;
  /** Available destinations */
  destinations?: IngressDestination[];
  /** Callback when destination is selected */
  onSelect$?: QRL<(selection: IngressSelection) => void>;
  /** Callback to cancel ingestion */
  onCancel$?: QRL<() => void>;
  /** Compact mode */
  compact?: boolean;
}

// ============================================================================
// Default Destinations
// ============================================================================

const DEFAULT_DESTINATIONS: IngressDestination[] = [
  {
    id: 'kb-main',
    name: 'Knowledge Base',
    type: 'knowledge_base',
    description: 'Primary knowledge repository',
    icon: '[K]',
    available: true,
  },
  {
    id: 'sota-catalog',
    name: 'SOTA Catalog',
    type: 'sota_catalog',
    description: 'State-of-the-art research catalog',
    icon: '[S]',
    available: true,
  },
  {
    id: 'memory-short',
    name: 'Short-term Memory',
    type: 'memory_bank',
    description: 'Ephemeral working memory',
    icon: '[M]',
    available: true,
  },
  {
    id: 'memory-long',
    name: 'Long-term Memory',
    type: 'memory_bank',
    description: 'Persistent memory storage',
    icon: '[L]',
    available: true,
  },
  {
    id: 'archive',
    name: 'Archive',
    type: 'archive',
    description: 'Cold storage archive',
    icon: '[A]',
    available: true,
  },
];

// ============================================================================
// Main Component
// ============================================================================

export const PortalDestinationSelector = component$<PortalDestinationSelectorProps>((props) => {
  const {
    contentId,
    contentTitle,
    destinations = DEFAULT_DESTINATIONS,
    onSelect$,
    onCancel$,
    compact = false,
  } = props;

  const selectedId = useSignal<string | null>(null);
  const isSubmitting = useSignal(false);

  const handleSelect = $((destId: string) => {
    selectedId.value = destId;
  });

  const handleConfirm = $(async () => {
    if (!selectedId.value || !onSelect$) return;

    isSubmitting.value = true;

    const selection: IngressSelection = {
      destinationId: selectedId.value,
      contentId,
      timestamp: new Date().toISOString(),
    };

    await onSelect$(selection);
    isSubmitting.value = false;
  });

  const typeColors: Record<IngressDestinationType, string> = {
    knowledge_base: 'border-blue-500/30 bg-blue-500/10',
    sota_catalog: 'border-purple-500/30 bg-purple-500/10',
    memory_bank: 'border-green-500/30 bg-green-500/10',
    archive: 'border-gray-500/30 bg-gray-500/10',
    custom: 'border-orange-500/30 bg-orange-500/10',
  };

  return (
    <div class={`rounded-lg border border-border bg-card ${compact ? 'p-3' : 'p-4'}`}>
      {/* Header */}
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class={`font-semibold ${compact ? 'text-sm' : ''}`}>
            Select Ingress Destination
          </h3>
          {contentTitle && (
            <p class="text-xs text-muted-foreground mt-1 truncate max-w-xs">
              {contentTitle}
            </p>
          )}
        </div>
        {onCancel$ && (
          <button
            onClick$={onCancel$}
            class="text-xs px-2 py-1 rounded bg-muted/50 text-muted-foreground hover:bg-muted transition-colors"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Destinations Grid */}
      <div class={`grid ${compact ? 'grid-cols-2 gap-2' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3'}`}>
        {destinations.map((dest) => (
          <button
            key={dest.id}
            onClick$={() => handleSelect(dest.id)}
            disabled={!dest.available}
            class={`
              p-3 rounded-lg border text-left transition-all
              ${selectedId.value === dest.id
                ? 'border-primary bg-primary/10 ring-2 ring-primary/30'
                : typeColors[dest.type]}
              ${dest.available
                ? 'hover:border-primary/50 cursor-pointer'
                : 'opacity-50 cursor-not-allowed'}
            `}
          >
            <div class="flex items-center gap-2 mb-1">
              <span class="font-mono text-xs">{dest.icon}</span>
              <span class={`font-medium ${compact ? 'text-xs' : 'text-sm'}`}>
                {dest.name}
              </span>
            </div>
            {!compact && dest.description && (
              <p class="text-xs text-muted-foreground">{dest.description}</p>
            )}
          </button>
        ))}
      </div>

      {/* Confirm Button */}
      <div class="mt-4 flex justify-end">
        <button
          onClick$={handleConfirm}
          disabled={!selectedId.value || isSubmitting.value}
          class={`
            px-4 py-2 rounded text-sm font-medium transition-colors
            ${selectedId.value && !isSubmitting.value
              ? 'bg-primary text-primary-foreground hover:bg-primary/90'
              : 'bg-muted text-muted-foreground cursor-not-allowed'}
          `}
        >
          {isSubmitting.value ? 'Ingesting...' : 'Confirm Ingress'}
        </button>
      </div>
    </div>
  );
});

export default PortalDestinationSelector;
