/**
 * FreshnessBadge - Shows data freshness based on timestamp age
 * ============================================================
 *
 * Displays the age of data with color-coded status:
 * - Fresh (green): < 60 seconds
 * - Recent (cyan): < 5 minutes
 * - Stale (amber): < 30 minutes
 * - Old (rose): > 30 minutes
 *
 * Optionally shows source label for data provenance.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';

export interface FreshnessBadgeProps {
  /** ISO 8601 timestamp string or Unix timestamp (seconds or ms) */
  timestamp?: string | number | null;
  /** Optional source label (e.g., "lanes.json", "/api/agents") */
  source?: string;
  /** Show relative time (e.g., "2m ago") instead of just status */
  showRelative?: boolean;
  /** TTL thresholds in seconds (defaults: fresh=60, recent=300, stale=1800) */
  ttlFresh?: number;
  ttlRecent?: number;
  ttlStale?: number;
  /** Additional CSS classes */
  class?: string;
}

type FreshnessLevel = 'fresh' | 'recent' | 'stale' | 'old' | 'unknown';

const freshnessColors: Record<FreshnessLevel, string> = {
  fresh: 'glass-title-neon-emerald',
  recent: 'glass-title-neon',
  stale: 'glass-title-neon-amber',
  old: 'glass-title-neon-rose',
  unknown: 'text-glass-text-muted',
};

const freshnessGlowColors: Record<FreshnessLevel, string> = {
  fresh: 'shadow-[0_0_4px_var(--glass-accent-emerald)]',
  recent: 'shadow-[0_0_4px_var(--glass-accent-cyan)]',
  stale: 'shadow-[0_0_4px_var(--glass-accent-amber)]',
  old: 'shadow-[0_0_4px_var(--glass-accent-rose)]',
  unknown: '',
};

/**
 * Parse timestamp to Unix seconds
 */
function parseTimestamp(ts: string | number | null | undefined): number | null {
  if (ts === null || ts === undefined) return null;

  if (typeof ts === 'number') {
    // If > 1e12, assume milliseconds
    return ts > 1e12 ? ts / 1000 : ts;
  }

  if (typeof ts === 'string') {
    // Try ISO 8601 parse
    const parsed = Date.parse(ts);
    if (!isNaN(parsed)) return parsed / 1000;

    // Try Unix timestamp string
    const num = parseFloat(ts);
    if (!isNaN(num)) return num > 1e12 ? num / 1000 : num;
  }

  return null;
}

/**
 * Format relative time (e.g., "2m", "1h", "3d")
 */
function formatRelativeTime(ageSeconds: number): string {
  if (ageSeconds < 0) return 'future';
  if (ageSeconds < 60) return `${Math.floor(ageSeconds)}s`;
  if (ageSeconds < 3600) return `${Math.floor(ageSeconds / 60)}m`;
  if (ageSeconds < 86400) return `${Math.floor(ageSeconds / 3600)}h`;
  return `${Math.floor(ageSeconds / 86400)}d`;
}

/**
 * Get freshness level from age
 */
function getFreshnessLevel(
  ageSeconds: number,
  ttlFresh: number,
  ttlRecent: number,
  ttlStale: number
): FreshnessLevel {
  if (ageSeconds < 0) return 'unknown';
  if (ageSeconds <= ttlFresh) return 'fresh';
  if (ageSeconds <= ttlRecent) return 'recent';
  if (ageSeconds <= ttlStale) return 'stale';
  return 'old';
}

export const FreshnessBadge = component$<FreshnessBadgeProps>((props) => {
  const {
    timestamp,
    source,
    showRelative = true,
    ttlFresh = 60,
    ttlRecent = 300,
    ttlStale = 1800,
  } = props;

  const age = useSignal<number | null>(null);
  const level = useSignal<FreshnessLevel>('unknown');

  // Update age every second
  useVisibleTask$(({ cleanup }) => {
    const update = () => {
      const ts = parseTimestamp(timestamp);
      if (ts === null) {
        age.value = null;
        level.value = 'unknown';
        return;
      }

      const now = Date.now() / 1000;
      const ageSeconds = now - ts;
      age.value = ageSeconds;
      level.value = getFreshnessLevel(ageSeconds, ttlFresh, ttlRecent, ttlStale);
    };

    update();
    const interval = setInterval(update, 1000);

    cleanup(() => clearInterval(interval));
  });

  const statusText = () => {
    if (level.value === 'unknown') return '?';
    if (!showRelative) return level.value;
    if (age.value === null) return '?';
    return formatRelativeTime(age.value);
  };

  const statusIcon = () => {
    switch (level.value) {
      case 'fresh': return '●';
      case 'recent': return '◐';
      case 'stale': return '○';
      case 'old': return '◌';
      default: return '?';
    }
  };

  return (
    <span
      class={[
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-mono',
        'border border-glass-border/30 bg-glass-surface/50',
        freshnessColors[level.value],
        freshnessGlowColors[level.value],
        props.class,
      ].filter(Boolean).join(' ')}
      title={[
        `Status: ${level.value}`,
        age.value !== null ? `Age: ${formatRelativeTime(age.value)} ago` : null,
        source ? `Source: ${source}` : null,
      ].filter(Boolean).join('\n')}
    >
      <span class="opacity-75">{statusIcon()}</span>
      <span>{statusText()}</span>
      {source && (
        <span class="text-glass-text-muted opacity-60 ml-0.5">
          [{source}]
        </span>
      )}
    </span>
  );
});

/**
 * SourceLabel - Simple source indicator without freshness
 */
export interface SourceLabelProps {
  source: string;
  class?: string;
}

export const SourceLabel = component$<SourceLabelProps>((props) => {
  return (
    <span
      class={[
        'text-[8px] text-glass-text-muted opacity-50 font-mono',
        props.class,
      ].filter(Boolean).join(' ')}
      title={`Data source: ${props.source}`}
    >
      [{props.source}]
    </span>
  );
});

/**
 * DataFreshnessIndicator - Combined freshness badge with tooltip details
 */
export interface DataFreshnessIndicatorProps {
  timestamp?: string | number | null;
  source?: string;
  label?: string;
  class?: string;
}

export const DataFreshnessIndicator = component$<DataFreshnessIndicatorProps>((props) => {
  const { timestamp, source, label } = props;

  return (
    <div
      class={[
        'inline-flex items-center gap-1.5',
        props.class,
      ].filter(Boolean).join(' ')}
    >
      {label && (
        <span class="text-[9px] text-glass-text-muted">{label}:</span>
      )}
      <FreshnessBadge timestamp={timestamp} source={source} />
    </div>
  );
});
