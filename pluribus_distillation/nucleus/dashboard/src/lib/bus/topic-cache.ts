import type { BusEvent } from '../state/types';

export interface TopicCacheOptions {
  maxEntries: number;
  maxEventBytes: number;
}

export const DEFAULT_TOPIC_CACHE_OPTIONS: TopicCacheOptions = {
  maxEntries: 2000,
  maxEventBytes: 64 * 1024,
};

function safeJsonByteLength(value: unknown): number {
  try {
    const json = JSON.stringify(value);
    if (typeof TextEncoder !== 'undefined') {
      return new TextEncoder().encode(json).length;
    }
    const buf = (globalThis as any)?.Buffer;
    if (buf?.byteLength) return buf.byteLength(json, 'utf8');
    return json.length;
  } catch {
    return Number.POSITIVE_INFINITY;
  }
}

export function normalizeBusEventForCache(event: BusEvent, maxEventBytes: number): BusEvent {
  const maxBytes = Math.max(1024, Math.floor(Number(maxEventBytes) || DEFAULT_TOPIC_CACHE_OPTIONS.maxEventBytes));
  const approxBytes = safeJsonByteLength(event);
  if (Number.isFinite(approxBytes) && approxBytes <= maxBytes) return event;

  const data = (event as any)?.data;
  const dataType = data === null ? 'null' : Array.isArray(data) ? 'array' : typeof data;
  const keys = data && typeof data === 'object' && !Array.isArray(data) ? Object.keys(data).slice(0, 25) : [];

  return {
    ...event,
    data: {
      truncated: true,
      approx_bytes: Number.isFinite(approxBytes) ? approxBytes : null,
      data_type: dataType,
      keys,
    },
  } as BusEvent;
}

export function rememberLastByTopic(
  cache: Map<string, BusEvent>,
  event: BusEvent,
  options: TopicCacheOptions = DEFAULT_TOPIC_CACHE_OPTIONS,
): void {
  if (!event || typeof (event as any).topic !== 'string') return;

  const maxEntriesRaw = Number(options.maxEntries);
  const maxEntries = Number.isFinite(maxEntriesRaw) ? Math.max(0, Math.floor(maxEntriesRaw)) : DEFAULT_TOPIC_CACHE_OPTIONS.maxEntries;
  if (maxEntries <= 0) return;

  const normalized = normalizeBusEventForCache(event, options.maxEventBytes);
  const topic = String((event as any).topic);

  if (cache.has(topic)) cache.delete(topic);
  cache.set(topic, normalized);

  while (cache.size > maxEntries) {
    const first = cache.keys().next().value as string | undefined;
    if (!first) break;
    cache.delete(first);
  }
}
