/**
 * HistorySearch - Full-text search component for lane history entries
 *
 * Phase 2, Iteration 12 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Full-text search in history notes
 * - Commit hash search (detects hex patterns like abc123)
 * - Highlighted matching text in results
 * - Result count display
 * - Debounced search input
 * - Qwik-native patterns (component$, useSignal, useComputed$)
 */

import {
  component$,
  useSignal,
  useComputed$,
  useTask$,
} from '@builder.io/qwik';

// M3 Components - HistorySearch
import '@material/web/elevation/elevation.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/iconbutton/icon-button.js';

// ============================================================================
// Types
// ============================================================================

export interface SearchableHistoryEntry {
  ts: string;
  laneId: string;
  laneName: string;
  wip_pct: number;
  note: string;
}

export interface HistorySearchProps {
  /** Array of history entries to search through */
  entries: SearchableHistoryEntry[];
  /** Placeholder text for search input */
  placeholder?: string;
  /** Debounce delay in ms (default: 200) */
  debounceMs?: number;
  /** Maximum results to display (default: 50) */
  maxResults?: number;
  /** Callback when search results change */
  onResultsChange$?: (results: SearchableHistoryEntry[]) => void;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Escape special regex characters in a string
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Check if a string looks like a commit hash (7+ hex characters)
 */
function isCommitHashPattern(str: string): boolean {
  return /^[a-f0-9]{7,40}$/i.test(str.trim());
}

/**
 * Format a timestamp into a human-readable date/time
 */
function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    if (isNaN(date.getTime())) {
      return ts.slice(0, 16).replace('T', ' ');
    }
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = months[date.getMonth()];
    const day = date.getDate();
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${month} ${day} ${hours}:${minutes}`;
  } catch {
    return ts.slice(0, 16).replace('T', ' ');
  }
}

/**
 * Get color class based on WIP percentage
 */
function wipColor(pct: number): string {
  if (pct >= 90) return 'text-emerald-400';
  if (pct >= 60) return 'text-cyan-400';
  if (pct >= 30) return 'text-amber-400';
  return 'text-red-400';
}

/**
 * Get background color class based on WIP percentage
 */
function wipBgColor(pct: number): string {
  if (pct >= 90) return 'bg-emerald-500/20 border-emerald-500/30';
  if (pct >= 60) return 'bg-cyan-500/20 border-cyan-500/30';
  if (pct >= 30) return 'bg-amber-500/20 border-amber-500/30';
  return 'bg-red-500/20 border-red-500/30';
}

/**
 * Highlight matching text in a string
 * Returns an array of JSX elements with highlighted spans
 */
function highlightMatches(text: string, query: string): (string | { highlight: string })[] {
  if (!query || !text) return [text];

  try {
    const escapedQuery = escapeRegex(query);
    const regex = new RegExp(`(${escapedQuery})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, index) => {
      if (part.toLowerCase() === query.toLowerCase()) {
        return { highlight: part };
      }
      return part;
    }).filter(p => p !== '');
  } catch {
    return [text];
  }
}

// ============================================================================
// Component
// ============================================================================

export const HistorySearch = component$<HistorySearchProps>(
  ({
    entries,
    placeholder = 'Search history notes, commit hashes...',
    debounceMs = 200,
    maxResults = 50,
    onResultsChange$,
  }) => {
    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    const searchQuery = useSignal('');
    const debouncedQuery = useSignal('');
    const isSearching = useSignal(false);

    // -------------------------------------------------------------------------
    // Debounce Effect
    // -------------------------------------------------------------------------
    useTask$(({ track, cleanup }) => {
      track(() => searchQuery.value);

      const timeoutId = setTimeout(() => {
        debouncedQuery.value = searchQuery.value;
        isSearching.value = false;
      }, debounceMs);

      isSearching.value = searchQuery.value !== debouncedQuery.value;

      cleanup(() => clearTimeout(timeoutId));
    });

    // -------------------------------------------------------------------------
    // Computed Values
    // -------------------------------------------------------------------------
    const searchResults = useComputed$(() => {
      const query = debouncedQuery.value.trim().toLowerCase();

      if (!query) {
        return { results: [], query: '', isCommitSearch: false };
      }

      const isCommitSearch = isCommitHashPattern(query);

      const results = entries.filter((entry) => {
        const noteText = entry.note.toLowerCase();

        // Basic text search in note
        if (noteText.includes(query)) {
          return true;
        }

        // If it looks like a commit hash, also search for partial matches
        // in the note that might contain commit references
        if (isCommitSearch) {
          // Match commit-like patterns in the note
          const commitPattern = /[a-f0-9]{7,40}/gi;
          const matches = noteText.match(commitPattern);
          if (matches) {
            return matches.some(m => m.includes(query));
          }
        }

        // Also search in lane name
        if (entry.laneName.toLowerCase().includes(query)) {
          return true;
        }

        return false;
      }).slice(0, maxResults);

      return { results, query, isCommitSearch };
    });

    // Notify parent of results change
    useTask$(({ track }) => {
      track(() => searchResults.value);

      if (onResultsChange$) {
        onResultsChange$(searchResults.value.results);
      }
    });

    const stats = useComputed$(() => {
      const total = entries.length;
      const resultCount = searchResults.value.results.length;
      const hasQuery = debouncedQuery.value.trim().length > 0;
      return { total, resultCount, hasQuery };
    });

    // -------------------------------------------------------------------------
    // Render
    // -------------------------------------------------------------------------
    return (
      <div class="rounded-lg border border-border bg-card">
        {/* Header */}
        <div class="flex items-center justify-between p-3 border-b border-border/50">
          <div class="flex items-center gap-2">
            <span class="text-sm font-semibold text-muted-foreground">HISTORY SEARCH</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
              {stats.value.total} entries
            </span>
          </div>
          {stats.value.hasQuery && (
            <span class={`text-[10px] px-2 py-0.5 rounded border ${
              stats.value.resultCount > 0
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                : 'bg-red-500/20 text-red-400 border-red-500/30'
            }`}>
              {stats.value.resultCount} {stats.value.resultCount === 1 ? 'match' : 'matches'}
            </span>
          )}
        </div>

        {/* Search Input */}
        <div class="p-3 border-b border-border/30">
          <div class="relative">
            <input
              type="text"
              value={searchQuery.value}
              onInput$={(e) => {
                searchQuery.value = (e.target as HTMLInputElement).value;
              }}
              placeholder={placeholder}
              class="w-full px-3 py-2 text-xs bg-black/20 border border-border/50 rounded-md text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-colors"
            />
            {/* Search icon or loading indicator */}
            <div class="absolute right-3 top-1/2 -translate-y-1/2">
              {isSearching.value ? (
                <div class="w-3 h-3 border border-primary/30 border-t-primary rounded-full animate-spin" />
              ) : (
                <svg
                  class="w-3.5 h-3.5 text-muted-foreground/50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              )}
            </div>
          </div>

          {/* Search hints */}
          {!stats.value.hasQuery && (
            <div class="mt-2 flex flex-wrap gap-1">
              <span class="text-[9px] text-muted-foreground/60">Try:</span>
              <button
                onClick$={() => { searchQuery.value = 'fix'; }}
                class="text-[9px] px-1.5 py-0.5 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground border border-border/30 transition-colors"
              >
                fix
              </button>
              <button
                onClick$={() => { searchQuery.value = 'complete'; }}
                class="text-[9px] px-1.5 py-0.5 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground border border-border/30 transition-colors"
              >
                complete
              </button>
              <button
                onClick$={() => { searchQuery.value = 'blocked'; }}
                class="text-[9px] px-1.5 py-0.5 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground border border-border/30 transition-colors"
              >
                blocked
              </button>
            </div>
          )}

          {/* Commit search indicator */}
          {searchResults.value.isCommitSearch && stats.value.hasQuery && (
            <div class="mt-2">
              <span class="text-[9px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                Commit hash search mode
              </span>
            </div>
          )}
        </div>

        {/* Results */}
        <div class="max-h-[400px] overflow-y-auto">
          {stats.value.hasQuery ? (
            searchResults.value.results.length > 0 ? (
              <div class="divide-y divide-border/30">
                {searchResults.value.results.map((entry, index) => {
                  const highlighted = highlightMatches(entry.note, searchResults.value.query);

                  return (
                    <div key={`${entry.laneId}-${entry.ts}-${index}`} class="p-3 hover:bg-muted/10 transition-colors">
                      {/* Entry header */}
                      <div class="flex items-center gap-2 mb-1">
                        <span class="text-[10px] text-muted-foreground/70 mono">
                          {formatTimestamp(entry.ts)}
                        </span>
                        <span class={`text-[10px] font-bold px-1.5 py-0.5 rounded ${wipBgColor(entry.wip_pct)} ${wipColor(entry.wip_pct)}`}>
                          {entry.wip_pct}%
                        </span>
                        <span class="text-[10px] px-1.5 py-0.5 rounded bg-muted/30 text-muted-foreground border border-border/30">
                          {entry.laneName}
                        </span>
                      </div>

                      {/* Note with highlighted matches */}
                      <div class="text-[11px] text-foreground/80 leading-relaxed">
                        {highlighted.map((part, i) => {
                          if (typeof part === 'object' && 'highlight' in part) {
                            return (
                              <mark
                                key={i}
                                class="bg-amber-500/30 text-amber-200 px-0.5 rounded"
                              >
                                {part.highlight}
                              </mark>
                            );
                          }
                          return <span key={i}>{part}</span>;
                        })}
                      </div>
                    </div>
                  );
                })}

                {/* More results indicator */}
                {searchResults.value.results.length >= maxResults && (
                  <div class="p-3 text-center">
                    <span class="text-[9px] text-muted-foreground/60">
                      Showing first {maxResults} results. Refine your search for more specific matches.
                    </span>
                  </div>
                )}
              </div>
            ) : (
              /* No results */
              <div class="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <svg
                  class="w-8 h-8 mb-2 text-muted-foreground/30"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="1.5"
                    d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <div class="text-xs mb-1">No matches found</div>
                <div class="text-[10px] text-muted-foreground/60">
                  Try a different search term or check spelling
                </div>
                <button
                  onClick$={() => { searchQuery.value = ''; }}
                  class="mt-3 text-[10px] px-3 py-1 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground border border-border/30 transition-colors"
                >
                  Clear search
                </button>
              </div>
            )
          ) : (
            /* Empty state - no query */
            <div class="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <svg
                class="w-8 h-8 mb-2 text-muted-foreground/30"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="1.5"
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
              <div class="text-xs mb-1">Search history entries</div>
              <div class="text-[10px] text-muted-foreground/60 text-center max-w-[200px]">
                Enter a search term to find matching notes, commit hashes, or lane names
              </div>
            </div>
          )}
        </div>

        {/* Footer stats */}
        {stats.value.hasQuery && stats.value.resultCount > 0 && (
          <div class="p-2 border-t border-border/50 flex items-center justify-between">
            <span class="text-[9px] text-muted-foreground/60">
              Found {stats.value.resultCount} of {stats.value.total} entries
            </span>
            <button
              onClick$={() => { searchQuery.value = ''; }}
              class="text-[9px] px-2 py-0.5 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground border border-border/30 transition-colors"
            >
              Clear
            </button>
          </div>
        )}
      </div>
    );
  }
);

export default HistorySearch;
