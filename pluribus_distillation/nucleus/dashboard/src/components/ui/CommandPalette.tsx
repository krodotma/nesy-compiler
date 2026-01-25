/**
 * CommandPalette - Keyboard-driven Command Interface
 * Phase 4, Step 118 - Menu & Navigation Refinement
 *
 * Cmd+K style command palette with fuzzy search
 */

import { component$, useSignal, useVisibleTask$, useComputed$, type PropFunction } from '@builder.io/qwik';

// M3 Components
import '@material/web/list/list.js';
import '@material/web/list/list-item.js';

export interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon?: string;
  shortcut?: string;
  category?: string;
  keywords?: string[];
  action?: () => void;
}

export interface CommandPaletteProps {
  /** Available commands */
  commands: CommandItem[];
  /** Whether the palette is open */
  open?: boolean;
  /** Callback when palette closes */
  onClose$?: PropFunction<() => void>;
  /** Callback when command is selected */
  onSelect$?: PropFunction<(command: CommandItem) => void>;
  /** Placeholder text */
  placeholder?: string;
  /** Maximum visible results */
  maxResults?: number;
  /** Additional class names */
  class?: string;
}

/**
 * Simple fuzzy matching function
 */
function fuzzyMatch(query: string, text: string): boolean {
  if (!query) return true;
  const lowerQuery = query.toLowerCase();
  const lowerText = text.toLowerCase();

  let queryIndex = 0;
  for (let i = 0; i < lowerText.length && queryIndex < lowerQuery.length; i++) {
    if (lowerText[i] === lowerQuery[queryIndex]) {
      queryIndex++;
    }
  }
  return queryIndex === lowerQuery.length;
}

/**
 * Calculate match score for sorting
 */
function matchScore(query: string, item: CommandItem): number {
  if (!query) return 0;
  const lowerQuery = query.toLowerCase();
  const label = item.label.toLowerCase();
  const description = (item.description || '').toLowerCase();
  const keywords = (item.keywords || []).join(' ').toLowerCase();

  let score = 0;

  // Exact match on label
  if (label === lowerQuery) score += 100;
  // Starts with query
  else if (label.startsWith(lowerQuery)) score += 50;
  // Contains query
  else if (label.includes(lowerQuery)) score += 25;

  // Keyword matches
  if (keywords.includes(lowerQuery)) score += 15;

  // Description contains
  if (description.includes(lowerQuery)) score += 5;

  return score;
}

/**
 * Command Palette Component
 *
 * Usage:
 * ```tsx
 * <CommandPalette
 *   open={paletteOpen.value}
 *   commands={[
 *     { id: 'home', label: 'Go to Home', icon: '🏠', shortcut: '⌘H' },
 *     { id: 'search', label: 'Search', icon: '🔍', shortcut: '⌘F' },
 *   ]}
 *   onSelect$={(cmd) => handleCommand(cmd)}
 *   onClose$={() => paletteOpen.value = false}
 * />
 * ```
 */
export const CommandPalette = component$<CommandPaletteProps>((props) => {
  const query = useSignal('');
  const selectedIndex = useSignal(0);
  const inputRef = useSignal<HTMLInputElement>();
  const maxResults = props.maxResults || 10;

  // Filter and sort commands
  const filteredCommands = useComputed$(() => {
    const q = query.value.trim();
    let results = props.commands;

    if (q) {
      // Filter by fuzzy match
      results = results.filter(cmd =>
        fuzzyMatch(q, cmd.label) ||
        fuzzyMatch(q, cmd.description || '') ||
        (cmd.keywords || []).some(k => fuzzyMatch(q, k))
      );

      // Sort by relevance score
      results = results.sort((a, b) => matchScore(q, b) - matchScore(q, a));
    }

    // Limit results
    return results.slice(0, maxResults);
  });

  // Group commands by category
  const groupedCommands = useComputed$(() => {
    const groups: Record<string, CommandItem[]> = {};

    filteredCommands.value.forEach(cmd => {
      const category = cmd.category || 'Commands';
      if (!groups[category]) groups[category] = [];
      groups[category].push(cmd);
    });

    return groups;
  });

  // Handle keyboard navigation
  const handleKeyDown = $((e: KeyboardEvent) => {
    const commands = filteredCommands.value;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        selectedIndex.value = Math.min(selectedIndex.value + 1, commands.length - 1);
        break;
      case 'ArrowUp':
        e.preventDefault();
        selectedIndex.value = Math.max(selectedIndex.value - 1, 0);
        break;
      case 'Enter':
        e.preventDefault();
        const selected = commands[selectedIndex.value];
        if (selected) {
          props.onSelect$?.(selected);
          selected.action?.();
          props.onClose$?.();
        }
        break;
      case 'Escape':
        e.preventDefault();
        props.onClose$?.();
        break;
    }
  });

  // Reset selection when query changes
  useVisibleTask$(({ track }) => {
    track(() => query.value);
    selectedIndex.value = 0;
  });

  // Focus input when opened
  useVisibleTask$(({ track }) => {
    track(() => props.open);
    if (props.open && inputRef.value) {
      inputRef.value.focus();
      query.value = '';
    }
  });

  // Global keyboard shortcut (Cmd/Ctrl + K)
  useVisibleTask$(({ cleanup }) => {
    if (typeof window === 'undefined') return;

    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (!props.open) {
          // This should trigger the parent to open the palette
          window.dispatchEvent(new CustomEvent('pluribus:command-palette', { detail: { action: 'toggle' } }));
        }
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    cleanup(() => window.removeEventListener('keydown', handleGlobalKeyDown));
  });

  if (!props.open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        class="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100] glass-animate-enter"
        onClick$={props.onClose$}
      />

      {/* Palette */}
      <div
        class={`
          fixed
          top-[20%]
          left-1/2
          -translate-x-1/2
          w-full
          max-w-xl
          z-[101]
          glass-surface-overlay
          rounded-2xl
          overflow-hidden
          shadow-2xl
          glass-animate-enter
          ${props.class || ''}
        `}
      >
        {/* Search input */}
        <div class="flex items-center gap-3 px-4 py-3 border-b border-[var(--glass-border)]">
          <span class="text-[var(--glass-text-tertiary)]">🔍</span>
          <input
            ref={inputRef}
            type="text"
            value={query.value}
            onInput$={(e) => query.value = (e.target as HTMLInputElement).value}
            onKeyDown$={handleKeyDown}
            placeholder={props.placeholder || 'Type a command or search...'}
            class={`
              flex-grow
              bg-transparent
              text-sm
              text-[var(--glass-text-primary)]
              placeholder:text-[var(--glass-text-tertiary)]
              outline-none
            `}
          />
          <kbd class="hidden sm:block text-[10px] px-2 py-1 rounded bg-[var(--glass-bg-card)] text-[var(--glass-text-tertiary)] border border-[var(--glass-border)]">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div class="max-h-[60vh] overflow-y-auto">
          {filteredCommands.value.length === 0 ? (
            <div class="px-4 py-8 text-center text-sm text-[var(--glass-text-tertiary)]">
              No commands found
            </div>
          ) : (
            <div class="py-2">
              {Object.entries(groupedCommands.value).map(([category, commands]) => (
                <div key={category}>
                  {/* Category header */}
                  <div class="px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--glass-text-tertiary)]">
                    {category}
                  </div>

                  {/* Commands in category */}
                  {commands.map((cmd) => {
                    const globalIndex = filteredCommands.value.indexOf(cmd);
                    const isSelected = globalIndex === selectedIndex.value;

                    return (
                      <button
                        key={cmd.id}
                        class={`
                          w-full
                          flex items-center gap-3
                          px-4 py-3
                          text-left
                          glass-transition-colors
                          ${isSelected
                            ? 'bg-[var(--glass-accent-cyan)]/15 text-[var(--glass-accent-cyan)]'
                            : 'hover:bg-[var(--glass-state-layer-hover)] text-[var(--glass-text-secondary)]'
                          }
                        `}
                        onClick$={() => {
                          props.onSelect$?.(cmd);
                          cmd.action?.();
                          props.onClose$?.();
                        }}
                        onMouseEnter$={() => selectedIndex.value = globalIndex}
                      >
                        {/* Icon */}
                        {cmd.icon && (
                          <span class="text-lg w-8 text-center flex-shrink-0">{cmd.icon}</span>
                        )}

                        {/* Label & description */}
                        <div class="flex-grow min-w-0">
                          <div class="text-sm font-medium truncate">{cmd.label}</div>
                          {cmd.description && (
                            <div class="text-xs text-[var(--glass-text-tertiary)] truncate">
                              {cmd.description}
                            </div>
                          )}
                        </div>

                        {/* Shortcut */}
                        {cmd.shortcut && (
                          <kbd class="text-[10px] px-2 py-1 rounded bg-[var(--glass-bg-card)] text-[var(--glass-text-tertiary)] border border-[var(--glass-border)] flex-shrink-0">
                            {cmd.shortcut}
                          </kbd>
                        )}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer hint */}
        <div class="px-4 py-2 border-t border-[var(--glass-border)] flex items-center justify-between text-[10px] text-[var(--glass-text-tertiary)]">
          <div class="flex items-center gap-3">
            <span><kbd>↑↓</kbd> Navigate</span>
            <span><kbd>↵</kbd> Select</span>
            <span><kbd>ESC</kbd> Close</span>
          </div>
          <div>
            {filteredCommands.value.length} result{filteredCommands.value.length !== 1 ? 's' : ''}
          </div>
        </div>
      </div>
    </>
  );
});

export default CommandPalette;
