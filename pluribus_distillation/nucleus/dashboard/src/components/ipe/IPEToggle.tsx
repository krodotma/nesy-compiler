/**
 * IPEToggle.tsx
 *
 * Floating toggle button for In-Place Editor mode.
 * Draggable, persists position, cycles through: Off → Inspect → Edit → Off
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  useOn,
  $,
  type QRL,
} from '@builder.io/qwik';
import type { IPEMode, IPEPreferences } from '../../lib/ipe';

interface IPEToggleProps {
  /** Callback when mode changes */
  onModeChange$?: QRL<(mode: IPEMode) => void>;
  /** Initial mode */
  initialMode?: IPEMode;
}

const STORAGE_KEY = 'pluribus.ipe.toggle';
const MODES: IPEMode[] = ['off', 'inspect', 'edit'];

const MODE_ICONS: Record<IPEMode, string> = {
  off: '✎',
  inspect: '🔍',
  edit: '✏️',
};

const MODE_CLASSES: Record<IPEMode, string> = {
  off: 'bg-black/50 text-muted-foreground border-glass-border hover:bg-black/80 hover:text-white hover:border-glass-border-hover',
  inspect: 'glass-bg-accent-cyan text-black border-glass-accent-cyan shadow-[0_0_15px_rgba(0,255,255,0.4)]',
  edit: 'glass-bg-accent-purple text-white border-glass-accent-purple shadow-[0_0_15px_rgba(168,85,247,0.4)]',
};

const MODE_LABELS: Record<IPEMode, string> = {
  off: 'IPE Off',
  inspect: 'Inspect',
  edit: 'Edit Mode',
};

export const IPEToggle = component$<IPEToggleProps>(({
  onModeChange$,
  initialMode = 'off',
}) => {
  const mode = useSignal<IPEMode>(initialMode);
  const isDragging = useSignal(false);
  const showTooltip = useSignal(false);

  const position = useStore({
    x: -1,  // -1 = default position (bottom-right)
    y: -1,
  });

  const dragOffset = useStore({
    x: 0,
    y: 0,
  });

  // Load saved position and register keyboard shortcut
  useVisibleTask$(({ cleanup }) => {
    // Load position from localStorage
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const data = JSON.parse(saved) as { x: number; y: number };
        position.x = data.x;
        position.y = data.y;
      }
    } catch {}

    // Keyboard shortcut: Ctrl+Shift+I
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'I') {
        e.preventDefault();
        cycleMode();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    cleanup(() => document.removeEventListener('keydown', handleKeyDown));
  });

  // Cycle through modes
  const cycleMode = $(() => {
    const currentIndex = MODES.indexOf(mode.value);
    const nextIndex = (currentIndex + 1) % MODES.length;
    mode.value = MODES[nextIndex];
    onModeChange$?.(mode.value);

    // Emit bus event
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('ipe:mode:change', {
        detail: { mode: mode.value }
      }));
    }
  });

  // Handle drag start
  const handleMouseDown = $((e: MouseEvent) => {
    if (e.button !== 0) return; // Left click only

    isDragging.value = true;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    dragOffset.x = e.clientX - rect.left;
    dragOffset.y = e.clientY - rect.top;
  });

  // Handle drag move
  useOn('document:mousemove', $((e: Event) => {
    if (!isDragging.value) return;
    const me = e as MouseEvent;

    const newX = me.clientX - dragOffset.x;
    const newY = me.clientY - dragOffset.y;

    // Clamp to viewport
    position.x = Math.max(0, Math.min(window.innerWidth - 56, newX));
    position.y = Math.max(0, Math.min(window.innerHeight - 56, newY));
  }));

  // Handle drag end
  useOn('document:mouseup', $(() => {
    if (!isDragging.value) return;
    isDragging.value = false;

    // Save position
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ x: position.x, y: position.y }));
    } catch {}
  }));

  // Compute position style
  const positionStyle = () => {
    if (position.x < 0 || position.y < 0) {
      // Default: bottom-right corner
      return { bottom: '24px', right: '24px' };
    }
    return {
      left: `${position.x}px`,
      top: `${position.y}px`,
    };
  };

  return (
    <div
      class={[
        'fixed z-[9999] select-none',
        isDragging.value ? 'cursor-grabbing' : 'cursor-grab',
      ]}
      style={positionStyle()}
    >
      {/* Main toggle button */}
      <button
        type="button"
        class={[
          'w-14 h-14 rounded-full shadow-lg transition-all duration-300',
          'flex items-center justify-center text-xl backdrop-blur-md border-2',
          MODE_CLASSES[mode.value],
          isDragging.value ? 'scale-110 cursor-grabbing' : 'cursor-grab hover:scale-105',
        ]}
        onMouseDown$={handleMouseDown}
        onClick$={cycleMode}
        onMouseEnter$={() => { showTooltip.value = true; }}
        onMouseLeave$={() => { showTooltip.value = false; }}
        title={`${MODE_LABELS[mode.value]} (Ctrl+Shift+I)`}
      >
        <span class="pointer-events-none">{MODE_ICONS[mode.value]}</span>

        {/* Mode indicator ring */}
        {mode.value !== 'off' && (
          <span
            class={[
              'absolute inset-0 rounded-full border-2 animate-ping',
              mode.value === 'inspect' ? 'border-blue-400' : 'border-purple-400',
            ]}
            style={{ animationDuration: '2s' }}
          />
        )}
      </button>

      {/* Tooltip */}
      {showTooltip.value && !isDragging.value && (
        <div
          class={[
            'absolute bottom-full mb-2 left-1/2 -translate-x-1/2',
            'px-3 py-1.5 rounded-lg text-sm whitespace-nowrap',
            'bg-gray-900/95 text-white shadow-xl',
            'border border-[var(--glass-border)] backdrop-blur-sm',
          ]}
        >
          <div class="font-medium">{MODE_LABELS[mode.value]}</div>
          <div class="text-xs text-gray-400 mt-0.5">
            Ctrl+Shift+I to toggle
          </div>
          {/* Tooltip arrow */}
          <div
            class="absolute top-full left-1/2 -translate-x-1/2 -mt-px"
            style={{
              borderLeft: '6px solid transparent',
              borderRight: '6px solid transparent',
              borderTop: '6px solid rgb(17 24 39 / 0.95)',
            }}
          />
        </div>
      )}

      {/* Mode indicator badge */}
      {mode.value !== 'off' && (
        <div
          class={[
            'absolute -top-1 -right-1 px-2 py-0.5 rounded-full text-xs font-bold',
            'shadow-lg border border-[var(--glass-border-active)]',
            mode.value === 'inspect'
              ? 'bg-blue-500 text-white'
              : 'bg-purple-500 text-white',
          ]}
        >
          {mode.value === 'inspect' ? 'I' : 'E'}
        </div>
      )}
    </div>
  );
});

export default IPEToggle;
