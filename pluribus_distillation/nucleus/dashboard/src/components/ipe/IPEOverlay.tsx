/**
 * IPEOverlay.tsx
 *
 * Visual overlay system for In-Place Editor inspection.
 * Shows element boundaries and quick info on hover when IPE mode is active.
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';
import {
  captureContextSafe,
  type IPEContext,
  type IPEMode,
} from '../../lib/ipe';

interface IPEOverlayProps {
  /** Current IPE mode */
  mode: IPEMode;
  /** Callback when element is selected for editing */
  onSelect$?: QRL<(context: IPEContext) => void>;
}

interface OverlayState {
  visible: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
  tagName: string;
  componentName: string;
  instanceId: string;
  tokenCount: number;
  isShader: boolean;
}

const INITIAL_STATE: OverlayState = {
  visible: false,
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  tagName: '',
  componentName: '',
  instanceId: '',
  tokenCount: 0,
  isShader: false,
};

export const IPEOverlay = component$<IPEOverlayProps>(({
  mode,
  onSelect$,
}) => {
  const overlay = useStore<OverlayState>({ ...INITIAL_STATE });
  const tooltip = useStore({
    visible: false,
    x: 0,
    y: 0,
  });
  const currentContext = useSignal<IPEContext | null>(null);
  const hoveredElement = useSignal<Element | null>(null);

  // Set up mouse tracking when mode is active
  useVisibleTask$(({ track, cleanup }) => {
    const currentMode = track(() => mode);

    if (currentMode === 'off') {
      overlay.visible = false;
      return;
    }

    // Handle mouse move
    const handleMouseMove = (e: MouseEvent) => {
      const target = document.elementFromPoint(e.clientX, e.clientY);

      // Skip if hovering over IPE UI itself
      if (target?.closest('[data-ipe-ui]')) {
        overlay.visible = false;
        return;
      }

      // Skip if no target or same target
      if (!target || target === hoveredElement.value) {
        // Update tooltip position
        if (overlay.visible) {
          tooltip.x = e.clientX + 16;
          tooltip.y = e.clientY + 16;
        }
        return;
      }

      hoveredElement.value = target;

      // Capture context
      const context = captureContextSafe(target);
      if (!context) {
        overlay.visible = false;
        return;
      }

      currentContext.value = context;

      // Update overlay position
      const rect = target.getBoundingClientRect();
      overlay.visible = true;
      overlay.x = rect.left;
      overlay.y = rect.top;
      overlay.width = rect.width;
      overlay.height = rect.height;
      overlay.tagName = target.tagName.toLowerCase();
      overlay.componentName = context.componentName || '';
      overlay.instanceId = context.instanceId;
      overlay.tokenCount = Object.keys(context.cssVariables).length;
      overlay.isShader = context.elementType === 'shader' || context.elementType === 'canvas';

      // Update tooltip position
      tooltip.visible = true;
      tooltip.x = e.clientX + 16;
      tooltip.y = e.clientY + 16;
    };

    // Handle mouse leave viewport
    const handleMouseLeave = () => {
      overlay.visible = false;
      tooltip.visible = false;
      hoveredElement.value = null;
    };

    // Handle click in edit mode
    const handleClick = (e: MouseEvent) => {
      if (currentMode !== 'edit' || !currentContext.value) return;

      const target = e.target as Element;
      if (target?.closest('[data-ipe-ui]')) return;

      e.preventDefault();
      e.stopPropagation();

      onSelect$?.(currentContext.value);
    };

    document.addEventListener('mousemove', handleMouseMove, { passive: true });
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('click', handleClick, { capture: true });

    // Add cursor style to body
    document.body.style.cursor = currentMode === 'inspect' ? 'crosshair' : 'pointer';

    cleanup(() => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('click', handleClick, { capture: true });
      document.body.style.cursor = '';
    });
  });

  // Don't render if mode is off
  if (mode === 'off') return null;

  // Use CSS variables for theme-aware highlighting
  const borderColor = mode === 'inspect'
    ? 'border-[var(--glass-accent-cyan)]'
    : 'border-[var(--glass-accent-purple)]';

  const bgColor = mode === 'inspect'
    ? 'bg-[var(--glass-accent-cyan-subtle)]'
    : 'bg-[var(--glass-accent-purple-bg)]';

  return (
    <div data-ipe-ui class="pointer-events-none fixed inset-0 z-[9998]">
      {/* Highlight box */}
      {overlay.visible && (
        <div
          class={[
            'absolute border-2 transition-all duration-75 shadow-[0_0_20px_rgba(0,0,0,0.3)]',
            borderColor,
            bgColor,
          ]}
          style={{
            left: `${overlay.x}px`,
            top: `${overlay.y}px`,
            width: `${overlay.width}px`,
            height: `${overlay.height}px`,
          }}
        >
          {/* Corner markers */}
          <div class={`absolute -top-1 -left-1 w-2 h-2 ${borderColor.replace('border-', 'bg-')}`} />
          <div class={`absolute -top-1 -right-1 w-2 h-2 ${borderColor.replace('border-', 'bg-')}`} />
          <div class={`absolute -bottom-1 -left-1 w-2 h-2 ${borderColor.replace('border-', 'bg-')}`} />
          <div class={`absolute -bottom-1 -right-1 w-2 h-2 ${borderColor.replace('border-', 'bg-')}`} />

          {/* Size indicator */}
          <div
            class={[
              'absolute -bottom-6 left-0 px-1.5 py-0.5 text-xs font-mono',
              'rounded bg-gray-900/90 text-gray-300',
            ]}
          >
            {Math.round(overlay.width)} × {Math.round(overlay.height)}
          </div>
        </div>
      )}

      {/* Info tooltip */}
      {tooltip.visible && overlay.visible && (
        <div
          class={[
            'absolute px-3 py-2 rounded-lg shadow-xl',
            'glass-panel bg-black/80 text-white text-sm',
            'border border-[var(--glass-border)] backdrop-blur-md',
            'max-w-xs z-[9999]',
          ]}
          style={{
            left: `${Math.min(tooltip.x, window.innerWidth - 200)}px`,
            top: `${Math.min(tooltip.y, window.innerHeight - 120)}px`,
          }}
        >
          {/* Element tag */}
          <div class="flex items-center gap-2">
            <span class="font-mono text-[var(--glass-accent-cyan)]">&lt;{overlay.tagName}&gt;</span>
            {overlay.componentName && (
              <span class="px-1.5 py-0.5 rounded bg-[var(--glass-accent-purple-bg)] text-[var(--glass-accent-purple)] text-xs">
                {overlay.componentName}
              </span>
            )}
          </div>

          {/* Instance ID */}
          <div class="text-xs text-gray-400 mt-1 font-mono">
            {overlay.instanceId}
          </div>

          {/* Quick stats */}
          <div class="flex gap-3 mt-2 text-xs">
            {overlay.tokenCount > 0 && (
              <span class="text-green-400">
                {overlay.tokenCount} tokens
              </span>
            )}
            {overlay.isShader && (
              <span class="text-orange-400">
                WebGL Shader
              </span>
            )}
          </div>

          {/* Action hint */}
          {mode === 'edit' && (
            <div class="mt-2 pt-2 border-t border-[var(--glass-border)] text-xs text-gray-400">
              Click to edit
            </div>
          )}
        </div>
      )}

      {/* Mode indicator banner */}
      <div
        class={[
          'fixed top-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-full',
          'text-sm font-medium shadow-lg backdrop-blur-sm',
          'border border-[var(--glass-border-active)]',
          mode === 'inspect'
            ? 'bg-blue-600/90 text-white'
            : 'bg-purple-600/90 text-white',
        ]}
      >
        IPE: {mode === 'inspect' ? 'Inspect Mode' : 'Edit Mode'}
        <span class="ml-2 text-white/60 text-xs">
          {mode === 'inspect' ? 'Hover to inspect' : 'Click to edit'}
        </span>
      </div>
    </div>
  );
});

export default IPEOverlay;
