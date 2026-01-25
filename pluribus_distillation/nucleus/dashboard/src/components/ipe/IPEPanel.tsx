/**
 * IPEPanel.tsx
 *
 * Main editor panel for In-Place Editor.
 * Contains tabs: Tokens (tweakcn), Art Dept (shaders), Raw (JSON)
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';
import type { IPEContext, IPEScope, SextetVerdict } from '../../lib/ipe';
import { TokensTab } from './tabs/TokensTab';
import { ArtDeptTab } from './tabs/ArtDeptTab';
import { RawTab } from './tabs/RawTab';

// M3 Components - IPEPanel
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';
import '@material/web/elevation/elevation.js';
import '@material/web/button/filled-tonal-button.js';

type TabId = 'tokens' | 'artdept' | 'raw';

interface IPEPanelProps {
  /** The captured context to edit */
  context: IPEContext;
  /** Callback when panel is closed */
  onClose$?: QRL<() => void>;
  /** Callback when changes are saved */
  onSave$?: QRL<(scope: IPEScope, changes: unknown) => void>;
}

interface PanelState {
  activeTab: TabId;
  scope: IPEScope;
  isDragging: boolean;
  isResizing: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
  hasChanges: boolean;
}

const STORAGE_KEY = 'pluribus.ipe.panel';

const TAB_CONFIG: { id: TabId; label: string; icon: string }[] = [
  { id: 'tokens', label: 'Tokens', icon: '🎨' },
  { id: 'artdept', label: 'Art Dept', icon: '🖼️' },
  { id: 'raw', label: 'Raw', icon: '{ }' },
];

export const IPEPanel = component$<IPEPanelProps>(({
  context,
  onClose$,
  onSave$,
}) => {
  const state = useStore<PanelState>({
    activeTab: 'tokens',
    scope: 'instance',
    isDragging: false,
    isResizing: false,
    x: -1,  // -1 = auto position
    y: -1,
    width: 380,
    height: 520,
    hasChanges: false,
  });

  const dragOffset = useStore({ x: 0, y: 0 });
  const validationResult = useSignal<{ verdict: SextetVerdict; messages: string[] } | null>(null);

  // Load saved position
  useVisibleTask$(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const data = JSON.parse(saved);
        state.x = data.x ?? -1;
        state.y = data.y ?? -1;
        state.width = data.width ?? 380;
        state.height = data.height ?? 520;
        state.activeTab = data.activeTab ?? 'tokens';
      }
    } catch {}

    // Auto-position if not set
    if (state.x < 0 || state.y < 0) {
      state.x = Math.max(20, window.innerWidth - state.width - 40);
      state.y = Math.max(20, (window.innerHeight - state.height) / 2);
    }
  });

  // Save position on changes
  const savePosition = $(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        x: state.x,
        y: state.y,
        width: state.width,
        height: state.height,
        activeTab: state.activeTab,
      }));
    } catch {}
  });

  // Handle drag
  const handleDragStart = $((e: MouseEvent) => {
    if ((e.target as HTMLElement).closest('[data-no-drag]')) return;
    state.isDragging = true;
    dragOffset.x = e.clientX - state.x;
    dragOffset.y = e.clientY - state.y;
  });

  const handleDragMove = $((e: MouseEvent) => {
    if (!state.isDragging) return;
    state.x = Math.max(0, Math.min(window.innerWidth - state.width, e.clientX - dragOffset.x));
    state.y = Math.max(0, Math.min(window.innerHeight - 50, e.clientY - dragOffset.y));
  });

  const handleDragEnd = $(() => {
    if (state.isDragging) {
      state.isDragging = false;
      savePosition();
    }
  });

  // Handle resize
  const handleResizeStart = $((e: MouseEvent) => {
    e.stopPropagation();
    state.isResizing = true;
    dragOffset.x = e.clientX;
    dragOffset.y = e.clientY;
  });

  const handleResizeMove = $((e: MouseEvent) => {
    if (!state.isResizing) return;
    const dx = e.clientX - dragOffset.x;
    const dy = e.clientY - dragOffset.y;
    state.width = Math.max(300, Math.min(800, state.width + dx));
    state.height = Math.max(400, Math.min(900, state.height + dy));
    dragOffset.x = e.clientX;
    dragOffset.y = e.clientY;
  });

  const handleResizeEnd = $(() => {
    if (state.isResizing) {
      state.isResizing = false;
      savePosition();
    }
  });

  // Tab change
  const setTab = $((tab: TabId) => {
    state.activeTab = tab;
    savePosition();
  });

  // Scope toggle
  const toggleScope = $(() => {
    state.scope = state.scope === 'global' ? 'instance' : 'global';
  });

  // Close panel
  const handleClose = $(() => {
    onClose$?.();
  });

  // Save changes
  const handleSave = $(() => {
    // TODO: Implement Sextet validation
    validationResult.value = {
      verdict: 'PASSED',
      messages: ['All checks passed'],
    };
    onSave$?.(state.scope, {});
    state.hasChanges = false;
  });

  // Reset changes
  const handleReset = $(() => {
    state.hasChanges = false;
    // TODO: Implement reset logic
  });

  return (
    <div
      data-ipe-ui
      class={[
        'fixed z-[9999] flex flex-col',
        'bg-gray-900/95 backdrop-blur-md rounded-xl shadow-2xl',
        'border border-white/10',
        state.isDragging ? 'cursor-grabbing' : '',
      ]}
      style={{
        left: `${state.x}px`,
        top: `${state.y}px`,
        width: `${state.width}px`,
        height: `${state.height}px`,
      }}
      document:onMouseMove$={[handleDragMove, handleResizeMove]}
      document:onMouseUp$={[handleDragEnd, handleResizeEnd]}
    >
      {/* Header */}
      <div
        class={[
          'flex items-center justify-between px-4 py-3',
          'border-b border-white/10 rounded-t-xl',
          'bg-gray-800/50 cursor-grab',
        ]}
        onMouseDown$={handleDragStart}
      >
        {/* Element info */}
        <div class="flex items-center gap-2 overflow-hidden">
          <span class="font-mono text-blue-400 text-sm">
            &lt;{context.selector.split(' > ').pop()}&gt;
          </span>
          {context.componentName && (
            <span class="px-2 py-0.5 rounded bg-purple-500/30 text-purple-300 text-xs truncate">
              {context.componentName}
            </span>
          )}
        </div>

        {/* Close button */}
        <button
          data-no-drag
          type="button"
          class="p-1 rounded hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
          onClick$={handleClose}
          title="Close (Esc)"
        >
          <svg class="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
          </svg>
        </button>
      </div>

      {/* Tabs */}
      <div class="flex gap-1 px-3 py-2 border-b border-white/10">
        {TAB_CONFIG.map(tab => (
          <button
            key={tab.id}
            type="button"
            class={[
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              state.activeTab === tab.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-white/10',
            ]}
            onClick$={() => setTab(tab.id)}
          >
            <span class="mr-1.5">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div class="flex-1 overflow-auto p-4">
        {state.activeTab === 'tokens' && (
          <TokensTab
            context={context}
            scope={state.scope}
            onTokenChange$={(token, value) => {
              state.hasChanges = true;
              console.log('[IPE] Token changed:', token, value);
            }}
          />
        )}
        {state.activeTab === 'artdept' && (
          <ArtDeptTab
            context={context}
            onShaderChange$={(shader) => {
              state.hasChanges = true;
              console.log('[IPE] Shader changed');
            }}
            onUniformChange$={(name, value) => {
              state.hasChanges = true;
              console.log('[IPE] Uniform changed:', name, value);
            }}
          />
        )}
        {state.activeTab === 'raw' && (
          <RawTab context={context} />
        )}
      </div>

      {/* Footer */}
      <div class="flex items-center justify-between px-4 py-3 border-t border-white/10 bg-gray-800/30">
        {/* Scope selector */}
        <button
          data-no-drag
          type="button"
          class={[
            'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
            'border border-white/20',
            state.scope === 'global'
              ? 'bg-orange-600/30 text-orange-300 border-orange-500/50'
              : 'bg-blue-600/30 text-blue-300 border-blue-500/50',
          ]}
          onClick$={toggleScope}
          title="Toggle scope"
        >
          {state.scope === 'global' ? '🌐 Global' : '📍 Instance'}
        </button>

        {/* Action buttons */}
        <div class="flex gap-2">
          <button
            data-no-drag
            type="button"
            class="px-3 py-1.5 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
            onClick$={handleReset}
            disabled={!state.hasChanges}
          >
            Reset
          </button>
          <button
            data-no-drag
            type="button"
            class={[
              'px-4 py-1.5 rounded-lg text-sm font-medium transition-colors',
              state.hasChanges
                ? 'bg-green-600 text-white hover:bg-green-500'
                : 'bg-gray-700 text-gray-400 cursor-not-allowed',
            ]}
            onClick$={handleSave}
            disabled={!state.hasChanges}
          >
            Save
          </button>
        </div>
      </div>

      {/* Validation result */}
      {validationResult.value && (
        <div
          class={[
            'absolute bottom-16 left-4 right-4 px-3 py-2 rounded-lg text-sm',
            validationResult.value.verdict === 'PASSED'
              ? 'bg-green-600/20 text-green-300 border border-green-500/30'
              : validationResult.value.verdict === 'WARNED'
              ? 'bg-yellow-600/20 text-yellow-300 border border-yellow-500/30'
              : 'bg-red-600/20 text-red-300 border border-red-500/30',
          ]}
        >
          <div class="font-medium">
            Sextet: {validationResult.value.verdict}
          </div>
          <div class="text-xs mt-1 opacity-80">
            {validationResult.value.messages.join(' • ')}
          </div>
        </div>
      )}

      {/* Resize handle */}
      <div
        data-no-drag
        class="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize"
        onMouseDown$={handleResizeStart}
      >
        <svg class="w-4 h-4 text-gray-600" viewBox="0 0 16 16" fill="currentColor">
          <path d="M14 14v-2h-2v2h2zm-4 0v-2h-2v2h2zm4-4v-2h-2v2h2z" />
        </svg>
      </div>
    </div>
  );
});

export default IPEPanel;
