/**
 * IPERoot.tsx
 *
 * Root component for the In-Place Editor system.
 * Manages state and coordinates Toggle, Overlay, and Panel components.
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  $,
} from '@builder.io/qwik';
import type { IPEContext, IPEMode, IPEScope } from '../../lib/ipe';
import { 
  saveInstance, 
  loadInstances, 
  syncToRhizome, 
  generateInstanceCSS 
} from '../../lib/ipe';
import { IPEToggle } from './IPEToggle';
import { IPEOverlay } from './IPEOverlay';
import { IPEPanel } from './IPEPanel';

interface IPERootProps {
  /** Enable IPE system (can be disabled for production) */
  enabled?: boolean;
}

export const IPERoot = component$<IPERootProps>(({ enabled = true }) => {
  const mode = useSignal<IPEMode>('off');
  const selectedContext = useSignal<IPEContext | null>(null);

  // Apply saved overrides on boot
  useVisibleTask$(() => {
    const instances = loadInstances();
    let css = '';
    
    for (const [id, data] of Object.entries(instances)) {
      if (data.styles) {
        css += generateInstanceCSS(id, data.styles);
      }
    }

    if (css) {
      const styleId = 'ipe-overrides';
      let styleEl = document.getElementById(styleId);
      if (!styleEl) {
        styleEl = document.createElement('style');
        styleEl.id = styleId;
        document.head.appendChild(styleEl);
      }
      styleEl.textContent = css;
      console.log(`[IPE] Applied ${Object.keys(instances).length} instance overrides`);
    }
  });

  // Listen for external mode changes
  useVisibleTask$(({ cleanup }) => {
    const handleModeChange = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.mode) {
        mode.value = detail.mode;
        // Clear selection when mode changes to off
        if (detail.mode === 'off') {
          selectedContext.value = null;
        }
      }
    };

    // Listen for Escape key to close panel or exit mode
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (selectedContext.value) {
          // Close panel first
          selectedContext.value = null;
        } else if (mode.value !== 'off') {
          // Then exit mode
          mode.value = 'off';
        }
      }
    };

    window.addEventListener('ipe:mode:change', handleModeChange);
    document.addEventListener('keydown', handleKeyDown);

    cleanup(() => {
      window.removeEventListener('ipe:mode:change', handleModeChange);
      document.removeEventListener('keydown', handleKeyDown);
    });
  });

  // Handle mode toggle from button
  const handleModeChange = $((newMode: IPEMode) => {
    mode.value = newMode;
    if (newMode === 'off') {
      selectedContext.value = null;
    }
  });

  // Handle element selection
  const handleSelect = $((context: IPEContext) => {
    selectedContext.value = context;
  });

  // Handle panel close
  const handlePanelClose = $(() => {
    selectedContext.value = null;
  });

  // Handle real-time preview
  const handlePreview = $((scope: IPEScope, changes: Record<string, any>) => {
    if (!selectedContext.value) return;
    
    const { styles } = changes;
    if (!styles) return;

    const instanceId = selectedContext.value.instanceId;
    const css = generateInstanceCSS(instanceId, styles);
    
    // Use a separate style tag for preview to avoid flickering or conflict with saved persistence
    const previewId = 'ipe-preview';
    let previewEl = document.getElementById(previewId);
    if (!previewEl) {
      previewEl = document.createElement('style');
      previewEl.id = previewId;
      document.head.appendChild(previewEl);
    }
    
    // For preview, we only want the CURRENT element's styles.
    // Real implementation might need to merge with existing previews if multiple active?
    // For now, single active preview is fine.
    previewEl.textContent = css;
  });

  // Handle save
  const handleSave = $((scope: IPEScope, changes: Record<string, any>) => {
    console.log('[IPE] Save:', { scope, changes });

    // Extract purpose and style changes (assuming changes *contains* style changes or is the style object itself?)
    // In TokensTab, we were modifying `tokens` store directly, but not passing it up?
    // Wait, in IPEPanel, `onSave$?.(state.scope, { purpose: purpose.value });`
    // We need to pass the actual token changes from TokensTab up to IPEPanel first!
    // Actually, looking at TokensTab, it modifies a local `tokens` store.
    // We need to implement a mechanism to gather the changes.
    // For now, let's assume `changes` contains `{ styles: ..., purpose: ... }`
    // But `TokensTab` logic in previous turns didn't seem to pass `tokens` back up.
    // IPEPanel tracks `hasChanges` but doesn't seem to hold the changed data centrally.
    // TokensTab holds `tokens` store.
    // This requires a refactor of IPEPanel to hold state or request state.
    // However, for this specific replacement, I will implement the logic assuming IPEPanel *will* be fixed to pass `{ styles: ..., purpose: ... }`
    
    // Correction: `changes` passed from IPEPanel is currently just `{ purpose: ... }`.
    // The `TokensTab` changes are stranded. 
    // I will fix `IPEPanel` in the NEXT step to properly aggregate changes.
    // This step just sets up `handleSave` to expect the right structure.

    const { styles, purpose } = changes;

    if (selectedContext.value && styles) {
      // 1. Save to local storage
      const instanceId = selectedContext.value.instanceId;
      
      saveInstance({
        instanceId,
        selector: selectedContext.value.selector,
        styles: styles, 
        purpose: purpose,
        createdAt: new Date().toISOString(), // Required by interface
        updatedAt: new Date().toISOString()
      });

      // 2. Apply immediately to DOM (Live Edit)
      const css = generateInstanceCSS(instanceId, styles);
      const styleId = 'ipe-overrides';
      const styleEl = document.getElementById(styleId);
      if (styleEl) {
        // Simple append for live feedback (in reality, we should manage a stylesheet registry)
        styleEl.textContent += css; 
      }

      // 3. Sync to Rhizome (Background)
      syncToRhizome({ actor: 'ipe-user' }).catch(err => 
        console.error('[IPE] Rhizome sync failed:', err)
      );
    }

    // Emit bus event
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('ipe:save', {
        detail: {
          scope,
          changes,
          context: selectedContext.value,
        },
      }));
    }

    // Close panel after save
    selectedContext.value = null;
  });

  if (!enabled) return null;

  return (
    <>
      {/* Floating toggle button */}
      <IPEToggle
        initialMode={mode.value}
        onModeChange$={handleModeChange}
      />

      {/* Overlay for inspect/edit modes */}
      <IPEOverlay
        mode={mode.value}
        onSelect$={handleSelect}
      />

      {/* Editor panel when element is selected */}
      {selectedContext.value && (
        <IPEPanel
          context={selectedContext.value}
          onClose$={handlePanelClose}
          onSave$={handleSave}
          onPreview$={handlePreview}
        />
      )}
    </>
  );
});

export default IPERoot;
