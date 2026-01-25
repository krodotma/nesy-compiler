/**
 * ShaderControl - A/B Testing UI for Art Department
 *
 * Small floating control to rate and switch shaders.
 * Shows current shader name, like/dislike buttons, and "next" button.
 */

import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import {
  getShaderPrefs,
  likeShader,
  dislikeShader,
  blacklistShader,
  type ShaderPref,
} from '../../lib/art/shader-preferences';

export const ShaderControl = component$(() => {
  const currentId = useSignal<string>('');
  const currentName = useSignal<string>('Default');
  const currentPref = useSignal<ShaderPref | null>(null);
  const expanded = useSignal(false);
  const feedback = useSignal<string>('');

  useVisibleTask$(() => {
    // Load initial state
    const prefs = getShaderPrefs();
    if (prefs.currentId) {
      currentId.value = prefs.currentId;
      currentName.value = prefs.currentName || 'Unknown';
      currentPref.value = prefs.shaders[prefs.currentId] || null;
    }

    // Listen for shader changes
    const onShaderChange = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.id) {
        currentId.value = detail.id;
        currentName.value = detail.name || 'Unknown';
        currentPref.value = detail.pref || null;
      }
    };
    window.addEventListener('pluribus:shader:current', onShaderChange);
    return () => window.removeEventListener('pluribus:shader:current', onShaderChange);
  });

  const handleLike = $(() => {
    if (!currentId.value) return;
    const updated = likeShader(currentId.value);
    if (updated) {
      currentPref.value = { ...updated };
      feedback.value = '+1';
      setTimeout(() => (feedback.value = ''), 1000);
    }
  });

  const handleDislike = $(() => {
    if (!currentId.value) return;
    const updated = dislikeShader(currentId.value);
    if (updated) {
      currentPref.value = { ...updated };
      feedback.value = '-1';
      setTimeout(() => (feedback.value = ''), 1000);
    }
  });

  const handleBlacklist = $(() => {
    if (!currentId.value) return;
    blacklistShader(currentId.value);
    feedback.value = 'Blacklisted';
    // Request next shader
    window.dispatchEvent(new CustomEvent('pluribus:art:request', {
      detail: { reason: 'blacklist_current' }
    }));
    setTimeout(() => (feedback.value = ''), 1500);
  });

  const handleNext = $(() => {
    window.dispatchEvent(new CustomEvent('pluribus:art:request', {
      detail: { reason: 'manual_next' }
    }));
    feedback.value = 'Loading...';
    setTimeout(() => (feedback.value = ''), 1500);
  });

  return (
    <div
      class={`fixed bottom-4 right-4 z-50 transition-all duration-300 ${
        expanded.value ? 'w-64' : 'w-auto'
      }`}
    >
      {/* Collapsed: Just an icon */}
      {!expanded.value && (
        <button
          onClick$={() => (expanded.value = true)}
          class="bg-black/50 backdrop-blur-sm border border-[var(--glass-border)] rounded-full p-2 hover:bg-black/70 transition-colors"
          title="Shader Controls"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 2a10 10 0 0 1 0 20" />
            <path d="M12 2a10 10 0 0 0 0 20" opacity="0.5" />
          </svg>
        </button>
      )}

      {/* Expanded: Full controls */}
      {expanded.value && (
        <div class="bg-black/70 backdrop-blur-md border border-[var(--glass-border)] rounded-lg p-3 shadow-xl">
          {/* Header */}
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs text-white/50 uppercase tracking-wider">Shader</span>
            <button
              onClick$={() => (expanded.value = false)}
              class="text-white/50 hover:text-white"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Current shader name */}
          <div class="text-sm text-white font-medium truncate mb-2" title={currentName.value}>
            {currentName.value}
          </div>

          {/* Stats */}
          {currentPref.value && (
            <div class="text-xs text-white/40 mb-3 flex gap-3">
              <span title="Likes">+{currentPref.value.likes}</span>
              <span title="Dislikes">-{currentPref.value.dislikes}</span>
              <span title="Times seen">#{currentPref.value.seen}</span>
            </div>
          )}

          {/* Feedback */}
          {feedback.value && (
            <div class="text-xs text-cyan-400 mb-2 animate-pulse">{feedback.value}</div>
          )}

          {/* Action buttons */}
          <div class="flex gap-2">
            <button
              onClick$={handleLike}
              class="flex-1 py-1.5 px-2 bg-green-600/20 border border-green-500/30 rounded text-green-400 text-xs hover:bg-green-600/30 transition-colors"
              title="Like this shader"
            >
              Like
            </button>
            <button
              onClick$={handleDislike}
              class="flex-1 py-1.5 px-2 bg-red-600/20 border border-red-500/30 rounded text-red-400 text-xs hover:bg-red-600/30 transition-colors"
              title="Dislike this shader"
            >
              Dislike
            </button>
            <button
              onClick$={handleNext}
              class="flex-1 py-1.5 px-2 bg-blue-600/20 border border-blue-500/30 rounded text-blue-400 text-xs hover:bg-blue-600/30 transition-colors"
              title="Show next shader"
            >
              Next
            </button>
          </div>

          {/* Blacklist button */}
          <button
            onClick$={handleBlacklist}
            class="w-full mt-2 py-1 text-xs text-white/30 hover:text-red-400 transition-colors"
            title="Never show this shader again"
          >
            Never show again
          </button>
        </div>
      )}
    </div>
  );
});
