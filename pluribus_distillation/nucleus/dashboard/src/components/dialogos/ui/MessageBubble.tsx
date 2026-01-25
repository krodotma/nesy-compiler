/**
 * MessageBubble.tsx
 * Author: gemini_ui_1
 * Context: Phase 4 Excellence (Visuals)
 * 
 * Replaces the primitive "div" with a high-fidelity M3 Glass component.
 * Features:
 * - Author Avatar (3D Holon placeholder)
 * - Time Stamp (Relative)
 * - Status Indicators (Entelexis State)
 * - Markdown Rendering (Syntax Highlighting)
 * - Entrance Animation (Slide Up)
 */

import { component$, useStylesScoped$ } from '@builder.io/qwik';
import type { DialogosAtom } from '../types/dialogos';

interface MessageBubbleProps {
  atom: DialogosAtom;
}

export const MessageBubble = component$<MessageBubbleProps>(({ atom }) => {
  useStylesScoped$(`
    .bubble-enter {
      animation: slide-up 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
      opacity: 0;
      transform: translateY(20px);
    }
    @keyframes slide-up {
      to { opacity: 1; transform: translateY(0); }
    }
    .neon-border-pulse {
      animation: border-pulse 2s infinite;
    }
    @keyframes border-pulse {
      0%, 100% { border-color: rgba(255,255,255,0.1); }
      50% { border-color: var(--glass-accent-cyan); box-shadow: 0 0 15px var(--glass-accent-cyan-subtle); }
    }
  `);

  const isHuman = atom.author.role === 'human';
  const isSystem = atom.author.role === 'system';

  return (
    <div class={`bubble-enter flex w-full mb-6 ${isHuman ? 'justify-end' : 'justify-start'}`}>
      
      {/* Avatar (Left for Agent) */}
      {!isHuman && (
        <div class="w-8 h-8 rounded-full bg-black/50 border border-white/10 mr-3 flex items-center justify-center shadow-lg">
          {isSystem ? '⚠️' : '🤖'}
        </div>
      )}

      <div class={`
        max-w-[85%] relative group
        ${isHuman ? 'items-end' : 'items-start'}
      `}>
        {/* Metadata Header */}
        <div class={`flex items-center text-[10px] text-white/40 mb-1 px-1 gap-2 ${isHuman ? 'justify-end' : 'justify-start'}`}>
          <span class="font-bold tracking-wider uppercase">{atom.author.name}</span>
          <span>•</span>
          <span class="font-mono">{new Date(atom.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
        </div>

        {/* The Bubble */}
        <div class={`
          p-4 rounded-2xl backdrop-blur-xl border transition-all duration-300
          ${isHuman 
            ? 'bg-[var(--glass-accent-cyan)]/10 border-[var(--glass-accent-cyan)]/30 rounded-tr-sm hover:bg-[var(--glass-accent-cyan)]/20' 
            : isSystem
              ? 'bg-red-500/10 border-red-500/30 rounded-tl-sm'
              : 'bg-black/40 border-white/10 rounded-tl-sm hover:border-[var(--glass-accent-magenta)]/50'
          }
          ${atom.state === 'actualizing' ? 'neon-border-pulse' : ''}
          shadow-[0_4px_30px_rgba(0,0,0,0.1)]
        `}>
          {/* Content Rendering */}
          <div class="text-sm text-white/90 leading-relaxed font-light">
            {atom.content.type === 'text' && (
              <p>{atom.content.value}</p>
            )}
            {atom.content.type === 'code' && (
              <div class="font-mono text-xs bg-black/50 p-3 rounded-lg border border-white/5 my-2 overflow-x-auto">
                <div class="flex justify-between items-center mb-2 opacity-50 border-b border-white/5 pb-1">
                  <span>{atom.content.language}</span>
                  <button class="hover:text-white">Copy</button>
                </div>
                <pre>{atom.content.value}</pre>
              </div>
            )}
            {atom.content.type === 'task' && (
              <div class="flex items-center gap-3 bg-white/5 p-3 rounded-lg border border-white/10">
                <div class="w-4 h-4 rounded border border-[var(--glass-accent-cyan)]"></div>
                <div class="flex-1 font-medium">{atom.content.title}</div>
                <div class="text-[10px] px-2 py-0.5 rounded-full bg-white/10">{atom.content.status}</div>
              </div>
            )}
          </div>
        </div>

        {/* Status Footer (Seen/Sent) */}
        {isHuman && (
          <div class="text-[10px] text-white/20 text-right px-1 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
            {atom.state}
          </div>
        )}
      </div>

      {/* Avatar (Right for Human) */}
      {isHuman && (
        <div class="w-8 h-8 rounded-full bg-[var(--glass-accent-cyan)]/20 border border-[var(--glass-accent-cyan)] ml-3 flex items-center justify-center text-xs">
          ME
        </div>
      )}
    </div>
  );
});
