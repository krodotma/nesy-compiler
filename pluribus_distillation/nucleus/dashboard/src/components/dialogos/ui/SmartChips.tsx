/**
 * SmartChips.tsx
 * Author: gemini_interaction_1
 * Context: Phase 2 UX Innovation
 * 
 * Context-aware suggestion chips that float *outside* the input.
 * Implements "Staggered Entrance" physics.
 */

import { component$, useStylesScoped$ } from '@builder.io/qwik';

interface SmartChipsProps {
  intent: string; // 'query' | 'mutation' | 'task'
  onSelect$: (text: string) => void;
}

export const SmartChips = component$<SmartChipsProps>(({ intent, onSelect$ }) => {
  useStylesScoped$(`
    .chip-enter {
      animation: chip-pop 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
      opacity: 0;
      transform: translateY(10px) scale(0.9);
    }
    @keyframes chip-pop {
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
  `);

  const suggestions = getSuggestions(intent);

  return (
    <div class="flex gap-2 mb-2 overflow-x-auto scrollbar-hide px-4 mask-fade-sides">
      {suggestions.map((s, i) => (
        <button
          key={s.label}
          onClick$={() => onSelect$(s.value)}
          class="chip-enter px-3 py-1.5 rounded-full 
                 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-[var(--glass-accent-cyan)]
                 text-xs text-white/70 hover:text-white transition-all duration-200
                 whitespace-nowrap flex items-center gap-2 backdrop-blur-md"
          style={{ animationDelay: `${i * 0.05}s` }}
        >
          <span class="opacity-50">{s.icon}</span>
          {s.label}
        </button>
      ))}
    </div>
  );
});

function getSuggestions(intent: string) {
  switch (intent) {
    case 'mutation':
      return [
        { label: 'Fix Bug', value: '/fix ', icon: '🐛' },
        { label: 'Refactor', value: '/refactor ', icon: '🔨' },
        { label: 'Add Test', value: '/test ', icon: '🧪' }
      ];
    case 'task':
      return [
        { label: 'New Epic', value: '/task --type=epic ', icon: '🏔️' },
        { label: 'To Inbox', value: '/task --lane=inbox ', icon: '📥' },
        { label: 'Schedule', value: '/schedule ', icon: '📅' }
      ];
    case 'reflection':
      return [
        { label: 'Analyze SOTA', value: '/sota ', icon: '🧠' },
        { label: 'Why?', value: 'Why is this happening?', icon: '🤔' }
      ];
    default:
      return [
        { label: 'Create Task', value: '/task ', icon: '✅' },
        { label: 'Fix Code', value: '/fix ', icon: '💻' },
        { label: 'Research', value: '/research ', icon: '🔎' }
      ];
  }
}
