/**
 * SpeakerEmbeddingSelector.tsx
 * 
 * Phase B Step 14: Speaker embedding selector (avatar voices)
 * PBDESIGN: Voice profile cards with preview capability
 */

import { component$, useSignal } from '@builder.io/qwik';

interface VoiceProfile {
    id: string;
    name: string;
    description: string;
    avatar?: string;
    tags: string[];
}

interface SpeakerEmbeddingSelectorProps {
    profiles: VoiceProfile[];
    selectedId: string | null;
    onSelect$: (id: string) => void;
    onPreview$: (id: string) => void;
}

export const SpeakerEmbeddingSelector = component$<SpeakerEmbeddingSelectorProps>(({
    profiles,
    selectedId,
    onSelect$,
    onPreview$
}) => {
    const previewingId = useSignal<string | null>(null);

    return (
        <div class="bg-black/40 backdrop-blur-xl rounded-2xl border border-[var(--glass-border)] p-4 space-y-4">
            {/* Header */}
            <div class="flex items-center justify-between">
                <h3 class="text-sm font-semibold text-white/80">Voice Profile</h3>
                <span class="text-xs text-white/40">{profiles.length} available</span>
            </div>

            {/* Profile Grid */}
            <div class="grid grid-cols-2 gap-3">
                {profiles.map((profile) => {
                    const isSelected = profile.id === selectedId;
                    const isPreviewing = profile.id === previewingId.value;

                    return (
                        <button
                            key={profile.id}
                            onClick$={() => onSelect$(profile.id)}
                            class={`relative p-3 rounded-xl text-left transition-all duration-300 ${isSelected
                                    ? 'bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-500/40 shadow-[0_0_20px_rgba(6,182,212,0.2)]'
                                    : 'bg-white/5 border border-[var(--glass-border)] hover:bg-white/10 hover:border-[var(--glass-border-hover)]'
                                }`}
                        >
                            {/* Avatar */}
                            <div class={`w-10 h-10 rounded-full mb-2 flex items-center justify-center text-lg ${isSelected ? 'bg-cyan-500/30' : 'bg-white/10'
                                }`}>
                                {profile.avatar || profile.name[0].toUpperCase()}
                            </div>

                            {/* Info */}
                            <div class="space-y-1">
                                <span class={`text-sm font-medium ${isSelected ? 'text-cyan-300' : 'text-white/80'}`}>
                                    {profile.name}
                                </span>
                                <p class="text-[10px] text-white/40 line-clamp-2">
                                    {profile.description}
                                </p>
                            </div>

                            {/* Tags */}
                            <div class="flex flex-wrap gap-1 mt-2">
                                {profile.tags.slice(0, 2).map((tag) => (
                                    <span key={tag} class="px-1.5 py-0.5 text-[9px] bg-white/10 rounded text-white/50">
                                        {tag}
                                    </span>
                                ))}
                            </div>

                            {/* Preview Button */}
                            <button
                                onClick$={(e) => {
                                    e.stopPropagation();
                                    previewingId.value = profile.id;
                                    onPreview$(profile.id);
                                    setTimeout(() => previewingId.value = null, 2000);
                                }}
                                class={`absolute top-2 right-2 w-7 h-7 rounded-full flex items-center justify-center transition-all ${isPreviewing
                                        ? 'bg-green-500/30 text-green-400'
                                        : 'bg-white/10 text-white/50 hover:bg-white/20 hover:text-white/70'
                                    }`}
                            >
                                {isPreviewing ? (
                                    <div class="w-3 h-3 border-2 border-green-400 border-t-transparent rounded-full animate-spin" />
                                ) : (
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                        <polygon points="5 3 19 12 5 21 5 3" />
                                    </svg>
                                )}
                            </button>

                            {/* Selected indicator */}
                            {isSelected && (
                                <div class="absolute -top-1 -right-1 w-4 h-4 bg-cyan-500 rounded-full flex items-center justify-center">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3">
                                        <polyline points="20 6 9 17 4 12" />
                                    </svg>
                                </div>
                            )}
                        </button>
                    );
                })}
            </div>
        </div>
    );
});

export default SpeakerEmbeddingSelector;
