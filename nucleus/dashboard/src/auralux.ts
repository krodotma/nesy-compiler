/**
 * Auralux Dashboard Components Index
 * Re-exports all Auralux frontend integration components.
 */

// Contexts
export { AuraluxProvider, AuraluxContext, useAuralux } from './contexts/AuraluxContext';
export type { AuraluxContextValue, AuraluxState, VADState, PipelineMetrics } from './contexts/AuraluxContext';

// Hooks
export { useVisemeLipSync, createThreeMorphAdapter } from './hooks/useVisemeLipSync';
export type { MorphTargetRef, LipSyncOptions } from './hooks/useVisemeLipSync';

export { useAvatarController } from './hooks/useAvatarController';
export type { UseAvatarControllerOptions, UseAvatarControllerReturn } from './hooks/useAvatarController';

// Components
export { VoiceHUD } from './components/VoiceHUD';

// Re-export core types from auralux
export type { VisemeFrame, PhonemeInput, VisemeName } from '../auralux/viseme_mapper';
export { phonemeToViseme, mapPhonemeSequence, getVisemeWeightsAtTime, VISEME_NAMES } from '../auralux/viseme_mapper';
export { AvatarController } from '../auralux/avatar_controller';
export type { MorphTargetMesh, AvatarControllerConfig, AvatarState } from '../auralux/avatar_controller';
