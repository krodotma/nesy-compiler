/**
 * Art Department Components - Visual Effects & Generative Art
 * ===========================================================
 *
 * This module exports all art/visual effect components for the dashboard.
 *
 * Components:
 * - HeartbeatOmega: Real-time stability visualization
 * - FractalGradientCanvas: Algorithmic layered gradient backgrounds
 * - ChromaticVisualizer: Color/sound visualization
 * - AgentHUD: Agent status heads-up display
 */

// Generative backgrounds
export {
  FractalGradientCanvas,
  FractalGradientSection,
  FractalGradientPresets,
} from './FractalGradientCanvas';

// Real-time visualizers
export { HeartbeatOmega } from './HeartbeatOmega';

// Chromatic effects (lazy-loaded in production)
// export { ChromaticVisualizer } from '../chromatic/ChromaticVisualizer';
// export { AgentHUD } from '../chromatic/AgentHUD';
