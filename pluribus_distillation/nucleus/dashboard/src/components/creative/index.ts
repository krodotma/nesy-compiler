/**
 * Creative Tools Components Index
 *
 * Re-exports all creative visualization and animation tools:
 * - CablesGLEmbed: Node-based WebGL visual editor (cables.gl)
 * - P5Canvas: Creative coding library (p5.js)
 * - ManimPreview: Mathematical animations (ManimCE)
 */

// Cables.gl WebGL Integration
export { CablesGLEmbed, CABLES_PRESETS } from './CablesGLEmbed';
export type {
  CablesGLEmbedProps,
  CablesPatch,
  CablesVariable,
  CablesEventPayload,
} from './CablesGLEmbed';

// p5.js Creative Coding
export { P5Canvas, P5_PRESETS } from './P5Canvas';
export type {
  P5CanvasProps,
  P5SketchFn,
  P5Instance,
  P5EventPayload,
} from './P5Canvas';

// ManimCE Mathematical Animations
export { ManimPreview, MANIM_TEMPLATES } from './ManimPreview';
export type {
  ManimPreviewProps,
  ManimRenderRequest,
  ManimRenderResult,
  ManimEventPayload,
} from './ManimPreview';
