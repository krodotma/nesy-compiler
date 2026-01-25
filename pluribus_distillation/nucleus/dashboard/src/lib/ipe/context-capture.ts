/**
 * IPE Context Capture Engine
 *
 * Intelligent element introspection for the In-Place Editor.
 * Captures context from any element: HTML, Web Components, Canvas/WebGL, SVG.
 *
 * Implements AUOM observer codec: Encode (capture) → Transform (edit) → Decode (apply)
 */

// ============================================================================
// Types
// ============================================================================

export type ElementType = 'html' | 'webcomponent' | 'canvas' | 'shader' | 'svg';

export interface UniformValue {
  type: 'float' | 'int' | 'vec2' | 'vec3' | 'vec4' | 'mat4' | 'sampler2D';
  value: number | number[];
  min?: number;
  max?: number;
  step?: number;
}

export interface ShaderContext {
  vertexSource?: string;
  fragmentSource?: string;
  uniforms: Record<string, UniformValue>;
  canvasSize: [number, number];
  webglVersion: 1 | 2;
  programInfo?: {
    activeUniforms: number;
    activeAttributes: number;
  };
}

export interface IPEContext {
  // Identity
  elementType: ElementType;
  selector: string;
  instanceId: string;

  // CSS Layer
  computedStyles: Record<string, string>;
  tailwindClasses: string[];
  cssVariables: Record<string, string>;

  // Shader Layer (canvas/shader elements)
  shaderContext?: ShaderContext;

  // Component Layer (Qwik/Web Components)
  componentName?: string;
  props?: Record<string, unknown>;
  dataAttributes: Record<string, string>;

  // Ancestry
  parentTokens: string[];
  ancestorSelectors: string[];

  // Metadata
  capturedAt: string;
  capturedBy: string;

  // Bounds for overlay positioning
  bounds: DOMRect;
}

// ============================================================================
// Constants
// ============================================================================

/** CSS properties relevant for design token editing */
const RELEVANT_CSS_PROPERTIES = [
  // Colors
  'color', 'background-color', 'border-color', 'outline-color',
  'fill', 'stroke', 'box-shadow', 'text-shadow',
  // Typography
  'font-family', 'font-size', 'font-weight', 'line-height',
  'letter-spacing', 'text-transform',
  // Spacing
  'padding', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
  'margin', 'margin-top', 'margin-right', 'margin-bottom', 'margin-left',
  'gap', 'row-gap', 'column-gap',
  // Borders
  'border-radius', 'border-width', 'border-style',
  // Layout
  'width', 'height', 'max-width', 'max-height', 'min-width', 'min-height',
  // Effects
  'opacity', 'filter', 'backdrop-filter', 'transform',
  'transition', 'animation',
];

/** Design token CSS variable prefixes */
const TOKEN_PREFIXES = [
  '--background', '--foreground', '--card', '--popover', '--primary',
  '--secondary', '--muted', '--accent', '--destructive', '--border',
  '--input', '--ring', '--radius', '--font',
];

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate a stable, deterministic instance ID from element path
 */
export function generateInstanceId(element: Element): string {
  const path = getElementPath(element);
  return `ipe-${hashString(path).slice(0, 12)}`;
}

/**
 * Simple string hash (djb2 algorithm)
 */
function hashString(str: string): string {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
  }
  return Math.abs(hash).toString(36);
}

/**
 * Build CSS selector path to element
 */
export function getElementPath(element: Element): string {
  const parts: string[] = [];
  let current: Element | null = element;

  while (current && current !== document.body && current !== document.documentElement) {
    let selector = current.tagName.toLowerCase();

    // Add ID if present (most specific)
    if (current.id) {
      selector += `#${CSS.escape(current.id)}`;
    } else if (current.className && typeof current.className === 'string') {
      // Add first class for specificity
      const firstClass = current.className.trim().split(/\s+/)[0];
      if (firstClass) {
        selector += `.${CSS.escape(firstClass)}`;
      }
    }

    // Add nth-child if siblings have same tag
    const parent = current.parentElement;
    if (parent) {
      const siblings = Array.from(parent.children);
      const sameTagSiblings = siblings.filter(s => s.tagName === current!.tagName);
      if (sameTagSiblings.length > 1) {
        const index = siblings.indexOf(current);
        selector += `:nth-child(${index + 1})`;
      }
    }

    parts.unshift(selector);
    current = current.parentElement;
  }

  return parts.join(' > ');
}

/**
 * Detect element type
 */
export function detectElementType(element: Element): ElementType {
  const tagName = element.tagName.toLowerCase();

  // SVG elements
  if (element instanceof SVGElement || tagName === 'svg') {
    return 'svg';
  }

  // Canvas with WebGL context
  if (element instanceof HTMLCanvasElement) {
    const gl = element.getContext('webgl2') || element.getContext('webgl');
    if (gl) {
      return 'shader';
    }
    return 'canvas';
  }

  // Web Components (custom elements)
  if (tagName.includes('-') || element.hasAttribute('data-qwik-key')) {
    return 'webcomponent';
  }

  return 'html';
}

/**
 * Extract Tailwind classes from element
 */
export function extractTailwindClasses(element: Element): string[] {
  if (!element.className || typeof element.className !== 'string') {
    return [];
  }

  const allClasses = element.className.trim().split(/\s+/);

  // Tailwind class patterns (simplified detection)
  const twPatterns = [
    /^(bg|text|border|ring|shadow|rounded|p|px|py|pt|pb|pl|pr|m|mx|my|mt|mb|ml|mr)-/,
    /^(w|h|min-w|min-h|max-w|max-h)-/,
    /^(flex|grid|block|inline|hidden)/,
    /^(gap|space)-/,
    /^(font|tracking|leading)-/,
    /^(opacity|blur|brightness|contrast)-/,
    /^(hover|focus|active|disabled|dark):/,
    /^(sm|md|lg|xl|2xl):/,
  ];

  return allClasses.filter(cls =>
    twPatterns.some(pattern => pattern.test(cls))
  );
}

/**
 * Extract CSS variables (design tokens) in scope for element
 */
export function extractCSSVariables(element: Element): Record<string, string> {
  const variables: Record<string, string> = {};
  const computed = getComputedStyle(element);

  // Check each token prefix
  for (const prefix of TOKEN_PREFIXES) {
    const value = computed.getPropertyValue(prefix).trim();
    if (value) {
      variables[prefix] = value;
    }
  }

  // Also check for any custom property on the element's style
  if (element instanceof HTMLElement && element.style.cssText) {
    const matches = element.style.cssText.matchAll(/(--.+?):\s*([^;]+)/g);
    for (const match of matches) {
      variables[match[1]] = match[2].trim();
    }
  }

  return variables;
}

/**
 * Extract parent tokens (inherited CSS variables)
 */
export function extractParentTokens(element: Element): string[] {
  const tokens: string[] = [];
  let current = element.parentElement;

  while (current) {
    if (current instanceof HTMLElement && current.style.cssText) {
      const matches = current.style.cssText.matchAll(/(--[^:]+):/g);
      for (const match of matches) {
        if (!tokens.includes(match[1])) {
          tokens.push(match[1]);
        }
      }
    }
    current = current.parentElement;
  }

  return tokens;
}

/**
 * Extract data attributes
 */
export function extractDataAttributes(element: Element): Record<string, string> {
  const dataAttrs: Record<string, string> = {};

  for (const attr of element.attributes) {
    if (attr.name.startsWith('data-')) {
      dataAttrs[attr.name] = attr.value;
    }
  }

  return dataAttrs;
}

/**
 * Extract relevant computed styles
 */
export function extractComputedStyles(element: Element): Record<string, string> {
  const styles: Record<string, string> = {};
  const computed = getComputedStyle(element);

  for (const prop of RELEVANT_CSS_PROPERTIES) {
    const value = computed.getPropertyValue(prop);
    if (value && value !== 'none' && value !== 'auto' && value !== 'normal') {
      styles[prop] = value;
    }
  }

  return styles;
}

/**
 * Detect Qwik component name from element
 */
export function detectComponentName(element: Element): string | undefined {
  // Qwik uses data-qwik-key attribute
  const qwikKey = element.getAttribute('data-qwik-key');
  if (qwikKey) {
    // Extract component name from key (format: component_hash)
    const match = qwikKey.match(/^([A-Za-z]+)/);
    if (match) {
      return match[1];
    }
  }

  // Custom element tag name
  const tagName = element.tagName.toLowerCase();
  if (tagName.includes('-')) {
    return tagName;
  }

  // Check for common component wrappers
  const className = element.className;
  if (typeof className === 'string') {
    // Look for BEM-style component names
    const match = className.match(/^([A-Z][a-zA-Z]+)/);
    if (match) {
      return match[1];
    }
  }

  return undefined;
}

/**
 * Extract ancestor selectors for context
 */
export function extractAncestorSelectors(element: Element, depth = 3): string[] {
  const selectors: string[] = [];
  let current = element.parentElement;
  let count = 0;

  while (current && count < depth) {
    let selector = current.tagName.toLowerCase();
    if (current.id) {
      selector += `#${current.id}`;
    } else if (current.className && typeof current.className === 'string') {
      const firstClass = current.className.trim().split(/\s+/)[0];
      if (firstClass) {
        selector += `.${firstClass}`;
      }
    }
    selectors.push(selector);
    current = current.parentElement;
    count++;
  }

  return selectors;
}

// ============================================================================
// WebGL/Shader Introspection
// ============================================================================

/**
 * Introspect WebGL context for shader information
 */
export function introspectWebGL(canvas: HTMLCanvasElement): ShaderContext | undefined {
  const gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null
    || canvas.getContext('webgl') as WebGLRenderingContext | null;

  if (!gl) return undefined;

  const isWebGL2 = gl instanceof WebGL2RenderingContext;

  const context: ShaderContext = {
    uniforms: {},
    canvasSize: [canvas.width, canvas.height],
    webglVersion: isWebGL2 ? 2 : 1,
  };

  // Try to get current program
  const program = gl.getParameter(gl.CURRENT_PROGRAM) as WebGLProgram | null;
  if (!program) {
    return context;
  }

  // Get program info
  const activeUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
  const activeAttributes = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES) as number;

  context.programInfo = { activeUniforms, activeAttributes };

  // Extract uniform values
  for (let i = 0; i < activeUniforms; i++) {
    const info = gl.getActiveUniform(program, i);
    if (!info) continue;

    const location = gl.getUniformLocation(program, info.name);
    if (!location) continue;

    const uniformValue = extractUniformValue(gl, location, info);
    if (uniformValue) {
      context.uniforms[info.name] = uniformValue;
    }
  }

  // Try to extract shader source from debug extension (if available)
  const debugInfo = gl.getExtension('WEBGL_debug_shaders');
  if (debugInfo) {
    const shaders = gl.getAttachedShaders(program);
    if (shaders) {
      for (const shader of shaders) {
        const type = gl.getShaderParameter(shader, gl.SHADER_TYPE);
        const source = gl.getShaderSource(shader);
        if (type === gl.VERTEX_SHADER) {
          context.vertexSource = source || undefined;
        } else if (type === gl.FRAGMENT_SHADER) {
          context.fragmentSource = source || undefined;
        }
      }
    }
  }

  return context;
}

/**
 * Extract uniform value based on type
 */
function extractUniformValue(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  location: WebGLUniformLocation,
  info: WebGLActiveInfo
): UniformValue | undefined {
  const uniformValue: UniformValue = {
    type: 'float',
    value: 0,
  };

  try {
    const value = gl.getUniform(gl.getParameter(gl.CURRENT_PROGRAM) as WebGLProgram, location);

    switch (info.type) {
      case gl.FLOAT:
        uniformValue.type = 'float';
        uniformValue.value = value as number;
        // Set reasonable defaults for common uniforms
        if (info.name.includes('Time') || info.name.includes('time')) {
          uniformValue.min = 0;
          uniformValue.max = 100;
          uniformValue.step = 0.1;
        } else {
          uniformValue.min = 0;
          uniformValue.max = 10;
          uniformValue.step = 0.01;
        }
        break;

      case gl.INT:
        uniformValue.type = 'int';
        uniformValue.value = value as number;
        uniformValue.min = 0;
        uniformValue.max = 100;
        uniformValue.step = 1;
        break;

      case gl.FLOAT_VEC2:
        uniformValue.type = 'vec2';
        uniformValue.value = Array.from(value as Float32Array);
        break;

      case gl.FLOAT_VEC3:
        uniformValue.type = 'vec3';
        uniformValue.value = Array.from(value as Float32Array);
        break;

      case gl.FLOAT_VEC4:
        uniformValue.type = 'vec4';
        uniformValue.value = Array.from(value as Float32Array);
        break;

      case gl.FLOAT_MAT4:
        uniformValue.type = 'mat4';
        uniformValue.value = Array.from(value as Float32Array);
        break;

      case gl.SAMPLER_2D:
        uniformValue.type = 'sampler2D';
        uniformValue.value = value as number;
        break;

      default:
        return undefined;
    }

    return uniformValue;
  } catch {
    return undefined;
  }
}

// ============================================================================
// Main Capture Function
// ============================================================================

/**
 * Capture full context of an element for IPE editing
 */
export function captureContext(
  element: Element,
  actor: string = 'ipe'
): IPEContext {
  const elementType = detectElementType(element);

  const context: IPEContext = {
    // Identity
    elementType,
    selector: getElementPath(element),
    instanceId: generateInstanceId(element),

    // CSS Layer
    computedStyles: extractComputedStyles(element),
    tailwindClasses: extractTailwindClasses(element),
    cssVariables: extractCSSVariables(element),

    // Component Layer
    componentName: detectComponentName(element),
    dataAttributes: extractDataAttributes(element),

    // Ancestry
    parentTokens: extractParentTokens(element),
    ancestorSelectors: extractAncestorSelectors(element),

    // Metadata
    capturedAt: new Date().toISOString(),
    capturedBy: actor,

    // Bounds
    bounds: element.getBoundingClientRect(),
  };

  // Add shader context for canvas/shader elements
  if ((elementType === 'canvas' || elementType === 'shader') && element instanceof HTMLCanvasElement) {
    context.shaderContext = introspectWebGL(element);
  }

  return context;
}

/**
 * Capture context with validation (for production use)
 */
export function captureContextSafe(
  element: Element,
  actor: string = 'ipe'
): IPEContext | null {
  try {
    return captureContext(element, actor);
  } catch (error) {
    console.error('[IPE] Context capture failed:', error);
    return null;
  }
}

// ============================================================================
// Context Comparison (for AUOM distortion budget)
// ============================================================================

export interface DistortionReport {
  colorDelta: number;      // 0-1 normalized color difference
  layoutShift: number;     // Cumulative layout shift
  propertyChanges: number; // Count of changed properties
  withinBudget: boolean;
}

/**
 * Compare two contexts to compute distortion (AUOM verification)
 */
export function computeDistortion(
  original: IPEContext,
  modified: IPEContext,
  budget: { maxColorDelta?: number; maxLayoutShift?: number; maxPropertyChanges?: number } = {}
): DistortionReport {
  const { maxColorDelta = 0.3, maxLayoutShift = 0.1, maxPropertyChanges = 10 } = budget;

  // Count property changes
  let propertyChanges = 0;
  const allKeys = new Set([
    ...Object.keys(original.computedStyles),
    ...Object.keys(modified.computedStyles),
  ]);

  for (const key of allKeys) {
    if (original.computedStyles[key] !== modified.computedStyles[key]) {
      propertyChanges++;
    }
  }

  // Calculate layout shift (simplified)
  const layoutShift = Math.abs(
    (modified.bounds.width - original.bounds.width) / original.bounds.width +
    (modified.bounds.height - original.bounds.height) / original.bounds.height
  ) / 2;

  // Calculate color delta (simplified - would use CIE2000 in production)
  let colorDelta = 0;
  const colorProps = ['color', 'background-color', 'border-color'];
  for (const prop of colorProps) {
    if (original.computedStyles[prop] !== modified.computedStyles[prop]) {
      colorDelta += 0.1; // Simplified metric
    }
  }
  colorDelta = Math.min(1, colorDelta);

  return {
    colorDelta,
    layoutShift,
    propertyChanges,
    withinBudget:
      colorDelta <= maxColorDelta &&
      layoutShift <= maxLayoutShift &&
      propertyChanges <= maxPropertyChanges,
  };
}

// ============================================================================
// Export default
// ============================================================================

export default {
  captureContext,
  captureContextSafe,
  generateInstanceId,
  getElementPath,
  detectElementType,
  extractTailwindClasses,
  extractCSSVariables,
  introspectWebGL,
  computeDistortion,
};
