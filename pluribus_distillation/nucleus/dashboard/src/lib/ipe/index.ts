/**
 * IPE (In-Place Editor) Module
 *
 * Exports all IPE functionality for dashboard integration.
 */

// Context capture
export {
  captureContext,
  captureContextSafe,
  generateInstanceId,
  getElementPath,
  detectElementType,
  extractTailwindClasses,
  extractCSSVariables,
  extractParentTokens,
  extractDataAttributes,
  extractComputedStyles,
  detectComponentName,
  extractAncestorSelectors,
  introspectWebGL,
  computeDistortion,
} from './context-capture';

// Token bridge (IPE ↔ tweakcn)
export {
  contextToThemeProps,
  themePropsToCSS,
  extractThemeFromRoot,
  applyThemeToRoot,
  applyThemeToElement,
  generateInstanceCSS,
  parseHSL,
  formatHSL,
  hslToRgb,
  contrastRatio,
  meetsWCAGAA,
  themePropsAdiff,
  mergeWithDefaults,
  CSS_VAR_TO_THEME,
  THEME_TO_CSS_VAR,
  DEFAULT_THEME,
} from './token-bridge';

// Types
export type {
  ElementType,
  UniformValue,
  ShaderContext,
  IPEContext,
  DistortionReport,
  IPEMode,
  IPEScope,
  SextetVerdict,
  IPEModeChangeEvent,
  IPEElementSelectEvent,
  IPETokenEditEvent,
  IPEShaderEditEvent,
  IPEChange,
  IPESaveEvent,
  IPEBusEvent,
  IPEInstanceOverride,
  IPELocalStorage,
  IPEPreferences,
  SextetGateResult,
  SextetValidationResult,
  AllowedCSSProperty,
} from './context-schema';

export type {
  ThemeStyleProps,
  ThemeStyles,
} from './token-bridge';

// Instance Manager
export {
  getInstanceManager,
  findElementByInstanceId,
  hasInstanceId,
  getElementInstanceId,
  type InstanceOverride,
  type InstanceManager,
} from './instance-manager';

// Persistence
export {
  loadFromStorage,
  saveToStorage,
  clearStorage,
  saveInstance,
  loadInstances,
  removeInstance,
  pushUndo,
  popUndo,
  clearUndo,
  savePreferences,
  loadPreferences,
  syncToRhizome,
  loadFromRhizome,
  mergeWithRhizome,
  saveGlobalTokens,
} from './persistence';

// Constants
export {
  ALLOWED_CSS_PROPERTIES,
  DEFAULT_IPE_PREFERENCES,
  DEFAULT_IPE_STORAGE,
} from './context-schema';
