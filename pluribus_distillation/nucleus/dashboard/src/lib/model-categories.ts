/**
 * Model Categories - Fine-grained categorization and portfolio selection for WebLLM models
 *
 * This module provides:
 * - ModelPurpose taxonomy for categorizing models by use case
 * - Category-based model selection functions
 * - Optimal model portfolio selector for diverse coverage
 */

// ============================================================================
// TYPES
// ============================================================================

/**
 * Model purpose categories for fine-grained classification
 * - 'vision': Visual understanding models (image analysis, OCR)
 * - 'language': General-purpose language models
 * - 'multimodal': Models with multiple input/output modalities
 * - 'coding': Code-specialized models (generation, completion, refactoring)
 * - 'reasoning': Reasoning-focused models (logic, math, planning)
 */
export type ModelPurpose = 'vision' | 'language' | 'multimodal' | 'coding' | 'reasoning';

/**
 * Category metadata for display and selection
 */
export interface CategoryInfo {
  purpose: ModelPurpose;
  label: string;
  description: string;
  color: string;
  icon: string;
  priority: number; // Lower = higher priority for portfolio selection
}

/**
 * Portfolio selection preferences
 */
export interface PortfolioPreferences {
  minCategories?: number; // Minimum number of different categories (default: 3)
  maxModels?: number; // Maximum models to select (default: 5)
  prioritizeVision?: boolean; // Ensure at least one vision model
  prioritizeFast?: boolean; // Prefer faster models when available
  deviceMemoryGB?: number; // Available memory for size constraints
}

// ============================================================================
// CONSTANTS
// ============================================================================

/**
 * Category definitions with metadata
 */
export const CATEGORY_INFO: Record<ModelPurpose, CategoryInfo> = {
  vision: {
    purpose: 'vision',
    label: 'Vision',
    description: 'Image and visual understanding',
    color: 'purple',
    icon: '👁️',
    priority: 1, // High priority - unique capability
  },
  coding: {
    purpose: 'coding',
    label: 'Coding',
    description: 'Code generation and analysis',
    color: 'green',
    icon: '💻',
    priority: 2,
  },
  reasoning: {
    purpose: 'reasoning',
    label: 'Reasoning',
    description: 'Logic, math, and planning',
    color: 'blue',
    icon: '🧮',
    priority: 3,
  },
  language: {
    purpose: 'language',
    label: 'Language',
    description: 'General text generation',
    color: 'cyan',
    icon: '💬',
    priority: 4,
  },
  multimodal: {
    purpose: 'multimodal',
    label: 'Multimodal',
    description: 'Multiple input/output types',
    color: 'orange',
    icon: '🎭',
    priority: 5,
  },
};

/**
 * Badge styles for category display
 */
export const CATEGORY_BADGE_STYLES: Record<ModelPurpose, string> = {
  vision: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  coding: 'bg-green-500/20 text-green-400 border-green-500/30',
  reasoning: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  language: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  multimodal: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get category info for a purpose
 */
export function getCategoryInfo(purpose: ModelPurpose): CategoryInfo {
  return CATEGORY_INFO[purpose];
}

/**
 * Get badge style class for a category
 */
export function getCategoryBadgeStyle(purpose: ModelPurpose): string {
  return CATEGORY_BADGE_STYLES[purpose] || CATEGORY_BADGE_STYLES.language;
}

/**
 * Get all available categories sorted by priority
 */
export function getOrderedCategories(): ModelPurpose[] {
  return (Object.values(CATEGORY_INFO) as CategoryInfo[])
    .sort((a, b) => a.priority - b.priority)
    .map((c) => c.purpose);
}

/**
 * Calculate category coverage from a list of model purposes
 */
export function calculateCategoryCoverage(purposes: ModelPurpose[]): {
  covered: ModelPurpose[];
  missing: ModelPurpose[];
  coverage: number;
} {
  const allCategories = getOrderedCategories();
  const uniquePurposes = [...new Set(purposes)];
  const covered = allCategories.filter((c) => uniquePurposes.includes(c));
  const missing = allCategories.filter((c) => !uniquePurposes.includes(c));
  const coverage = allCategories.length > 0 ? (covered.length / allCategories.length) * 100 : 0;

  return { covered, missing, coverage };
}

/**
 * Format category coverage as a display string
 */
export function formatCategoryCoverage(purposes: ModelPurpose[]): string {
  const { covered, coverage } = calculateCategoryCoverage(purposes);
  const icons = covered.map((p) => CATEGORY_INFO[p].icon).join(' ');
  return `${icons} ${Math.round(coverage)}% coverage (${covered.length}/5 categories)`;
}

// ============================================================================
// PORTFOLIO SELECTION
// ============================================================================

export interface ModelForPortfolio {
  baseId: string;
  name: string;
  purpose: ModelPurpose;
  vramMB: number;
  speedRating: number;
  category: string;
}

/**
 * Select an optimal portfolio of models ensuring category diversity
 *
 * Algorithm:
 * 1. Group models by purpose
 * 2. Select at least one from each priority category (vision, coding, reasoning)
 * 3. Fill remaining slots with fast language models
 * 4. Respect memory constraints
 */
export function selectOptimalPortfolio(
  models: ModelForPortfolio[],
  preferences: PortfolioPreferences = {}
): ModelForPortfolio[] {
  const {
    minCategories = 3,
    maxModels = 5,
    prioritizeVision = true,
    prioritizeFast = true,
    deviceMemoryGB = 16,
  } = preferences;

  // Group by purpose
  const byPurpose = new Map<ModelPurpose, ModelForPortfolio[]>();
  for (const model of models) {
    const list = byPurpose.get(model.purpose) || [];
    list.push(model);
    byPurpose.set(model.purpose, list);
  }

  // Sort each group by speed (if prioritizeFast) then by memory
  for (const [purpose, list] of byPurpose) {
    list.sort((a, b) => {
      if (prioritizeFast && a.speedRating !== b.speedRating) {
        return b.speedRating - a.speedRating; // Higher speed first
      }
      return a.vramMB - b.vramMB; // Lower memory first
    });
    byPurpose.set(purpose, list);
  }

  const selected: ModelForPortfolio[] = [];
  const selectedPurposes = new Set<ModelPurpose>();
  const maxVramMB = deviceMemoryGB * 1024 * 0.7; // Use 70% of available memory max

  // Priority order for category selection
  const priorityOrder: ModelPurpose[] = prioritizeVision
    ? ['vision', 'coding', 'reasoning', 'language', 'multimodal']
    : ['language', 'coding', 'reasoning', 'vision', 'multimodal'];

  // First pass: ensure minimum category coverage
  for (const purpose of priorityOrder) {
    if (selected.length >= maxModels) break;
    if (selectedPurposes.size >= minCategories && !priorityOrder.slice(0, 3).includes(purpose)) {
      continue;
    }

    const candidates = byPurpose.get(purpose) || [];
    for (const candidate of candidates) {
      const totalVram = selected.reduce((sum, m) => sum + m.vramMB, 0) + candidate.vramMB;
      if (totalVram <= maxVramMB) {
        selected.push(candidate);
        selectedPurposes.add(purpose);
        break;
      }
    }
  }

  // Second pass: fill remaining slots with best available
  if (selected.length < maxModels) {
    const allCandidates = models
      .filter((m) => !selected.some((s) => s.baseId === m.baseId))
      .sort((a, b) => {
        if (prioritizeFast && a.speedRating !== b.speedRating) {
          return b.speedRating - a.speedRating;
        }
        return a.vramMB - b.vramMB;
      });

    for (const candidate of allCandidates) {
      if (selected.length >= maxModels) break;
      const totalVram = selected.reduce((sum, m) => sum + m.vramMB, 0) + candidate.vramMB;
      if (totalVram <= maxVramMB) {
        selected.push(candidate);
        selectedPurposes.add(candidate.purpose);
      }
    }
  }

  return selected;
}

/**
 * Validate portfolio meets minimum requirements
 */
export function validatePortfolio(
  models: ModelForPortfolio[],
  requirements: { minCategories?: number; requireVision?: boolean; requireCoding?: boolean } = {}
): { valid: boolean; issues: string[] } {
  const { minCategories = 3, requireVision = false, requireCoding = false } = requirements;
  const issues: string[] = [];

  const purposes = new Set(models.map((m) => m.purpose));

  if (purposes.size < minCategories) {
    issues.push(`Insufficient category coverage: ${purposes.size}/${minCategories}`);
  }

  if (requireVision && !purposes.has('vision')) {
    issues.push('Missing required vision model');
  }

  if (requireCoding && !purposes.has('coding')) {
    issues.push('Missing required coding model');
  }

  return { valid: issues.length === 0, issues };
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  CATEGORY_INFO,
  CATEGORY_BADGE_STYLES,
  getCategoryInfo,
  getCategoryBadgeStyle,
  getOrderedCategories,
  calculateCategoryCoverage,
  formatCategoryCoverage,
  selectOptimalPortfolio,
  validatePortfolio,
};
