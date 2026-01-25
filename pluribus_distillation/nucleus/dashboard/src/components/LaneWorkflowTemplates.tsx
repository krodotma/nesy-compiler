/**
 * LaneWorkflowTemplates - Predefined Workflow Templates
 *
 * Phase 8, Iteration 66 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Template library
 * - Template editor
 * - Import/export
 * - Template application
 * - Custom template creation
 */

import {
  component$,
  useSignal,
  $,
  type QRL,
} from '@builder.io/qwik';
import { Button } from './ui/Button';
import { Input } from './ui/Input';

// ============================================================================
// Types
// ============================================================================

export interface WorkflowStep {
  id: string;
  name: string;
  description?: string;
  defaultOwner?: string;
  estimatedDays?: number;
  dependencies?: string[];
  tags?: string[];
  autoTransition?: {
    onComplete?: string;
    condition?: string;
  };
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: 'development' | 'deployment' | 'testing' | 'custom';
  steps: WorkflowStep[];
  metadata?: {
    author?: string;
    version?: string;
    createdAt?: string;
    updatedAt?: string;
  };
}

export interface LaneWorkflowTemplatesProps {
  /** Available templates */
  templates?: WorkflowTemplate[];
  /** Callback when template is applied */
  onApplyTemplate$?: QRL<(template: WorkflowTemplate, options: ApplyOptions) => void>;
  /** Callback when template is saved */
  onSaveTemplate$?: QRL<(template: WorkflowTemplate) => void>;
  /** Callback when template is deleted */
  onDeleteTemplate$?: QRL<(templateId: string) => void>;
}

export interface ApplyOptions {
  prefix?: string;
  ownerOverrides?: Record<string, string>;
  skipSteps?: string[];
}

// ============================================================================
// Built-in Templates
// ============================================================================

const BUILTIN_TEMPLATES: WorkflowTemplate[] = [
  {
    id: 'feature-development',
    name: 'Feature Development',
    description: 'Standard feature development workflow with planning, implementation, and review',
    category: 'development',
    steps: [
      { id: 'planning', name: 'Planning & Design', estimatedDays: 2, tags: ['planning'] },
      { id: 'implementation', name: 'Implementation', estimatedDays: 5, dependencies: ['planning'], tags: ['coding'] },
      { id: 'testing', name: 'Testing', estimatedDays: 2, dependencies: ['implementation'], tags: ['testing'] },
      { id: 'review', name: 'Code Review', estimatedDays: 1, dependencies: ['testing'], tags: ['review'] },
      { id: 'merge', name: 'Merge & Deploy', estimatedDays: 1, dependencies: ['review'], tags: ['deployment'] },
    ],
    metadata: { author: 'system', version: '1.0' },
  },
  {
    id: 'bug-fix',
    name: 'Bug Fix',
    description: 'Quick bug fix workflow with investigation and verification',
    category: 'development',
    steps: [
      { id: 'investigate', name: 'Investigation', estimatedDays: 1, tags: ['debug'] },
      { id: 'fix', name: 'Fix Implementation', estimatedDays: 2, dependencies: ['investigate'], tags: ['coding'] },
      { id: 'verify', name: 'Verification', estimatedDays: 1, dependencies: ['fix'], tags: ['testing'] },
    ],
    metadata: { author: 'system', version: '1.0' },
  },
  {
    id: 'deployment-pipeline',
    name: 'Deployment Pipeline',
    description: 'Multi-stage deployment workflow with staging and production',
    category: 'deployment',
    steps: [
      { id: 'build', name: 'Build & Package', estimatedDays: 1, tags: ['build'] },
      { id: 'staging', name: 'Staging Deploy', estimatedDays: 1, dependencies: ['build'], tags: ['staging'] },
      { id: 'staging-test', name: 'Staging Tests', estimatedDays: 1, dependencies: ['staging'], tags: ['testing'] },
      { id: 'production', name: 'Production Deploy', estimatedDays: 1, dependencies: ['staging-test'], tags: ['production'] },
      { id: 'verify', name: 'Production Verify', estimatedDays: 1, dependencies: ['production'], tags: ['verification'] },
    ],
    metadata: { author: 'system', version: '1.0' },
  },
  {
    id: 'test-suite',
    name: 'Test Suite Execution',
    description: 'Comprehensive testing workflow with unit, integration, and E2E tests',
    category: 'testing',
    steps: [
      { id: 'unit', name: 'Unit Tests', estimatedDays: 1, tags: ['unit-test'] },
      { id: 'integration', name: 'Integration Tests', estimatedDays: 1, tags: ['integration-test'] },
      { id: 'e2e', name: 'E2E Tests', estimatedDays: 2, tags: ['e2e-test'] },
      { id: 'report', name: 'Report Generation', estimatedDays: 1, dependencies: ['unit', 'integration', 'e2e'], tags: ['reporting'] },
    ],
    metadata: { author: 'system', version: '1.0' },
  },
];

// ============================================================================
// Component
// ============================================================================

export const LaneWorkflowTemplates = component$<LaneWorkflowTemplatesProps>(({
  templates: propTemplates,
  onApplyTemplate$,
  onSaveTemplate$,
  onDeleteTemplate$,
}) => {
  // Merge built-in and custom templates
  const allTemplates = [...BUILTIN_TEMPLATES, ...(propTemplates || [])];

  // State
  const selectedTemplateId = useSignal<string | null>(null);
  const showEditor = useSignal(false);
  const showApplyModal = useSignal(false);
  const categoryFilter = useSignal<string>('all');
  const searchQuery = useSignal('');

  const applyOptions = useSignal<ApplyOptions>({
    prefix: '',
    ownerOverrides: {},
    skipSteps: [],
  });

  const editingTemplate = useSignal<WorkflowTemplate | null>(null);

  // Filter templates
  const filteredTemplates = allTemplates.filter(t => {
    if (categoryFilter.value !== 'all' && t.category !== categoryFilter.value) return false;
    if (searchQuery.value) {
      const q = searchQuery.value.toLowerCase();
      return t.name.toLowerCase().includes(q) || t.description.toLowerCase().includes(q);
    }
    return true;
  });

  // Get selected template
  const selectedTemplate = allTemplates.find(t => t.id === selectedTemplateId.value);

  // Apply template
  const applyTemplate = $(async () => {
    if (!selectedTemplate || !onApplyTemplate$) return;
    await onApplyTemplate$(selectedTemplate, applyOptions.value);
    showApplyModal.value = false;
    applyOptions.value = { prefix: '', ownerOverrides: {}, skipSteps: [] };
  });

  // Export template
  const exportTemplate = $((template: WorkflowTemplate) => {
    const data = JSON.stringify(template, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `template-${template.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Import template
  const importTemplate = $(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const text = await file.text();
      try {
        const template = JSON.parse(text) as WorkflowTemplate;
        if (onSaveTemplate$) {
          await onSaveTemplate$(template);
        }
      } catch {
        console.error('Failed to parse template');
      }
    };
    input.click();
  });

  // Create new template
  const createTemplate = $(() => {
    editingTemplate.value = {
      id: `custom-${Date.now()}`,
      name: 'New Template',
      description: '',
      category: 'custom',
      steps: [],
      metadata: { author: 'user', version: '1.0', createdAt: new Date().toISOString() },
    };
    showEditor.value = true;
  });

  const getCategoryColor = (cat: string) => {
    switch (cat) {
      case 'development': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'deployment': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
      case 'testing': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      default: return 'bg-muted/20 text-muted-foreground border-border/30';
    }
  };

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">WORKFLOW TEMPLATES</span>
          <span class="text-[9px] px-2 py-0.5 rounded bg-muted/20 text-muted-foreground">
            {allTemplates.length} templates
          </span>
        </div>
        <div class="flex items-center gap-1">
          <Button
            variant="tonal"
            class="text-[9px] h-6 px-2 py-1"
            onClick$={importTemplate}
          >
            Import
          </Button>
          <Button
            variant="primary"
            class="text-[9px] h-6 px-2 py-1"
            onClick$={createTemplate}
          >
            + New
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div class="flex items-center gap-2 p-2 border-b border-border/30">
        <Input
          type="search"
          label=""
          value={searchQuery.value}
          onInput$={(_, el) => { searchQuery.value = el.value; }}
          placeholder="Search templates..."
          class="flex-1"
        />
        <select
          value={categoryFilter.value}
          onChange$={(e) => { categoryFilter.value = (e.target as HTMLSelectElement).value; }}
          class="px-2 py-1 text-[10px] rounded bg-card border border-border/50 h-10"
        >
          <option value="all">All Categories</option>
          <option value="development">Development</option>
          <option value="deployment">Deployment</option>
          <option value="testing">Testing</option>
          <option value="custom">Custom</option>
        </select>
      </div>

      {/* Template list */}
      <div class="grid grid-cols-2 gap-2 p-2 max-h-[300px] overflow-y-auto">
        {filteredTemplates.map(template => (
          <div
            key={template.id}
            onClick$={() => { selectedTemplateId.value = template.id; }}
            class={`p-3 rounded border cursor-pointer transition-colors ${
              selectedTemplateId.value === template.id
                ? 'bg-primary/10 border-primary/30'
                : 'bg-muted/5 border-border/30 hover:bg-muted/10'
            }`}
          >
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs font-medium text-foreground">{template.name}</span>
              <span class={`text-[8px] px-1.5 py-0.5 rounded border ${getCategoryColor(template.category)}`}>
                {template.category}
              </span>
            </div>
            <div class="text-[9px] text-muted-foreground mb-2 line-clamp-2">
              {template.description}
            </div>
            <div class="flex items-center justify-between text-[8px]">
              <span class="text-muted-foreground">{template.steps.length} steps</span>
              {template.metadata?.version && (
                <span class="text-muted-foreground/50">v{template.metadata.version}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Selected template details */}
      {selectedTemplate && (
        <div class="p-3 border-t border-border/50 bg-muted/5">
          <div class="flex items-center justify-between mb-3">
            <span class="text-xs font-semibold text-foreground">{selectedTemplate.name}</span>
            <div class="flex items-center gap-1">
              <Button
                variant="secondary"
                class="text-[9px] h-6 px-2 py-1"
                onClick$={() => exportTemplate(selectedTemplate)}
              >
                Export
              </Button>
              <Button
                variant="primary"
                class="text-[9px] h-6 px-2 py-1"
                onClick$={() => { showApplyModal.value = true; }}
              >
                Apply Template
              </Button>
            </div>
          </div>

          {/* Steps preview */}
          <div class="space-y-1">
            {selectedTemplate.steps.map((step, index) => (
              <div key={step.id} class="flex items-center gap-2 text-[10px]">
                <span class="w-4 h-4 rounded-full bg-muted/30 flex items-center justify-center text-muted-foreground">
                  {index + 1}
                </span>
                <span class="flex-1 text-foreground">{step.name}</span>
                {step.estimatedDays && (
                  <span class="text-muted-foreground">{step.estimatedDays}d</span>
                )}
                {step.dependencies && step.dependencies.length > 0 && (
                  <span class="text-[8px] text-cyan-400">
                    \u2190 {step.dependencies.length}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Apply Modal */}
      {showApplyModal.value && selectedTemplate && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-96">
            <div class="text-xs font-semibold text-foreground mb-4">
              Apply: {selectedTemplate.name}
            </div>

            <div class="space-y-3">
              <div>
                <Input
                  label="Lane Name Prefix"
                  value={applyOptions.value.prefix}
                  onInput$={(_, el) => {
                    applyOptions.value = { ...applyOptions.value, prefix: el.value };
                  }}
                  placeholder="e.g., feature-xyz-"
                />
              </div>

              <div>
                <div class="text-[9px] text-muted-foreground mb-1">Steps to Create ({selectedTemplate.steps.length})</div>
                <div class="space-y-1 max-h-[150px] overflow-y-auto">
                  {selectedTemplate.steps.map(step => (
                    <label key={step.id} class="flex items-center gap-2 text-[10px]">
                      <input
                        type="checkbox"
                        checked={!applyOptions.value.skipSteps?.includes(step.id)}
                        onChange$={(e) => {
                          const skip = new Set(applyOptions.value.skipSteps || []);
                          if ((e.target as HTMLInputElement).checked) {
                            skip.delete(step.id);
                          } else {
                            skip.add(step.id);
                          }
                          applyOptions.value = { ...applyOptions.value, skipSteps: Array.from(skip) };
                        }}
                      />
                      <span class="text-foreground">{step.name}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div class="flex items-center gap-2 mt-4">
              <Button
                variant="secondary"
                class="flex-1 text-xs"
                onClick$={() => { showApplyModal.value = false; }}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                class="flex-1 text-xs"
                onClick$={applyTemplate}
              >
                Apply
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default LaneWorkflowTemplates;