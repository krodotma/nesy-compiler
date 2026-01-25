/**
 * PublicToolBuilder.tsx - Simplified Tool Builder for Public Use
 *
 * A wizard-style interface for non-technical users to create custom tools.
 * Features: Template gallery, configuration wizard, live preview, export.
 *
 * Flow: Template -> Configure -> Preview -> Export
 */

import { component$, useSignal, useStore, useComputed$, $, type QRL } from '@builder.io/qwik';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Card } from './ui/Card';

// ============================================================================ 
// Types & Interfaces
// ============================================================================ 

export interface ToolTemplate {
  id: string;
  name: string;
  icon: string;
  description: string;
  category: 'automation' | 'data' | 'notification' | 'integration' | 'utility';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  defaults: Partial<ToolConfig>;
  fields: FieldConfig[];
  example?: string;
}

export interface FieldConfig {
  id: string;
  label: string;
  type: 'text' | 'textarea' | 'select' | 'toggle' | 'number' | 'tags';
  placeholder?: string;
  hint?: string;
  required?: boolean;
  options?: { value: string; label: string }[];
  validation?: 'url' | 'email' | 'identifier' | 'none';
}

export interface ToolConfig {
  name: string;
  description: string;
  trigger: 'manual' | 'schedule' | 'event' | 'webhook';
  triggerConfig: Record<string, string>;
  action: string;
  actionConfig: Record<string, string>;
  outputs: string[];
  tags: string[];
  enabled: boolean;
}

interface ValidationResult {
  valid: boolean;
  errors: Record<string, string>;
}

interface PublicToolBuilderProps {
  isOpen?: boolean;
  onClose$?: QRL<() => void>;
  onExport$?: QRL<(config: ToolConfig, format: 'json' | 'yaml') => void>;
}

// ============================================================================ 
// Templates Gallery
// ============================================================================ 

const TEMPLATES: ToolTemplate[] = [
  {
    id: 'notification',
    name: 'Send Notification',
    icon: 'bell',
    description: 'Send a notification when something happens',
    category: 'notification',
    difficulty: 'beginner',
    defaults: {
      trigger: 'event',
      action: 'notify',
    },
    fields: [
      { id: 'event_topic', label: 'When this event occurs', type: 'text', placeholder: 'e.g., task.completed', required: true },
      { id: 'message', label: 'Notification message', type: 'textarea', placeholder: 'The task has been completed!', required: true },
      { id: 'channel', label: 'Delivery channel', type: 'select', options: [
        { value: 'in-app', label: 'In-App Notification' },
        { value: 'email', label: 'Email' },
        { value: 'webhook', label: 'Webhook/Slack' },
      ]},
    ],
    example: 'Notify me via Slack when a new file is uploaded',
  },
  {
    id: 'scheduler',
    name: 'Scheduled Task',
    icon: 'clock',
    description: 'Run an action on a schedule',
    category: 'automation',
    difficulty: 'beginner',
    defaults: {
      trigger: 'schedule',
      action: 'run_task',
    },
    fields: [
      { id: 'schedule', label: 'Run every', type: 'select', options: [
        { value: '1h', label: 'Hour' },
        { value: '1d', label: 'Day' },
        { value: '1w', label: 'Week' },
        { value: 'custom', label: 'Custom (cron)' },
      ], required: true },
      { id: 'cron', label: 'Custom schedule (cron)', type: 'text', placeholder: '0 9 * * MON-FRI', hint: 'Only needed for custom schedule' },
      { id: 'task_name', label: 'Task to run', type: 'text', placeholder: 'cleanup_logs', required: true },
      { id: 'timeout', label: 'Timeout (seconds)', type: 'number', placeholder: '60' },
    ],
    example: 'Clean up old logs every day at 3 AM',
  },
  {
    id: 'data_transform',
    name: 'Data Transformer',
    icon: 'shuffle',
    description: 'Transform data from one format to another',
    category: 'data',
    difficulty: 'intermediate',
    defaults: {
      trigger: 'event',
      action: 'transform',
    },
    fields: [
      { id: 'source_topic', label: 'Source event', type: 'text', placeholder: 'data.incoming', required: true },
      { id: 'input_format', label: 'Input format', type: 'select', options: [
        { value: 'json', label: 'JSON' },
        { value: 'csv', label: 'CSV' },
        { value: 'xml', label: 'XML' },
        { value: 'text', label: 'Plain Text' },
      ], required: true },
      { id: 'output_format', label: 'Output format', type: 'select', options: [
        { value: 'json', label: 'JSON' },
        { value: 'csv', label: 'CSV' },
        { value: 'xml', label: 'XML' },
        { value: 'text', label: 'Plain Text' },
      ], required: true },
      { id: 'mapping', label: 'Field mappings', type: 'textarea', placeholder: 'source.name -> output.fullName', hint: 'One mapping per line' },
      { id: 'output_topic', label: 'Output event', type: 'text', placeholder: 'data.transformed', required: true },
    ],
    example: 'Convert incoming CSV data to JSON format',
  },
  {
    id: 'webhook',
    name: 'Webhook Receiver',
    icon: 'globe',
    description: 'Receive and process incoming webhooks',
    category: 'integration',
    difficulty: 'intermediate',
    defaults: {
      trigger: 'webhook',
      action: 'process',
    },
    fields: [
      { id: 'endpoint', label: 'Webhook path', type: 'text', placeholder: '/webhook/my-integration', required: true, validation: 'identifier' },
      { id: 'method', label: 'HTTP method', type: 'select', options: [
        { value: 'POST', label: 'POST' },
        { value: 'PUT', label: 'PUT' },
        { value: 'GET', label: 'GET' },
      ]},
      { id: 'secret', label: 'Secret (for validation)', type: 'text', placeholder: 'optional-webhook-secret', hint: 'Used to verify incoming requests' },
      { id: 'forward_to', label: 'Forward to event', type: 'text', placeholder: 'webhook.received', required: true },
    ],
    example: 'Receive GitHub webhooks and trigger builds',
  },
  {
    id: 'api_call',
    name: 'API Caller',
    icon: 'send',
    description: 'Call an external API when triggered',
    category: 'integration',
    difficulty: 'intermediate',
    defaults: {
      trigger: 'event',
      action: 'api_call',
    },
    fields: [
      { id: 'trigger_event', label: 'Trigger on event', type: 'text', placeholder: 'order.placed', required: true },
      { id: 'url', label: 'API URL', type: 'text', placeholder: 'https://api.example.com/endpoint', required: true, validation: 'url' },
      { id: 'method', label: 'HTTP method', type: 'select', options: [
        { value: 'GET', label: 'GET' },
        { value: 'POST', label: 'POST' },
        { value: 'PUT', label: 'PUT' },
        { value: 'DELETE', label: 'DELETE' },
      ]},
      { id: 'headers', label: 'Headers (JSON)', type: 'textarea', placeholder: '{"Authorization": "Bearer ..."}' },
      { id: 'body_template', label: 'Body template', type: 'textarea', placeholder: '{"order_id": "${event.id}"}', hint: 'Use ${event.field} for dynamic values' },
    ],
    example: 'Send order data to fulfillment API when order is placed',
  },
  {
    id: 'file_watcher',
    name: 'File Watcher',
    icon: 'folder',
    description: 'Watch for file changes and react',
    category: 'automation',
    difficulty: 'beginner',
    defaults: {
      trigger: 'event',
      action: 'file_action',
    },
    fields: [
      { id: 'watch_path', label: 'Path to watch', type: 'text', placeholder: '/data/uploads', required: true },
      { id: 'pattern', label: 'File pattern', type: 'text', placeholder: '*.csv', hint: 'Glob pattern to match files' },
      { id: 'events', label: 'Watch for', type: 'select', options: [
        { value: 'create', label: 'New files' },
        { value: 'modify', label: 'Modified files' },
        { value: 'delete', label: 'Deleted files' },
        { value: 'all', label: 'All changes' },
      ], required: true },
      { id: 'output_event', label: 'Emit event', type: 'text', placeholder: 'file.changed', required: true },
    ],
    example: 'Process new CSV files when they are added to uploads folder',
  },
  {
    id: 'aggregator',
    name: 'Event Aggregator',
    icon: 'layers',
    description: 'Collect and summarize multiple events',
    category: 'data',
    difficulty: 'advanced',
    defaults: {
      trigger: 'event',
      action: 'aggregate',
    },
    fields: [
      { id: 'source_events', label: 'Source events', type: 'tags', placeholder: 'Add event topics...', required: true, hint: 'Events to aggregate' },
      { id: 'window', label: 'Time window', type: 'select', options: [
        { value: '1m', label: '1 minute' },
        { value: '5m', label: '5 minutes' },
        { value: '1h', label: '1 hour' },
        { value: '1d', label: '1 day' },
      ], required: true },
      { id: 'aggregation', label: 'Aggregation type', type: 'select', options: [
        { value: 'count', label: 'Count' },
        { value: 'sum', label: 'Sum' },
        { value: 'avg', label: 'Average' },
        { value: 'collect', label: 'Collect all' },
      ], required: true },
      { id: 'group_by', label: 'Group by field', type: 'text', placeholder: 'user_id', hint: 'Optional field to group results' },
      { id: 'output_event', label: 'Output event', type: 'text', placeholder: 'metrics.aggregated', required: true },
    ],
    example: 'Count page views per user every 5 minutes',
  },
  {
    id: 'filter',
    name: 'Event Filter',
    icon: 'filter',
    description: 'Filter events based on conditions',
    category: 'utility',
    difficulty: 'beginner',
    defaults: {
      trigger: 'event',
      action: 'filter',
    },
    fields: [
      { id: 'source_event', label: 'Source event', type: 'text', placeholder: 'orders.all', required: true },
      { id: 'condition', label: 'Filter condition', type: 'textarea', placeholder: 'event.amount > 100 && event.status == "pending"', required: true, hint: 'JavaScript-like expression' },
      { id: 'pass_event', label: 'Event when matched', type: 'text', placeholder: 'orders.high_value', required: true },
      { id: 'fail_event', label: 'Event when not matched', type: 'text', placeholder: 'orders.low_value', hint: 'Optional' },
    ],
    example: 'Route high-value orders to priority queue',
  },
];

const CATEGORY_INFO: Record<string, { color: string; label: string }> = {
  automation: { color: 'blue', label: 'Automation' },
  data: { color: 'purple', label: 'Data Processing' },
  notification: { color: 'yellow', label: 'Notifications' },
  integration: { color: 'green', label: 'Integrations' },
  utility: { color: 'gray', label: 'Utilities' },
};

const DIFFICULTY_INFO: Record<string, { color: string; dots: number }> = {
  beginner: { color: 'green', dots: 1 },
  intermediate: { color: 'yellow', dots: 2 },
  advanced: { color: 'orange', dots: 3 },
};

const ICONS: Record<string, string> = {
  bell: 'M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9',
  clock: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z',
  shuffle: 'M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4',
  globe: 'M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9',
  send: 'M12 19l9 2-9-18-9 18 9-2zm0 0v-8',
  folder: 'M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z',
  layers: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10',
  filter: 'M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z',
};

const STEPS = [
  { id: 'template', label: 'Choose Template', icon: 'grid' },
  { id: 'configure', label: 'Configure', icon: 'settings' },
  { id: 'preview', label: 'Preview', icon: 'eye' },
  { id: 'export', label: 'Export', icon: 'download' },
];

// ============================================================================ 
// Helper Components
// ============================================================================ 

const Icon = component$<{ name: string; class?: string }>(({ name, class: className }) => {
  const path = ICONS[name] || '';
  return (
    <svg class={className || 'w-5 h-5'} fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
      <path stroke-linecap="round" stroke-linejoin="round" d={path} />
    </svg>
  );
});

const DifficultyDots = component$<{ level: 'beginner' | 'intermediate' | 'advanced' }>(({ level }) => {
  const info = DIFFICULTY_INFO[level];
  return (
    <div class="flex items-center gap-1">
      {[1, 2, 3].map((i) => (
        <span
          key={i}
          class={`w-2 h-2 rounded-full ${ 
            i <= info.dots
              ? info.color === 'green' ? 'bg-green-400'
                : info.color === 'yellow' ? 'bg-yellow-400'
                : 'bg-orange-400'
              : 'bg-muted/30'
          }`}
        />
      ))}
      <span class="text-xs text-muted-foreground ml-1 capitalize">{level}</span>
    </div>
  );
});

const TagsInput = component$<{ 
  value: string[];
  onChange$: QRL<(tags: string[]) => void>;
  placeholder?: string;
}>(({ value, onChange$, placeholder }) => {
  const inputValue = useSignal('');

  const addTag = $(() => {
    const tag = inputValue.value.trim();
    if (tag && !value.includes(tag)) {
      onChange$([...value, tag]);
      inputValue.value = '';
    }
  });

  const removeTag = $((tag: string) => {
    onChange$(value.filter((t) => t !== tag));
  });

  return (
    <div class="space-y-2">
      <div class="flex flex-wrap gap-1">
        {value.map((tag) => (
          <span
            key={tag}
            class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-primary/20 text-primary"
          >
            {tag}
            <button
              type="button"
              onClick$={() => removeTag(tag)}
              class="hover:text-red-400"
            >
              x
            </button>
          </span>
        ))}
      </div>
      <div class="flex gap-2">
        <Input
          value={inputValue.value}
          onInput$={(_, el) => (inputValue.value = el.value)}
          onKeyDown$={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              addTag();
            }
          }}
          placeholder={placeholder}
          class="flex-1"
        />
        <Button
          variant="tonal"
          onClick$={addTag}
          class="h-[56px]"
        >
          Add
        </Button>
      </div>
    </div>
  );
});

// ============================================================================ 
// Main Component
// ============================================================================ 

export const PublicToolBuilder = component$<PublicToolBuilderProps>(({
  isOpen = true,
  onClose$,
  onExport$,
}) => {
  const currentStep = useSignal(0);
  const selectedTemplate = useSignal<ToolTemplate | null>(null);
  const searchQuery = useSignal('');
  const categoryFilter = useSignal<string>('all');
  const validationErrors = useSignal<Record<string, string>>({});
  const exportFormat = useSignal<'json' | 'yaml'>('json');

  const config = useStore<ToolConfig>({
    name: '',
    description: '',
    trigger: 'manual',
    triggerConfig: {},
    action: '',
    actionConfig: {},
    outputs: [],
    tags: [],
    enabled: true,
  });

  // Filter templates based on search and category
  const filteredTemplates = useComputed$(() => {
    return TEMPLATES.filter((t) => {
      const matchesSearch = searchQuery.value === '' ||
        t.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
        t.description.toLowerCase().includes(searchQuery.value.toLowerCase());
      const matchesCategory = categoryFilter.value === 'all' || t.category === categoryFilter.value;
      return matchesSearch && matchesCategory;
    });
  });

  // Generate preview output
  const previewOutput = useComputed$(() => {
    const template = selectedTemplate.value;
    if (!template) return '';

    const output: Record<string, unknown> = {
      name: config.name || `my_${template.id}_tool`,
      description: config.description || template.description,
      template: template.id,
      trigger: {
        type: config.trigger,
        config: { ...config.triggerConfig },
      },
      action: {
        type: config.action || template.defaults.action,
        config: { ...config.actionConfig },
      },
      outputs: config.outputs.length > 0 ? config.outputs : undefined,
      tags: config.tags.length > 0 ? config.tags : undefined,
      enabled: config.enabled,
      version: '1.0.0',
      created_at: new Date().toISOString(),
    };

    if (exportFormat.value === 'yaml') {
      return jsonToYaml(output);
    }
    return JSON.stringify(output, null, 2);
  });

  const selectTemplate = $((template: ToolTemplate) => {
    selectedTemplate.value = template;
    config.trigger = template.defaults.trigger || 'manual';
    config.action = template.defaults.action || '';
    config.name = '';
    config.description = template.description;
    config.triggerConfig = {};
    config.actionConfig = {};
    currentStep.value = 1;
  });

  const validateConfig = $(() => {
    const errors: Record<string, string> = {};
    const template = selectedTemplate.value;
    if (!template) return { valid: false, errors: { template: 'No template selected' } };

    if (!config.name.trim()) {
      errors.name = 'Tool name is required';
    } else if (!/^[a-z0-9_-]+$/i.test(config.name)) {
      errors.name = 'Name can only contain letters, numbers, underscores, and hyphens';
    }

    for (const field of template.fields) {
      if (field.required) {
        const value = config.actionConfig[field.id];
        if (!value || value.trim() === '') {
          errors[field.id] = `${field.label} is required`;
        }
      }

      const value = config.actionConfig[field.id];
      if (value && field.validation) {
        if (field.validation === 'url' && !/^https?:\/\/.+/.test(value)) {
          errors[field.id] = 'Please enter a valid URL starting with http:// or https://';
        }
        if (field.validation === 'email' && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
          errors[field.id] = 'Please enter a valid email address';
        }
        if (field.validation === 'identifier' && !/^[a-z0-9_/-]+$/i.test(value)) {
          errors[field.id] = 'Only letters, numbers, underscores, hyphens, and slashes allowed';
        }
      }
    }

    validationErrors.value = errors;
    return { valid: Object.keys(errors).length === 0, errors };
  });

  const nextStep = $(async () => {
    if (currentStep.value === 1) {
      const result = await validateConfig();
      if (!result.valid) return;
    }
    if (currentStep.value < STEPS.length - 1) {
      currentStep.value++;
    }
  });

  const prevStep = $(() => {
    if (currentStep.value > 0) {
      currentStep.value--;
      validationErrors.value = {};
    }
  });

  const goToStep = $((step: number) => {
    if (step === 0) {
      currentStep.value = 0;
    } else if (step < currentStep.value) {
      currentStep.value = step;
      validationErrors.value = {};
    }
  });

  const copyToClipboard = $(async () => {
    try {
      await navigator.clipboard.writeText(previewOutput.value);
    } catch (e) {
      console.error('Failed to copy:', e);
    }
  });

  const downloadConfig = $(() => {
    const blob = new Blob([previewOutput.value], {
      type: exportFormat.value === 'json' ? 'application/json' : 'text/yaml',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${config.name || 'tool'}.${exportFormat.value}`;
    a.click();
    URL.revokeObjectURL(url);
  });

  const handleExport = $(async () => {
    if (onExport$) {
      await onExport$(config, exportFormat.value);
    } else {
      await downloadConfig();
    }
  });

  const reset = $(() => {
    currentStep.value = 0;
    selectedTemplate.value = null;
    config.name = '';
    config.description = '';
    config.trigger = 'manual';
    config.triggerConfig = {};
    config.action = '';
    config.actionConfig = {};
    config.outputs = [];
    config.tags = [];
    config.enabled = true;
    validationErrors.value = {};
  });

  if (!isOpen) return null;

  return (
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <Card class="w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl">
        {/* Header */}
        <div class="flex items-center justify-between border-b border-border px-6 py-4 shrink-0">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
              <svg class="w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <div>
              <h2 class="text-lg font-semibold">Tool Builder</h2>
              <p class="text-xs text-muted-foreground">
                {STEPS[currentStep.value].label}
              </p>
            </div>
          </div>
          {onClose$ && (
            <Button variant="icon" icon="close" onClick$={onClose$} class="rounded-lg p-2 hover:bg-muted/30 text-muted-foreground transition-colors" />
          )}
        </div>

        {/* Step Indicator */}
        <div class="flex items-center justify-between px-6 py-3 border-b border-border/50 bg-muted/5 shrink-0">
          {STEPS.map((step, i) => (
            <button
              key={step.id}
              onClick$={() => goToStep(i)}
              disabled={i > currentStep.value && (i === 0 || selectedTemplate.value === null)}
              class={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${ 
                i === currentStep.value
                  ? 'bg-primary/20 text-primary'
                  : i < currentStep.value
                  ? 'text-green-400 hover:bg-green-500/10 cursor-pointer'
                  : 'text-muted-foreground/50 cursor-not-allowed'
              }`}
            >
              <span class={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${ 
                i === currentStep.value
                  ? 'bg-primary text-primary-foreground'
                  : i < currentStep.value
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-muted/30'
              }`}>
                {i < currentStep.value ? '!' : i + 1}
              </span>
              <span class="hidden sm:inline">{step.label}</span>
            </button>
          ))}
        </div>

        {/* Content Area */}
        <div class="flex-1 overflow-y-auto">
          {/* Step 0: Template Selection */}
          {currentStep.value === 0 && (
            <div class="p-6 space-y-6">
              <div class="text-center max-w-xl mx-auto space-y-2">
                <h3 class="text-xl font-semibold">Choose a Template</h3>
                <p class="text-muted-foreground">
                  Start with a pre-built template and customize it to your needs.
                  No coding required!
                </p>
              </div>

              {/* Search and Filter */}
              <div class="flex flex-col sm:flex-row gap-3">
                <Input
                  value={searchQuery.value}
                  onInput$={(_, el) => (searchQuery.value = el.value)}
                  placeholder="Search templates..."
                  class="flex-1"
                  icon="search"
                />
                <div class="flex gap-1 flex-wrap">
                  <Button
                    variant={categoryFilter.value === 'all' ? 'tonal' : 'text'}
                    onClick$={() => (categoryFilter.value = 'all')}
                    class="h-10 text-xs"
                  >
                    All
                  </Button>
                  {Object.entries(CATEGORY_INFO).map(([key, { label }]) => (
                    <Button
                      key={key}
                      variant={categoryFilter.value === key ? 'tonal' : 'text'}
                      onClick$={() => (categoryFilter.value = key)}
                      class="h-10 text-xs"
                    >
                      {label}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Template Grid */}
              <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredTemplates.value.map((template) => {
                  const catInfo = CATEGORY_INFO[template.category];
                  return (
                    <Card
                      key={template.id}
                      variant="outlined"
                      class="group relative p-5 hover:border-primary/50 hover:bg-primary/5 text-left transition-all cursor-pointer"
                      onClick$={() => selectTemplate(template)}
                    >
                      <div class="flex items-start gap-3">
                        <div class={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${ 
                          catInfo.color === 'blue' ? 'bg-blue-500/20 text-blue-400' :
                          catInfo.color === 'purple' ? 'bg-purple-500/20 text-purple-400' :
                          catInfo.color === 'yellow' ? 'bg-yellow-500/20 text-yellow-400' :
                          catInfo.color === 'green' ? 'bg-green-500/20 text-green-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          <Icon name={template.icon} />
                        </div>
                        <div class="flex-1 min-w-0">
                          <h4 class="font-medium text-sm group-hover:text-primary transition-colors">
                            {template.name}
                          </h4>
                          <p class="text-xs text-muted-foreground mt-1 line-clamp-2">
                            {template.description}
                          </p>
                        </div>
                      </div>
                      <div class="flex items-center justify-between mt-4">
                        <span class={`text-[10px] px-2 py-0.5 rounded-full font-medium ${ 
                          catInfo.color === 'blue' ? 'bg-blue-500/10 text-blue-400' :
                          catInfo.color === 'purple' ? 'bg-purple-500/10 text-purple-400' :
                          catInfo.color === 'yellow' ? 'bg-yellow-500/10 text-yellow-400' :
                          catInfo.color === 'green' ? 'bg-green-500/10 text-green-400' :
                          'bg-gray-500/10 text-gray-400'
                        }`}>
                          {catInfo.label}
                        </span>
                        <DifficultyDots level={template.difficulty} />
                      </div>
                      {template.example && (
                        <p class="text-[10px] text-muted-foreground/60 mt-3 italic border-t border-border/30 pt-2">
                          e.g., "{template.example}"
                        </p>
                      )}
                    </Card>
                  );
                })}
              </div>

              {filteredTemplates.value.length === 0 && (
                <div class="text-center py-12 text-muted-foreground">
                  <p>No templates match your search.</p>
                  <button
                    onClick$={() => {
                      searchQuery.value = '';
                      categoryFilter.value = 'all';
                    }}
                    class="text-primary mt-2 hover:underline"
                  >
                    Clear filters
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Step 1: Configure */}
          {currentStep.value === 1 && selectedTemplate.value && (
            <div class="p-6 space-y-6">
              <div class="flex items-center gap-3 pb-4 border-b border-border/50">
                <div class={`w-10 h-10 rounded-lg flex items-center justify-center ${ 
                  CATEGORY_INFO[selectedTemplate.value.category].color === 'blue' ? 'bg-blue-500/20 text-blue-400' :
                  CATEGORY_INFO[selectedTemplate.value.category].color === 'purple' ? 'bg-purple-500/20 text-purple-400' :
                  CATEGORY_INFO[selectedTemplate.value.category].color === 'yellow' ? 'bg-yellow-500/20 text-yellow-400' :
                  CATEGORY_INFO[selectedTemplate.value.category].color === 'green' ? 'bg-green-500/20 text-green-400' :
                  'bg-gray-500/20 text-gray-400'
                }`}>
                  <Icon name={selectedTemplate.value.icon} />
                </div>
                <div>
                  <h3 class="font-semibold">{selectedTemplate.value.name}</h3>
                  <p class="text-xs text-muted-foreground">{selectedTemplate.value.description}</p>
                </div>
              </div>

              {/* Basic Info */}
              <div class="space-y-4">
                <h4 class="text-sm font-medium flex items-center gap-2">
                  <span class="w-6 h-6 rounded bg-primary/20 text-primary flex items-center justify-center text-xs">1</span>
                  Basic Information
                </h4>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 pl-8">
                  <div class="space-y-1.5">
                    <Input
                      label="Tool Name *"
                      value={config.name}
                      onInput$={(_, el) => (config.name = el.value)}
                      placeholder="my_awesome_tool"
                      error={validationErrors.value.name}
                    />
                    <p class="text-[10px] text-muted-foreground">Use lowercase letters, numbers, and underscores</p>
                  </div>
                  <div class="space-y-1.5">
                    <Input
                      label="Description"
                      value={config.description}
                      onInput$={(_, el) => (config.description = el.value)}
                      placeholder="What does this tool do?"
                    />
                  </div>
                </div>
              </div>

              {/* Template-specific Fields */}
              <div class="space-y-4">
                <h4 class="text-sm font-medium flex items-center gap-2">
                  <span class="w-6 h-6 rounded bg-primary/20 text-primary flex items-center justify-center text-xs">2</span>
                  Configuration
                </h4>
                <div class="space-y-4 pl-8">
                  {selectedTemplate.value.fields.map((field) => (
                    <div key={field.id} class="space-y-1.5">
                      <label class="text-xs font-medium text-muted-foreground flex items-center gap-1">
                        {field.label}
                        {field.required && <span class="text-red-400">*</span>}
                      </label>

                      {field.type === 'text' && (
                        <Input
                          label=""
                          value={config.actionConfig[field.id] || ''}
                          onInput$={(_, el) => (config.actionConfig[field.id] = el.value)}
                          placeholder={field.placeholder}
                          error={validationErrors.value[field.id]}
                        />
                      )}

                      {field.type === 'number' && (
                        <Input
                          type="number"
                          label=""
                          value={config.actionConfig[field.id] || ''}
                          onInput$={(_, el) => (config.actionConfig[field.id] = el.value)}
                          placeholder={field.placeholder}
                          error={validationErrors.value[field.id]}
                        />
                      )}

                      {field.type === 'textarea' && (
                        <Input
                          type="textarea"
                          label=""
                          value={config.actionConfig[field.id] || ''}
                          onInput$={(_, el) => (config.actionConfig[field.id] = el.value)}
                          placeholder={field.placeholder}
                          error={validationErrors.value[field.id]}
                        />
                      )}

                      {field.type === 'select' && (
                        <select
                          value={config.actionConfig[field.id] || ''}
                          onChange$={(e) => (config.actionConfig[field.id] = (e.target as HTMLSelectElement).value)}
                          class={`w-full px-3 py-2.5 rounded-lg bg-background border text-sm h-[56px] ${ 
                            validationErrors.value[field.id] ? 'border-red-500/50' : 'border-border'
                          }`}
                        >
                          <option value="">Select...</option>
                          {field.options?.map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      )}

                      {field.type === 'toggle' && (
                        <label class="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={config.actionConfig[field.id] === 'true'}
                            onChange$={(e) => (config.actionConfig[field.id] = (e.target as HTMLInputElement).checked ? 'true' : 'false')}
                            class="rounded"
                          />
                          <span class="text-sm">Enabled</span>
                        </label>
                      )}

                      {field.type === 'tags' && (
                        <TagsInput
                          value={(config.actionConfig[field.id] || '').split(',').filter(Boolean)}
                          onChange$={$((tags: string[]) => {
                            config.actionConfig[field.id] = tags.join(',');
                          })}
                          placeholder={field.placeholder}
                        />
                      )}

                      {field.hint && !validationErrors.value[field.id] && (
                        <p class="text-[10px] text-muted-foreground">{field.hint}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Tags */}
              <div class="space-y-4">
                <h4 class="text-sm font-medium flex items-center gap-2">
                  <span class="w-6 h-6 rounded bg-primary/20 text-primary flex items-center justify-center text-xs">3</span>
                  Tags (Optional)
                </h4>
                <div class="pl-8">
                  <TagsInput
                    value={config.tags}
                    onChange$={$((tags: string[]) => {
                      config.tags = tags;
                    })}
                    placeholder="Add tags for organization..."
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Preview */}
          {currentStep.value === 2 && (
            <div class="p-6 space-y-6">
              <div class="text-center max-w-xl mx-auto space-y-2">
                <h3 class="text-xl font-semibold">Preview Your Tool</h3>
                <p class="text-muted-foreground">
                  Review the configuration before exporting. You can switch between JSON and YAML formats.
                </p>
              </div>

              {/* Format Toggle */}
              <div class="flex justify-center">
                <div class="inline-flex rounded-lg border border-border overflow-hidden">
                  <Button
                    variant={exportFormat.value === 'json' ? 'tonal' : 'text'}
                    onClick$={() => (exportFormat.value = 'json')}
                    class="rounded-r-none"
                  >
                    JSON
                  </Button>
                  <Button
                    variant={exportFormat.value === 'yaml' ? 'tonal' : 'text'}
                    onClick$={() => (exportFormat.value = 'yaml')}
                    class="rounded-l-none"
                  >
                    YAML
                  </Button>
                </div>
              </div>

              {/* Code Preview */}
              <div class="relative">
                <div class="absolute right-2 top-2 z-10">
                  <Button
                    variant="tonal"
                    onClick$={copyToClipboard}
                    class="text-xs h-8"
                  >
                    Copy
                  </Button>
                </div>
                <pre class="p-4 rounded-xl bg-background border border-border overflow-x-auto text-sm font-mono">
                  <code>{previewOutput.value}</code>
                </pre>
              </div>

              {/* Summary Cards */}
              {selectedTemplate.value && (
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <Card padding="p-4" variant="filled">
                    <div class="text-xs text-muted-foreground mb-1">Template</div>
                    <div class="font-medium">{selectedTemplate.value.name}</div>
                  </Card>
                  <Card padding="p-4" variant="filled">
                    <div class="text-xs text-muted-foreground mb-1">Trigger</div>
                    <div class="font-medium capitalize">{config.trigger}</div>
                  </Card>
                  <Card padding="p-4" variant="filled">
                    <div class="text-xs text-muted-foreground mb-1">Fields Configured</div>
                    <div class="font-medium">
                      {Object.values(config.actionConfig).filter(Boolean).length} / {selectedTemplate.value.fields.length}
                    </div>
                  </Card>
                </div>
              )}
            </div>
          )}

          {/* Step 3: Export */}
          {currentStep.value === 3 && (
            <div class="p-6 space-y-6">
              <div class="text-center max-w-xl mx-auto space-y-2">
                <div class="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-4">
                  <svg class="w-8 h-8 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 class="text-xl font-semibold">Your Tool is Ready!</h3>
                <p class="text-muted-foreground">
                  Choose how you want to save or share your tool configuration.
                </p>
              </div>

              {/* Export Options */}
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-lg mx-auto">
                <Card
                  variant="outlined"
                  class="p-5 hover:border-primary/50 hover:bg-primary/5 text-left transition-all cursor-pointer group"
                  onClick$={() => {
                    exportFormat.value = 'json';
                    downloadConfig();
                  }}
                >
                  <div class="w-10 h-10 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                    <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                  </div>
                  <h4 class="font-medium">Download JSON</h4>
                  <p class="text-xs text-muted-foreground mt-1">
                    Standard format for most integrations
                  </p>
                </Card>

                <Card
                  variant="outlined"
                  class="p-5 hover:border-primary/50 hover:bg-primary/5 text-left transition-all cursor-pointer group"
                  onClick$={() => {
                    exportFormat.value = 'yaml';
                    downloadConfig();
                  }}
                >
                  <div class="w-10 h-10 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                    <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                  </div>
                  <h4 class="font-medium">Download YAML</h4>
                  <p class="text-xs text-muted-foreground mt-1">
                    Human-readable format for config files
                  </p>
                </Card>

                <Card
                  variant="outlined"
                  class="p-5 hover:border-primary/50 hover:bg-primary/5 text-left transition-all cursor-pointer group"
                  onClick$={copyToClipboard}
                >
                  <div class="w-10 h-10 rounded-lg bg-green-500/20 text-green-400 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                    <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                    </svg>
                  </div>
                  <h4 class="font-medium">Copy to Clipboard</h4>
                  <p class="text-xs text-muted-foreground mt-1">
                    Paste anywhere you need it
                  </p>
                </Card>

                {onExport$ && (
                  <Card
                    variant="outlined"
                    class="p-5 border-primary/50 bg-primary/10 hover:bg-primary/20 text-left transition-all cursor-pointer group"
                    onClick$={handleExport}
                  >
                    <div class="w-10 h-10 rounded-lg bg-primary/30 text-primary flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                      <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <h4 class="font-medium">Deploy Tool</h4>
                    <p class="text-xs text-muted-foreground mt-1">
                      Activate this tool immediately
                    </p>
                  </Card>
                )}
              </div>

              {/* Create Another */}
              <div class="text-center pt-4">
                <Button variant="text" onClick$={reset}>
                  Create another tool
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div class="flex items-center justify-between border-t border-border px-6 py-4 shrink-0 bg-muted/5">
          <Button variant="secondary" onClick$={reset} class="h-8 text-xs">
            Start Over
          </Button>
          <div class="flex items-center gap-3">
            {currentStep.value > 0 && (
              <Button variant="secondary" onClick$={prevStep} class="h-9 text-sm">
                ← Back
              </Button>
            )}
            {currentStep.value < STEPS.length - 1 ? (
              <Button variant="primary" onClick$={nextStep} disabled={currentStep.value === 0 && !selectedTemplate.value} class="h-9 text-sm">
                Next →
              </Button>
            ) : (
              <Button variant="primary" onClick$={publish} class="h-9 text-sm bg-green-500/20 text-green-300 border-green-500/30 hover:bg-green-500/30">
                🚀 Publish
              </Button>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
});

// ============================================================================ 
// Utility Functions
// ============================================================================ 

function jsonToYaml(obj: Record<string, unknown>, indent = 0): string {
  const spaces = '  '.repeat(indent);
  let result = '';

  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined || value === null) continue;

    if (typeof value === 'object' && !Array.isArray(value)) {
      result += `${spaces}${key}:\n`;
      result += jsonToYaml(value as Record<string, unknown>, indent + 1);
    } else if (Array.isArray(value)) {
      if (value.length === 0) continue;
      result += `${spaces}${key}:\n`;
      for (const item of value) {
        if (typeof item === 'object') {
          result += `${spaces}  -\n`;
          result += jsonToYaml(item as Record<string, unknown>, indent + 2);
        } else {
          result += `${spaces}  - ${item}\n`;
        }
      }
    } else if (typeof value === 'string') {
      // Quote strings that might need it
      const needsQuotes = value.includes(':') || value.includes('#') || value.includes('\n') || value === '';
      result += `${spaces}${key}: ${needsQuotes ? `"${value.replace(/"/g, '\"')}"` : value}\n`;
    } else {
      result += `${spaces}${key}: ${value}\n`;
    }
  }

  return result;
}

export default PublicToolBuilder;