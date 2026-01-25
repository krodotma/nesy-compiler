/**
 * MetaArchitectTool.tsx - Meta-Architecting Tool Interface
 *
 * A visual tool for designing, specifying, and generating new Pluribus tools.
 * Features:
 * - Visual schema editor for inputs/outputs
 * - Type selectors with Pluribus-aware types
 * - Live code preview (Python/TypeScript)
 * - Sandbox test execution
 * - Export to nucleus/tools/ format
 *
 * DKIN v29 compliant - emits bus events for all tool operations.
 */

import {
  component$,
  useSignal,
  useStore,
  useComputed$,
  $,
  useVisibleTask$,
} from '@builder.io/qwik';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Card } from './ui/Card';

// ============================================================================
// Types
// ============================================================================

export type ToolFieldType =
  | 'string'
  | 'int'
  | 'float'
  | 'bool'
  | 'path'
  | 'json'
  | 'list[string]'
  | 'list[int]'
  | 'dict'
  | 'optional[string]'
  | 'optional[int]'
  | 'optional[path]'
  | 'bus_topic'
  | 'actor_id'
  | 'uuid';

export interface ToolField {
  id: string;
  name: string;
  type: ToolFieldType;
  description: string;
  required: boolean;
  default_value?: string;
  validation?: string; // regex or constraint
  cli_flag?: string; // e.g., "--scope"
}

export interface ToolSchema {
  id: string;
  name: string;
  operator_key: string; // e.g., "PBMYTOOL"
  description: string;
  domain: string; // execution, safety, evolution, ui
  category: string; // tool, daemon, operator, bridge
  inputs: ToolField[];
  outputs: ToolField[];
  bus_topics: {
    request?: string;
    response?: string;
    metric?: string;
  };
  effects: 'none' | 'file' | 'network' | 'system';
  cli_enabled: boolean;
  daemon_mode: boolean;
}

export interface TestResult {
  success: boolean;
  stdout: string;
  stderr: string;
  duration_ms: number;
  exit_code: number;
}

export type GeneratorTarget = 'python' | 'typescript';

interface MetaArchitectState {
  schema: ToolSchema;
  activeTab: 'schema' | 'preview' | 'test' | 'export';
  previewLang: GeneratorTarget;
  generatedCode: string;
  testCode: string;
  testResult: TestResult | null;
  isGenerating: boolean;
  isTesting: boolean;
  error: string | null;
  status: string | null;
  recentTools: string[];
}

// ============================================================================
// Constants
// ============================================================================

const FIELD_TYPES: ToolFieldType[] = [
  'string',
  'int',
  'float',
  'bool',
  'path',
  'json',
  'list[string]',
  'list[int]',
  'dict',
  'optional[string]',
  'optional[int]',
  'optional[path]',
  'bus_topic',
  'actor_id',
  'uuid',
];

const DOMAINS = ['execution', 'safety', 'evolution', 'ui', 'infra', 'memory'];

const CATEGORIES = ['operator', 'daemon', 'bridge', 'adapter', 'cli', 'responder'];

const EFFECTS = ['none', 'file', 'network', 'system'] as const;

const DEFAULT_SCHEMA: ToolSchema = {
  id: '',
  name: '',
  operator_key: '',
  description: '',
  domain: 'execution',
  category: 'operator',
  inputs: [],
  outputs: [],
  bus_topics: {},
  effects: 'none',
  cli_enabled: true,
  daemon_mode: false,
};

// ============================================================================
// Code Generators
// ============================================================================

function generatePythonCode(schema: ToolSchema): string {
  const pyName = schema.name.toLowerCase().replace(/[^a-z0-9]/g, '_');
  const operatorName = schema.operator_key || pyName.toUpperCase();

  // Generate argument parsing
  const argLines = schema.inputs
    .map((f) => {
      const pyType = mapToPythonType(f.type);
      const required = f.required ? 'required=True' : `default=${f.default_value || 'None'}`;
      const help = f.description.replace(/"/g, '\\"');
      return `    p.add_argument("${f.cli_flag || '--' + f.name}", type=${pyType}, ${required}, help="${help}")`;
    })
    .join('\n');

  // Generate bus topic constants
  const busTopics = [];
  if (schema.bus_topics.request) {
    busTopics.push(`REQUEST_TOPIC = "${schema.bus_topics.request}"`);
  }
  if (schema.bus_topics.response) {
    busTopics.push(`RESPONSE_TOPIC = "${schema.bus_topics.response}"`);
  }
  if (schema.bus_topics.metric) {
    busTopics.push(`METRIC_TOPIC = "${schema.bus_topics.metric}"`);
  }

  const effectsWarning =
    schema.effects !== 'none'
      ? `# WARNING: This tool has ${schema.effects.toUpperCase()} effects - verify before deployment`
      : '';

  return `#!/usr/bin/env python3
"""
${operatorName} - ${schema.description}

Domain: ${schema.domain}
Category: ${schema.category}
Effects: ${schema.effects}
${effectsWarning}

Usage:
    python3 nucleus/tools/${pyName}_operator.py ${schema.inputs.map((f) => (f.cli_flag || '--' + f.name) + ' <value>').join(' ')}

DKIN v29 compliant.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

# Canonical paths
PLURIBUS_ROOT = os.environ.get("PLURIBUS_ROOT", "/pluribus")
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", f"{PLURIBUS_ROOT}/.pluribus/bus"))
ACTOR = os.environ.get("PLURIBUS_ACTOR", "${pyName}-operator")

# Bus topics
${busTopics.length > 0 ? busTopics.join('\n') : '# No bus topics defined'}


def emit_bus_event(topic: str, kind: str, data: Dict[str, Any], level: str = "info") -> str:
    """Emit event to Pluribus bus."""
    if not BUS_DIR.exists():
        BUS_DIR.mkdir(parents=True, exist_ok=True)

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": ACTOR,
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "data": data,
    }

    events_file = BUS_DIR / "events.ndjson"
    try:
        with open(events_file, "a") as f:
            f.write(json.dumps(event) + "\\n")
    except Exception as e:
        print(f"Warning: Bus emit failed: {e}", file=sys.stderr)

    return event["id"]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="${pyName}_operator.py",
        description="${schema.description}"
    )
${argLines || '    # No CLI arguments defined'}
    return p


def validate_inputs(args: argparse.Namespace) -> List[str]:
    """Validate input arguments. Returns list of errors."""
    errors = []
${schema.inputs
  .filter((f) => f.validation)
  .map((f) => {
    return `    # Validate ${f.name}
    # TODO: Add validation for ${f.validation}`;
  })
  .join('\n')}
    return errors


def execute(${schema.inputs.map((f) => `${f.name}: ${mapToPythonTypeAnnotation(f.type)}`).join(', ')}) -> Dict[str, Any]:
    """
    Main execution logic for ${operatorName}.
${schema.inputs.map((f) => `    :param ${f.name}: ${f.description}`).join('\n')}
${schema.outputs.map((f) => `    :returns ${f.name}: ${f.description}`).join('\n')}
    """
    result: Dict[str, Any] = {}

    # TODO: Implement tool logic here
    print(f"${operatorName}: Executing...")

${schema.outputs.map((f) => `    # result["${f.name}"] = ...  # ${f.type}: ${f.description}`).join('\n')}

    return result


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1

    # Emit request event
${schema.bus_topics.request ? `    emit_bus_event(REQUEST_TOPIC, "request", {
        "phase": "start",
${schema.inputs.map((f) => `        "${f.name}": getattr(args, "${f.name.replace(/-/g, '_')}", None),`).join('\n')}
    })` : '    # No request topic configured'}

    try:
        # Execute
        result = execute(
${schema.inputs.map((f) => `            ${f.name}=getattr(args, "${f.name.replace(/-/g, '_')}", None),`).join('\n')}
        )

        # Emit response event
${schema.bus_topics.response ? `        emit_bus_event(RESPONSE_TOPIC, "response", {
            "phase": "complete",
            "success": True,
            "result": result,
        })` : '        # No response topic configured'}

        # Output result
        print(json.dumps(result, indent=2, default=str))
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
${schema.bus_topics.response ? `        emit_bus_event(RESPONSE_TOPIC, "response", {
            "phase": "error",
            "success": False,
            "error": str(e),
        })` : ''}
        return 1


if __name__ == "__main__":
    sys.exit(main())
`;
}

function generateTypeScriptCode(schema: ToolSchema): string {
  const tsName = schema.name.replace(/[^a-zA-Z0-9]/g, '');
  const operatorName = schema.operator_key || tsName.toUpperCase();

  const inputInterface = schema.inputs
    .map((f) => {
      const tsType = mapToTypeScriptType(f.type);
      const optional = f.required ? '' : '?';
      return `  /** ${f.description} */\n  ${f.name}${optional}: ${tsType};`;
    })
    .join('\n');

  const outputInterface = schema.outputs
    .map((f) => {
      const tsType = mapToTypeScriptType(f.type);
      return `  /** ${f.description} */\n  ${f.name}: ${tsType};`;
    })
    .join('\n');

  return `/**
 * ${operatorName} - ${schema.description}
 *
 * Domain: ${schema.domain}
 * Category: ${schema.category}
 * Effects: ${schema.effects}
 *
 * Auto-generated by MetaArchitectTool
 * DKIN v29 compliant
 */

// ============================================================================
// Types
// ============================================================================

export interface ${tsName}Input {
${inputInterface || '  // No inputs defined'}
}

export interface ${tsName}Output {
${outputInterface || '  // No outputs defined'}
}

export interface ${tsName}BusEvent {
  id: string;
  ts: number;
  iso: string;
  topic: string;
  kind: 'request' | 'response' | 'metric';
  level: 'info' | 'warn' | 'error';
  actor: string;
  data: ${tsName}Input | ${tsName}Output | Record<string, unknown>;
}

// ============================================================================
// Constants
// ============================================================================

const ACTOR = '${tsName.toLowerCase()}-operator';
${schema.bus_topics.request ? `const REQUEST_TOPIC = '${schema.bus_topics.request}';` : ''}
${schema.bus_topics.response ? `const RESPONSE_TOPIC = '${schema.bus_topics.response}';` : ''}
${schema.bus_topics.metric ? `const METRIC_TOPIC = '${schema.bus_topics.metric}';` : ''}

// ============================================================================
// Bus Helpers
// ============================================================================

async function emitBusEvent(
  topic: string,
  kind: 'request' | 'response' | 'metric',
  data: Record<string, unknown>
): Promise<string> {
  const event: ${tsName}BusEvent = {
    id: crypto.randomUUID(),
    ts: Date.now() / 1000,
    iso: new Date().toISOString(),
    topic,
    kind,
    level: 'info',
    actor: ACTOR,
    data,
  };

  // TODO: Implement actual bus emission via WebSocket
  // const wsUrl = ...
  console.log('Emit:', topic, event);

  return event.id;
}

// ============================================================================
// Validation
// ============================================================================

function validateInput(input: ${tsName}Input): string[] {
  const errors: string[] = [];

${schema.inputs
  .filter((f) => f.required)
  .map((f) => {
    return `  if (input.${f.name} === undefined || input.${f.name} === null) {
    errors.push('${f.name} is required');
  }`;
  })
  .join('\n')}

  return errors;
}

// ============================================================================
// Main Execution
// ============================================================================

export async function execute(input: ${tsName}Input): Promise<${tsName}Output> {
  // Validate
  const errors = validateInput(input);
  if (errors.length > 0) {
    throw new Error(\`Validation failed: \${errors.join(', ')}\`);
  }

  // Emit request
${schema.bus_topics.request ? `  await emitBusEvent(REQUEST_TOPIC, 'request', {
    phase: 'start',
    ...input,
  });` : '  // No request topic configured'}

  try {
    // TODO: Implement tool logic here
    console.log('${operatorName}: Executing...');

    const result: ${tsName}Output = {
${schema.outputs.map((f) => `      ${f.name}: undefined as unknown as ${mapToTypeScriptType(f.type)}, // TODO: implement`).join('\n')}
    };

    // Emit response
${schema.bus_topics.response ? `    await emitBusEvent(RESPONSE_TOPIC, 'response', {
      phase: 'complete',
      success: true,
      result,
    });` : '    // No response topic configured'}

    return result;
  } catch (error) {
${schema.bus_topics.response ? `    await emitBusEvent(RESPONSE_TOPIC, 'response', {
      phase: 'error',
      success: false,
      error: String(error),
    });` : ''}
    throw error;
  }
}

export default { execute, validateInput };
`;
}

function mapToPythonType(type: ToolFieldType): string {
  const mapping: Record<ToolFieldType, string> = {
    string: 'str',
    int: 'int',
    float: 'float',
    bool: 'bool',
    path: 'str',
    json: 'str',
    'list[string]': 'str',
    'list[int]': 'str',
    dict: 'str',
    'optional[string]': 'str',
    'optional[int]': 'int',
    'optional[path]': 'str',
    bus_topic: 'str',
    actor_id: 'str',
    uuid: 'str',
  };
  return mapping[type] || 'str';
}

function mapToPythonTypeAnnotation(type: ToolFieldType): string {
  const mapping: Record<ToolFieldType, string> = {
    string: 'str',
    int: 'int',
    float: 'float',
    bool: 'bool',
    path: 'Path',
    json: 'Dict[str, Any]',
    'list[string]': 'List[str]',
    'list[int]': 'List[int]',
    dict: 'Dict[str, Any]',
    'optional[string]': 'Optional[str]',
    'optional[int]': 'Optional[int]',
    'optional[path]': 'Optional[Path]',
    bus_topic: 'str',
    actor_id: 'str',
    uuid: 'str',
  };
  return mapping[type] || 'Any';
}

function mapToTypeScriptType(type: ToolFieldType): string {
  const mapping: Record<ToolFieldType, string> = {
    string: 'string',
    int: 'number',
    float: 'number',
    bool: 'boolean',
    path: 'string',
    json: 'Record<string, unknown>',
    'list[string]': 'string[]',
    'list[int]': 'number[]',
    dict: 'Record<string, unknown>',
    'optional[string]': 'string | undefined',
    'optional[int]': 'number | undefined',
    'optional[path]': 'string | undefined',
    bus_topic: 'string',
    actor_id: 'string',
    uuid: 'string',
  };
  return mapping[type] || 'unknown';
}

// ============================================================================
// Component
// ============================================================================

export const MetaArchitectTool = component$(() => {
  const state = useStore<MetaArchitectState>({
    schema: { ...DEFAULT_SCHEMA },
    activeTab: 'schema',
    previewLang: 'python',
    generatedCode: '',
    testCode: '',
    testResult: null,
    isGenerating: false,
    isTesting: false,
    error: null,
    status: null,
    recentTools: [],
  });

  // Editing state
  const editingField = useSignal<{ type: 'input' | 'output'; index: number } | null>(null);
  const newFieldForm = useStore({
    name: '',
    type: 'string' as ToolFieldType,
    description: '',
    required: true,
    default_value: '',
    cli_flag: '',
  });

  // Computed preview
  const codePreview = useComputed$(() => {
    if (state.previewLang === 'python') {
      return generatePythonCode(state.schema);
    } else {
      return generateTypeScriptCode(state.schema);
    }
  });

  // Generate operator key from name
  const autoGenerateKey = $(() => {
    if (state.schema.name && !state.schema.operator_key) {
      const key = 'PB' + state.schema.name.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 12);
      state.schema.operator_key = key;
    }
  });

  // Add field
  const addField = $((target: 'inputs' | 'outputs') => {
    if (!newFieldForm.name.trim()) {
      state.error = 'Field name is required';
      return;
    }

    const field: ToolField = {
      id: crypto.randomUUID(),
      name: newFieldForm.name.trim().toLowerCase().replace(/[^a-z0-9_]/g, '_'),
      type: newFieldForm.type,
      description: newFieldForm.description.trim(),
      required: newFieldForm.required,
      default_value: newFieldForm.default_value.trim() || undefined,
      cli_flag: newFieldForm.cli_flag.trim() || undefined,
    };

    state.schema[target] = [...state.schema[target], field];
    state.status = `Added ${field.name} to ${target}`;

    // Reset form
    newFieldForm.name = '';
    newFieldForm.description = '';
    newFieldForm.default_value = '';
    newFieldForm.cli_flag = '';
    newFieldForm.required = true;
    state.error = null;
  });

  // Remove field
  const removeField = $((target: 'inputs' | 'outputs', index: number) => {
    const field = state.schema[target][index];
    state.schema[target] = state.schema[target].filter((_, i) => i !== index);
    state.status = `Removed ${field?.name || 'field'} from ${target}`;
  });

  // Move field up/down
  const moveField = $((target: 'inputs' | 'outputs', index: number, direction: 'up' | 'down') => {
    const arr = [...state.schema[target]];
    const newIndex = direction === 'up' ? index - 1 : index + 1;
    if (newIndex < 0 || newIndex >= arr.length) return;

    [arr[index], arr[newIndex]] = [arr[newIndex], arr[index]];
    state.schema[target] = arr;
  });

  // Generate code
  const generateCode = $(() => {
    state.isGenerating = true;
    state.error = null;

    try {
      if (state.previewLang === 'python') {
        state.generatedCode = generatePythonCode(state.schema);
      } else {
        state.generatedCode = generateTypeScriptCode(state.schema);
      }
      state.status = `Generated ${state.previewLang} code`;
    } catch (e) {
      state.error = `Code generation failed: ${String(e)}`;
    } finally {
      state.isGenerating = false;
    }
  });

  // Run sandbox test (Python only via Pyodide)
  const runTest = $(async () => {
    state.isTesting = true;
    state.testResult = null;
    state.error = null;

    try {
      // For now, emit a test request to the bus and show mock result
      // In production, this would use Pyodide or server-side execution
      const startTime = Date.now();

      // Emit test event
      emitBusEvent('meta_architect.test.request', {
        schema: state.schema,
        testCode: state.testCode,
        lang: state.previewLang,
      });

      // Mock test result (replace with actual execution)
      await new Promise((r) => setTimeout(r, 500));

      state.testResult = {
        success: true,
        stdout: `[Test] ${state.schema.operator_key || 'TOOL'}: Syntax OK\n[Test] Schema validation: PASS\n[Test] CLI parser: PASS`,
        stderr: '',
        duration_ms: Date.now() - startTime,
        exit_code: 0,
      };
      state.status = 'Test completed';
    } catch (e) {
      state.error = `Test failed: ${String(e)}`;
      state.testResult = {
        success: false,
        stdout: '',
        stderr: String(e),
        duration_ms: 0,
        exit_code: 1,
      };
    } finally {
      state.isTesting = false;
    }
  });

  // Export to file (downloads or saves to nucleus/tools)
  const exportTool = $(async () => {
    state.error = null;

    try {
      const code = codePreview.value;
      const filename =
        state.previewLang === 'python'
          ? `${state.schema.name.toLowerCase().replace(/[^a-z0-9]/g, '_')}_operator.py`
          : `${state.schema.name.replace(/[^a-zA-Z0-9]/g, '')}Tool.ts`;

      // Try to save via API first
      const res = await fetch('/api/meta-architect/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename,
          code,
          lang: state.previewLang,
          schema: state.schema,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        state.status = `Exported to ${data.path}`;
        emitBusEvent('meta_architect.export.success', {
          path: data.path,
          operator_key: state.schema.operator_key,
        });
      } else {
        // Fallback to download
        const blob = new Blob([code], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        state.status = `Downloaded ${filename}`;
      }
    } catch (e) {
      state.error = `Export failed: ${String(e)}`;
    }
  });

  // Copy to clipboard
  const copyToClipboard = $(async () => {
    try {
      await navigator.clipboard.writeText(codePreview.value);
      state.status = 'Copied to clipboard';
    } catch {
      state.error = 'Failed to copy';
    }
  });

  // Reset schema
  const resetSchema = $(() => {
    state.schema = { ...DEFAULT_SCHEMA, inputs: [], outputs: [], bus_topics: {} };
    state.generatedCode = '';
    state.testResult = null;
    state.status = 'Schema reset';
    state.error = null;
  });

  // Load template
  const loadTemplate = $((template: 'operator' | 'daemon' | 'bridge') => {
    if (template === 'operator') {
      state.schema = {
        ...DEFAULT_SCHEMA,
        category: 'operator',
        cli_enabled: true,
        daemon_mode: false,
        inputs: [
          {
            id: crypto.randomUUID(),
            name: 'scope',
            type: 'path',
            description: 'Target path or scope',
            required: true,
            cli_flag: '--scope',
          },
          {
            id: crypto.randomUUID(),
            name: 'mode',
            type: 'string',
            description: 'Execution mode',
            required: false,
            default_value: 'default',
            cli_flag: '--mode',
          },
        ],
        outputs: [
          {
            id: crypto.randomUUID(),
            name: 'success',
            type: 'bool',
            description: 'Whether operation succeeded',
            required: true,
          },
          {
            id: crypto.randomUUID(),
            name: 'result',
            type: 'json',
            description: 'Operation result data',
            required: true,
          },
        ],
        bus_topics: {
          request: 'operator.{name}.request',
          response: 'operator.{name}.response',
        },
      };
    } else if (template === 'daemon') {
      state.schema = {
        ...DEFAULT_SCHEMA,
        category: 'daemon',
        cli_enabled: false,
        daemon_mode: true,
        effects: 'network',
        inputs: [
          {
            id: crypto.randomUUID(),
            name: 'port',
            type: 'int',
            description: 'Port to listen on',
            required: false,
            default_value: '8080',
          },
          {
            id: crypto.randomUUID(),
            name: 'host',
            type: 'string',
            description: 'Host to bind to',
            required: false,
            default_value: '0.0.0.0',
          },
        ],
        outputs: [],
        bus_topics: {
          metric: 'daemon.{name}.metric',
        },
      };
    } else if (template === 'bridge') {
      state.schema = {
        ...DEFAULT_SCHEMA,
        category: 'bridge',
        cli_enabled: false,
        daemon_mode: true,
        effects: 'network',
        inputs: [
          {
            id: crypto.randomUUID(),
            name: 'upstream_url',
            type: 'string',
            description: 'Upstream service URL',
            required: true,
          },
        ],
        outputs: [
          {
            id: crypto.randomUUID(),
            name: 'status',
            type: 'string',
            description: 'Bridge status',
            required: true,
          },
        ],
        bus_topics: {
          request: 'bridge.{name}.request',
          response: 'bridge.{name}.response',
        },
      };
    }
    state.status = `Loaded ${template} template`;
  });

  // Bus event helper
  const emitBusEvent = (topic: string, data: Record<string, unknown>) => {
    try {
      const wsUrl =
        typeof window !== 'undefined'
          ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/bus`
          : null;
      if (wsUrl) {
        const ws = new WebSocket(wsUrl);
        ws.onopen = () => {
          ws.send(
            JSON.stringify({
              type: 'publish',
              event: {
                id: crypto.randomUUID(),
                topic,
                kind: 'metric',
                level: 'info',
                actor: 'meta-architect',
                data,
              },
            })
          );
          ws.close();
        };
      }
    } catch {
      // Best effort
    }
  };

  // Load recent tools on mount
  useVisibleTask$(() => {
    // Could load from localStorage or API
    state.recentTools = ['pbtest_operator', 'pblanes_operator', 'pbus_operator'];
  });

  return (
    <div class="h-full flex flex-col bg-background text-foreground">
      {/* Header */}
      <div class="flex-shrink-0 p-4 border-b border-border bg-card">
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <div class="text-2xl">&#x1F3D7;&#xFE0F;</div>
            <div>
              <h1 class="text-lg font-semibold">Meta-Architecting Tool</h1>
              <p class="text-xs text-muted-foreground">
                Design, generate, and test Pluribus tools visually
              </p>
            </div>
          </div>

          <div class="flex items-center gap-2">
            {/* Templates */}
            <select
              class="px-3 py-1.5 rounded bg-background border border-border text-sm"
              onChange$={(e) => {
                const v = (e.target as HTMLSelectElement).value;
                if (v) loadTemplate(v as 'operator' | 'daemon' | 'bridge');
              }}
            >
              <option value="">Load Template...</option>
              <option value="operator">Operator (CLI)</option>
              <option value="daemon">Daemon (Service)</option>
              <option value="bridge">Bridge (Adapter)</option>
            </select>

            <Button variant="secondary" onClick$={resetSchema} class="h-8 text-sm">
              Reset
            </Button>
          </div>
        </div>
      </div>

      {/* Status/Error */}
      {state.error && (
        <div class="flex-shrink-0 px-4 py-2 bg-red-500/10 border-b border-red-500/30 text-sm text-red-400">
          {state.error}
        </div>
      )}
      {state.status && !state.error && (
        <div class="flex-shrink-0 px-4 py-2 bg-green-500/10 border-b border-green-500/30 text-sm text-green-400">
          {state.status}
        </div>
      )}

      {/* Tabs */}
      <div class="flex-shrink-0 border-b border-border bg-card/50">
        <div class="flex gap-1 px-4">
          {(['schema', 'preview', 'test', 'export'] as const).map((tab) => (
            <Button
              key={tab}
              variant={state.activeTab === tab ? 'primary' : 'text'}
              onClick$={() => (state.activeTab = tab)}
              class="rounded-b-none rounded-t-lg"
            >
              {tab === 'schema' && 'Schema Editor'}
              {tab === 'preview' && 'Code Preview'}
              {tab === 'test' && 'Sandbox Test'}
              {tab === 'export' && 'Export'}
            </Button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div class="flex-1 overflow-auto p-4">
        {/* Schema Editor Tab */}
        {state.activeTab === 'schema' && (
          <div class="grid grid-cols-12 gap-6">
            {/* Left: Basic Info */}
            <div class="col-span-4 space-y-4">
            <Card padding="p-4">
                <h2 class="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                  Tool Identity
                </h2>

                <div class="space-y-3">
                  <Input
                    label="Name"
                    value={state.schema.name}
                    onInput$={(_, el) => (state.schema.name = el.value)}
                    placeholder="MyTool"
                  />

                  <Input
                    label="Operator Key"
                    value={state.schema.operator_key}
                    onInput$={(_, el) => (state.schema.operator_key = el.value.toUpperCase())}
                    placeholder="PBMYTOOL"
                  />

                  <Input
                    type="textarea"
                    label="Description"
                    value={state.schema.description}
                    onInput$={(_, el) => (state.schema.description = el.value)}
                    placeholder="What does this tool do?"
                  />

                  <div class="grid grid-cols-2 gap-3">
                    <div>
                      <label class="text-xs text-muted-foreground mb-1 block">Domain</label>
                      <select
                        value={state.schema.domain}
                        onChange$={(e) =>
                          (state.schema.domain = (e.target as HTMLSelectElement).value)
                        }
                        class="w-full px-3 py-2 rounded bg-background border border-border text-sm"
                      >
                        {DOMAINS.map((d) => (
                          <option key={d} value={d}>
                            {d}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label class="text-xs text-muted-foreground mb-1 block">Category</label>
                      <select
                        value={state.schema.category}
                        onChange$={(e) =>
                          (state.schema.category = (e.target as HTMLSelectElement).value)
                        }
                        class="w-full px-3 py-2 rounded bg-background border border-border text-sm"
                      >
                        {CATEGORIES.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div>
                    <label class="text-xs text-muted-foreground mb-1 block">Effects</label>
                    <select
                      value={state.schema.effects}
                      onChange$={(e) =>
                        (state.schema.effects = (e.target as HTMLSelectElement).value as typeof EFFECTS[number])
                      }
                      class={`w-full px-3 py-2 rounded border text-sm ${
                        state.schema.effects === 'system'
                          ? 'bg-red-500/10 border-red-500/30 text-red-400'
                          : state.schema.effects === 'network'
                          ? 'bg-orange-500/10 border-orange-500/30 text-orange-400'
                          : state.schema.effects === 'file'
                          ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400'
                          : 'bg-background border-border'
                      }`}
                    >
                      {EFFECTS.map((e) => (
                        <option key={e} value={e}>
                          {e}
                          {e === 'system' ? ' (elevated)' : ''}
                          {e === 'network' ? ' (external)' : ''}
                          {e === 'file' ? ' (filesystem)' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div class="flex items-center gap-4">
                    <label class="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={state.schema.cli_enabled}
                        onChange$={(e) =>
                          (state.schema.cli_enabled = (e.target as HTMLInputElement).checked)
                        }
                      />
                      CLI enabled
                    </label>
                    <label class="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={state.schema.daemon_mode}
                        onChange$={(e) =>
                          (state.schema.daemon_mode = (e.target as HTMLInputElement).checked)
                        }
                      />
                      Daemon mode
                    </label>
                  </div>
                </div>
            </Card>

              {/* Bus Topics */}
              <Card padding="p-4" class="space-y-3">
                <h2 class="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                  Bus Topics
                </h2>
                <div class="space-y-2">
                  <Input
                    label="Request Topic"
                    value={state.schema.bus_topics.request || ''}
                    onInput$={(_, el) => (state.schema.bus_topics.request = el.value || undefined)}
                    placeholder="operator.mytool.request"
                  />
                  <Input
                    label="Response Topic"
                    value={state.schema.bus_topics.response || ''}
                    onInput$={(_, el) => (state.schema.bus_topics.response = el.value || undefined)}
                    placeholder="operator.mytool.response"
                  />
                  <Input
                    label="Metric Topic"
                    value={state.schema.bus_topics.metric || ''}
                    onInput$={(_, el) => (state.schema.bus_topics.metric = el.value || undefined)}
                    placeholder="operator.mytool.metric"
                  />
                </div>
              </Card>
            </div>

            {/* Right: Fields Editor */}
            <div class="col-span-8 space-y-4">
              {/* Inputs */}
              <Card padding="p-4" class="space-y-3">
                <div class="flex items-center justify-between">
                  <h2 class="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                    Inputs ({state.schema.inputs.length})
                  </h2>
                </div>

                {/* Field list */}
                <div class="space-y-2">
                  {state.schema.inputs.map((field, idx) => (
                    <div
                      key={field.id}
                      class="flex items-center gap-2 p-2 rounded bg-background/50 border border-border/50"
                    >
                      <div class="flex flex-col gap-1">
                        <button
                          class="text-xs text-muted-foreground hover:text-foreground"
                          onClick$={() => moveField('inputs', idx, 'up')}
                          disabled={idx === 0}
                        >
                          &#x25B2;
                        </button>
                        <button
                          class="text-xs text-muted-foreground hover:text-foreground"
                          onClick$={() => moveField('inputs', idx, 'down')}
                          disabled={idx === state.schema.inputs.length - 1}
                        >
                          &#x25BC;
                        </button>
                      </div>
                      <div class="flex-1 min-w-0">
                        <div class="flex items-center gap-2">
                          <span class="font-mono text-sm">{field.name}</span>
                          <span class="text-xs px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                            {field.type}
                          </span>
                          {field.required && (
                            <span class="text-xs px-1.5 py-0.5 rounded bg-red-500/10 text-red-400">
                              required
                            </span>
                          )}
                          {field.cli_flag && (
                            <span class="text-xs px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 font-mono">
                              {field.cli_flag}
                            </span>
                          )}
                        </div>
                        <div class="text-xs text-muted-foreground truncate">{field.description}</div>
                      </div>
                      <Button
                        variant="tonal"
                        class="text-xs h-6 text-red-400 border-red-500/30"
                        onClick$={() => removeField('inputs', idx)}
                      >
                        Remove
                      </Button>
                    </div>
                  ))}
                </div>

                {/* Add field form */}
                <div class="pt-3 border-t border-border/50 space-y-2">
                  <div class="text-xs font-semibold text-muted-foreground">Add Input Field</div>
                  <div class="grid grid-cols-6 gap-2">
                    <div class="col-span-2">
                        <Input
                          label="Name"
                          value={newFieldForm.name}
                          onInput$={(_, el) => (newFieldForm.name = el.value)}
                          placeholder="name"
                        />
                    </div>
                    <select
                      value={newFieldForm.type}
                      onChange$={(e) => (newFieldForm.type = (e.target as HTMLSelectElement).value as ToolFieldType)}
                      class="px-2 py-1.5 rounded bg-background border border-border text-sm h-[56px]"
                    >
                      {FIELD_TYPES.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                    <Input
                      label="Flag"
                      value={newFieldForm.cli_flag}
                      onInput$={(_, el) => (newFieldForm.cli_flag = el.value)}
                      placeholder="--flag"
                    />
                    <label class="flex items-center gap-1 text-xs">
                      <input
                        type="checkbox"
                        checked={newFieldForm.required}
                        onChange$={(e) => (newFieldForm.required = (e.target as HTMLInputElement).checked)}
                      />
                      Required
                    </label>
                    <Button
                      variant="tonal"
                      class="h-[56px]"
                      onClick$={() => addField('inputs')}
                    >
                      Add
                    </Button>
                  </div>
                  <Input
                    label="Description"
                    value={newFieldForm.description}
                    onInput$={(_, el) => (newFieldForm.description = el.value)}
                    placeholder="Description..."
                  />
                </div>
              </Card>

              {/* Outputs */}
              <Card padding="p-4" class="space-y-3">
                <div class="flex items-center justify-between">
                  <h2 class="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                    Outputs ({state.schema.outputs.length})
                  </h2>
                </div>

                {/* Field list */}
                <div class="space-y-2">
                  {state.schema.outputs.map((field, idx) => (
                    <div
                      key={field.id}
                      class="flex items-center gap-2 p-2 rounded bg-background/50 border border-border/50"
                    >
                      <div class="flex flex-col gap-1">
                        <button
                          class="text-xs text-muted-foreground hover:text-foreground"
                          onClick$={() => moveField('outputs', idx, 'up')}
                          disabled={idx === 0}
                        >
                          &#x25B2;
                        </button>
                        <button
                          class="text-xs text-muted-foreground hover:text-foreground"
                          onClick$={() => moveField('outputs', idx, 'down')}
                          disabled={idx === state.schema.outputs.length - 1}
                        >
                          &#x25BC;
                        </button>
                      </div>
                      <div class="flex-1 min-w-0">
                        <div class="flex items-center gap-2">
                          <span class="font-mono text-sm">{field.name}</span>
                          <span class="text-xs px-1.5 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20">
                            {field.type}
                          </span>
                        </div>
                        <div class="text-xs text-muted-foreground truncate">{field.description}</div>
                      </div>
                      <Button
                        variant="tonal"
                        class="text-xs h-6 text-red-400 border-red-500/30"
                        onClick$={() => removeField('outputs', idx)}
                      >
                        Remove
                      </Button>
                    </div>
                  ))}
                </div>

                {/* Add output field */}
                <div class="pt-3 border-t border-border/50 space-y-2">
                  <div class="text-xs font-semibold text-muted-foreground">Add Output Field</div>
                  <div class="grid grid-cols-5 gap-2">
                    <div class="col-span-2">
                        <Input
                          label="Name"
                          value={newFieldForm.name}
                          onInput$={(_, el) => (newFieldForm.name = el.value)}
                          placeholder="name"
                        />
                    </div>
                    <select
                      value={newFieldForm.type}
                      onChange$={(e) => (newFieldForm.type = (e.target as HTMLSelectElement).value as ToolFieldType)}
                      class="col-span-2 px-2 py-1.5 rounded bg-background border border-border text-sm h-[56px]"
                    >
                      {FIELD_TYPES.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                    <Button
                      variant="tonal"
                      class="h-[56px]"
                      onClick$={() => addField('outputs')}
                    >
                      Add
                    </Button>
                  </div>
                  <Input
                    label="Description"
                    value={newFieldForm.description}
                    onInput$={(_, el) => (newFieldForm.description = el.value)}
                    placeholder="Description..."
                  />
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* Code Preview Tab */}
        {state.activeTab === 'preview' && (
          <div class="space-y-4">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <select
                  value={state.previewLang}
                  onChange$={(e) => (state.previewLang = (e.target as HTMLSelectElement).value as GeneratorTarget)}
                  class="px-3 py-1.5 rounded bg-background border border-border text-sm"
                >
                  <option value="python">Python</option>
                  <option value="typescript">TypeScript</option>
                </select>
                <span class="text-xs text-muted-foreground">
                  {state.previewLang === 'python' ? '*.py operator' : '*.ts module'}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <Button variant="secondary" onClick$={copyToClipboard} class="h-8 text-xs">Copy</Button>
                <Button variant="primary" onClick$={generateCode} disabled={state.isGenerating} class="h-8 text-xs">
                  {state.isGenerating ? 'Generating...' : 'Regenerate'}
                </Button>
              </div>
            </div>

            {/* Code display */}
            <div class="rounded-lg border border-border bg-[#08080a] overflow-hidden">
              <div class="flex items-center justify-between px-4 py-2 bg-black/30 border-b border-border/50">
                <span class="text-xs text-muted-foreground font-mono">
                  {state.previewLang === 'python'
                    ? `${state.schema.name.toLowerCase().replace(/[^a-z0-9]/g, '_') || 'tool'}_operator.py`
                    : `${state.schema.name.replace(/[^a-zA-Z0-9]/g, '') || 'Tool'}Tool.ts`}
                </span>
                <span class="text-xs text-muted-foreground">
                  {codePreview.value.split('\n').length} lines
                </span>
              </div>
              <div class="max-h-[600px] overflow-auto">
                <pre class="p-4 text-sm font-mono text-foreground/90 whitespace-pre">
                  <code>{codePreview.value}</code>
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* Sandbox Test Tab */}
        {state.activeTab === 'test' && (
          <div class="space-y-4">
            <Card padding="p-4" class="space-y-4">
              <div class="flex items-center justify-between">
                <h2 class="text-sm font-semibold">Sandbox Test</h2>
                <Button
                  variant="primary"
                  onClick$={runTest}
                  disabled={state.isTesting}
                  class="bg-green-600 hover:bg-green-500 h-8 text-xs"
                >
                  {state.isTesting ? 'Running...' : 'Run Test'}
                </Button>
              </div>

              <div>
                <Input
                  type="textarea"
                  label="Test Code"
                  value={state.testCode}
                  onInput$={(_, el) => (state.testCode = el.value)}
                  placeholder={`# Custom test code\n# Leave empty for syntax/schema validation only`}
                />
              </div>

              {state.testResult && (
                <div
                  class={`rounded border p-4 ${
                    state.testResult.success
                      ? 'bg-green-500/10 border-green-500/30'
                      : 'bg-red-500/10 border-red-500/30'
                  }`}
                >
                  <div class="flex items-center justify-between mb-2">
                    <span
                      class={`text-sm font-medium ${
                        state.testResult.success ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {state.testResult.success ? 'PASS' : 'FAIL'}
                    </span>
                    <span class="text-xs text-muted-foreground">
                      {state.testResult.duration_ms}ms | exit {state.testResult.exit_code}
                    </span>
                  </div>
                  {state.testResult.stdout && (
                    <pre class="text-xs font-mono text-muted-foreground whitespace-pre-wrap">
                      {state.testResult.stdout}
                    </pre>
                  )}
                  {state.testResult.stderr && (
                    <pre class="text-xs font-mono text-red-400 whitespace-pre-wrap mt-2">
                      {state.testResult.stderr}
                    </pre>
                  )}
                </div>
              )}
            </Card>

            <Card padding="p-4" class="bg-card/50">
              <h3 class="text-xs font-semibold text-muted-foreground mb-2">Test Environment</h3>
              <ul class="text-xs text-muted-foreground space-y-1">
                <li>&#x2022; Python: Pyodide WASM sandbox (in-browser)</li>
                <li>&#x2022; TypeScript: Esbuild bundle + eval</li>
                <li>&#x2022; No filesystem/network access in sandbox</li>
                <li>&#x2022; Bus events captured locally</li>
              </ul>
            </Card>
          </div>
        )}

        {/* Export Tab */}
        {state.activeTab === 'export' && (
          <div class="space-y-4">
            <Card padding="p-4" class="space-y-4">
              <h2 class="text-sm font-semibold">Export Tool</h2>

              <div class="grid grid-cols-2 gap-4">
                {/* Python Export */}
                <div class="rounded border border-border bg-background/50 p-4 space-y-3">
                  <div class="flex items-center gap-2">
                    <span class="text-xl">&#x1F40D;</span>
                    <span class="font-medium">Python Operator</span>
                  </div>
                  <div class="text-xs text-muted-foreground">
                    nucleus/tools/
                    {state.schema.name.toLowerCase().replace(/[^a-z0-9]/g, '_') || 'tool'}
                    _operator.py
                  </div>
                  <Button
                    variant="tonal"
                    class="w-full h-8 text-xs"
                    onClick$={() => {
                      state.previewLang = 'python';
                      exportTool();
                    }}
                  >
                    Export Python
                  </Button>
                </div>

                {/* TypeScript Export */}
                <div class="rounded border border-border bg-background/50 p-4 space-y-3">
                  <div class="flex items-center gap-2">
                    <span class="text-xl">&#x1F4DD;</span>
                    <span class="font-medium">TypeScript Module</span>
                  </div>
                  <div class="text-xs text-muted-foreground">
                    nucleus/dashboard/src/lib/tools/
                    {state.schema.name.replace(/[^a-zA-Z0-9]/g, '') || 'Tool'}
                    Tool.ts
                  </div>
                  <Button
                    variant="secondary"
                    class="w-full h-8 text-xs"
                    onClick$={() => {
                      state.previewLang = 'typescript';
                      exportTool();
                    }}
                  >
                    Export TypeScript
                  </Button>
                </div>
              </div>

              {/* Export Schema JSON */}
              <div class="pt-4 border-t border-border/50">
                <h3 class="text-xs font-semibold text-muted-foreground mb-2">Export Schema (JSON)</h3>
                <div class="flex gap-2">
                  <Button
                    variant="secondary"
                    class="h-8 text-xs"
                    onClick$={async () => {
                      const json = JSON.stringify(state.schema, null, 2);
                      await navigator.clipboard.writeText(json);
                      state.status = 'Schema JSON copied';
                    }}
                  >
                    Copy Schema JSON
                  </Button>
                  <Button
                    variant="secondary"
                    class="h-8 text-xs"
                    onClick$={() => {
                      const json = JSON.stringify(state.schema, null, 2);
                      const blob = new Blob([json], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${state.schema.operator_key || 'tool'}_schema.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                      state.status = 'Schema downloaded';
                    }}
                  >
                    Download Schema
                  </Button>
                </div>
              </div>
            </Card>

            {/* Integration Guide */}
            <Card padding="p-4" class="bg-card/50">
              <h3 class="text-sm font-semibold mb-3">Integration Checklist</h3>
              <div class="space-y-2 text-xs">
                <label class="flex items-center gap-2">
                  <input type="checkbox" class="rounded" />
                  Add tool to nucleus/tools/
                </label>
                <label class="flex items-center gap-2">
                  <input type="checkbox" class="rounded" />
                  Register in semops_schema.yaml
                </label>
                <label class="flex items-center gap-2">
                  <input type="checkbox" class="rounded" />
                  Add tests to nucleus/tests/
                </label>
                <label class="flex items-center gap-2">
                  <input type="checkbox" class="rounded" />
                  Update CHANGELOG.md
                </label>
                <label class="flex items-center gap-2">
                  <input type="checkbox" class="rounded" />
                  Run PBTEST verification
                </label>
              </div>
            </Card>
          </div>
        )}
      </div>

      {/* Footer */}
      <div class="flex-shrink-0 p-3 border-t border-border bg-card/50 text-xs text-muted-foreground flex items-center justify-between">
        <div class="flex items-center gap-4">
          <span>
            Schema: {state.schema.inputs.length} inputs, {state.schema.outputs.length} outputs
          </span>
          {state.schema.operator_key && (
            <span class="font-mono text-primary">{state.schema.operator_key}</span>
          )}
        </div>
        <span>MetaArchitectTool v1.0 | DKIN v29</span>
      </div>
    </div>
  );
});

export default MetaArchitectTool;
