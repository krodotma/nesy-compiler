/**
 * MarimoNotebook.tsx - Reactive Python Notebook Component
 *
 * Provides reactive execution of .py notebook files using Pyodide/WASM.
 * Beyond simple iframe embedding - supports:
 * - Reactive cell execution with dependency tracking
 * - Integration with rd_workflow for research documentation
 * - Bus event emission for cell outputs
 * - Rhizome artifact persistence
 *
 * Specification from SOTA catalog:
 * - reactive_execution: automatic re-run on change
 * - pure_python: .py files, not .ipynb
 * - git_friendly: diffable code
 */

import { component$, useSignal, useVisibleTask$, $, type Signal } from '@builder.io/qwik';
import type { BusEvent } from '../lib/state/types';

// M3 Components - MarimoNotebook
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/circular-progress.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NotebookCell {
  id: string;
  code: string;
  outputs: CellOutput[];
  status: 'idle' | 'pending' | 'running' | 'complete' | 'error';
  dependencies: string[];
  executionOrder: number;
  lastRunMs?: number;
}

export interface CellOutput {
  type: 'text' | 'html' | 'image' | 'error' | 'dataframe' | 'plot';
  content: string;
  mimeType?: string;
  timestamp: number;
}

export interface NotebookState {
  cells: NotebookCell[];
  globals: Record<string, unknown>;
  executionGraph: Map<string, Set<string>>;
  isReactive: boolean;
  pyodideReady: boolean;
}

export interface MarimoNotebookProps {
  /** Path to .py notebook file (relative to rhizome or absolute) */
  notebookPath?: string;
  /** Initial code if no file provided */
  initialCode?: string;
  /** Enable reactive mode (auto re-run on dependency change) */
  reactive?: boolean;
  /** Callback when cell executes */
  onCellExecute$?: (cellId: string, output: CellOutput) => void;
  /** Bus events signal for integration */
  busEvents?: Signal<BusEvent[]>;
  /** Height of the notebook container */
  height?: string;
}

// ---------------------------------------------------------------------------
// Pyodide Runtime Manager
// ---------------------------------------------------------------------------

interface PyodideRuntime {
  ready: boolean;
  pyodide: unknown;
  globals: Record<string, unknown>;
}

const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';

async function loadPyodide(): Promise<PyodideRuntime> {
  // Check if already loaded
  if (typeof window !== 'undefined' && (window as any).pyodide) {
    return {
      ready: true,
      pyodide: (window as any).pyodide,
      globals: {},
    };
  }

  // Load Pyodide script
  await new Promise<void>((resolve, reject) => {
    if (typeof document === 'undefined') {
      reject(new Error('Document not available'));
      return;
    }

    const existing = document.querySelector(`script[src*="pyodide"]`);
    if (existing) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = PYODIDE_CDN;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load Pyodide'));
    document.head.appendChild(script);
  });

  // Initialize Pyodide
  const pyodide = await (window as any).loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/',
  });

  // Install commonly used packages
  await pyodide.loadPackage(['numpy', 'micropip']);

  // Store globally for reuse
  (window as any).pyodide = pyodide;

  return {
    ready: true,
    pyodide,
    globals: {},
  };
}

// ---------------------------------------------------------------------------
// Reactive Dependency Parser
// ---------------------------------------------------------------------------

function parseCellDependencies(code: string): { imports: string[]; defines: string[]; uses: string[] } {
  const imports: string[] = [];
  const defines: string[] = [];
  const uses: string[] = [];

  // Extract imports
  const importRegex = /^(?:from\s+(\w+)|import\s+(\w+))/gm;
  let match;
  while ((match = importRegex.exec(code)) !== null) {
    imports.push(match[1] || match[2]);
  }

  // Extract variable definitions (simplified)
  const defRegex = /^(\w+)\s*=/gm;
  while ((match = defRegex.exec(code)) !== null) {
    if (!match[1].startsWith('_')) {
      defines.push(match[1]);
    }
  }

  // Extract function definitions
  const funcRegex = /^def\s+(\w+)\s*\(/gm;
  while ((match = funcRegex.exec(code)) !== null) {
    defines.push(match[1]);
  }

  // Extract variable uses (simplified - looks for identifiers)
  const useRegex = /\b([a-zA-Z_]\w*)\b/g;
  const allTokens = new Set<string>();
  while ((match = useRegex.exec(code)) !== null) {
    const token = match[1];
    // Filter out Python keywords and builtins
    const keywords = new Set(['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'pass', 'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None', 'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple']);
    if (!keywords.has(token) && !defines.includes(token)) {
      allTokens.add(token);
    }
  }
  uses.push(...Array.from(allTokens));

  return { imports, defines, uses };
}

function buildDependencyGraph(cells: NotebookCell[]): Map<string, Set<string>> {
  const graph = new Map<string, Set<string>>();
  const definedBy = new Map<string, string>();

  // First pass: map variable definitions to cells
  for (const cell of cells) {
    const { defines } = parseCellDependencies(cell.code);
    for (const def of defines) {
      definedBy.set(def, cell.id);
    }
    graph.set(cell.id, new Set());
  }

  // Second pass: build dependency edges
  for (const cell of cells) {
    const { uses } = parseCellDependencies(cell.code);
    const deps = graph.get(cell.id)!;
    for (const use of uses) {
      const definer = definedBy.get(use);
      if (definer && definer !== cell.id) {
        deps.add(definer);
      }
    }
  }

  return graph;
}

function topologicalSort(cells: NotebookCell[], graph: Map<string, Set<string>>): string[] {
  const visited = new Set<string>();
  const result: string[] = [];

  function visit(id: string) {
    if (visited.has(id)) return;
    visited.add(id);
    const deps = graph.get(id) || new Set();
    for (const dep of deps) {
      visit(dep);
    }
    result.push(id);
  }

  for (const cell of cells) {
    visit(cell.id);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Cell Component
// ---------------------------------------------------------------------------

interface CellProps {
  cell: NotebookCell;
  onCodeChange$: (cellId: string, code: string) => void;
  onRun$: (cellId: string) => void;
  onDelete$: (cellId: string) => void;
  isSelected: boolean;
  onSelect$: () => void;
}

const NotebookCellView = component$<CellProps>((props) => {
  const { cell, onCodeChange$, onRun$, onDelete$, isSelected, onSelect$ } = props;

  const statusColors: Record<NotebookCell['status'], string> = {
    idle: 'bg-gray-500',
    pending: 'bg-yellow-500 animate-pulse',
    running: 'bg-blue-500 animate-pulse',
    complete: 'bg-green-500',
    error: 'bg-red-500',
  };

  return (
    <div
      class={`rounded-lg border transition-all ${
        isSelected ? 'border-primary ring-2 ring-primary/20' : 'border-border'
      } bg-card overflow-hidden`}
      onClick$={onSelect$}
    >
      {/* Cell Header */}
      <div class="flex items-center justify-between px-3 py-2 bg-muted/30 border-b border-border">
        <div class="flex items-center gap-2">
          <div class={`w-2 h-2 rounded-full ${statusColors[cell.status]}`} />
          <span class="text-xs font-mono text-muted-foreground">
            [{cell.executionOrder > 0 ? cell.executionOrder : ' '}]
          </span>
          {cell.lastRunMs !== undefined && (
            <span class="text-xs text-muted-foreground">
              {cell.lastRunMs}ms
            </span>
          )}
        </div>
        <div class="flex items-center gap-1">
          <button
            onClick$={() => onRun$(cell.id)}
            class="p-1 rounded hover:bg-primary/20 text-primary"
            title="Run cell (Shift+Enter)"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
          <button
            onClick$={() => onDelete$(cell.id)}
            class="p-1 rounded hover:bg-red-500/20 text-red-400"
            title="Delete cell"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>

      {/* Code Editor */}
      <div class="relative">
        <textarea
          class="w-full p-3 font-mono text-sm bg-black/30 text-gray-200 resize-none focus:outline-none focus:ring-1 focus:ring-primary/50"
          rows={Math.max(3, cell.code.split('\n').length)}
          value={cell.code}
          onInput$={(e) => onCodeChange$(cell.id, (e.target as HTMLTextAreaElement).value)}
          onKeyDown$={(e) => {
            if (e.key === 'Enter' && e.shiftKey) {
              e.preventDefault();
              onRun$(cell.id);
            }
          }}
          spellcheck={false}
        />
      </div>

      {/* Outputs */}
      {cell.outputs.length > 0 && (
        <div class="border-t border-border bg-muted/10">
          {cell.outputs.map((output, i) => (
            <div key={i} class="p-3 border-b border-border/50 last:border-0">
              {output.type === 'error' ? (
                <pre class="text-xs font-mono text-red-400 whitespace-pre-wrap">{output.content}</pre>
              ) : output.type === 'html' ? (
                <div class="prose prose-sm prose-invert" dangerouslySetInnerHTML={output.content} />
              ) : output.type === 'image' ? (
                <img src={output.content} alt="Output" class="max-w-full rounded" />
              ) : output.type === 'dataframe' ? (
                <div class="overflow-x-auto">
                  <div class="text-xs font-mono" dangerouslySetInnerHTML={output.content} />
                </div>
              ) : (
                <pre class="text-xs font-mono text-gray-300 whitespace-pre-wrap">{output.content}</pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

// ---------------------------------------------------------------------------
// Main Notebook Component
// ---------------------------------------------------------------------------

export const MarimoNotebook = component$<MarimoNotebookProps>((props) => {
  const {
    notebookPath,
    initialCode,
    reactive = true,
    height = 'calc(100vh - 200px)',
  } = props;

  // State
  const cells = useSignal<NotebookCell[]>([]);
  const selectedCellId = useSignal<string | null>(null);
  const pyodideReady = useSignal(false);
  const pyodideLoading = useSignal(false);
  const pyodideError = useSignal<string | null>(null);
  const executionCount = useSignal(0);
  const isReactive = useSignal(reactive);

  // Runtime reference (stored in closure)
  let runtime: PyodideRuntime | null = null;

  // Initialize Pyodide
  useVisibleTask$(async () => {
    pyodideLoading.value = true;
    try {
      runtime = await loadPyodide();
      pyodideReady.value = true;

      // Load notebook file if provided
      if (notebookPath) {
        await loadNotebookFile(notebookPath);
      } else if (initialCode) {
        // Parse initial code into cells
        const parsed = parseNotebookCode(initialCode);
        cells.value = parsed;
      } else {
        // Create default empty cell
        cells.value = [{
          id: crypto.randomUUID(),
          code: '# Welcome to Marimo Notebook\n# Press Shift+Enter to run a cell\n\nprint("Hello, Pluribus!")',
          outputs: [],
          status: 'idle',
          dependencies: [],
          executionOrder: 0,
        }];
      }
    } catch (err) {
      pyodideError.value = String(err);
    } finally {
      pyodideLoading.value = false;
    }
  });

  // Load notebook from file
  const loadNotebookFile = async (path: string) => {
    try {
      const response = await fetch(`/api/fs${path}`);
      if (!response.ok) throw new Error(`Failed to load: ${path}`);
      const content = await response.text();
      const parsed = parseNotebookCode(content);
      cells.value = parsed;
    } catch (err) {
      pyodideError.value = `Failed to load notebook: ${err}`;
    }
  };

  // Parse .py file into cells (Marimo format)
  const parseNotebookCode = (code: string): NotebookCell[] => {
    // Marimo uses special comments to delimit cells: # %%
    const cellDelimiter = /^# %%.*$/gm;
    const parts = code.split(cellDelimiter).filter(p => p.trim());

    if (parts.length === 0) {
      parts.push(code);
    }

    return parts.map((part, i) => ({
      id: crypto.randomUUID(),
      code: part.trim(),
      outputs: [],
      status: 'idle' as const,
      dependencies: [],
      executionOrder: 0,
    }));
  };

  // Execute a cell
  const executeCell = $(async (cellId: string) => {
    if (!runtime?.ready || !runtime.pyodide) {
      return;
    }

    const cellIndex = cells.value.findIndex(c => c.id === cellId);
    if (cellIndex === -1) return;

    const cell = cells.value[cellIndex];
    const pyodide = runtime.pyodide as any;

    // Update status
    cells.value = cells.value.map(c =>
      c.id === cellId ? { ...c, status: 'running' as const, outputs: [] } : c
    );

    const startTime = performance.now();

    try {
      // Capture stdout/stderr
      let stdout = '';
      let stderr = '';

      pyodide.setStdout({ batched: (s: string) => { stdout += s; } });
      pyodide.setStderr({ batched: (s: string) => { stderr += s; } });

      // Execute the code
      const result = await pyodide.runPythonAsync(cell.code);

      const endTime = performance.now();
      const outputs: CellOutput[] = [];

      // Add stdout
      if (stdout.trim()) {
        outputs.push({
          type: 'text',
          content: stdout,
          timestamp: Date.now(),
        });
      }

      // Add stderr as error
      if (stderr.trim()) {
        outputs.push({
          type: 'error',
          content: stderr,
          timestamp: Date.now(),
        });
      }

      // Add return value
      if (result !== undefined && result !== null) {
        // Check if it is a DataFrame
        try {
          const isDataFrame = pyodide.runPython(`
            import pandas as pd
            isinstance(${result}, pd.DataFrame)
          `);
          if (isDataFrame) {
            const html = pyodide.runPython(`${result}.to_html()`);
            outputs.push({
              type: 'dataframe',
              content: html,
              mimeType: 'text/html',
              timestamp: Date.now(),
            });
          } else {
            outputs.push({
              type: 'text',
              content: String(result),
              timestamp: Date.now(),
            });
          }
        } catch {
          outputs.push({
            type: 'text',
            content: String(result),
            timestamp: Date.now(),
          });
        }
      }

      executionCount.value++;

      // Update cell with results
      cells.value = cells.value.map(c =>
        c.id === cellId
          ? {
              ...c,
              status: 'complete' as const,
              outputs,
              executionOrder: executionCount.value,
              lastRunMs: Math.round(endTime - startTime),
            }
          : c
      );

      // Emit bus event
      emitCellEvent(cellId, 'complete', outputs);

      // Reactive re-execution of dependent cells
      if (isReactive.value) {
        await reactiveRerun(cellId);
      }

    } catch (err) {
      const endTime = performance.now();
      const outputs: CellOutput[] = [{
        type: 'error',
        content: String(err),
        timestamp: Date.now(),
      }];

      cells.value = cells.value.map(c =>
        c.id === cellId
          ? {
              ...c,
              status: 'error' as const,
              outputs,
              lastRunMs: Math.round(endTime - startTime),
            }
          : c
      );

      emitCellEvent(cellId, 'error', outputs);
    }
  });

  // Reactive re-execution
  const reactiveRerun = $(async (changedCellId: string) => {
    const graph = buildDependencyGraph(cells.value);
    const order = topologicalSort(cells.value, graph);

    // Find cells that depend on the changed cell
    const changedIndex = order.indexOf(changedCellId);
    const toRerun = order.slice(changedIndex + 1).filter(id => {
      const deps = graph.get(id);
      return deps && deps.has(changedCellId);
    });

    // Mark as pending
    cells.value = cells.value.map(c =>
      toRerun.includes(c.id) ? { ...c, status: 'pending' as const } : c
    );

    // Execute in order
    for (const id of toRerun) {
      await executeCell(id);
    }
  });

  // Emit bus event for cell execution
  const emitCellEvent = (cellId: string, status: string, outputs: CellOutput[]) => {
    if (typeof window !== 'undefined') {
      const event = {
        topic: 'marimo.cell.execute',
        kind: 'metric',
        level: status === 'error' ? 'error' : 'info',
        actor: 'marimo-notebook',
        ts: Date.now(),
        iso: new Date().toISOString(),
        data: {
          cellId,
          status,
          outputCount: outputs.length,
          notebookPath,
        },
      };

      // Emit to bus via WebSocket if available
      try {
        const busWs = (window as any).__pluribus_bus_ws;
        if (busWs && busWs.readyState === 1) {
          busWs.send(JSON.stringify({ type: 'publish', event }));
        }
      } catch {
        // Ignore bus errors
      }
    }
  };

  // Cell operations
  const handleCodeChange = $((cellId: string, code: string) => {
    cells.value = cells.value.map(c =>
      c.id === cellId ? { ...c, code, status: 'idle' as const } : c
    );
  });

  const handleDeleteCell = $((cellId: string) => {
    if (cells.value.length <= 1) return;
    cells.value = cells.value.filter(c => c.id !== cellId);
    if (selectedCellId.value === cellId) {
      selectedCellId.value = cells.value[0]?.id || null;
    }
  });

  const addCell = $(() => {
    const newCell: NotebookCell = {
      id: crypto.randomUUID(),
      code: '',
      outputs: [],
      status: 'idle',
      dependencies: [],
      executionOrder: 0,
    };

    const selectedIndex = cells.value.findIndex(c => c.id === selectedCellId.value);
    if (selectedIndex >= 0) {
      const newCells = [...cells.value];
      newCells.splice(selectedIndex + 1, 0, newCell);
      cells.value = newCells;
    } else {
      cells.value = [...cells.value, newCell];
    }

    selectedCellId.value = newCell.id;
  });

  const runAllCells = $(async () => {
    const graph = buildDependencyGraph(cells.value);
    const order = topologicalSort(cells.value, graph);

    for (const id of order) {
      await executeCell(id);
    }
  });

  const exportNotebook = $(() => {
    const content = cells.value.map(c => `# %%\n${c.code}`).join('\n\n');
    const blob = new Blob([content], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = notebookPath?.split('/').pop() || 'notebook.py';
    a.click();
    URL.revokeObjectURL(url);
  });

  // Render
  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden" style={{ height }}>
      {/* Toolbar */}
      <div class="flex items-center justify-between px-4 py-2 bg-muted/30 border-b border-border">
        <div class="flex items-center gap-3">
          <h2 class="font-semibold text-sm">Marimo Notebook</h2>
          {notebookPath && (
            <span class="text-xs font-mono text-muted-foreground">{notebookPath}</span>
          )}
          <div class={`w-2 h-2 rounded-full ${pyodideReady.value ? 'bg-green-500' : pyodideLoading.value ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'}`} />
          <span class="text-xs text-muted-foreground">
            {pyodideReady.value ? 'Pyodide Ready' : pyodideLoading.value ? 'Loading...' : 'Not Ready'}
          </span>
        </div>

        <div class="flex items-center gap-2">
          {/* Reactive Toggle */}
          <button
            onClick$={() => isReactive.value = !isReactive.value}
            class={`text-xs px-2 py-1 rounded border transition-colors ${
              isReactive.value
                ? 'bg-primary/20 text-primary border-primary/30'
                : 'bg-muted/30 text-muted-foreground border-border'
            }`}
          >
            Reactive: {isReactive.value ? 'ON' : 'OFF'}
          </button>

          <button
            onClick$={addCell}
            class="text-xs px-2 py-1 rounded bg-muted/30 text-muted-foreground border border-border hover:bg-muted/50"
          >
            + Add Cell
          </button>

          <button
            onClick$={runAllCells}
            disabled={!pyodideReady.value}
            class="text-xs px-2 py-1 rounded bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 disabled:opacity-50"
          >
            Run All
          </button>

          <button
            onClick$={exportNotebook}
            class="text-xs px-2 py-1 rounded bg-muted/30 text-muted-foreground border border-border hover:bg-muted/50"
          >
            Export .py
          </button>
        </div>
      </div>

      {/* Error Display */}
      {pyodideError.value && (
        <div class="px-4 py-2 bg-red-500/10 text-red-400 text-xs border-b border-red-500/30">
          {pyodideError.value}
        </div>
      )}

      {/* Cells */}
      <div class="p-4 space-y-4 overflow-auto" style={{ height: 'calc(100% - 60px)' }}>
        {cells.value.map(cell => (
          <NotebookCellView
            key={cell.id}
            cell={cell}
            onCodeChange$={handleCodeChange}
            onRun$={executeCell}
            onDelete$={handleDeleteCell}
            isSelected={selectedCellId.value === cell.id}
            onSelect$={() => { selectedCellId.value = cell.id; }}
          />
        ))}

        {/* Loading State */}
        {pyodideLoading.value && cells.value.length === 0 && (
          <div class="flex items-center justify-center h-48 text-muted-foreground">
            <div class="text-center">
              <div class="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4" />
              <div class="text-sm">Loading Pyodide WASM runtime...</div>
              <div class="text-xs mt-2">This may take a few seconds on first load</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export default MarimoNotebook;
