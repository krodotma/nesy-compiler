/**
 * PyodideCell.tsx - In-Browser Python Execution Cell
 *
 * Implements Pyodide (cPython via WASM) integration for the Pluribus dashboard.
 * Features:
 * - In-browser Python execution without server roundtrip
 * - micropip package installation
 * - Output bridging to Pluribus bus events
 * - Persistent execution context per session
 */

import {
  component$,
  useSignal,
  useStore,
  $,
  useVisibleTask$,
  noSerialize,
  type NoSerialize,
} from '@builder.io/qwik';

// Types for Pyodide
interface PyodideInterface {
  runPythonAsync(code: string): Promise<unknown>;
  loadPackage(packages: string | string[]): Promise<void>;
  loadPackagesFromImports(code: string): Promise<void>;
  globals: {
    get(name: string): unknown;
    set(name: string, value: unknown): void;
  };
  runPython(code: string): unknown;
  version: string;
}

interface MicroPip {
  install(packages: string | string[]): Promise<void>;
}

interface PyodideCellState {
  status: 'uninitialized' | 'loading' | 'ready' | 'executing' | 'error';
  loadProgress: string;
  pyodide: NoSerialize<PyodideInterface> | null;
  output: string[];
  error: string | null;
  executionCount: number;
  installedPackages: string[];
}

interface CellHistoryItem {
  code: string;
  output: string;
  error: string | null;
  timestamp: number;
}

// Load Pyodide from CDN
async function loadPyodideRuntime(): Promise<PyodideInterface> {
  // Dynamically load the Pyodide script
  if (typeof window !== 'undefined' && !(window as unknown as { loadPyodide?: unknown }).loadPyodide) {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js';
    script.async = true;
    document.head.appendChild(script);

    await new Promise<void>((resolve, reject) => {
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load Pyodide script'));
    });
  }

  const loadPyodide = (window as unknown as { loadPyodide: (config: { indexURL: string }) => Promise<PyodideInterface> }).loadPyodide;
  return await loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.1/full/',
  });
}

export const PyodideCell = component$(() => {
  const state = useStore<PyodideCellState>({
    status: 'uninitialized',
    loadProgress: '',
    pyodide: null,
    output: [],
    error: null,
    executionCount: 0,
    installedPackages: [],
  });

  const codeInput = useSignal('# Python code here\nprint("Hello from Pyodide!")');
  const packageInput = useSignal('');
  const history = useStore<{ items: CellHistoryItem[] }>({ items: [] });
  const expanded = useSignal(true);
  const showInstaller = useSignal(false);

  // Initialize Pyodide
  const initPyodide = $(async () => {
    if (state.status === 'loading' || state.pyodide) return;

    state.status = 'loading';
    state.loadProgress = 'Loading Pyodide runtime...';
    state.error = null;

    try {
      const pyodide = await loadPyodideRuntime();

      // Set up stdout/stderr capture
      state.loadProgress = 'Configuring output capture...';
      await pyodide.runPythonAsync(`
import sys
from io import StringIO

class OutputCapture:
    def __init__(self):
        self.buffer = StringIO()

    def write(self, text):
        self.buffer.write(text)
        return len(text)

    def flush(self):
        pass

    def getvalue(self):
        return self.buffer.getvalue()

    def clear(self):
        self.buffer = StringIO()

_stdout_capture = OutputCapture()
_stderr_capture = OutputCapture()
sys.stdout = _stdout_capture
sys.stderr = _stderr_capture

def _get_output():
    out = _stdout_capture.getvalue()
    err = _stderr_capture.getvalue()
    _stdout_capture.clear()
    _stderr_capture.clear()
    return out, err
`);

      state.pyodide = noSerialize(pyodide);
      state.status = 'ready';
      state.loadProgress = `Pyodide ${pyodide.version} ready`;
      state.output = [`Pyodide ${pyodide.version} initialized successfully.`];
    } catch (err) {
      state.status = 'error';
      state.error = err instanceof Error ? err.message : 'Failed to load Pyodide';
    }
  });

  // Execute Python code
  const executeCode = $(async () => {
    if (!state.pyodide || state.status !== 'ready') return;

    const code = codeInput.value.trim();
    if (!code) return;

    state.status = 'executing';
    state.error = null;
    const startTime = Date.now();

    try {
      const pyodide = state.pyodide as unknown as PyodideInterface;

      // Auto-install packages from imports
      await pyodide.loadPackagesFromImports(code);

      // Execute the code
      const result = await pyodide.runPythonAsync(code);

      // Get captured output
      const [stdout, stderr] = pyodide.runPython('_get_output()') as [string, string];

      let outputText = '';
      if (stdout) outputText += stdout;
      if (stderr) outputText += `[stderr] ${stderr}`;
      if (result !== undefined && result !== null) {
        outputText += `=> ${String(result)}`;
      }
      if (!outputText) outputText = '(no output)';

      state.output = [...state.output, `In [${state.executionCount}]: ${code.split('\n')[0]}...`, outputText];
      state.executionCount++;

      // Add to history
      history.items = [
        ...history.items,
        {
          code,
          output: outputText,
          error: null,
          timestamp: startTime,
        },
      ].slice(-20); // Keep last 20

      state.status = 'ready';

      // Emit bus event (if bus client available)
      emitBusEvent('pyodide.cell.execute', {
        code: code.slice(0, 200),
        output: outputText.slice(0, 500),
        executionCount: state.executionCount,
        duration_ms: Date.now() - startTime,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      state.error = errorMsg;
      state.output = [...state.output, `Error: ${errorMsg}`];
      state.status = 'ready';

      history.items = [
        ...history.items,
        {
          code,
          output: '',
          error: errorMsg,
          timestamp: startTime,
        },
      ].slice(-20);
    }
  });

  // Install package via micropip
  const installPackage = $(async () => {
    if (!state.pyodide || state.status !== 'ready') return;

    const pkg = packageInput.value.trim();
    if (!pkg) return;

    state.status = 'executing';
    state.loadProgress = `Installing ${pkg}...`;

    try {
      const pyodide = state.pyodide as unknown as PyodideInterface;

      // Load micropip if not already loaded
      await pyodide.loadPackage('micropip');
      const micropip = pyodide.globals.get('micropip') as MicroPip;

      await micropip.install(pkg);

      state.installedPackages = [...state.installedPackages, pkg];
      state.output = [...state.output, `Installed: ${pkg}`];
      state.status = 'ready';
      state.loadProgress = '';
      packageInput.value = '';

      emitBusEvent('pyodide.package.install', { package: pkg, success: true });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      state.error = `Failed to install ${pkg}: ${errorMsg}`;
      state.status = 'ready';
      state.loadProgress = '';

      emitBusEvent('pyodide.package.install', { package: pkg, success: false, error: errorMsg });
    }
  });

  // Clear output
  const clearOutput = $(() => {
    state.output = [];
    state.error = null;
  });

  // Emit bus event helper
  const emitBusEvent = (topic: string, data: Record<string, unknown>) => {
    // Fire-and-forget event emission via WebSocket if available
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
                actor: 'pyodide-cell',
                data,
              },
            })
          );
          ws.close();
        };
      }
    } catch {
      // Bus emission is best-effort
    }
  };

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div
        class="flex items-center gap-3 p-3 bg-black/30 cursor-pointer hover:bg-black/40 transition-colors"
        onClick$={() => (expanded.value = !expanded.value)}
      >
        <span
          class={`w-3 h-3 rounded-full ${
            state.status === 'ready'
              ? 'bg-green-400'
              : state.status === 'loading' || state.status === 'executing'
              ? 'bg-yellow-400 animate-pulse'
              : state.status === 'error'
              ? 'bg-red-400'
              : 'bg-gray-400'
          }`}
        />
        <span class="text-lg">&#x1F40D;</span>
        <div class="flex-1">
          <div class="font-medium">Pyodide Python Cell</div>
          <div class="text-xs text-muted-foreground">
            {state.status === 'ready'
              ? `Ready - ${state.executionCount} executions`
              : state.status === 'loading'
              ? state.loadProgress
              : state.status === 'executing'
              ? 'Executing...'
              : state.status === 'error'
              ? state.error
              : 'Click to initialize'}
          </div>
        </div>
        {state.installedPackages.length > 0 && (
          <span class="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-400">
            {state.installedPackages.length} pkgs
          </span>
        )}
        <span class={`text-muted-foreground transition-transform ${expanded.value ? 'rotate-180' : ''}`}>
          &#x25BC;
        </span>
      </div>

      {/* Main Content */}
      {expanded.value && (
        <div class="border-t border-border/50">
          {/* Initialize Button (if not loaded) */}
          {state.status === 'uninitialized' && (
            <div class="p-4 text-center">
              <button
                class="px-4 py-2 rounded bg-green-600 hover:bg-green-500 text-white font-medium"
                onClick$={initPyodide}
              >
                Initialize Pyodide Runtime
              </button>
              <div class="text-xs text-muted-foreground mt-2">
                Downloads ~15MB WASM runtime (cached after first load)
              </div>
            </div>
          )}

          {/* Loading State */}
          {state.status === 'loading' && (
            <div class="p-4 text-center">
              <div class="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full mx-auto mb-2" />
              <div class="text-sm text-muted-foreground">{state.loadProgress}</div>
            </div>
          )}

          {/* Ready State */}
          {(state.status === 'ready' || state.status === 'executing') && (
            <div class="flex flex-col">
              {/* Toolbar */}
              <div class="flex items-center gap-2 p-2 bg-black/20 border-b border-border/30">
                <button
                  class="px-3 py-1 rounded text-xs font-medium bg-green-600 hover:bg-green-500 text-white disabled:opacity-50"
                  onClick$={executeCode}
                  disabled={state.status === 'executing'}
                >
                  &#x25B6; Run
                </button>
                <button
                  class="px-3 py-1 rounded text-xs font-medium bg-gray-600 hover:bg-gray-500 text-white"
                  onClick$={clearOutput}
                >
                  Clear
                </button>
                <button
                  class={`px-3 py-1 rounded text-xs font-medium ${
                    showInstaller.value ? 'bg-purple-600' : 'bg-gray-600 hover:bg-gray-500'
                  } text-white`}
                  onClick$={() => (showInstaller.value = !showInstaller.value)}
                >
                  &#x1F4E6; Packages
                </button>
                <div class="flex-1" />
                <span class="text-xs text-muted-foreground mono">
                  {state.status === 'executing' ? 'Running...' : `In [${state.executionCount}]`}
                </span>
              </div>

              {/* Package Installer */}
              {showInstaller.value && (
                <div class="flex items-center gap-2 p-2 bg-purple-500/10 border-b border-purple-500/30">
                  <input
                    type="text"
                    class="flex-1 bg-black/50 border border-border/50 rounded px-2 py-1 text-xs"
                    placeholder="Package name (e.g., numpy, pandas, sympy)"
                    value={packageInput.value}
                    onInput$={(e) => (packageInput.value = (e.target as HTMLInputElement).value)}
                    onKeyPress$={(e) => e.key === 'Enter' && installPackage()}
                  />
                  <button
                    class="px-3 py-1 rounded text-xs font-medium bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-50"
                    onClick$={installPackage}
                    disabled={state.status === 'executing'}
                  >
                    Install
                  </button>
                </div>
              )}

              {/* Code Editor */}
              <div class="p-2">
                <textarea
                  class="w-full h-32 bg-black/50 border border-border/50 rounded p-2 font-mono text-sm resize-y"
                  value={codeInput.value}
                  onInput$={(e) => (codeInput.value = (e.target as HTMLTextAreaElement).value)}
                  onKeyDown$={(e) => {
                    if (e.ctrlKey && e.key === 'Enter') {
                      executeCode();
                    }
                  }}
                  placeholder="Enter Python code... (Ctrl+Enter to run)"
                  spellcheck={false}
                />
              </div>

              {/* Output */}
              <div class="border-t border-border/30 bg-black/30">
                <div class="p-2 text-xs text-muted-foreground border-b border-border/30">Output</div>
                <div class="p-2 font-mono text-xs max-h-48 overflow-y-auto">
                  {state.output.length === 0 ? (
                    <div class="text-muted-foreground">(no output yet)</div>
                  ) : (
                    state.output.map((line, i) => (
                      <div key={i} class="whitespace-pre-wrap break-words">
                        {line}
                      </div>
                    ))
                  )}
                  {state.error && <div class="text-red-400 mt-1">Error: {state.error}</div>}
                </div>
              </div>

              {/* Installed Packages */}
              {state.installedPackages.length > 0 && (
                <div class="p-2 border-t border-border/30 bg-black/20">
                  <div class="text-xs text-muted-foreground mb-1">Installed:</div>
                  <div class="flex flex-wrap gap-1">
                    {state.installedPackages.map((pkg) => (
                      <span
                        key={pkg}
                        class="px-2 py-0.5 rounded text-xs bg-purple-500/20 text-purple-400"
                      >
                        {pkg}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Error State */}
          {state.status === 'error' && !state.pyodide && (
            <div class="p-4 bg-red-500/10 border-t border-red-500/30">
              <div class="text-sm text-red-400 font-medium mb-2">Failed to initialize Pyodide</div>
              <div class="text-xs text-muted-foreground mb-3">{state.error}</div>
              <button
                class="px-3 py-1 rounded text-xs font-medium bg-red-600 hover:bg-red-500 text-white"
                onClick$={() => {
                  state.status = 'uninitialized';
                  state.error = null;
                }}
              >
                Retry
              </button>
            </div>
          )}

          {/* Footer */}
          <div class="p-2 border-t border-border/30 bg-black/30 text-xs text-muted-foreground flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span>&#x1F512;</span>
              <span>100% local execution (WASM sandbox)</span>
            </div>
            <span class="mono">pyodide v0.25.1</span>
          </div>
        </div>
      )}
    </div>
  );
});

export default PyodideCell;
