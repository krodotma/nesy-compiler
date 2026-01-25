/**
 * Python CLI Wrapper Utilities
 * ============================
 *
 * Utilities for executing Python CLI tools from Qwik City API routes.
 * Uses child_process.spawn for subprocess execution.
 *
 * DKIN: v29 | Protocol: metaingest/v1
 */

import { spawn, type ChildProcess } from 'child_process';
import type { CLIResult } from '../types';

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * Default paths for MetaIngest CLI tools
 */
export const CLI_PATHS = {
  ONTOLOGY_EVOLVER: '/pluribus/nucleus/tools/ontology_evolver.py',
  SEMANTIC_DRIFT_TRACKER: '/pluribus/nucleus/tools/semantic_drift_tracker.py',
  KNOWLEDGE_GRAPH_INGESTOR: '/pluribus/nucleus/tools/knowledge_graph_ingestor.py',
  SOTA_MUTATION_ENGINE: '/pluribus/nucleus/tools/sota_mutation_engine.py',
  SOTA_INGESTION: '/pluribus/nucleus/tools/sota_ingestion.py',
  METAINGEST_PIPELINE: '/pluribus/nucleus/tools/metaingest_pipeline.py',
} as const;

/**
 * Default working directory
 */
export const DEFAULT_CWD = '/pluribus';

/**
 * Default timeout in milliseconds
 */
export const DEFAULT_TIMEOUT = 30000;

// =============================================================================
// CLI EXECUTION
// =============================================================================

/**
 * Options for CLI execution
 */
export interface CLIOptions {
  /** Timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Working directory (default: /pluribus) */
  cwd?: string;
  /** Environment variables to pass */
  env?: Record<string, string>;
  /** Input to pipe to stdin */
  stdin?: string;
}

/**
 * Execute a Python CLI script with arguments.
 *
 * @param script - Path to Python script
 * @param args - Command line arguments
 * @param options - Execution options
 * @returns Promise resolving to CLI result
 *
 * @example
 * const result = await runPythonCLI(
 *   CLI_PATHS.ONTOLOGY_EVOLVER,
 *   ['terms', '--limit', '50']
 * );
 */
export async function runPythonCLI(
  script: string,
  args: string[],
  options: CLIOptions = {}
): Promise<CLIResult> {
  const {
    timeout = DEFAULT_TIMEOUT,
    cwd = DEFAULT_CWD,
    env,
    stdin,
  } = options;

  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    let killed = false;

    const proc: ChildProcess = spawn('python3', [script, ...args], {
      cwd,
      env: { ...process.env, ...env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Timeout handler
    const timer = setTimeout(() => {
      killed = true;
      proc.kill('SIGTERM');
      reject(new Error(`CLI timeout after ${timeout}ms: ${script} ${args.join(' ')}`));
    }, timeout);

    // Collect stdout
    proc.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    // Collect stderr
    proc.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    // Handle process close
    proc.on('close', (code: number | null) => {
      clearTimeout(timer);
      if (!killed) {
        resolve({
          stdout: stdout.trim(),
          stderr: stderr.trim(),
          exitCode: code ?? 0,
        });
      }
    });

    // Handle process error
    proc.on('error', (err: Error) => {
      clearTimeout(timer);
      if (!killed) {
        reject(new Error(`CLI spawn error: ${err.message}`));
      }
    });

    // Write stdin if provided
    if (stdin && proc.stdin) {
      proc.stdin.write(stdin);
      proc.stdin.end();
    }
  });
}

// =============================================================================
// OUTPUT PARSING
// =============================================================================

/**
 * Parse JSON output from CLI result.
 *
 * @param result - CLI execution result
 * @returns Parsed JSON data
 * @throws Error if CLI failed or JSON is invalid
 *
 * @example
 * const result = await runPythonCLI(script, ['--json']);
 * const data = parseJSONOutput<MyType>(result);
 */
export function parseJSONOutput<T>(result: CLIResult): T {
  if (result.exitCode !== 0) {
    throw new CLIError(
      `CLI exited with code ${result.exitCode}`,
      result.exitCode,
      result.stderr
    );
  }

  try {
    return JSON.parse(result.stdout) as T;
  } catch (e) {
    throw new CLIError(
      `Failed to parse JSON output: ${e instanceof Error ? e.message : 'Unknown error'}`,
      0,
      result.stdout.substring(0, 200)
    );
  }
}

/**
 * Try to parse JSON output, return null on failure.
 *
 * @param result - CLI execution result
 * @returns Parsed JSON data or null
 */
export function tryParseJSONOutput<T>(result: CLIResult): T | null {
  try {
    return parseJSONOutput<T>(result);
  } catch {
    return null;
  }
}

/**
 * Parse tabular text output into array of records.
 * Handles output like:
 * ```
 * term1  0.85  active
 * term2  0.72  active
 * ```
 *
 * @param output - Text output
 * @param headers - Column headers
 * @param delimiter - Column delimiter (default: whitespace)
 */
export function parseTabularOutput(
  output: string,
  headers: string[],
  delimiter: RegExp = /\s+/
): Array<Record<string, string>> {
  const lines = output.split('\n').filter(line =>
    line.trim() && !line.startsWith('-') && !line.startsWith('=')
  );

  return lines.map(line => {
    const values = line.trim().split(delimiter);
    const record: Record<string, string> = {};

    headers.forEach((header, i) => {
      record[header] = values[i] || '';
    });

    return record;
  });
}

/**
 * Parse key-value output like:
 * ```
 * Total Terms: 42
 * Active: 35
 * ```
 *
 * @param output - Text output
 * @param separator - Key-value separator (default: ':')
 */
export function parseKeyValueOutput(
  output: string,
  separator: string = ':'
): Record<string, string> {
  const result: Record<string, string> = {};

  output.split('\n').forEach(line => {
    const sepIndex = line.indexOf(separator);
    if (sepIndex > 0) {
      const key = line.substring(0, sepIndex).trim();
      const value = line.substring(sepIndex + 1).trim();
      if (key) {
        result[key] = value;
      }
    }
  });

  return result;
}

// =============================================================================
// ERROR HANDLING
// =============================================================================

/**
 * CLI execution error
 */
export class CLIError extends Error {
  constructor(
    message: string,
    public readonly exitCode: number,
    public readonly stderr: string
  ) {
    super(message);
    this.name = 'CLIError';
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      exitCode: this.exitCode,
      stderr: this.stderr,
    };
  }
}

/**
 * Check if error is a CLI error
 */
export function isCLIError(error: unknown): error is CLIError {
  return error instanceof CLIError;
}

/**
 * Check if error is a timeout error
 */
export function isTimeoutError(error: unknown): error is Error {
  return error instanceof Error && error.message.includes('timeout');
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Run ontology evolver command
 */
export async function runOntologyEvolver(
  command: string,
  args: string[] = [],
  options?: CLIOptions
): Promise<CLIResult> {
  return runPythonCLI(CLI_PATHS.ONTOLOGY_EVOLVER, [command, ...args], options);
}

/**
 * Run semantic drift tracker command
 */
export async function runDriftTracker(
  command: string,
  args: string[] = [],
  options?: CLIOptions
): Promise<CLIResult> {
  return runPythonCLI(CLI_PATHS.SEMANTIC_DRIFT_TRACKER, [command, ...args], options);
}

/**
 * Run knowledge graph ingestor command
 */
export async function runKnowledgeGraph(
  command: string,
  args: string[] = [],
  options?: CLIOptions
): Promise<CLIResult> {
  return runPythonCLI(CLI_PATHS.KNOWLEDGE_GRAPH_INGESTOR, [command, ...args], options);
}

/**
 * Run SOTA mutation engine command
 */
export async function runSOTAMutation(
  command: string,
  args: string[] = [],
  options?: CLIOptions
): Promise<CLIResult> {
  return runPythonCLI(CLI_PATHS.SOTA_MUTATION_ENGINE, [command, ...args], options);
}

/**
 * Run MetaIngest pipeline command
 */
export async function runPipeline(
  command: string,
  args: string[] = [],
  options?: CLIOptions
): Promise<CLIResult> {
  return runPythonCLI(CLI_PATHS.METAINGEST_PIPELINE, [command, ...args], options);
}

// =============================================================================
// ARGUMENT BUILDING
// =============================================================================

/**
 * Build CLI arguments from an object.
 *
 * @example
 * buildArgs({ limit: 50, json: true, term: 'foo' })
 * // Returns: ['--limit', '50', '--json', '--term', 'foo']
 */
export function buildArgs(params: Record<string, unknown>): string[] {
  const args: string[] = [];

  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) {
      continue;
    }

    const flag = `--${key.replace(/_/g, '-')}`;

    if (typeof value === 'boolean') {
      if (value) {
        args.push(flag);
      }
    } else if (Array.isArray(value)) {
      for (const item of value) {
        args.push(flag, String(item));
      }
    } else {
      args.push(flag, String(value));
    }
  }

  return args;
}
