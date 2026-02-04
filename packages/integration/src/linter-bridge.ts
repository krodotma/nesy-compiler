/**
 * LinterBridge: Ingest Ground Truth from Static Analyzers
 *
 * Step 11 of NeSy Evolution Phase 1.5 (Sensor Fusion)
 *
 * Runs external linters (ESLint, Ruff, TSC) and ingests their output
 * as Graph Node properties. This provides "ground truth" from compilers
 * that the neural components should respect.
 *
 * Principle: Symbolic > Neural (trust the compiler)
 */

import { spawn } from 'child_process';
import * as path from 'path';

export type LinterType = 'eslint' | 'ruff' | 'tsc' | 'biome';

export type Severity = 'error' | 'warning' | 'info' | 'hint';

export interface LintViolation {
  ruleId: string;
  message: string;
  severity: Severity;
  line: number;
  column: number;
  endLine?: number;
  endColumn?: number;
  source?: string;
  fix?: {
    range: [number, number];
    text: string;
  };
}

export interface LintResult {
  filePath: string;
  violations: LintViolation[];
  errorCount: number;
  warningCount: number;
  fixableCount: number;
}

export interface LinterConfig {
  /** Working directory for running linters */
  cwd: string;
  /** Timeout in milliseconds */
  timeout: number;
  /** Additional arguments to pass to linters */
  extraArgs?: string[];
}

const DEFAULT_CONFIG: LinterConfig = {
  cwd: process.cwd(),
  timeout: 60000,
};

/**
 * Run ESLint and parse JSON output.
 */
export async function runESLint(
  files: string[],
  config?: Partial<LinterConfig>
): Promise<LintResult[]> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  const args = [
    '--format', 'json',
    '--no-error-on-unmatched-pattern',
    ...(cfg.extraArgs || []),
    ...files,
  ];

  try {
    const output = await runCommand('eslint', args, cfg);
    const results = JSON.parse(output) as ESLintOutput[];

    return results.map(r => ({
      filePath: r.filePath,
      violations: r.messages.map(m => ({
        ruleId: m.ruleId || 'unknown',
        message: m.message,
        severity: eslintSeverity(m.severity),
        line: m.line,
        column: m.column,
        endLine: m.endLine,
        endColumn: m.endColumn,
        source: m.source,
        fix: m.fix,
      })),
      errorCount: r.errorCount,
      warningCount: r.warningCount,
      fixableCount: r.fixableErrorCount + r.fixableWarningCount,
    }));
  } catch (error) {
    console.error('[LinterBridge] ESLint failed:', error);
    return [];
  }
}

interface ESLintOutput {
  filePath: string;
  messages: Array<{
    ruleId: string | null;
    message: string;
    severity: 1 | 2;
    line: number;
    column: number;
    endLine?: number;
    endColumn?: number;
    source?: string;
    fix?: { range: [number, number]; text: string };
  }>;
  errorCount: number;
  warningCount: number;
  fixableErrorCount: number;
  fixableWarningCount: number;
}

function eslintSeverity(level: 1 | 2): Severity {
  return level === 2 ? 'error' : 'warning';
}

/**
 * Run Ruff (Python linter) and parse JSON output.
 */
export async function runRuff(
  files: string[],
  config?: Partial<LinterConfig>
): Promise<LintResult[]> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  const args = [
    'check',
    '--output-format', 'json',
    ...(cfg.extraArgs || []),
    ...files,
  ];

  try {
    const output = await runCommand('ruff', args, cfg);
    const violations = JSON.parse(output) as RuffViolation[];

    // Group by file
    const byFile = new Map<string, LintViolation[]>();
    for (const v of violations) {
      const filePath = v.filename;
      if (!byFile.has(filePath)) {
        byFile.set(filePath, []);
      }
      byFile.get(filePath)!.push({
        ruleId: v.code,
        message: v.message,
        severity: ruffSeverity(v.code),
        line: v.location.row,
        column: v.location.column,
        endLine: v.end_location?.row,
        endColumn: v.end_location?.column,
        fix: v.fix ? {
          range: [0, 0], // Ruff doesn't provide byte ranges
          text: v.fix.message,
        } : undefined,
      });
    }

    return Array.from(byFile.entries()).map(([filePath, violations]) => ({
      filePath,
      violations,
      errorCount: violations.filter(v => v.severity === 'error').length,
      warningCount: violations.filter(v => v.severity === 'warning').length,
      fixableCount: violations.filter(v => v.fix).length,
    }));
  } catch (error) {
    console.error('[LinterBridge] Ruff failed:', error);
    return [];
  }
}

interface RuffViolation {
  code: string;
  message: string;
  filename: string;
  location: { row: number; column: number };
  end_location?: { row: number; column: number };
  fix?: { message: string; edits: unknown[] };
}

function ruffSeverity(code: string): Severity {
  // E/F codes are errors, W codes are warnings
  if (code.startsWith('E') || code.startsWith('F')) return 'error';
  if (code.startsWith('W')) return 'warning';
  return 'info';
}

/**
 * Run TypeScript compiler for type checking.
 */
export async function runTSC(
  configPath?: string,
  config?: Partial<LinterConfig>
): Promise<LintResult[]> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  const args = [
    '--noEmit',
    '--pretty', 'false',
    ...(configPath ? ['-p', configPath] : []),
    ...(cfg.extraArgs || []),
  ];

  try {
    const output = await runCommand('tsc', args, cfg);
    return parseTSCOutput(output, cfg.cwd);
  } catch (error) {
    // TSC exits with code 1 when there are errors, but still produces output
    if (typeof error === 'object' && error !== null && 'output' in error) {
      return parseTSCOutput((error as { output: string }).output, cfg.cwd);
    }
    console.error('[LinterBridge] TSC failed:', error);
    return [];
  }
}

function parseTSCOutput(output: string, cwd: string): LintResult[] {
  const results = new Map<string, LintViolation[]>();

  // Parse TSC output format: file(line,col): error TS1234: message
  const regex = /^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)$/gm;
  let match;

  while ((match = regex.exec(output)) !== null) {
    const [, file, line, col, severity, code, message] = match;
    const filePath = path.isAbsolute(file) ? file : path.join(cwd, file);

    if (!results.has(filePath)) {
      results.set(filePath, []);
    }

    results.get(filePath)!.push({
      ruleId: code,
      message,
      severity: severity as Severity,
      line: parseInt(line, 10),
      column: parseInt(col, 10),
    });
  }

  return Array.from(results.entries()).map(([filePath, violations]) => ({
    filePath,
    violations,
    errorCount: violations.filter(v => v.severity === 'error').length,
    warningCount: violations.filter(v => v.severity === 'warning').length,
    fixableCount: 0, // TSC doesn't provide auto-fixes
  }));
}

/**
 * Run Biome (fast all-in-one linter).
 */
export async function runBiome(
  files: string[],
  config?: Partial<LinterConfig>
): Promise<LintResult[]> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  const args = [
    'lint',
    '--reporter', 'json',
    ...(cfg.extraArgs || []),
    ...files,
  ];

  try {
    const output = await runCommand('biome', args, cfg);
    const data = JSON.parse(output) as BiomeOutput;

    return data.diagnostics.map(d => ({
      filePath: d.location.path.file,
      violations: [{
        ruleId: d.category,
        message: d.message,
        severity: biomeSeverity(d.severity),
        line: d.location.span?.start.line || 1,
        column: d.location.span?.start.column || 1,
        endLine: d.location.span?.end.line,
        endColumn: d.location.span?.end.column,
      }],
      errorCount: d.severity === 'error' ? 1 : 0,
      warningCount: d.severity === 'warning' ? 1 : 0,
      fixableCount: 0,
    }));
  } catch (error) {
    console.error('[LinterBridge] Biome failed:', error);
    return [];
  }
}

interface BiomeOutput {
  diagnostics: Array<{
    category: string;
    message: string;
    severity: 'error' | 'warning' | 'information' | 'hint';
    location: {
      path: { file: string };
      span?: {
        start: { line: number; column: number };
        end: { line: number; column: number };
      };
    };
  }>;
}

function biomeSeverity(level: string): Severity {
  switch (level) {
    case 'error': return 'error';
    case 'warning': return 'warning';
    case 'information': return 'info';
    case 'hint': return 'hint';
    default: return 'info';
  }
}

/**
 * Run all available linters on a codebase.
 */
export async function runAllLinters(
  files: string[],
  config?: Partial<LinterConfig>
): Promise<Map<LinterType, LintResult[]>> {
  const results = new Map<LinterType, LintResult[]>();

  const tsFiles = files.filter(f => f.endsWith('.ts') || f.endsWith('.tsx') || f.endsWith('.js') || f.endsWith('.jsx'));
  const pyFiles = files.filter(f => f.endsWith('.py'));

  // Run linters in parallel
  const [eslintResults, ruffResults] = await Promise.all([
    tsFiles.length > 0 ? runESLint(tsFiles, config) : Promise.resolve([]),
    pyFiles.length > 0 ? runRuff(pyFiles, config) : Promise.resolve([]),
  ]);

  if (eslintResults.length > 0) results.set('eslint', eslintResults);
  if (ruffResults.length > 0) results.set('ruff', ruffResults);

  return results;
}

/**
 * Aggregate lint results into a summary for graph ingestion.
 */
export function aggregateLintResults(
  results: LintResult[]
): {
  totalErrors: number;
  totalWarnings: number;
  violationsByRule: Map<string, number>;
  fileScores: Map<string, number>;
} {
  const violationsByRule = new Map<string, number>();
  const fileScores = new Map<string, number>();
  let totalErrors = 0;
  let totalWarnings = 0;

  for (const result of results) {
    totalErrors += result.errorCount;
    totalWarnings += result.warningCount;

    // Calculate file quality score (100 - penalties)
    const penalty = result.errorCount * 10 + result.warningCount * 2;
    const score = Math.max(0, 100 - penalty);
    fileScores.set(result.filePath, score);

    for (const v of result.violations) {
      const count = violationsByRule.get(v.ruleId) || 0;
      violationsByRule.set(v.ruleId, count + 1);
    }
  }

  return { totalErrors, totalWarnings, violationsByRule, fileScores };
}

/**
 * Helper to run a command and capture output.
 */
function runCommand(
  cmd: string,
  args: string[],
  config: LinterConfig
): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      cwd: config.cwd,
      timeout: config.timeout,
      shell: true,
    });

    let stdout = '';
    let stderr = '';

    proc.stdout?.on('data', (data) => { stdout += data; });
    proc.stderr?.on('data', (data) => { stderr += data; });

    proc.on('close', (code) => {
      if (code === 0 || stdout) {
        resolve(stdout);
      } else {
        reject({ code, stderr, output: stdout });
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });
  });
}
