/**
 * Lanes Performance Benchmark
 *
 * Phase 9, Iteration 74 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Performance benchmarks
 * - Large dataset simulation
 * - Memory profiling
 * - Render performance tracking
 * - Operation timing
 */

import type { Lane, LanesState, LaneAction } from './store';

// ============================================================================
// Types
// ============================================================================

export interface BenchmarkResult {
  name: string;
  duration: number;
  iterations: number;
  avgDuration: number;
  minDuration: number;
  maxDuration: number;
  opsPerSecond: number;
  memoryUsed?: number;
  timestamp: number;
}

export interface BenchmarkSuite {
  name: string;
  results: BenchmarkResult[];
  totalDuration: number;
  startedAt: number;
  completedAt: number;
}

export interface BenchmarkConfig {
  /** Number of warmup iterations */
  warmupIterations: number;
  /** Number of measured iterations */
  iterations: number;
  /** Timeout per benchmark (ms) */
  timeout: number;
  /** Enable memory measurement */
  measureMemory: boolean;
  /** Log progress */
  verbose: boolean;
}

export interface LaneGeneratorConfig {
  /** Number of lanes to generate */
  count: number;
  /** Percentage with blockers (0-100) */
  blockerRate: number;
  /** Maximum WIP percentage */
  maxWip: number;
  /** Status distribution */
  statusDistribution: {
    green: number;
    yellow: number;
    red: number;
  };
  /** Number of agents to assign from */
  agentCount: number;
  /** Tags per lane */
  tagsPerLane: number;
}

// ============================================================================
// Default Configs
// ============================================================================

const DEFAULT_BENCHMARK_CONFIG: BenchmarkConfig = {
  warmupIterations: 3,
  iterations: 10,
  timeout: 30000,
  measureMemory: true,
  verbose: false,
};

const DEFAULT_LANE_GENERATOR_CONFIG: LaneGeneratorConfig = {
  count: 100,
  blockerRate: 20,
  maxWip: 100,
  statusDistribution: { green: 60, yellow: 25, red: 15 },
  agentCount: 10,
  tagsPerLane: 3,
};

// ============================================================================
// Lane Data Generator
// ============================================================================

const SAMPLE_NAMES = [
  'Feature Implementation',
  'Bug Fix',
  'Performance Optimization',
  'Security Patch',
  'Documentation',
  'Testing',
  'Code Review',
  'Deployment',
  'Infrastructure',
  'Refactoring',
  'API Development',
  'UI Enhancement',
  'Database Migration',
  'Cache Optimization',
  'Monitoring Setup',
];

const SAMPLE_BLOCKERS = [
  'Waiting for review',
  'Dependency issue',
  'Test failures',
  'Build broken',
  'Resource unavailable',
  'External service down',
  'Merge conflict',
  'Missing requirements',
  'Security review pending',
  'Performance regression',
];

const SAMPLE_TAGS = [
  'frontend',
  'backend',
  'urgent',
  'tech-debt',
  'feature',
  'bugfix',
  'infra',
  'security',
  'performance',
  'documentation',
];

export function generateLanes(config: Partial<LaneGeneratorConfig> = {}): Lane[] {
  const cfg = { ...DEFAULT_LANE_GENERATOR_CONFIG, ...config };
  const lanes: Lane[] = [];

  // Generate agents
  const agents = Array.from({ length: cfg.agentCount }, (_, i) => `agent-${i + 1}`);

  for (let i = 0; i < cfg.count; i++) {
    // Determine status based on distribution
    const statusRoll = Math.random() * 100;
    let status: Lane['status'];
    if (statusRoll < cfg.statusDistribution.green) {
      status = 'green';
    } else if (statusRoll < cfg.statusDistribution.green + cfg.statusDistribution.yellow) {
      status = 'yellow';
    } else {
      status = 'red';
    }

    // Generate blockers
    const hasBlocker = Math.random() * 100 < cfg.blockerRate;
    const blockers: string[] = [];
    if (hasBlocker) {
      const blockerCount = Math.floor(Math.random() * 3) + 1;
      for (let j = 0; j < blockerCount; j++) {
        blockers.push(SAMPLE_BLOCKERS[Math.floor(Math.random() * SAMPLE_BLOCKERS.length)]);
      }
    }

    // Generate tags
    const tags: string[] = [];
    const shuffledTags = [...SAMPLE_TAGS].sort(() => Math.random() - 0.5);
    for (let j = 0; j < cfg.tagsPerLane; j++) {
      tags.push(shuffledTags[j]);
    }

    lanes.push({
      id: `lane-${i + 1}`,
      name: `${SAMPLE_NAMES[i % SAMPLE_NAMES.length]} ${Math.floor(i / SAMPLE_NAMES.length) + 1}`,
      status,
      wip_pct: Math.floor(Math.random() * cfg.maxWip),
      owner: agents[Math.floor(Math.random() * agents.length)],
      blockers,
      tags,
      updated: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
    });
  }

  return lanes;
}

export function generateLanesState(laneCount: number = 100): LanesState {
  const lanes = generateLanes({ count: laneCount });

  // Extract unique agents
  const agents = [...new Set(lanes.map(l => l.owner))].map(id => ({
    id,
    name: id,
    avatar: undefined,
    status: 'active' as const,
  }));

  return {
    lanes,
    agents,
    selectedLaneId: null,
    filter: {
      status: null,
      owner: null,
      search: '',
      tags: [],
    },
    sort: {
      field: 'updated',
      direction: 'desc',
    },
    isLoading: false,
    error: null,
    lastSync: Date.now(),
  };
}

// ============================================================================
// Benchmark Runner
// ============================================================================

export class BenchmarkRunner {
  private config: BenchmarkConfig;
  private results: BenchmarkResult[] = [];

  constructor(config: Partial<BenchmarkConfig> = {}) {
    this.config = { ...DEFAULT_BENCHMARK_CONFIG, ...config };
  }

  /**
   * Run a benchmark
   */
  async run(name: string, fn: () => void | Promise<void>): Promise<BenchmarkResult> {
    const durations: number[] = [];
    let memoryBefore = 0;
    let memoryAfter = 0;

    // Measure initial memory
    if (this.config.measureMemory && typeof performance !== 'undefined') {
      memoryBefore = (performance as any).memory?.usedJSHeapSize || 0;
    }

    // Warmup
    for (let i = 0; i < this.config.warmupIterations; i++) {
      await fn();
    }

    // Measured iterations
    for (let i = 0; i < this.config.iterations; i++) {
      const start = performance.now();
      await fn();
      const end = performance.now();
      durations.push(end - start);

      if (this.config.verbose) {
        console.log(`[Benchmark] ${name} iteration ${i + 1}: ${(end - start).toFixed(2)}ms`);
      }
    }

    // Measure final memory
    if (this.config.measureMemory && typeof performance !== 'undefined') {
      memoryAfter = (performance as any).memory?.usedJSHeapSize || 0;
    }

    const totalDuration = durations.reduce((a, b) => a + b, 0);
    const result: BenchmarkResult = {
      name,
      duration: totalDuration,
      iterations: this.config.iterations,
      avgDuration: totalDuration / this.config.iterations,
      minDuration: Math.min(...durations),
      maxDuration: Math.max(...durations),
      opsPerSecond: (this.config.iterations / totalDuration) * 1000,
      memoryUsed: memoryAfter - memoryBefore,
      timestamp: Date.now(),
    };

    this.results.push(result);
    return result;
  }

  /**
   * Get all results
   */
  getResults(): BenchmarkResult[] {
    return [...this.results];
  }

  /**
   * Clear results
   */
  clear(): void {
    this.results = [];
  }

  /**
   * Format results as table
   */
  formatResults(): string {
    const lines: string[] = [];
    lines.push('┌─────────────────────────────────────────────────────────────────────┐');
    lines.push('│ Benchmark Results                                                   │');
    lines.push('├───────────────────────────┬─────────┬─────────┬─────────┬──────────┤');
    lines.push('│ Name                      │ Avg(ms) │ Min(ms) │ Max(ms) │ Ops/sec  │');
    lines.push('├───────────────────────────┼─────────┼─────────┼─────────┼──────────┤');

    for (const result of this.results) {
      const name = result.name.padEnd(25).slice(0, 25);
      const avg = result.avgDuration.toFixed(2).padStart(7);
      const min = result.minDuration.toFixed(2).padStart(7);
      const max = result.maxDuration.toFixed(2).padStart(7);
      const ops = result.opsPerSecond.toFixed(1).padStart(8);
      lines.push(`│ ${name} │ ${avg} │ ${min} │ ${max} │ ${ops} │`);
    }

    lines.push('└───────────────────────────┴─────────┴─────────┴─────────┴──────────┘');
    return lines.join('\n');
  }
}

// ============================================================================
// Pre-built Benchmark Suites
// ============================================================================

export async function runLanesBenchmarkSuite(): Promise<BenchmarkSuite> {
  const runner = new BenchmarkRunner({ verbose: true });
  const startedAt = Date.now();

  // Generate test data
  const smallDataset = generateLanesState(100);
  const mediumDataset = generateLanesState(500);
  const largeDataset = generateLanesState(1000);
  const hugeDataset = generateLanesState(5000);

  // State creation benchmarks
  await runner.run('Generate 100 lanes', () => {
    generateLanes({ count: 100 });
  });

  await runner.run('Generate 1000 lanes', () => {
    generateLanes({ count: 1000 });
  });

  await runner.run('Generate 5000 lanes', () => {
    generateLanes({ count: 5000 });
  });

  // Filtering benchmarks
  await runner.run('Filter 100 lanes by status', () => {
    smallDataset.lanes.filter(l => l.status === 'red');
  });

  await runner.run('Filter 1000 lanes by status', () => {
    largeDataset.lanes.filter(l => l.status === 'red');
  });

  await runner.run('Filter 5000 lanes by status', () => {
    hugeDataset.lanes.filter(l => l.status === 'red');
  });

  // Search benchmarks
  await runner.run('Search 100 lanes', () => {
    const query = 'feature';
    smallDataset.lanes.filter(l =>
      l.name.toLowerCase().includes(query) ||
      l.owner.toLowerCase().includes(query)
    );
  });

  await runner.run('Search 1000 lanes', () => {
    const query = 'feature';
    largeDataset.lanes.filter(l =>
      l.name.toLowerCase().includes(query) ||
      l.owner.toLowerCase().includes(query)
    );
  });

  await runner.run('Search 5000 lanes', () => {
    const query = 'feature';
    hugeDataset.lanes.filter(l =>
      l.name.toLowerCase().includes(query) ||
      l.owner.toLowerCase().includes(query)
    );
  });

  // Sorting benchmarks
  await runner.run('Sort 100 lanes by WIP', () => {
    [...smallDataset.lanes].sort((a, b) => b.wip_pct - a.wip_pct);
  });

  await runner.run('Sort 1000 lanes by WIP', () => {
    [...largeDataset.lanes].sort((a, b) => b.wip_pct - a.wip_pct);
  });

  await runner.run('Sort 5000 lanes by WIP', () => {
    [...hugeDataset.lanes].sort((a, b) => b.wip_pct - a.wip_pct);
  });

  // Aggregation benchmarks
  await runner.run('Aggregate 100 lanes stats', () => {
    const stats = {
      total: smallDataset.lanes.length,
      byStatus: { green: 0, yellow: 0, red: 0 },
      avgWip: 0,
      withBlockers: 0,
    };
    for (const lane of smallDataset.lanes) {
      stats.byStatus[lane.status]++;
      stats.avgWip += lane.wip_pct;
      if (lane.blockers.length > 0) stats.withBlockers++;
    }
    stats.avgWip /= stats.total;
  });

  await runner.run('Aggregate 5000 lanes stats', () => {
    const stats = {
      total: hugeDataset.lanes.length,
      byStatus: { green: 0, yellow: 0, red: 0 },
      avgWip: 0,
      withBlockers: 0,
    };
    for (const lane of hugeDataset.lanes) {
      stats.byStatus[lane.status]++;
      stats.avgWip += lane.wip_pct;
      if (lane.blockers.length > 0) stats.withBlockers++;
    }
    stats.avgWip /= stats.total;
  });

  // Batch update benchmarks
  await runner.run('Batch update 100 lanes', () => {
    smallDataset.lanes.map(lane => ({
      ...lane,
      wip_pct: Math.min(100, lane.wip_pct + 5),
    }));
  });

  await runner.run('Batch update 1000 lanes', () => {
    largeDataset.lanes.map(lane => ({
      ...lane,
      wip_pct: Math.min(100, lane.wip_pct + 5),
    }));
  });

  // Deep clone benchmarks
  await runner.run('Deep clone 100 lanes', () => {
    JSON.parse(JSON.stringify(smallDataset.lanes));
  });

  await runner.run('Deep clone 1000 lanes', () => {
    JSON.parse(JSON.stringify(largeDataset.lanes));
  });

  const completedAt = Date.now();

  return {
    name: 'Lanes Performance Suite',
    results: runner.getResults(),
    totalDuration: completedAt - startedAt,
    startedAt,
    completedAt,
  };
}

// ============================================================================
// Render Performance Utilities
// ============================================================================

export interface RenderMetrics {
  fps: number;
  frameTime: number;
  frames: number;
  duration: number;
  jank: number; // Frames that took > 16.67ms
}

export function measureRenderPerformance(durationMs: number = 1000): Promise<RenderMetrics> {
  return new Promise((resolve) => {
    const frameTimes: number[] = [];
    let lastFrameTime = performance.now();
    let startTime = lastFrameTime;
    let animationId: number;

    const measureFrame = () => {
      const now = performance.now();
      const frameTime = now - lastFrameTime;
      frameTimes.push(frameTime);
      lastFrameTime = now;

      if (now - startTime < durationMs) {
        animationId = requestAnimationFrame(measureFrame);
      } else {
        // Calculate metrics
        const totalFrames = frameTimes.length;
        const totalTime = now - startTime;
        const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / totalFrames;
        const jankFrames = frameTimes.filter(t => t > 16.67).length;

        resolve({
          fps: (totalFrames / totalTime) * 1000,
          frameTime: avgFrameTime,
          frames: totalFrames,
          duration: totalTime,
          jank: jankFrames,
        });
      }
    };

    animationId = requestAnimationFrame(measureFrame);
  });
}

// ============================================================================
// Memory Profiling
// ============================================================================

export interface MemorySnapshot {
  usedHeap: number;
  totalHeap: number;
  heapLimit: number;
  timestamp: number;
}

export function getMemorySnapshot(): MemorySnapshot | null {
  if (typeof performance === 'undefined' || !(performance as any).memory) {
    return null;
  }

  const memory = (performance as any).memory;
  return {
    usedHeap: memory.usedJSHeapSize,
    totalHeap: memory.totalJSHeapSize,
    heapLimit: memory.jsHeapSizeLimit,
    timestamp: Date.now(),
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// ============================================================================
// Singleton
// ============================================================================

let globalRunner: BenchmarkRunner | null = null;

export function getGlobalBenchmarkRunner(config?: Partial<BenchmarkConfig>): BenchmarkRunner {
  if (!globalRunner) {
    globalRunner = new BenchmarkRunner(config);
  }
  return globalRunner;
}

export function resetGlobalBenchmarkRunner(): void {
  globalRunner = null;
}

export default BenchmarkRunner;
