/**
 * Lanes Load Testing
 *
 * Phase 9, Iteration 78 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Load testing utilities
 * - Concurrent user simulation
 * - Memory leak detection
 * - Stress testing
 * - Performance degradation monitoring
 */

import { generateLanes, generateLanesState, type BenchmarkResult } from './benchmark';
import type { Lane, LanesState } from './store';

// ============================================================================
// Types
// ============================================================================

export interface LoadTestConfig {
  /** Number of simulated concurrent users */
  concurrentUsers: number;
  /** Test duration in milliseconds */
  duration: number;
  /** Operations per second per user */
  opsPerSecond: number;
  /** Ramp-up time in milliseconds */
  rampUpTime: number;
  /** Types of operations to simulate */
  operations: LoadTestOperation[];
  /** Enable memory monitoring */
  monitorMemory: boolean;
  /** Memory sample interval in ms */
  memorySampleInterval: number;
  /** Callback for progress updates */
  onProgress?: (progress: LoadTestProgress) => void;
}

export type LoadTestOperation =
  | 'read'
  | 'filter'
  | 'search'
  | 'sort'
  | 'update'
  | 'create'
  | 'delete'
  | 'select';

export interface LoadTestProgress {
  elapsedMs: number;
  completedOperations: number;
  failedOperations: number;
  activeUsers: number;
  currentOpsPerSecond: number;
  memoryUsed?: number;
  percentComplete: number;
}

export interface LoadTestResult {
  config: LoadTestConfig;
  duration: number;
  totalOperations: number;
  successfulOperations: number;
  failedOperations: number;
  averageLatency: number;
  p50Latency: number;
  p95Latency: number;
  p99Latency: number;
  maxLatency: number;
  minLatency: number;
  operationsPerSecond: number;
  memorySnapshots: MemorySnapshot[];
  memoryLeakDetected: boolean;
  memoryLeakRate?: number; // bytes per second
  errors: LoadTestError[];
  degradationPoints: DegradationPoint[];
}

export interface MemorySnapshot {
  timestamp: number;
  usedHeap: number;
  totalHeap: number;
}

export interface LoadTestError {
  timestamp: number;
  operation: LoadTestOperation;
  message: string;
}

export interface DegradationPoint {
  timestamp: number;
  latencyBefore: number;
  latencyAfter: number;
  cause: string;
}

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: LoadTestConfig = {
  concurrentUsers: 10,
  duration: 30000, // 30 seconds
  opsPerSecond: 10,
  rampUpTime: 5000, // 5 seconds
  operations: ['read', 'filter', 'search', 'sort', 'update', 'select'],
  monitorMemory: true,
  memorySampleInterval: 1000,
};

// ============================================================================
// Load Test Runner
// ============================================================================

export class LoadTestRunner {
  private config: LoadTestConfig;
  private state: LanesState;
  private latencies: number[] = [];
  private errors: LoadTestError[] = [];
  private memorySnapshots: MemorySnapshot[] = [];
  private degradationPoints: DegradationPoint[] = [];
  private isRunning = false;
  private startTime = 0;
  private completedOps = 0;
  private failedOps = 0;

  constructor(config: Partial<LoadTestConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.state = generateLanesState(100);
  }

  /**
   * Run load test
   */
  async run(): Promise<LoadTestResult> {
    this.reset();
    this.isRunning = true;
    this.startTime = Date.now();

    // Start memory monitoring
    let memoryInterval: NodeJS.Timeout | undefined;
    if (this.config.monitorMemory) {
      memoryInterval = setInterval(() => {
        this.captureMemorySnapshot();
      }, this.config.memorySampleInterval);
    }

    // Start progress reporting
    const progressInterval = setInterval(() => {
      this.reportProgress();
    }, 500);

    try {
      // Ramp up users
      const usersPerStep = Math.ceil(this.config.concurrentUsers / 5);
      const stepDelay = this.config.rampUpTime / 5;

      const userPromises: Promise<void>[] = [];

      for (let i = 0; i < this.config.concurrentUsers; i++) {
        const delay = Math.floor(i / usersPerStep) * stepDelay;
        userPromises.push(this.simulateUser(i, delay));
      }

      // Wait for all users to complete
      await Promise.all(userPromises);

    } finally {
      this.isRunning = false;
      if (memoryInterval) clearInterval(memoryInterval);
      clearInterval(progressInterval);
    }

    return this.generateResult();
  }

  /**
   * Stop the load test
   */
  stop(): void {
    this.isRunning = false;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private reset(): void {
    this.latencies = [];
    this.errors = [];
    this.memorySnapshots = [];
    this.degradationPoints = [];
    this.completedOps = 0;
    this.failedOps = 0;
    this.state = generateLanesState(100);
  }

  private async simulateUser(userId: number, startDelay: number): Promise<void> {
    // Wait for ramp-up delay
    await this.sleep(startDelay);

    const endTime = this.startTime + this.config.duration;
    const opInterval = 1000 / this.config.opsPerSecond;

    while (this.isRunning && Date.now() < endTime) {
      const operation = this.randomOperation();
      const opStart = performance.now();

      try {
        await this.executeOperation(operation);
        const latency = performance.now() - opStart;
        this.latencies.push(latency);
        this.completedOps++;

        // Check for degradation
        this.checkDegradation(latency);

      } catch (err: any) {
        this.failedOps++;
        this.errors.push({
          timestamp: Date.now(),
          operation,
          message: err?.message || 'Unknown error',
        });
      }

      // Wait for next operation
      const elapsed = performance.now() - opStart;
      const waitTime = Math.max(0, opInterval - elapsed);
      await this.sleep(waitTime);
    }
  }

  private randomOperation(): LoadTestOperation {
    const ops = this.config.operations;
    return ops[Math.floor(Math.random() * ops.length)];
  }

  private async executeOperation(op: LoadTestOperation): Promise<void> {
    switch (op) {
      case 'read':
        this.readLanes();
        break;
      case 'filter':
        this.filterLanes();
        break;
      case 'search':
        this.searchLanes();
        break;
      case 'sort':
        this.sortLanes();
        break;
      case 'update':
        this.updateLane();
        break;
      case 'create':
        this.createLane();
        break;
      case 'delete':
        this.deleteLane();
        break;
      case 'select':
        this.selectLane();
        break;
    }
  }

  private readLanes(): Lane[] {
    return [...this.state.lanes];
  }

  private filterLanes(): Lane[] {
    const statuses: Lane['status'][] = ['green', 'yellow', 'red'];
    const status = statuses[Math.floor(Math.random() * statuses.length)];
    return this.state.lanes.filter(l => l.status === status);
  }

  private searchLanes(): Lane[] {
    const terms = ['feature', 'bug', 'test', 'deploy', 'review'];
    const term = terms[Math.floor(Math.random() * terms.length)];
    return this.state.lanes.filter(l =>
      l.name.toLowerCase().includes(term) ||
      l.owner.toLowerCase().includes(term)
    );
  }

  private sortLanes(): Lane[] {
    const fields: Array<keyof Lane> = ['name', 'status', 'wip_pct', 'updated'];
    const field = fields[Math.floor(Math.random() * fields.length)];
    return [...this.state.lanes].sort((a, b) => {
      const aVal = a[field];
      const bVal = b[field];
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return aVal.localeCompare(bVal);
      }
      return (aVal as number) - (bVal as number);
    });
  }

  private updateLane(): void {
    if (this.state.lanes.length === 0) return;
    const index = Math.floor(Math.random() * this.state.lanes.length);
    const lane = this.state.lanes[index];
    this.state.lanes[index] = {
      ...lane,
      wip_pct: Math.min(100, lane.wip_pct + Math.floor(Math.random() * 10)),
      updated: new Date().toISOString(),
    };
  }

  private createLane(): void {
    const newLane = generateLanes({ count: 1 })[0];
    this.state.lanes.push(newLane);
  }

  private deleteLane(): void {
    if (this.state.lanes.length > 10) {
      const index = Math.floor(Math.random() * this.state.lanes.length);
      this.state.lanes.splice(index, 1);
    }
  }

  private selectLane(): void {
    if (this.state.lanes.length === 0) return;
    const index = Math.floor(Math.random() * this.state.lanes.length);
    this.state.selectedLaneId = this.state.lanes[index].id;
  }

  private captureMemorySnapshot(): void {
    if (typeof performance !== 'undefined' && (performance as any).memory) {
      const memory = (performance as any).memory;
      this.memorySnapshots.push({
        timestamp: Date.now(),
        usedHeap: memory.usedJSHeapSize,
        totalHeap: memory.totalJSHeapSize,
      });
    }
  }

  private checkDegradation(latency: number): void {
    // Check if we have enough samples
    if (this.latencies.length < 100) return;

    // Get average of last 50 latencies
    const recentAvg = this.latencies.slice(-50).reduce((a, b) => a + b, 0) / 50;
    const previousAvg = this.latencies.slice(-100, -50).reduce((a, b) => a + b, 0) / 50;

    // Check for significant degradation (>50% increase)
    if (recentAvg > previousAvg * 1.5 && !this.isDuplicateDegradation()) {
      this.degradationPoints.push({
        timestamp: Date.now(),
        latencyBefore: previousAvg,
        latencyAfter: recentAvg,
        cause: this.analyzeDegradationCause(),
      });
    }
  }

  private isDuplicateDegradation(): boolean {
    if (this.degradationPoints.length === 0) return false;
    const lastPoint = this.degradationPoints[this.degradationPoints.length - 1];
    return Date.now() - lastPoint.timestamp < 5000; // Within 5 seconds
  }

  private analyzeDegradationCause(): string {
    // Check memory
    if (this.memorySnapshots.length >= 2) {
      const recent = this.memorySnapshots[this.memorySnapshots.length - 1];
      const previous = this.memorySnapshots[this.memorySnapshots.length - 2];
      const memoryGrowth = (recent.usedHeap - previous.usedHeap) / previous.usedHeap;

      if (memoryGrowth > 0.1) {
        return 'Memory pressure';
      }
    }

    // Check lane count
    if (this.state.lanes.length > 500) {
      return 'Large dataset size';
    }

    return 'Unknown cause';
  }

  private reportProgress(): void {
    if (!this.config.onProgress) return;

    const elapsed = Date.now() - this.startTime;
    const currentOps = this.latencies.slice(-10).length;
    const lastSecondLatencies = this.latencies.filter((_, i) =>
      i >= this.latencies.length - currentOps
    );

    this.config.onProgress({
      elapsedMs: elapsed,
      completedOperations: this.completedOps,
      failedOperations: this.failedOps,
      activeUsers: this.config.concurrentUsers,
      currentOpsPerSecond: currentOps,
      memoryUsed: this.memorySnapshots[this.memorySnapshots.length - 1]?.usedHeap,
      percentComplete: Math.min(100, (elapsed / this.config.duration) * 100),
    });
  }

  private generateResult(): LoadTestResult {
    const sortedLatencies = [...this.latencies].sort((a, b) => a - b);
    const duration = Date.now() - this.startTime;

    // Calculate percentiles
    const percentile = (p: number) => {
      const index = Math.floor(sortedLatencies.length * p);
      return sortedLatencies[index] || 0;
    };

    // Detect memory leak
    const { leakDetected, leakRate } = this.detectMemoryLeak();

    return {
      config: this.config,
      duration,
      totalOperations: this.completedOps + this.failedOps,
      successfulOperations: this.completedOps,
      failedOperations: this.failedOps,
      averageLatency: sortedLatencies.reduce((a, b) => a + b, 0) / sortedLatencies.length || 0,
      p50Latency: percentile(0.5),
      p95Latency: percentile(0.95),
      p99Latency: percentile(0.99),
      maxLatency: Math.max(...sortedLatencies, 0),
      minLatency: Math.min(...sortedLatencies, 0),
      operationsPerSecond: (this.completedOps / duration) * 1000,
      memorySnapshots: this.memorySnapshots,
      memoryLeakDetected: leakDetected,
      memoryLeakRate: leakRate,
      errors: this.errors,
      degradationPoints: this.degradationPoints,
    };
  }

  private detectMemoryLeak(): { leakDetected: boolean; leakRate?: number } {
    if (this.memorySnapshots.length < 10) {
      return { leakDetected: false };
    }

    // Linear regression on memory usage
    const n = this.memorySnapshots.length;
    const startTime = this.memorySnapshots[0].timestamp;

    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for (const snapshot of this.memorySnapshots) {
      const x = (snapshot.timestamp - startTime) / 1000; // seconds
      const y = snapshot.usedHeap;
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

    // If memory is growing more than 1MB per second, likely a leak
    const leakThreshold = 1024 * 1024;
    const leakDetected = slope > leakThreshold;

    return {
      leakDetected,
      leakRate: leakDetected ? slope : undefined,
    };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Report Generator
// ============================================================================

export function generateLoadTestReport(result: LoadTestResult): string {
  const lines: string[] = [];

  lines.push('╔════════════════════════════════════════════════════════════════════╗');
  lines.push('║                      LOAD TEST REPORT                              ║');
  lines.push('╚════════════════════════════════════════════════════════════════════╝');
  lines.push('');

  // Configuration
  lines.push('Configuration:');
  lines.push(`  Concurrent Users: ${result.config.concurrentUsers}`);
  lines.push(`  Duration: ${result.config.duration}ms`);
  lines.push(`  Target Ops/Sec: ${result.config.opsPerSecond * result.config.concurrentUsers}`);
  lines.push('');

  // Summary
  lines.push('Summary:');
  lines.push(`  Total Operations: ${result.totalOperations}`);
  lines.push(`  Successful: ${result.successfulOperations}`);
  lines.push(`  Failed: ${result.failedOperations}`);
  lines.push(`  Success Rate: ${((result.successfulOperations / result.totalOperations) * 100).toFixed(2)}%`);
  lines.push(`  Actual Ops/Sec: ${result.operationsPerSecond.toFixed(2)}`);
  lines.push('');

  // Latency
  lines.push('Latency (ms):');
  lines.push(`  Average: ${result.averageLatency.toFixed(2)}`);
  lines.push(`  P50: ${result.p50Latency.toFixed(2)}`);
  lines.push(`  P95: ${result.p95Latency.toFixed(2)}`);
  lines.push(`  P99: ${result.p99Latency.toFixed(2)}`);
  lines.push(`  Min: ${result.minLatency.toFixed(2)}`);
  lines.push(`  Max: ${result.maxLatency.toFixed(2)}`);
  lines.push('');

  // Memory
  if (result.memorySnapshots.length > 0) {
    const startMem = result.memorySnapshots[0].usedHeap;
    const endMem = result.memorySnapshots[result.memorySnapshots.length - 1].usedHeap;
    const memoryGrowth = endMem - startMem;

    lines.push('Memory:');
    lines.push(`  Start: ${formatBytes(startMem)}`);
    lines.push(`  End: ${formatBytes(endMem)}`);
    lines.push(`  Growth: ${formatBytes(memoryGrowth)}`);
    lines.push(`  Leak Detected: ${result.memoryLeakDetected ? 'YES ⚠' : 'No'}`);
    if (result.memoryLeakRate) {
      lines.push(`  Leak Rate: ${formatBytes(result.memoryLeakRate)}/sec`);
    }
    lines.push('');
  }

  // Degradation
  if (result.degradationPoints.length > 0) {
    lines.push('Performance Degradation Points:');
    for (const point of result.degradationPoints) {
      lines.push(`  ${new Date(point.timestamp).toISOString()}: ${point.latencyBefore.toFixed(2)}ms → ${point.latencyAfter.toFixed(2)}ms (${point.cause})`);
    }
    lines.push('');
  }

  // Errors
  if (result.errors.length > 0) {
    lines.push(`Errors (${result.errors.length}):`);
    const errorCounts: Record<string, number> = {};
    for (const error of result.errors) {
      const key = `${error.operation}: ${error.message}`;
      errorCounts[key] = (errorCounts[key] || 0) + 1;
    }
    for (const [msg, count] of Object.entries(errorCounts)) {
      lines.push(`  ${count}x ${msg}`);
    }
    lines.push('');
  }

  lines.push('═══════════════════════════════════════════════════════════════════════');

  return lines.join('\n');
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// ============================================================================
// Stress Test Scenarios
// ============================================================================

export async function runStressTestSuite(): Promise<Map<string, LoadTestResult>> {
  const results = new Map<string, LoadTestResult>();

  // Scenario 1: Low load baseline
  const lowLoadRunner = new LoadTestRunner({
    concurrentUsers: 5,
    duration: 10000,
    opsPerSecond: 5,
  });
  results.set('low-load', await lowLoadRunner.run());

  // Scenario 2: Medium load
  const mediumLoadRunner = new LoadTestRunner({
    concurrentUsers: 20,
    duration: 15000,
    opsPerSecond: 10,
  });
  results.set('medium-load', await mediumLoadRunner.run());

  // Scenario 3: High load
  const highLoadRunner = new LoadTestRunner({
    concurrentUsers: 50,
    duration: 20000,
    opsPerSecond: 20,
  });
  results.set('high-load', await highLoadRunner.run());

  // Scenario 4: Write-heavy
  const writeHeavyRunner = new LoadTestRunner({
    concurrentUsers: 10,
    duration: 15000,
    opsPerSecond: 10,
    operations: ['create', 'update', 'delete'],
  });
  results.set('write-heavy', await writeHeavyRunner.run());

  // Scenario 5: Read-heavy
  const readHeavyRunner = new LoadTestRunner({
    concurrentUsers: 30,
    duration: 15000,
    opsPerSecond: 50,
    operations: ['read', 'filter', 'search', 'sort'],
  });
  results.set('read-heavy', await readHeavyRunner.run());

  return results;
}

export default LoadTestRunner;
