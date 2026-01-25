/**
 * LoadingRegistry - Tracks real entry/exit for each loading stage
 * Each tick corresponds to actual code execution, not simulated timing
 */

import { LOADING_MANIFEST, MANIFEST_STATS, type LoadingStage } from "./LoadingManifest";

export interface StageStatus {
  stage: LoadingStage;
  status: "pending" | "loading" | "complete" | "error";
  startTime: number;
  endTime?: number;
  duration?: number;
  error?: string;
}

export interface LoadingProgress {
  completed: number;
  total: number;
  criticalCompleted: number;
  criticalTotal: number;
  percent: number;
  criticalPercent: number;
  current: string | null;
  currentDurationMs: number | null;
  bottlenecks: string[];
  isInteractive: boolean; // All critical stages complete
}

class LoadingRegistryClass {
  private stages: Map<string, StageStatus> = new Map();
  private listeners: Set<() => void> = new Set();
  private startupTime: number;

  constructor() {
    this.startupTime = typeof performance !== "undefined" ? performance.now() : Date.now();

    // Initialize all stages as pending
    LOADING_MANIFEST.forEach(stage => {
      this.stages.set(stage.id, {
        stage,
        status: "pending",
        startTime: 0
      });
    });
  }

  /**
   * Mark a stage as starting. Call when code begins executing.
   */
  entry(stageId: string): void {
    const status = this.stages.get(stageId);
    if (!status) {
      console.warn(`[LoadingRegistry] Unknown stage: ${stageId}`);
      return;
    }

    if (status.status !== "pending") {
      return; // Already started or complete
    }

    status.status = "loading";
    status.startTime = performance.now();

    console.log(
      `%c[LOAD] → ${status.stage.name}`,
      "color: #3498db; font-weight: bold;"
    );

    this.notify();
  }

  /**
   * Mark a stage as complete. Call when code finishes executing.
   */
  exit(stageId: string, error?: string): void {
    const status = this.stages.get(stageId);
    if (!status) {
      console.warn(`[LoadingRegistry] Unknown stage: ${stageId}`);
      return;
    }

    if (status.status === "complete" || status.status === "error") {
      return; // Already done
    }

    status.endTime = performance.now();
    status.duration = status.endTime - (status.startTime || this.startupTime);

    if (error) {
      status.status = "error";
      status.error = error;
      console.error(`%c[LOAD] ✗ ${status.stage.name} (ERROR: ${error})`, "color: #e74c3c;");
    } else {
      status.status = "complete";
      const color = status.duration > 5000 ? "#e74c3c" : status.duration > 2000 ? "#f39c12" : "#2ecc71";
      console.log(
        `%c[LOAD] ✓ ${status.stage.name} (${Math.round(status.duration)}ms)`,
        `color: ${color}; font-weight: bold;`
      );

      if (status.duration > 5000) {
        console.warn(`[BOTTLENECK] ${status.stage.name} took ${Math.round(status.duration / 1000)}s`);
      }
    }

    this.notify();

    // Check if all critical stages complete
    if (this.isInteractive()) {
      console.log(
        "%c[LOAD] ★ All critical stages complete - UI is interactive!",
        "color: #9b59b6; font-weight: bold; font-size: 14px;"
      );
      if (typeof window !== "undefined") {
        (window as any).__pluribusReady = true;
      }
    }
  }

  /**
   * Convenience: entry + exit in one call with async function
   */
  async track<T>(stageId: string, fn: () => Promise<T>): Promise<T> {
    this.entry(stageId);
    try {
      const result = await fn();
      this.exit(stageId);
      return result;
    } catch (err) {
      this.exit(stageId, String(err));
      throw err;
    }
  }

  /**
   * Get current loading progress
   */
  getProgress(): LoadingProgress {
    let completed = 0;
    let criticalCompleted = 0;
    let currentLoading: string | null = null;
    let currentDurationMs: number | null = null;
    const bottlenecks: string[] = [];
    let completedWeight = 0;
    let criticalCompletedWeight = 0;

    this.stages.forEach(status => {
      if (status.status === "complete") {
        completed++;
        completedWeight += status.stage.weight;
        if (status.stage.critical) {
          criticalCompleted++;
          criticalCompletedWeight += status.stage.weight;
        }
        if (status.duration && status.duration > 5000) {
          bottlenecks.push(`${status.stage.name}: ${Math.round(status.duration / 1000)}s`);
        }
      } else if (status.status === "loading" && !currentLoading) {
        currentLoading = status.stage.name;
        currentDurationMs = Math.round((performance.now() - (status.startTime || this.startupTime)));
      }
    });

    return {
      completed,
      total: MANIFEST_STATS.total,
      criticalCompleted,
      criticalTotal: MANIFEST_STATS.critical,
      percent: Math.round((completedWeight / MANIFEST_STATS.totalWeight) * 100),
      criticalPercent: Math.round((criticalCompletedWeight / MANIFEST_STATS.criticalWeight) * 100),
      current: currentLoading,
      currentDurationMs,
      bottlenecks,
      isInteractive: criticalCompleted >= MANIFEST_STATS.critical
    };
  }

  /**
   * Check if all critical stages are complete (UI is interactive)
   */
  isInteractive(): boolean {
    for (const status of this.stages.values()) {
      if (status.stage.critical && status.status !== "complete") {
        return false;
      }
    }
    return true;
  }

  /**
   * Subscribe to progress updates
   */
  subscribe(callback: () => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  private notify(): void {
    this.listeners.forEach(cb => {
      try { cb(); } catch {}
    });
  }

  /**
   * Get all stages for debugging
   */
  getStages(): Map<string, StageStatus> {
    return new Map(this.stages);
  }
}

// Proper singleton that works across Qwik chunks
let _instance: LoadingRegistryClass;

function getOrCreateRegistry(): LoadingRegistryClass {
  // On client, always use window global for shared state across chunks
  if (typeof window !== "undefined") {
    if (!(window as any).__loadingRegistry) {
      (window as any).__loadingRegistry = new LoadingRegistryClass();
    }
    return (window as any).__loadingRegistry;
  }
  // On server, use module-level instance
  if (!_instance) {
    _instance = new LoadingRegistryClass();
  }
  return _instance;
}

export const LoadingRegistry = getOrCreateRegistry();
