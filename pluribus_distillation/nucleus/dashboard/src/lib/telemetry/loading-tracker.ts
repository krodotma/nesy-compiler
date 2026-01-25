/**
 * LoadingTracker - Tracks actual component/system/feature loading
 * Provides real progress data for LoadingOverlay
 */

export interface LoadingItem {
  id: string;
  category: "component" | "system" | "websocket" | "data" | "feature";
  name: string;
  startTime: number;
  endTime?: number;
  status: "pending" | "loading" | "complete" | "error";
}

class LoadingTrackerClass {
  private items: Map<string, LoadingItem> = new Map();
  private listeners: Set<() => void> = new Set();
  
  // Define expected items and their weights for progress calculation
  private expectedItems: { id: string; category: LoadingItem["category"]; name: string; weight: number }[] = [
    { id: "qwik-hydration", category: "system", name: "Qwik Hydration", weight: 20 },
    { id: "header", category: "component", name: "Header Navigation", weight: 10 },
    { id: "websocket", category: "websocket", name: "Bus Connection", weight: 15 },
    { id: "menu-handlers", category: "feature", name: "Menu Handlers", weight: 10 },
    { id: "broadcast-channel", category: "system", name: "Broadcast Channel", weight: 5 },
    { id: "shadow-channel", category: "system", name: "Shadow Channel", weight: 5 },
    { id: "initial-data", category: "data", name: "Initial Data Fetch", weight: 15 },
    { id: "event-log", category: "component", name: "Event Log", weight: 10 },
    { id: "telemetry", category: "system", name: "Telemetry Init", weight: 10 },
  ];
  
  constructor() {
    // Initialize all expected items as pending
    this.expectedItems.forEach(item => {
      this.items.set(item.id, {
        id: item.id,
        category: item.category,
        name: item.name,
        startTime: 0,
        status: "pending"
      });
    });
  }

  start(id: string): void {
    const item = this.items.get(id);
    if (item) {
      item.status = "loading";
      item.startTime = performance.now();
      this.notify();
      console.log(`%c[LOAD] Starting: ${item.name}`, "color: #3498db;");
    }
  }

  complete(id: string): void {
    const item = this.items.get(id);
    if (item) {
      item.status = "complete";
      item.endTime = performance.now();
      const duration = item.endTime - item.startTime;
      this.notify();
      
      const color = duration > 5000 ? "#e74c3c" : duration > 2000 ? "#f39c12" : "#2ecc71";
      console.log(
        `%c[LOAD] Complete: ${item.name} (${Math.round(duration)}ms)`,
        `color: ${color}; font-weight: bold;`
      );
      
      if (duration > 5000) {
        console.warn(`[BOTTLENECK] ${item.name} took ${Math.round(duration / 1000)}s - needs optimization`);
      }
    }
  }

  error(id: string, err?: string): void {
    const item = this.items.get(id);
    if (item) {
      item.status = "error";
      item.endTime = performance.now();
      this.notify();
      console.error(`[LOAD] Error: ${item.name}`, err);
    }
  }

  getProgress(): number {
    let totalWeight = 0;
    let completedWeight = 0;
    
    this.expectedItems.forEach(expected => {
      totalWeight += expected.weight;
      const item = this.items.get(expected.id);
      if (item?.status === "complete") {
        completedWeight += expected.weight;
      } else if (item?.status === "loading") {
        completedWeight += expected.weight * 0.5; // Half credit for loading
      }
    });
    
    return totalWeight > 0 ? (completedWeight / totalWeight) * 100 : 0;
  }

  getCurrentLoading(): LoadingItem | null {
    for (const item of this.items.values()) {
      if (item.status === "loading") return item;
    }
    return null;
  }

  getStatus(): { complete: number; total: number; current: string; bottlenecks: string[] } {
    let complete = 0;
    const bottlenecks: string[] = [];
    
    this.items.forEach(item => {
      if (item.status === "complete") {
        complete++;
        if (item.endTime && item.startTime && (item.endTime - item.startTime) > 5000) {
          bottlenecks.push(`${item.name}: ${Math.round((item.endTime - item.startTime) / 1000)}s`);
        }
      }
    });
    
    const current = this.getCurrentLoading();
    
    return {
      complete,
      total: this.items.size,
      current: current?.name || "Waiting...",
      bottlenecks
    };
  }

  isComplete(): boolean {
    // Check if critical items are complete
    const critical = ["qwik-hydration", "header", "websocket", "menu-handlers"];
    return critical.every(id => this.items.get(id)?.status === "complete");
  }

  subscribe(cb: () => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  private notify(): void {
    this.listeners.forEach(cb => cb());
  }
}

// Global singleton
export const LoadingTracker = new LoadingTrackerClass();

// Expose to window for component access
if (typeof window !== "undefined") {
  (window as any).__loadingTracker = LoadingTracker;
}
