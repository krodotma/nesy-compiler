/**
 * Verbose Loading Tracker (v2 - Neon Stack Edition)
 * Maps 1:1 to actual code entry/exit points (39+ items).
 * Optimized for "Neon Stack" visualization with precise timing.
 */

export interface LoadingItem {
  id: string;
  type: "component" | "subsystem" | "flow";
  name: string;
  weight: number;
}

export const TRACKING_MANIFEST: LoadingItem[] = [
  // === CRITICAL SUBSYSTEMS ===
  { id: "sys:auth", type: "subsystem", name: "Authentication", weight: 10 },
  { id: "sys:websocket", type: "subsystem", name: "Bus Connection", weight: 15 },
  { id: "sys:telemetry", type: "subsystem", name: "Telemetry Core", weight: 5 },
  { id: "sys:omega", type: "subsystem", name: "Omega Channel", weight: 5 },
  { id: "sys:shadow", type: "subsystem", name: "Shadow Channel", weight: 5 },
  { id: "sys:service-worker", type: "subsystem", name: "Service Worker", weight: 3 },

  // === FLOW / HOOKS ===
  { id: "flow:menu", type: "flow", name: "Menu Handlers", weight: 5 },
  { id: "flow:theme", type: "flow", name: "Theme Restoration", weight: 3 },
  { id: "flow:sota", type: "flow", name: "SOTA Data Fetch", weight: 5 },
  { id: "flow:events", type: "flow", name: "Initial Event Sync", weight: 8 },
  { id: "flow:event-loop", type: "flow", name: "Event Loop Init", weight: 3 },

  // === COMPONENTS (Critical) ===
  { id: "comp:header", type: "component", name: "Global Header", weight: 8 },
  { id: "comp:nav", type: "component", name: "Bicameral Nav", weight: 8 },
  { id: "comp:bicameral-nav", type: "component", name: "Navigation", weight: 5 }, // Alias for safety

  // === COMPONENTS (Features) ===
  { id: "comp:terminal", type: "component", name: "Terminal Interface", weight: 8 },
  { id: "comp:git-view", type: "component", name: "Git Operations", weight: 5 },
  { id: "comp:semops-editor", type: "component", name: "SemOps Editor", weight: 5 },
  { id: "comp:bus-observatory", type: "component", name: "Bus Observatory", weight: 5 },
  { id: "comp:diagnostics", type: "component", name: "Diagnostics Panel", weight: 4 },
  { id: "comp:infer-cell", type: "component", name: "InferCell Grid", weight: 6 },
  { id: "comp:generative-canvas", type: "component", name: "Generative Canvas", weight: 4 },
  
  // === COMPONENTS (Widgets/Minor) ===
  { id: "comp:supermotd", type: "component", name: "Super MOTD", weight: 2 },
  { id: "comp:event-viz", type: "component", name: "Event Visualizer", weight: 3 },
  { id: "comp:agent-telemetry", type: "component", name: "Agent Telemetry", weight: 3 },
  { id: "comp:vnc-auth", type: "component", name: "VNC Auth", weight: 2 },
  { id: "comp:bus-pulse", type: "component", name: "Bus Pulse", weight: 2 },
  { id: "comp:dkin-monitor", type: "component", name: "DKIN Monitor", weight: 3 },
  { id: "comp:studio", type: "component", name: "Studio View", weight: 3 },
  { id: "comp:voice-speech", type: "component", name: "Voice Speech", weight: 2 },
  { id: "comp:auralux", type: "component", name: "Auralux Console", weight: 2 },
  { id: "comp:types-tree", type: "component", name: "Types Tree", weight: 3 },
  { id: "comp:memory-ingest", type: "component", name: "Memory Ingest", weight: 2 },
  { id: "comp:notifications", type: "component", name: "Notifications", weight: 2 },
  { id: "comp:code-segment", type: "component", name: "Code Segment", weight: 2 },
  { id: "comp:pb-lanes", type: "component", name: "PB Lanes", weight: 2 },
  { id: "comp:local-llm", type: "component", name: "Local LLM", weight: 3 },
  { id: "comp:marimo", type: "component", name: "Marimo Widget", weight: 2 },
  { id: "comp:code-viewer", type: "component", name: "Code Viewer", weight: 2 },
  { id: "comp:plurichat", type: "component", name: "PluriChat", weight: 2 },
  { id: "comp:lazy-webllm", type: "component", name: "WebLLM Engine", weight: 5 },
];

export interface TrackedItemState {
  status: "pending" | "active" | "done";
  start: number;
  end?: number;
  duration?: number;
  meta: LoadingItem;
}

export interface TrackerState {
  items: Map<string, TrackedItemState>;
  recentlyCompleted: TrackedItemState[]; // Queue for UI
  activeItems: string[];
  completedCount: number;
  totalCount: number;
  progressPercent: number;
}

class VerboseTracker {
  private state: TrackerState;
  private listeners: Set<(s: TrackerState) => void>;

  constructor() {
    this.listeners = new Set();
    this.state = {
      items: new Map(),
      recentlyCompleted: [],
      activeItems: [],
      completedCount: 0,
      totalCount: TRACKING_MANIFEST.length,
      progressPercent: 0,
    };
    
    // Initialize state
    TRACKING_MANIFEST.forEach(m => {
      this.state.items.set(m.id, { 
        status: "pending", 
        start: 0, 
        meta: m 
      });
    });
  }

  public entry(id: string) {
    const item = this.state.items.get(id);
    
    // Auto-register unknown items (for safety)
    if (!item) {
      if (id.startsWith("comp:") || id.startsWith("sys:") || id.startsWith("flow:")) {
        const type = id.split(":")[0] as any;
        const name = id.split(":")[1].replace(/-/g, " ").toUpperCase();
        const newItem: LoadingItem = { id, type, name, weight: 1 };
        
        TRACKING_MANIFEST.push(newItem);
        this.state.items.set(id, { status: "active", start: performance.now(), meta: newItem });
        this.state.totalCount++;
        this.state.activeItems.push(name);
        this.notify();
      }
      return;
    }

    if (item.status === "pending") {
      item.status = "active";
      item.start = performance.now();
      if (!this.state.activeItems.includes(item.meta.name)) {
        this.state.activeItems.push(item.meta.name);
      }
      this.notify();
    }
  }

  public exit(id: string) {
    const item = this.state.items.get(id);
    if (!item) return;

    if (item.status !== "done") {
      item.status = "done";
      item.end = performance.now();
      item.duration = item.end - item.start;
      
      // Remove from active
      this.state.activeItems = this.state.activeItems.filter(n => n !== item.meta.name);
      
      // Add to recent queue
      this.state.recentlyCompleted.push(item);
      
      this.recalc();
      this.notify();
    }
  }

  private recalc() {
    let completedWeight = 0;
    let totalWeight = 0;
    let completedCount = 0;

    // Use manifest for weights, but verify against items map state
    TRACKING_MANIFEST.forEach(m => {
      totalWeight += m.weight;
      if (this.state.items.get(m.id)?.status === "done") {
        completedWeight += m.weight;
        completedCount++;
      }
    });

    this.state.completedCount = completedCount;
    // Calculate percent based on weights for accuracy
    this.state.progressPercent = totalWeight > 0 
      ? Math.min(100, Math.round((completedWeight / totalWeight) * 100))
      : 0;
  }

  public subscribe(cb: (s: TrackerState) => void) {
    this.listeners.add(cb);
    cb(this.state); // Immediate update
    return () => this.listeners.delete(cb);
  }

  private notify() {
    this.listeners.forEach(cb => cb(this.state));
  }
}

// Window-first singleton for cross-chunk compatibility
let _trackerInstance: VerboseTracker;

function getOrCreateTracker(): VerboseTracker {
  if (typeof window !== "undefined") {
    if (!(window as any).__verboseTracker) {
      (window as any).__verboseTracker = new VerboseTracker();
    }
    return (window as any).__verboseTracker;
  }
  if (!_trackerInstance) {
    _trackerInstance = new VerboseTracker();
  }
  return _trackerInstance;
}

export const tracker = getOrCreateTracker();
