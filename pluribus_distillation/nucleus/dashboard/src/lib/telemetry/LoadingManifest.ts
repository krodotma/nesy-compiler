/**
 * LoadingManifest - Compile-time definition of all loading stages
 * Each stage maps to a real code entry/exit point
 * 
 * Total stages: 50
 * - 6 hooks
 * - 39 components
 * - 1 network
 * - 2 channels
 * - 2 system
 */

export interface LoadingStage {
  id: string;
  category: "hook" | "component" | "network" | "channel" | "system";
  name: string;
  weight: number; // Relative importance for progress calculation
  critical: boolean; // Must complete before UI is interactive
}

export const LOADING_MANIFEST: LoadingStage[] = [
  // === CRITICAL HOOKS (must complete for interactivity) ===
  { id: "hook:auth", category: "hook", name: "Auth Bootstrap", weight: 10, critical: true },
  { id: "hook:omega-channel", category: "channel", name: "Omega Channel", weight: 5, critical: true },
  { id: "hook:shadow-channel", category: "channel", name: "Shadow Channel", weight: 5, critical: true },
  { id: "hook:websocket", category: "network", name: "WebSocket Bus", weight: 15, critical: true },
  { id: "hook:menu-handlers", category: "hook", name: "Menu Handlers", weight: 10, critical: true },
  { id: "hook:theme", category: "hook", name: "Theme Init", weight: 3, critical: true },
  
  // === OTHER HOOKS ===
  { id: "hook:sota-data", category: "hook", name: "SOTA Data", weight: 5, critical: false },
  { id: "hook:events-fetch", category: "hook", name: "Events Fetch", weight: 5, critical: false },
  { id: "hook:telemetry", category: "system", name: "Telemetry", weight: 3, critical: false },
  { id: "hook:service-worker", category: "system", name: "Service Worker", weight: 2, critical: false },
  { id: "hook:event-loop", category: "hook", name: "Event Loop", weight: 3, critical: false },
  
  // === CRITICAL COMPONENTS ===
  { id: "comp:bicameral-nav", category: "component", name: "Navigation", weight: 8, critical: true },
  { id: "comp:header", category: "component", name: "Header", weight: 5, critical: true },
  
  // === HEAVY COMPONENTS ===
  { id: "comp:terminal", category: "component", name: "Terminal", weight: 5, critical: false },
  { id: "comp:git-view", category: "component", name: "Git View", weight: 5, critical: false },
  { id: "comp:semops-editor", category: "component", name: "Semops Editor", weight: 5, critical: false },
  { id: "comp:bus-observatory", category: "component", name: "Bus Observatory", weight: 5, critical: false },
  { id: "comp:diagnostics", category: "component", name: "Diagnostics", weight: 4, critical: false },
  { id: "comp:infer-cell", category: "component", name: "InferCell Grid", weight: 4, critical: false },
  { id: "comp:generative-canvas", category: "component", name: "Generative Canvas", weight: 4, critical: false },
  
  // === OTHER COMPONENTS (lighter weight) ===
  { id: "comp:supermotd", category: "component", name: "MOTD", weight: 2, critical: false },
  { id: "comp:event-viz", category: "component", name: "Event Viz", weight: 3, critical: false },
  { id: "comp:agent-telemetry", category: "component", name: "Agent Telemetry", weight: 3, critical: false },
  { id: "comp:agent-avatar", category: "component", name: "Agent Avatar", weight: 1, critical: false },
  { id: "comp:learning-tower", category: "component", name: "Learning Tower", weight: 2, critical: false },
  { id: "comp:vnc-auth", category: "component", name: "VNC Auth", weight: 2, critical: false },
  { id: "comp:bus-pulse", category: "component", name: "Bus Pulse", weight: 2, critical: false },
  { id: "comp:dkin-monitor", category: "component", name: "DKIN Monitor", weight: 3, critical: false },
  { id: "comp:metatest", category: "component", name: "MetaTest Dashboard", weight: 2, critical: false },
  { id: "comp:studio", category: "component", name: "Studio View", weight: 3, critical: false },
  { id: "comp:voice-speech", category: "component", name: "Voice Speech", weight: 2, critical: false },
  { id: "comp:auralux", category: "component", name: "Auralux Console", weight: 2, critical: false },
  { id: "comp:types-tree", category: "component", name: "Types Tree", weight: 3, critical: false },
  { id: "comp:leads", category: "component", name: "Leads", weight: 2, critical: false },
  { id: "comp:skills", category: "component", name: "Skills", weight: 2, critical: false },
  { id: "comp:memory-ingest", category: "component", name: "Memory Ingest", weight: 2, critical: false },
  { id: "comp:notifications", category: "component", name: "Notifications", weight: 2, critical: false },
  { id: "comp:notification-sidepanel", category: "component", name: "Notification Sidepanel", weight: 1, critical: false },
  { id: "comp:code-segment", category: "component", name: "Code Segment", weight: 2, critical: false },
  { id: "comp:pb-lanes", category: "component", name: "PB Lanes", weight: 2, critical: false },
  { id: "comp:edge-inference", category: "component", name: "Edge Inference", weight: 2, critical: false },
  { id: "comp:edge-catalog", category: "component", name: "Edge Catalog", weight: 2, critical: false },
  { id: "comp:local-llm", category: "component", name: "Local LLM", weight: 2, critical: false },
  { id: "comp:marimo", category: "component", name: "Marimo Widget", weight: 2, critical: false },
  { id: "comp:code-viewer", category: "component", name: "Code Viewer", weight: 2, critical: false },
  { id: "comp:portal-ingest", category: "component", name: "Portal Ingest", weight: 2, critical: false },
  { id: "comp:portal-inception", category: "component", name: "Portal Inception", weight: 2, critical: false },
  { id: "comp:portal-metrics", category: "component", name: "Portal Metrics", weight: 2, critical: false },
  { id: "comp:plurichat", category: "component", name: "PluriChat", weight: 2, critical: false },
  { id: "comp:lazy-webllm", category: "component", name: "LazyWebLLM", weight: 1, critical: false },
];

// Precomputed totals
export const MANIFEST_STATS = {
  total: LOADING_MANIFEST.length,
  critical: LOADING_MANIFEST.filter(s => s.critical).length,
  totalWeight: LOADING_MANIFEST.reduce((sum, s) => sum + s.weight, 0),
  criticalWeight: LOADING_MANIFEST.filter(s => s.critical).reduce((sum, s) => sum + s.weight, 0),
};

// Expose to window immediately
if (typeof window !== "undefined") {
  (window as any).__loadingManifest = { stages: LOADING_MANIFEST, stats: MANIFEST_STATS };
}
