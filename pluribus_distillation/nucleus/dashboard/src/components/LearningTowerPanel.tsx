/**
 * LearningTowerPanel - 8-Level Neurosymbolic Learning Architecture Visualization
 *
 * Displays the Learning Tower with real-time status for each level:
 *   L8: Ω (Reflexive Meta-Geometry)     ← Meta-learning, self-play, Gödel machines
 *   L7: DNA (Dual Neurosymbolic)        ← CRL, Q-learning, PPO, automata
 *   L6: Birkhoff Polytope               ← Sinkhorn, optimal transport, crystallization
 *   L5: mHC (Modern Hopfield)           ← Associative memory, contrastive, energy-based
 *   L4: Fiber Bundle Geometry           ← Parallel transport, gauge equivariant NNs
 *   L3: S^n/H^n (Spherical-Hyperbolic)  ← Manifold learning, geodesic, SLERP
 *   L2: Vec2Vec (Learned Embeddings)    ← SFT, LoRA, RLHF, DPO, adapters
 *   L1: Raw Input                       ← BPE, BM25, TF-IDF, AUOM tokenization
 *
 * @version 1.1.0
 * @update Connected to real orchestrator data via BroadcastChannel
 */

import { component$, useSignal, useVisibleTask$, useComputed$ } from "@builder.io/qwik";

type TowerStatus = "working" | "partial" | "planned" | "missing";

interface TowerLevel {
  level: number;
  name: string;
  shortName: string;
  description: string;
  paradigms: string[];
  status: TowerStatus;
  color: string;
  techStack: string[];
}

/** Real paradigm status from learning_orchestrator.py PARADIGM_STATUS */
const PARADIGM_STATUS: Record<string, TowerStatus> = {
  // L1: Tokenization
  bpe: "working",
  bm25: "working",
  tfidf: "working",
  auom: "working",
  // L2: Fine-tuning
  sft: "working",
  lora: "working",
  rlhf: "working",
  dpo: "working",
  kto: "partial",
  adapter: "working",
  prefix_tuning: "planned",
  // L3: Manifold
  spherical: "working",
  hyperbolic: "planned",
  contrastive: "partial",
  // L4: Geometry
  parallel_transport: "working",
  gauge_equivariant: "working",
  // L5: Memory
  hopfield: "working",
  energy_based: "working",
  associative: "working",
  // L6: Crystallization
  sinkhorn: "working",
  optimal_transport: "working",
  soft_sort: "working",
  gumbel_softmax: "working",
  // L7: RL
  q_learning: "partial",
  dqn: "working",
  ppo: "working",
  grpo: "working",
  sac: "working",
  curriculum_rl: "partial",
  // L8: Meta
  maml: "working",
  reptile: "working",
  self_play: "working",
  meta_learning: "working",
};

/** Calculate level status from paradigm statuses */
function calculateLevelStatus(paradigms: string[]): TowerStatus {
  const statuses = paradigms.map((p) => PARADIGM_STATUS[p.toLowerCase().replace(/[- ]/g, "_")] || "missing");
  const working = statuses.filter((s) => s === "working").length;
  const partial = statuses.filter((s) => s === "partial").length;

  if (working === statuses.length) return "working";
  if (working + partial > 0) return "partial";
  if (statuses.some((s) => s === "planned")) return "planned";
  return "missing";
}

/** Base level definitions without status - status is calculated dynamically */
const TOWER_LEVEL_DEFS: Omit<TowerLevel, "status">[] = [
  {
    level: 8,
    name: "Omega (Reflexive Meta-Geometry)",
    shortName: "Omega",
    description: "Self-modeling, Godel machines, coalgebraic self-modification",
    paradigms: ["MAML", "Reptile", "Self_Play", "Meta_Learning"],
    color: "hsl(280, 100%, 70%)", // Purple
    techStack: ["omega_metalearning.py", "MAML", "SelfPlay", "OmegaLoop"],
  },
  {
    level: 7,
    name: "DNA (Dual Neurosymbolic)",
    shortName: "DNA",
    description: "Buchi automata, Omega-regular languages, state machines",
    paradigms: ["Curriculum_RL", "Q_Learning", "PPO", "GRPO", "SAC"],
    color: "hsl(200, 100%, 70%)", // Cyan
    techStack: ["VERL", "ICRL", "Buchi acceptance"],
  },
  {
    level: 6,
    name: "Birkhoff Polytope",
    shortName: "Birkhoff",
    description: "Doubly-stochastic matrices, Sinkhorn normalization",
    paradigms: ["Sinkhorn", "Optimal_Transport", "Soft_Sort", "Gumbel_Softmax"],
    color: "hsl(45, 100%, 70%)", // Yellow
    techStack: ["sinkhorn_layer.py", "Sinkhorn", "OptTransport", "SoftSort"],
  },
  {
    level: 5,
    name: "mHC (Modern Hopfield)",
    shortName: "Hopfield",
    description: "Dense associative memory, attention = Hopfield update",
    paradigms: ["Hopfield", "Associative", "Energy_Based"],
    color: "hsl(160, 100%, 60%)", // Teal
    techStack: ["hopfield_memory.py", "DenseHopfield", "HopfieldAttention"],
  },
  {
    level: 4,
    name: "Fiber Bundle Geometry",
    shortName: "Fiber",
    description: "Parallel transport, connection nabla, curvature R",
    paradigms: ["Parallel_Transport", "Gauge_Equivariant"],
    color: "hsl(30, 100%, 70%)", // Orange
    techStack: ["fiber_bundle_geometry.py", "ParallelTransport", "Holonomy"],
  },
  {
    level: 3,
    name: "S^n/H^n (Spherical-Hyperbolic)",
    shortName: "Manifold",
    description: "Mobius transforms, geodesic distance, SLERP",
    paradigms: ["Spherical", "Hyperbolic", "Contrastive"],
    color: "hsl(120, 80%, 60%)", // Green
    techStack: ["N-sphere", "Poincare", "Sextet"],
  },
  {
    level: 2,
    name: "Vec2Vec (Learned Embeddings)",
    shortName: "Vec2Vec",
    description: "phi_embed: X -> R^d (pure neural maps)",
    paradigms: ["SFT", "LoRA", "RLHF", "DPO", "KTO"],
    color: "hsl(180, 80%, 60%)", // Cyan-Green
    techStack: ["VERL", "PEFT", "Adapters"],
  },
  {
    level: 1,
    name: "Raw Input",
    shortName: "Input",
    description: "Raw signals -> discrete tokens",
    paradigms: ["BPE", "BM25", "TF-IDF", "AUOM"],
    color: "hsl(220, 80%, 70%)", // Blue
    techStack: ["tiktoken", "rank_bm25", "AUOM"],
  },
];

/** Build tower levels with calculated status */
function buildTowerLevels(statusOverrides?: Record<number, TowerStatus>): TowerLevel[] {
  return TOWER_LEVEL_DEFS.map((def) => ({
    ...def,
    status: statusOverrides?.[def.level] ?? calculateLevelStatus(def.paradigms),
  }));
}

/** Default tower levels with calculated status */
const DEFAULT_TOWER_LEVELS: TowerLevel[] = buildTowerLevels();

const STATUS_ICONS: Record<string, string> = {
  working: "✅",
  partial: "⚠️",
  planned: "📋",
  missing: "❌",
};

const STATUS_COLORS: Record<string, string> = {
  working: "hsl(var(--color-success) / 0.8)",
  partial: "hsl(var(--color-warning) / 0.8)",
  planned: "hsl(var(--color-info) / 0.8)",
  missing: "hsl(var(--color-error) / 0.8)",
};

export const LearningTowerPanel = component$(() => {
  const expandedLevel = useSignal<number | null>(null);
  const animationPhase = useSignal(0);
  const towerLevels = useSignal<TowerLevel[]>(DEFAULT_TOWER_LEVELS);
  const loading = useSignal(true);
  const lastUpdate = useSignal<string | null>(null);
  const connectionStatus = useSignal<"connected" | "disconnected" | "error">("disconnected");

  // Computed summary stats from actual tower data
  const summaryStats = useComputed$(() => {
    const levels = towerLevels.value;
    return {
      working: levels.filter((l) => l.status === "working").length,
      partial: levels.filter((l) => l.status === "partial").length,
      planned: levels.filter((l) => l.status === "planned").length,
      missing: levels.filter((l) => l.status === "missing").length,
      total: levels.length,
    };
  });

  // Connect to real-time bus via BroadcastChannel (shares omega worker connection)
  useVisibleTask$(({ cleanup }) => {
    if (typeof window === "undefined") return;

    // Use BroadcastChannel to receive events from the shared omega worker
    const channel = new BroadcastChannel("pluribus-omega");

    channel.onmessage = (msg) => {
      if (msg.data.type === "BUS_EVENT" && msg.data.event) {
        const evt = msg.data.event;

        // Handle learning orchestrator events
        if (evt.topic === "learning.orchestrator.executed") {
          const data = evt.data;
          if (data && typeof data.level === "string") {
            // Extract level number from level name (e.g., "L1_RAW_INPUT" -> 1)
            const levelMatch = data.level.match(/L(\d)/);
            if (levelMatch) {
              const levelNum = parseInt(levelMatch[1], 10);
              const newStatus = data.success ? "working" : "partial";

              // Update the specific level status
              towerLevels.value = towerLevels.value.map((l) =>
                l.level === levelNum ? { ...l, status: newStatus as TowerStatus } : l
              );
              lastUpdate.value = new Date().toISOString();
            }
          }
        }

        // Handle level-specific events (learning.l1.executed, etc.)
        const levelEventMatch = evt.topic?.match(/^learning\.l(\d)\.executed$/);
        if (levelEventMatch) {
          const levelNum = parseInt(levelEventMatch[1], 10);
          lastUpdate.value = new Date().toISOString();
          // Could update paradigm-specific status here if needed
        }

        // Handle tower status sync event
        if (evt.topic === "learning.tower.status") {
          const statusData = evt.data?.levels;
          if (statusData && typeof statusData === "object") {
            const overrides: Record<number, TowerStatus> = {};
            for (const [key, val] of Object.entries(statusData)) {
              const levelMatch = key.match(/L(\d)/);
              if (levelMatch && val && typeof val === "object" && "health" in val) {
                overrides[parseInt(levelMatch[1], 10)] = (val as { health: TowerStatus }).health;
              }
            }
            towerLevels.value = buildTowerLevels(overrides);
            lastUpdate.value = new Date().toISOString();
          }
        }
      } else if (msg.data.type === "OMEGA_TICK") {
        connectionStatus.value = msg.data.state?.connected ? "connected" : "disconnected";
      }
    };

    // Mark as loaded after initial setup
    loading.value = false;
    connectionStatus.value = "connected";

    cleanup(() => {
      channel.close();
    });
  });

  // Animate the tower levels
  useVisibleTask$(({ cleanup }) => {
    const interval = setInterval(() => {
      animationPhase.value = (animationPhase.value + 1) % 100;
    }, 50);
    cleanup(() => clearInterval(interval));
  });

  return (
    <div class="glass-surface-elevated p-4 glass-hover-lift">
      <md-elevation></md-elevation>
      <h3 class="glass-section-header -mx-4 -mt-4 mb-3 flex items-center justify-between">
        <span>LEARNING TOWER v1.1</span>
        <span class="flex items-center gap-2">
          {loading.value && (
            <span class="text-[10px] text-white/40 animate-pulse">loading...</span>
          )}
          <span
            class={`w-2 h-2 rounded-full ${
              connectionStatus.value === "connected"
                ? "bg-green-400"
                : connectionStatus.value === "error"
                ? "bg-red-400"
                : "bg-yellow-400 animate-pulse"
            }`}
            title={`Bus: ${connectionStatus.value}`}
          />
        </span>
      </h3>

      {/* Tower visualization */}
      <div class="space-y-1">
        {towerLevels.value.map((level) => {
          const isExpanded = expandedLevel.value === level.level;
          const workingCount = level.paradigms.length;
          const barWidth = level.status === "working" ? 100 :
                           level.status === "partial" ? 60 :
                           level.status === "planned" ? 30 : 10;

          return (
            <div
              key={level.level}
              class="cursor-pointer transition-all duration-200"
              onClick$={() => {
                expandedLevel.value = isExpanded ? null : level.level;
              }}
            >
              {/* Level row */}
              <div
                class="flex items-center gap-2 p-2 rounded-md hover:bg-white/5 transition-colors"
                style={{
                  borderLeft: `3px solid ${level.color}`,
                }}
              >
                {/* Level number */}
                <div
                  class="w-8 h-8 flex items-center justify-center rounded-full text-xs font-bold"
                  style={{
                    background: `${level.color}20`,
                    color: level.color,
                  }}
                >
                  L{level.level}
                </div>

                {/* Level info */}
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-2">
                    <span class="text-xs font-medium text-white/90 truncate">
                      {level.shortName}
                    </span>
                    <span class="text-[10px] text-white/50">
                      {STATUS_ICONS[level.status]}
                    </span>
                  </div>

                  {/* Progress bar */}
                  <div class="h-1 bg-white/10 rounded-full mt-1 overflow-hidden">
                    <div
                      class="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${barWidth}%`,
                        background: `linear-gradient(90deg, ${level.color}, ${level.color}80)`,
                      }}
                    />
                  </div>
                </div>

                {/* Paradigm count */}
                <div class="text-[10px] text-white/40">
                  {workingCount} paradigms
                </div>

                {/* Expand indicator */}
                <div
                  class="text-white/30 text-xs transition-transform duration-200"
                  style={{
                    transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
                  }}
                >
                  ▼
                </div>
              </div>

              {/* Expanded details */}
              {isExpanded && (
                <div
                  class="ml-10 mt-1 p-3 rounded-md text-xs space-y-2"
                  style={{
                    background: `${level.color}10`,
                    borderLeft: `2px solid ${level.color}40`,
                  }}
                >
                  {/* Description */}
                  <div class="text-white/70">{level.description}</div>

                  {/* Paradigms */}
                  <div class="flex flex-wrap gap-1">
                    {level.paradigms.map((paradigm) => (
                      <span
                        key={paradigm}
                        class="px-2 py-0.5 rounded-full text-[10px]"
                        style={{
                          background: `${level.color}30`,
                          color: level.color,
                        }}
                      >
                        {paradigm}
                      </span>
                    ))}
                  </div>

                  {/* Tech stack */}
                  <div class="text-[10px] text-white/50">
                    <span class="text-white/30">Tech:</span>{" "}
                    {level.techStack.join(" • ")}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary stats - computed from actual tower data */}
      <div class="mt-4 pt-3 border-t border-cyan-500/10">
        <div class="grid grid-cols-4 gap-2 text-center">
          <div>
            <div class="text-lg font-bold text-green-400">{summaryStats.value.working}</div>
            <div class="text-[10px] text-white/40">Working</div>
          </div>
          <div>
            <div class="text-lg font-bold text-yellow-400">{summaryStats.value.partial}</div>
            <div class="text-[10px] text-white/40">Partial</div>
          </div>
          <div>
            <div class="text-lg font-bold text-red-400">{summaryStats.value.missing + summaryStats.value.planned}</div>
            <div class="text-[10px] text-white/40">Pending</div>
          </div>
          <div>
            <div class="text-lg font-bold text-white/80">{summaryStats.value.total}</div>
            <div class="text-[10px] text-white/40">Total</div>
          </div>
        </div>
      </div>

      {/* Learning flow indicator */}
      <div class="mt-3 text-[10px] text-white/30 text-center">
        {"L1 \u2192 L2 \u2192 L3 \u2192 ... \u2192 L8 (Upward: Discretization) \u2194 (Downward: Constraint)"}
        {lastUpdate.value && (
          <span class="ml-2 text-white/20">
            | updated {new Date(lastUpdate.value).toLocaleTimeString()}
          </span>
        )}
      </div>
    </div>
  );
});

export default LearningTowerPanel;
