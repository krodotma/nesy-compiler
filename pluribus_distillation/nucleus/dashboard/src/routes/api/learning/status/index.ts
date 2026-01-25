
import type { RequestHandler } from "@builder.io/qwik-city";

export const onGet: RequestHandler = async ({ json }) => {
    // Simulator for Learning Systems Status
    // In a real implementation, this would query the various agents/databases

    const systems = [
        { id: "ls-tower", status: "ok", label: "LearningTower", detail: "L8", meta: "neurosymbolic", uncertainty: 0.05 },
        { id: "ls-meta", status: "ok", label: "MetaLearner", detail: "Bayesian", meta: "⚡ active-inf", uncertainty: 0.12 },
        { id: "ls-opt", status: "ok", label: "OptimizationL", detail: "telemetry", meta: "levers", uncertainty: 0.02 },
        { id: "ls-portal", status: "ok", label: "Portal", detail: "AM/SM", meta: "metabolic", uncertainty: 0.01 },
        { id: "ls-ingest", status: "ok", label: "MetaIngest", detail: "bus→repo", meta: "streaming", uncertainty: 0.00 },
        { id: "ls-graphiti", status: "ok", label: "Graphiti", detail: "KG", meta: "causal", uncertainty: 0.08 },
        { id: "ls-mem0", status: "ok", label: "Mem0", detail: "episodic", meta: "agents", uncertainty: 0.15 }
    ];

    // Randomize slightly for "liveness"
    const randomized = systems.map(s => ({
        ...s,
        uncertainty: Math.max(0, Math.min(1, s.uncertainty + (Math.random() * 0.05 - 0.025)))
    }));

    json(200, {
        timestamp: new Date().toISOString(),
        systems: randomized
    });
};
