#!/usr/bin/env python3
"""Service Registry (Genotype Library) for Pluribus TUI.

Manages the "DNA" of the system—services, agents, and pipelines—treating them
as evolving genotypes within the CMP-LARGE framework.

Concepts:
- ServiceDef: The "Genotype" (static code, plan, DNA).
- ServiceInstance: The "Phenotype" (runtime behavior, process).
- Clade: A lineage of services evolving together.
- Gates: P/E/L/R/Q/Ω/ω checks for acceptance.

Registry is stored in .pluribus/services/registry.ndjson
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@dataclass
class ServiceDef:
    """Service definition (Genotype).
    
    In CMP-LARGE terms, this is the 'Genotype'—the static code/plan.
    """
    id: str
    name: str
    kind: str  # "port" | "composition" | "process"
    entry_point: str  # Python script or command
    description: str = ""
    port: int | None = None
    depends_on: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    auto_start: bool = False
    restart_policy: str = "never"  # "never" | "on_failure" | "always"
    health_check: str | None = None  # URL or command for health check
    created_iso: str = ""
    provenance: dict = field(default_factory=dict)
    
    # DNA / CMP-LARGE Extensions
    lineage: str = "orphan"      # The evolutionary branch/clade (e.g., "core.tui", "mcp.host")
    gates: dict[str, str] = field(default_factory=dict)  # P/E/L/R/Q + Omega/omega requirements
    omega_motif: bool = False    # Is this a recurring, stable pattern (evolutionary attractor)?
    cmp_score: float = 0.0       # Clade-Metaproductivity score (descendant promise)


@dataclass
class ServiceInstance:
    """Running service instance (Phenotype).
    
    In CMP-LARGE terms, this is the 'Phenotype'—the runtime behavior.
    """
    service_id: str
    instance_id: str
    pid: int | None = None
    port: int | None = None
    status: str = "stopped"  # "stopped" | "starting" | "running" | "error"
    started_iso: str = ""
    last_health_iso: str = ""
    health: str = "unknown"  # "unknown" | "healthy" | "unhealthy"
    error: str | None = None


@dataclass
class Clade:
    """A lineage of related services (Genes)."""
    id: str
    description: str
    services: list[str] = field(default_factory=list)
    cmp_aggregate: float = 0.0


# Built-in service definitions (The Ancestral Stubs)
BUILTIN_SERVICES: list[dict] = [
    {
        "id": "strp-monitor",
        "name": "STRp Monitor",
        "kind": "process",
        "entry_point": "nucleus/tools/strp_monitor.py",
        "description": "Primary operator TUI dashboard",
        "tags": ["tui", "monitor", "core"],
        "lineage": "core.tui",
        "omega_motif": True,
        "gates": {"L": "liveness_check", "Q": "visual_stability"},
    },
    {
        "id": "strp-worker",
        "name": "STRp Worker",
        "kind": "process",
        "entry_point": "nucleus/tools/strp_worker.py",
        "description": "Background task processor for STRp pipeline",
        "args": ["--provider", "auto"],
        "tags": ["worker", "strp", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.strp",
        "omega_motif": True,
        "gates": {"E": "task_completion", "R": "idempotency"},
    },
    {
        "id": "strp-curation-loop",
        "name": "Curation Loop",
        "kind": "process",
        "entry_point": "nucleus/tools/strp_curation_loop.py",
        "description": "Automated SOTA curation and distillation",
        "tags": ["curation", "strp", "core"],
        "lineage": "core.strp",
        "gates": {"P": "provenance_tracking"},
    },
    {
        "id": "kroma-vor",
        "name": "VOR Self-Verification",
        "kind": "process",
        "entry_point": "nucleus/tools/kroma_vor.py",
        "description": "Autonomous self-verification (VOR cycle)",
        "args": ["pluribuscheck"],
        "tags": ["verification", "kroma", "vor"],
        "lineage": "core.kroma",
        "omega_motif": True,
        "gates": {"omega": "recurrence_check"},
    },
    {
        "id": "mesh-status",
        "name": "Mesh Status",
        "kind": "process",
        "entry_point": "nucleus/tools/mesh_status.py",
        "description": "Agent mesh status summary",
        "tags": ["status", "mesh"],
        "lineage": "core.observability",
    },
    {
        "id": "pluribus-secops",
        "name": "SecOps Guard",
        "kind": "process",
        "entry_point": "nucleus/secops/enforcement.py",
        "description": "Active constitutional enforcement daemon. Audits bus for ring breaches and provenance gaps.",
        "tags": ["secops", "enforcement", "v26", "core"],
        "restart_policy": "always",
        "lineage": "core.security",
        "omega_motif": True,
        "gates": {"E": "signature_verification", "L": "realtime_audit"},
    },
    {
        "id": "provider-health-invariant",
        "name": "Provider Health Invariant",
        "kind": "process",
        "entry_point": "nucleus/tools/provider_health_invariant.py",
        "description": "Continuous liveness mesh that executes 14+ provider smoke tests and emits health metrics.",
        "tags": ["health", "providers", "v26", "core"],
        "restart_policy": "always",
        "lineage": "core.observability",
        "omega_motif": True,
        "gates": {"L": "automated_liveness"},
    },
    {
        "id": "pluribus-docs-audit",
        "name": "PBDOCS Audit",
        "kind": "process",
        "entry_point": "nucleus/tools/pbdocs_operator.py",
        "description": "Documentation coverage, staleness, and build audits for Pluribus.",
        "args": ["--full"],
        "tags": ["docs", "audit", "pbdocs", "observability"],
        "restart_policy": "on_failure",
        "lineage": "core.docs",
        "gates": {"Q": "docs_quality"},
    },
    {
        "id": "pluribus-kg-watcher",
        "name": "KG Provenance Watcher",
        "kind": "process",
        "entry_point": "nucleus/tools/graphiti_bridge.py",
        "args": ["watch"],
        "description": "Always-on fact extractor that binds provenance IDs from the bus to the Knowledge Graph.",
        "tags": ["kg", "provenance", "v26", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.memory",
        "omega_motif": True,
        "gates": {"P": "verifiable_provenance"},
    },
    {
        "id": "pluribuscheck-responder",
        "name": "PLURIBUSCHECK Responder",
        "kind": "process",
        "entry_point": "nucleus/tools/pluribus_check_responder.py",
        "description": "Always-on responder: watches pluribus.check.trigger and emits pluribus.check.report (session context fingerprints).",
        "args": ["--poll", "0.25"],
        "tags": ["pluribuscheck", "observability", "a2a", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.observability",
        "omega_motif": True,
        "gates": {"P": "append_only_evidence", "L": "trigger_response"},
    },
    {
        "id": "pbflush-responder",
        "name": "PBFLUSH Responder",
        "kind": "process",
        "entry_point": "nucleus/tools/pbflush_responder.py",
        "description": "Always-on responder: watches operator.pbflush.request and emits operator.pbflush.ack + infer_sync.response (no-messenger coordination).",
        "args": ["--poll", "0.25"],
        "tags": ["pbflush", "coordination", "observability", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.observability",
        "omega_motif": True,
        "gates": {"P": "append_only_evidence", "L": "trigger_response"},
    },
    {
        "id": "bus-mirror-daemon",
        "name": "Bus Mirror Daemon",
        "kind": "process",
        "entry_point": "nucleus/tools/bus_mirror_daemon.py",
        "description": "Continuously promotes fallback bus events into the canonical bus to prevent evidence fragmentation under sandboxing.",
        "args": ["--emit-bus", "--emit-interval-s", "10"],
        "tags": ["bus", "observability", "evidence", "mirror", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.observability",
        "omega_motif": True,
        "gates": {"P": "append_only_evidence", "L": "canonical_bus_coherent"},
    },
    {
        "id": "bus-mirror-daemon-reverse",
        "name": "Bus Mirror Daemon (Reverse)",
        "kind": "process",
        "entry_point": "nucleus/tools/bus_mirror_daemon.py",
        "description": "Optionally mirrors the canonical bus into the workspace-local bus so sandboxed agents can tail a coherent shared stream.",
        "args": [
            "--from-bus-dir",
            "/pluribus/.pluribus/bus",
            "--to-bus-dir",
            "/pluribus/.pluribus_local/bus",
            "--state-path",
            "/pluribus/.pluribus_local/bus/bus_mirror_state_from_canonical.json",
            "--start-at-end",
            "--emit-bus",
            "--emit-interval-s",
            "10",
            "--actor",
            "bus-mirror-reverse",
        ],
        "tags": ["bus", "observability", "evidence", "mirror", "reverse", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.observability",
        "omega_motif": True,
        "gates": {"P": "append_only_evidence", "L": "sandbox_bus_visibility"},
    },
    {
        "id": "metrics-exporter",
        "name": "Metrics Exporter",
        "kind": "port",
        "entry_point": "nucleus/tools/metrics_exporter.py",
        "description": "Prometheus metrics endpoint",
        "port": 9090,
        "tags": ["metrics", "observability"],
        "health_check": "http://localhost:9090/health",
        "lineage": "core.observability",
    },
    {
        "id": "microsandbox",
        "name": "Microsandbox Controller",
        "kind": "port",
        "entry_point": "microsandbox-ctl serve --port 8300",
        "description": "Firecracker-based sandboxing for agent code execution",
        "port": 8300,
        "tags": ["sandbox", "isolation", "security"],
        "restart_policy": "on_failure",
        "health_check": "curl -sf http://localhost:8300/health || exit 1",
        "lineage": "core.security",
        "gates": {"E": "isolation_boundary", "L": "no_ring0_write"},
    },
    {
        "id": "vllm-service",
        "name": "vLLM Inference Server",
        "kind": "port",
        "entry_point": "nucleus/tools/vllm_server.py",
        "description": "Local LLM inference via vLLM (OpenAI-compatible)",
        "port": 8000,
        "args": ["--model", "meta-llama/Llama-3.2-3B-Instruct", "--port", "8000"],
        "tags": ["vllm", "inference", "local", "sota"],
        "health_check": "http://localhost:8000/health",
        "lineage": "core.inference",
        "gates": {"E": "api_responsive"},
    },
    {
        "id": "mcp-rhizome",
        "name": "MCP Rhizome Server",
        "kind": "port",
        "entry_point": "nucleus/mcp/rhizome_server.py",
        "description": "MCP server for rhizome artifact management",
        "port": 9100,
        "tags": ["mcp", "rhizome"],
        "lineage": "mcp.servers",
    },
    {
        "id": "mcp-sota",
        "name": "MCP SOTA Server",
        "kind": "port",
        "entry_point": "nucleus/mcp/sota_server.py",
        "description": "MCP server for SOTA stream tracking",
        "port": 9101,
        "tags": ["mcp", "sota"],
        "lineage": "mcp.servers",
    },
    {
        "id": "mcp-kg",
        "name": "MCP KG Server",
        "kind": "port",
        "entry_point": "nucleus/mcp/kg_server.py",
        "description": "MCP server for knowledge graph operations",
        "port": 9102,
        "tags": ["mcp", "kg"],
        "lineage": "mcp.servers",
    },
    {
        "id": "mcp-host",
        "name": "MCP Headless Host",
        "kind": "process",
        "entry_point": "nucleus/mcp/host.py",
        "description": "Bus-driven MCP host (spawns stdio servers; no sockets required)",
        "args": ["daemon"],
        "tags": ["mcp", "host", "daemon"],
        "restart_policy": "on_failure",
        "lineage": "mcp.host",
        "omega_motif": True,
    },
    # Compositions (pipelines)
    {
        "id": "composition-distill",
        "name": "Distill Pipeline",
        "kind": "composition",
        "entry_point": "nucleus/compositions/distill.py",
        "description": "Full distillation pipeline: curate → extract → index",
        "depends_on": ["mcp-rhizome", "mcp-kg"],
        "tags": ["composition", "strp", "distill"],
        "lineage": "pipeline.strp",
    },
    {
        "id": "composition-verify",
        "name": "Verify Pipeline",
        "kind": "composition",
        "entry_point": "nucleus/compositions/verify.py",
        "description": "Full verification pipeline: liveness → omega → VOR",
        "depends_on": ["kroma-vor"],
        "tags": ["composition", "verification"],
        "lineage": "pipeline.verification",
        "gates": {"omega": "recurrence"},
    },
    # R&D Workflow
    {
        "id": "rd-workflow",
        "name": "R&D Workflow",
        "kind": "process",
        "entry_point": "nucleus/tools/rd_workflow.py",
        "description": "R&D intake workflow: drop → ingest → distill → track",
        "args": ["status"],
        "tags": ["rd", "workflow", "intake"],
        "lineage": "workflow.rd",
    },
    {
        "id": "services-tui",
        "name": "Services TUI",
        "kind": "process",
        "entry_point": "nucleus/tools/services_tui.py",
        "description": "Interactive service management dashboard",
        "tags": ["tui", "services", "monitor"],
        "lineage": "core.tui",
    },
    {
        "id": "dialogosd",
        "name": "Dialogos Daemon",
        "kind": "process",
        "entry_point": "nucleus/tools/dialogosd.py",
        "description": "Consumes dialogos.submit and emits dialogos.cell.* (notebook streaming outputs)",
        "tags": ["dialogos", "a2a", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.messaging",
        "omega_motif": True,
    },
    {
        "id": "meta-ingest",
        "name": "Meta Repo Ingest",
        "kind": "process",
        "entry_point": "nucleus/tools/meta_ingest.py",
        "description": "Normalize bus/dialogos/task ledger into append-only meta ledger + index",
        "tags": ["meta", "recovery", "ledger"],
        "restart_policy": "on_failure",
        "lineage": "core.recovery",
        "omega_motif": True,
    },
    {
        "id": "mabswarm",
        "name": "MAB Swarm Operator",
        "kind": "process",
        "entry_point": "nucleus/tools/mabswarm.py",
        "description": "Intelligent Membrane (Swarm Reflexes for the Bus)",
        "args": ["--daemon", "--emit-bus"],
        "tags": ["swarm", "core", "membrane"],
        "lineage": "core.swarm",
        "gates": {"L": "bus_reflex"},
        "omega_motif": True,
    },
    {
        "id": "sota-ingest",
        "name": "SOTA Ingest Tool",
        "kind": "process",
        "entry_point": "nucleus/tools/sota_ingest.py",
        "description": "Deep multimodal ingestion (URL/File -> Rhizome/Vector/KG)",
        "tags": ["sota", "ingest", "tool"],
        "lineage": "core.sota",
        "gates": {"P": "provenance", "E": "format_handling"},
        "omega_motif": False,
    },
    {
        "id": "sota-pulse",
        "name": "SOTA Pulse Daemon",
        "kind": "process",
        "entry_point": "nucleus/tools/sota_pulse.py",
        "description": "Recurring research trigger (12h) for SOTA ingestion",
        "tags": ["sota", "research", "daemon", "cron"],
        "lineage": "core.sota",
        "gates": {"L": "periodic_pulse"},
        "omega_motif": True,
    },
    {
        "id": "research-feeds",
        "name": "Research Feeds Daemon",
        "kind": "process",
        "entry_point": "nucleus/tools/research_feeds.py",
        "description": "Aggregated daemon for arXiv and industry labs feeds (daily/weekly cadence)",
        "args": ["daemon"],
        "tags": ["sota", "research", "feeds", "arxiv", "labs", "daemon"],
        "restart_policy": "on_failure",
        "lineage": "core.sota",
        "gates": {"L": "periodic_fetch", "P": "dedup_storage"},
        "omega_motif": True,
    },
    {
        "id": "arxiv-feed",
        "name": "arXiv Feed Fetcher",
        "kind": "process",
        "entry_point": "nucleus/tools/arxiv_feed.py",
        "description": "arXiv RSS parser (cs.AI, cs.LG, cs.CL, stat.ML) with dedup and RAG indexing",
        "args": ["daemon"],
        "tags": ["sota", "arxiv", "rss", "research"],
        "lineage": "core.sota",
        "gates": {"L": "daily_fetch", "P": "content_hash_dedup"},
        "omega_motif": False,
    },
    {
        "id": "labs-feed",
        "name": "Labs Feed Fetcher",
        "kind": "process",
        "entry_point": "nucleus/tools/labs_feed.py",
        "description": "Industry labs blog parser (DeepMind, OpenAI, Anthropic, Meta, MSR, HuggingFace)",
        "args": ["daemon"],
        "tags": ["sota", "labs", "rss", "research"],
        "lineage": "core.sota",
        "gates": {"L": "weekly_fetch", "P": "link_dedup"},
        "omega_motif": False,
    },
    {
        "id": "art-director",
        "name": "Art Director",
        "kind": "process",
        "entry_point": "nucleus/art_dept/director.py",
        "description": "Autonomous aesthetic supervisor (Mood/Entropy -> Generative UI)",
        "args": ["--daemon"],
        "tags": ["art", "ui", "daemon", "autonomous"],
        "lineage": "core.art",
        "gates": {"L": "aesthetic_liveness"},
        "omega_motif": True,
    },
    {
        "id": "realagents-op",
        "name": "REALAGENTS Operator",
        "kind": "process",
        "entry_point": "nucleus/tools/realagents_operator.py",
        "description": "Formal dispatcher for deep implementation tasks (REALAGENTS protocol)",
        "tags": ["dispatch", "coordination", "realagents"],
        "lineage": "core.realagents",
        "omega_motif": False,
    },
    {
        "id": "pluribus-studio",
        "name": "Pluribus Studio",
        "kind": "process",
        "entry_point": "nucleus/tools/studio_server.py",
        "description": "Neurosymbolic Flow Editor Backend (FastAPI/XYFlow Bridge)",
        "port": 9202,
        "tags": ["studio", "ui", "experimental"],
        "lineage": "core.studio",
        "gates": {"E": "api_responsive"},
        "omega_motif": False,
    },
    {
        "id": "infercell-activator",
        "name": "InferCell Activator",
        "kind": "process",
        "entry_point": "nucleus/tools/infercell_activator.py",
        "description": "Active supervisor for InferCell workspaces (realizes forks/merges)",
        "tags": ["infercell", "core", "daemon"],
        "lineage": "core.infercell",
        "gates": {"E": "workspace_creation"},
        "omega_motif": True,
    },
    {
        "id": "wasmedge-runtime",
        "name": "WasmEdge Runtime",
        "kind": "process",
        "entry_point": "nucleus/tools/wasmedge_runtime.py",
        "description": "WASM-based sandboxed execution via WasmEdge (wasi_nn for neural inference)",
        "args": ["--capabilities"],
        "tags": ["sandbox", "wasm", "isolation", "security", "sota"],
        "lineage": "core.security",
        "gates": {"E": "wasm_execution", "L": "sandbox_boundary"},
        "omega_motif": False,
    },
    # SOTA Dev Tools Integration
    {
        "id": "marimo-notebook",
        "name": "Marimo Notebook Runtime",
        "kind": "port",
        "entry_point": "nucleus/dashboard/src/components/MarimoNotebook.tsx",
        "description": "Reactive Python notebook component with Pyodide/WASM execution",
        "port": 5200,
        "tags": ["marimo", "notebook", "pyodide", "wasm", "sota", "rd_workflow"],
        "lineage": "sota.development",
        "gates": {"E": "pyodide_ready", "L": "cell_execution"},
        "omega_motif": False,
    },
    {
        "id": "prisma-adapter",
        "name": "Prisma DB Adapter",
        "kind": "composition",
        "entry_point": "nucleus/dashboard/src/lib/db/prisma_adapter.ts",
        "description": "Type-safe database adapter for Pluribus entities (NDJSON-backed)",
        "tags": ["prisma", "db", "typescript", "sota", "type-safe"],
        "lineage": "sota.development",
        "gates": {"E": "query_execution"},
        "omega_motif": False,
    },
    {
        "id": "lit-bus-monitor",
        "name": "Lit Bus Monitor Widget",
        "kind": "composition",
        "entry_point": "nucleus/dashboard/src/components/LitBusMonitor.ts",
        "description": "Standards-based web component for bus event monitoring (~5kb)",
        "tags": ["lit", "webcomponent", "monitor", "bus", "sota", "embeddable"],
        "lineage": "sota.ui",
        "gates": {"E": "websocket_connection", "L": "event_render"},
        "omega_motif": False,
    },
    {
        "id": "container-executor",
        "name": "Container Executor",
        "kind": "process",
        "entry_point": "nucleus/tools/container_executor.py",
        "description": "Microsandbox-ready container execution (Firecracker/gVisor/Docker)",
        "args": ["--capabilities"],
        "tags": ["sandbox", "container", "isolation", "security", "microsandbox"],
        "lineage": "core.security",
        "gates": {"E": "container_spawn", "L": "isolation_boundary"},
        "omega_motif": True,
    },
    {
        "id": "a2a-bridge",
        "name": "A2A Bridge",
        "kind": "process",
        "entry_point": "nucleus/tools/a2a_bridge.py",
        "description": "Forwards selected bus request topics into dialogos.submit (req_id-correlated)",
        "tags": ["dialogos", "a2a", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.messaging",
        "omega_motif": True,
    },
    {
        "id": "a2a-negotiate-daemon",
        "name": "A2A Negotiation Daemon",
        "kind": "process",
        "entry_point": "nucleus/tools/a2a/negotiate_daemon.py",
        "description": "Bus-native A2A negotiation responder (capabilities/agree/decline), idempotent by req_id",
        "tags": ["a2a", "negotiation", "daemon", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.messaging",
        "omega_motif": True,
    },
    {
        "id": "mcp-host-official",
        "name": "MCP Host (Official SDK)",
        "kind": "process",
        "entry_point": "nucleus/mcp/host_official.py",
        "description": "Bus-driven MCP host using the official MCP Python SDK (initialize handshake + stdio).",
        "args": ["daemon"],
        "tags": ["mcp", "official", "daemon", "core"],
        "restart_policy": "on_failure",
        "lineage": "core.mcp",
        "omega_motif": True,
        "gates": {"P": "append_only_evidence", "L": "request_response"},
    },
    {
        "id": "a2a-official-server",
        "name": "A2A Server (Official SDK)",
        "kind": "port",
        "entry_point": "nucleus/tools/a2a_official_server.py",
        "description": "HTTP A2A server using official a2a-sdk; routes prompts via PluriChat.",
        "port": 8141,
        "args": ["--provider", "auto"],
        "tags": ["a2a", "official", "server", "plurichat"],
        "lineage": "core.messaging",
        "omega_motif": False,
        "gates": {"E": "roundtrip"},
    },
    {
        "id": "adk-a2a-server",
        "name": "ADK→A2A Server (Official)",
        "kind": "port",
        "entry_point": "nucleus/tools/adk_a2a_server.py",
        "description": "Official Google ADK agent exposed as an A2A server (LLM via PluriChat, tools via MCP).",
        "port": 8142,
        "args": ["--provider", "auto"],
        "tags": ["adk", "a2a", "official", "mcp", "plurichat"],
        "lineage": "core.realagents",
        "omega_motif": False,
        "gates": {"E": "roundtrip"},
    },
    {
        "id": "pluribus-tui",
        "name": "Master TUI",
        "kind": "process",
        "entry_point": "nucleus/tools/pluribus_tui.py",
        "description": "Isomorphic Master Control TUI (Dashboard, Dialogos, Rhizome)",
        "tags": ["tui", "master", "core"],
        "lineage": "core.tui",
        "omega_motif": True,
    },
    # VPS Session and Dashboard
    {
        "id": "vps-session",
        "name": "VPS Session Manager",
        "kind": "process",
        "entry_point": "nucleus/tools/vps_session.py",
        "description": "Provider status and fallback mode management",
        "args": ["status"],
        "tags": ["vps", "session", "providers"],
        "lineage": "core.vps",
    },
    {
        "id": "vps-session-daemon",
        "name": "VPS Session Daemon",
        "kind": "process",
        "entry_point": "nucleus/tools/vps_session.py",
        "description": "Bus-controlled provider refresh + flow mode control (control plane for TUI/web)",
        "args": ["daemon", "--interval-s", "30"],
        "tags": ["vps", "session", "providers", "daemon", "control-plane"],
        "restart_policy": "on_failure",
        "lineage": "core.vps",
        "omega_motif": True,
        "gates": {"L": "heartbeat"},
    },
    {
        "id": "dashboard-bridge",
        "name": "Dashboard Bridge",
        "kind": "port",
        "entry_point": "nucleus/dashboard/src/lib/bus/bus-bridge.ts",
        "description": "WebSocket bridge for web/native dashboard",
        "port": 9200,
        "tags": ["dashboard", "bridge", "websocket"],
        "lineage": "core.dashboard",
        "gates": {"E": "websocket_handshake"},
    },
    # SOTA MLOps & Orchestration
    {
        "id": "tensorzero-gateway",
        "name": "TensorZero Gateway",
        "kind": "process",
        "entry_point": "nucleus/tools/tensorzero_gateway.py",
        "description": "Unified observability/routing gateway with experiment tracking and feedback loops",
        "args": ["daemon", "--emit-bus"],
        "tags": ["mlops", "gateway", "observability", "sota"],
        "lineage": "core.mlops",
        "gates": {"E": "gateway_health", "L": "metrics_emission"},
        "omega_motif": True,
    },
    {
        "id": "mastra-workflows",
        "name": "Mastra Workflow Engine",
        "kind": "process",
        "entry_point": "nucleus/tools/mastra_workflows.ts",
        "description": "TypeScript agent orchestration with real-time bus bridge (distill, hypothesize, debate)",
        "args": ["daemon", "--bus-dir", "/pluribus/.pluribus/bus"],
        "tags": ["orchestration", "typescript", "workflow", "sota"],
        "lineage": "core.orchestration",
        "gates": {"E": "workflow_execution", "L": "step_events"},
        "omega_motif": True,
    },
    {
        "id": "agent-lightning-trainer",
        "name": "Agent Lightning Trainer",
        "kind": "process",
        "entry_point": "nucleus/tools/agent_lightning_trainer.py",
        "description": "RL training framework with bus-first evidence rewards and omega-liveness gates",
        "args": ["daemon", "--emit-bus"],
        "tags": ["rl", "training", "mlops", "sota"],
        "lineage": "core.training",
        "gates": {"E": "episode_completion", "L": "liveness_healthy", "omega": "training_convergence"},
        "omega_motif": True,
    },
    # Browser Tools (SOTA)
    {
        "id": "browserless",
        "name": "Browserless Chrome API",
        "kind": "port",
        "entry_point": "nucleus/tools/browserless_client.py",
        "description": "HTTP Chrome API for web scraping, screenshots, and PDF generation",
        "port": 3000,
        "tags": ["browser", "scraping", "sota", "chrome"],
        "health_check": "http://localhost:3000/health",
        "lineage": "core.browser",
        "gates": {"E": "chrome_ready"},
        "omega_motif": False,
    },
    {
        "id": "pyodide-runtime",
        "name": "Pyodide WASM Runtime",
        "kind": "process",
        "entry_point": "nucleus/dashboard/src/components/PyodideCell.tsx",
        "description": "In-browser Python execution via WASM (cPython)",
        "tags": ["pyodide", "wasm", "python", "browser", "sota"],
        "lineage": "core.browser",
        "gates": {"E": "wasm_init"},
        "omega_motif": False,
    },
    # SOTA Coding Agents (Phase 3 Integration)
    {
        "id": "opencode",
        "name": "OpenCode TUI Agent",
        "kind": "process",
        "entry_point": "nucleus/tools/opencode_wrapper.py",
        "description": "TUI coding agent for rapid iteration (peer to plurichat)",
        "args": ["run", "--goal", "default"],
        "tags": ["agent", "swe", "coding", "sota"],
        "lineage": "sota.agents",
        "gates": {"E": "npm_available", "L": "task_completion"},
        "omega_motif": False,
    },
    {
        "id": "pr-agent",
        "name": "PR-Agent Code Review",
        "kind": "process",
        "entry_point": "nucleus/tools/pr_agent_wrapper.py",
        "description": "AI code review for pull requests (CI/CD integration)",
        "args": ["check"],
        "tags": ["review", "ci", "pr", "sota"],
        "lineage": "sota.agents",
        "gates": {"E": "pr_parse", "L": "review_complete"},
        "omega_motif": False,
    },
    {
        "id": "agent-s",
        "name": "Agent S GUI Automator",
        "kind": "process",
        "entry_point": "nucleus/tools/agent_s_adapter.py",
        "description": "GUI automation agent (OSWorld-grade CUA) via Agent-S",
        "args": ["check"],
        "tags": ["agent", "gui", "cua", "sota"],
        "lineage": "sota.agents",
        "gates": {"E": "grounding_ready", "L": "task_completion"},
        "omega_motif": False,
    },
    {
        "id": "agent0",
        "name": "Agent0 Self-Evolving Loop",
        "kind": "process",
        "entry_point": "nucleus/tools/agent0_adapter.py",
        "description": "Curriculum/executor self-evolution loop (Agent0)",
        "args": ["plan", "--goal", "default"],
        "tags": ["agent", "evolution", "sota"],
        "lineage": "sota.agents",
        "gates": {"E": "plan_created", "L": "ledger_append"},
        "omega_motif": False,
    },
    {
        "id": "maestro",
        "name": "Maestro Mobile E2E",
        "kind": "process",
        "entry_point": "nucleus/tools/maestro_adapter.py",
        "description": "Mobile/UI E2E flows via Maestro CLI",
        "args": ["check"],
        "tags": ["testing", "mobile", "e2e", "sota"],
        "lineage": "sota.testing",
        "gates": {"E": "maestro_cli", "L": "flow_complete"},
        "omega_motif": False,
    },
    # Microsoft SOTA Integrations
    {
        "id": "autogen-bridge",
        "name": "AutoGen Bridge",
        "kind": "process",
        "entry_point": "nucleus/tools/autogen_bridge.py",
        "description": "Bridges Microsoft AutoGen multi-agent conversations to Pluribus event bus",
        "args": ["demo"],
        "tags": ["autogen", "microsoft", "orchestration", "sota", "multi-agent"],
        "lineage": "sota.microsoft",
        "omega_motif": False,
        "gates": {"E": "conversation_completion", "P": "message_provenance"},
    },
    {
        "id": "semantic-kernel-adapter",
        "name": "Semantic Kernel Adapter",
        "kind": "process",
        "entry_point": "nucleus/tools/semantic_kernel_adapter.py",
        "description": "Bridges Microsoft Semantic Kernel planners and tools to Pluribus",
        "args": ["demo"],
        "tags": ["semantic-kernel", "microsoft", "planner", "sota", "orchestration"],
        "lineage": "sota.microsoft",
        "omega_motif": False,
        "gates": {"E": "plan_execution", "P": "step_provenance"},
    },
    # SOTA Vision & 3D Tools
    {
        "id": "vggt-inference",
        "name": "VGGT 3D Inference",
        "kind": "port",
        "entry_point": "nucleus/tools/vggt_inference.py",
        "description": "Single-view 3D inference: depth + normals + mesh from single image",
        "port": 9301,
        "args": ["serve", "--port", "9301"],
        "tags": ["sota", "vision", "3d", "inference", "vggt"],
        "health_check": "http://localhost:9301/health",
        "lineage": "sota.vision",
        "omega_motif": False,
        "gates": {"E": "inference_roundtrip", "L": "gpu_or_mock"},
    },
    {
        "id": "gen3c-video",
        "name": "GEN3C Video Generation",
        "kind": "port",
        "entry_point": "nucleus/tools/gen3c_video.py",
        "description": "3D-consistent video generation from sparse views",
        "port": 9302,
        "args": ["serve", "--port", "9302"],
        "tags": ["sota", "vision", "3d", "video", "gen3c"],
        "health_check": "http://localhost:9302/health",
        "lineage": "sota.vision",
        "omega_motif": False,
        "gates": {"E": "generation_roundtrip", "L": "gpu_or_mock"},
    },
    {
        "id": "live2diff-stream",
        "name": "Live2Diff Stylization",
        "kind": "port",
        "entry_point": "nucleus/tools/live2diff_stream.py",
        "description": "Real-time video stylization (~16 FPS)",
        "port": 9303,
        "args": ["serve", "--port", "9303"],
        "tags": ["sota", "vision", "video", "stylization", "live2diff"],
        "health_check": "http://localhost:9303/health",
        "lineage": "sota.vision",
        "omega_motif": False,
        "gates": {"E": "frame_roundtrip", "L": "fps_target"},
    },
    # Documentation System
    {
        "id": "pbdocs",
        "name": "PBDOCS Documentation System",
        "kind": "process",
        "entry_point": "docs/scripts/build.py",
        "description": "MkDocs-based documentation system with bus integration for audit and coverage events",
        "tags": ["docs", "mkdocs", "documentation", "core"],
        "lineage": "core.docs",
        "omega_motif": True,
        "gates": {"E": "build_success", "L": "coverage_threshold"},
    },
    {
        "id": "pbdocs-indexer",
        "name": "PBDOCS Indexer",
        "kind": "process",
        "entry_point": "nucleus/tools/docs_indexer.py",
        "description": "Indexes documentation into RAG vector store and Graphiti KG for semantic retrieval",
        "args": ["reindex"],
        "tags": ["docs", "indexer", "rag", "kg"],
        "lineage": "core.docs",
        "omega_motif": False,
        "gates": {"E": "index_complete"},
    },
]


class ServiceRegistry:
    """Service registry manager."""

    def __init__(self, root: Path):
        self.root = root
        self.pluribus_dir = root / ".pluribus"
        self.services_dir = self.pluribus_dir / "services"
        self.registry_path = self.services_dir / "registry.ndjson"
        self.instances_path = self.services_dir / "instances.ndjson"
        self.pid_dir = self.services_dir / "pids"
        self._services: dict[str, ServiceDef] = {}
        self._instances: dict[str, ServiceInstance] = {}
        self._procs: dict[str, subprocess.Popen] = {}

    def init(self) -> None:
        """Initialize service registry directories."""
        ensure_dir(self.services_dir)
        ensure_dir(self.pid_dir)
        self.registry_path.touch(exist_ok=True)
        self.instances_path.touch(exist_ok=True)

    def load(self) -> None:
        """Load services and instances from disk."""
        self._services.clear()
        self._instances.clear()

        # Load registered services
        for obj in iter_ndjson(self.registry_path):
            if obj.get("kind") == "service_def":
                svc = ServiceDef(
                    id=obj.get("id", ""),
                    name=obj.get("name", ""),
                    kind=obj.get("service_kind", "process"),
                    entry_point=obj.get("entry_point", ""),
                    description=obj.get("description", ""),
                    port=obj.get("port"),
                    depends_on=obj.get("depends_on", []),
                    env=obj.get("env", {}),
                    args=obj.get("args", []),
                    tags=obj.get("tags", []),
                    auto_start=obj.get("auto_start", False),
                    restart_policy=obj.get("restart_policy", "never"),
                    health_check=obj.get("health_check"),
                    created_iso=obj.get("created_iso", ""),
                    provenance=obj.get("provenance", {}),
                    lineage=obj.get("lineage", "orphan"),
                    gates=obj.get("gates", {}),
                    omega_motif=obj.get("omega_motif", False),
                    cmp_score=obj.get("cmp_score", 0.0),
                )
                self._services[svc.id] = svc

        # Load builtin services if not already registered
        for builtin in BUILTIN_SERVICES:
            if builtin["id"] not in self._services:
                svc = ServiceDef(
                    id=builtin["id"],
                    name=builtin["name"],
                    kind=builtin["kind"],
                    entry_point=builtin["entry_point"],
                    description=builtin.get("description", ""),
                    port=builtin.get("port"),
                    depends_on=builtin.get("depends_on", []),
                    env=builtin.get("env", {}),
                    args=builtin.get("args", []),
                    tags=builtin.get("tags", []),
                    auto_start=builtin.get("auto_start", False),
                    restart_policy=builtin.get("restart_policy", "never"),
                    health_check=builtin.get("health_check"),
                    created_iso="builtin",
                    provenance={"source": "builtin"},
                    lineage=builtin.get("lineage", "orphan"),
                    gates=builtin.get("gates", {}),
                    omega_motif=builtin.get("omega_motif", False),
                    cmp_score=builtin.get("cmp_score", 0.0),
                )
                self._services[svc.id] = svc

        # Load running instances
        for obj in iter_ndjson(self.instances_path):
            if obj.get("kind") == "service_instance":
                inst = ServiceInstance(
                    service_id=obj.get("service_id", ""),
                    instance_id=obj.get("instance_id", ""),
                    pid=obj.get("pid"),
                    port=obj.get("port"),
                    status=obj.get("status", "stopped"),
                    started_iso=obj.get("started_iso", ""),
                    last_health_iso=obj.get("last_health_iso", ""),
                    health=obj.get("health", "unknown"),
                    error=obj.get("error"),
                )
                self._instances[inst.instance_id] = inst

    def register_service(self, svc: ServiceDef) -> str:
        """Register a new service definition."""
        if not svc.id:
            svc.id = str(uuid.uuid4())
        svc.created_iso = now_iso_utc()
        svc.provenance = {"added_by": default_actor()}

        self._services[svc.id] = svc

        # Persist to registry
        rec = {
            **asdict(svc),
            "kind": "service_def",
            "ts": time.time(),
            "iso": now_iso_utc(),
            "service_kind": svc.kind,  # Avoid conflict with ndjson "kind"
        }
        append_ndjson(self.registry_path, rec)

        return svc.id

    def list_services(self) -> list[ServiceDef]:
        """List all registered services."""
        return list(self._services.values())

    def get_service(self, service_id: str) -> ServiceDef | None:
        """Get service by ID."""
        return self._services.get(service_id)

    def list_instances(self) -> list[ServiceInstance]:
        """List all service instances."""
        return list(self._instances.values())

    def start_service(self, service_id: str, port_override: int | None = None) -> ServiceInstance | None:
        """Start a service instance."""
        svc = self._services.get(service_id)
        if not svc:
            return None

        # Check dependencies
        for dep_id in svc.depends_on:
            dep_running = any(
                inst.service_id == dep_id and inst.status == "running"
                for inst in self._instances.values()
            )
            if not dep_running:
                # Try to start dependency first
                dep_inst = self.start_service(dep_id)
                if not dep_inst:
                    return None

        # Create instance
        instance_id = str(uuid.uuid4())[:8]
        inst = ServiceInstance(
            service_id=service_id,
            instance_id=instance_id,
            port=port_override or svc.port,
            status="starting",
            started_iso=now_iso_utc(),
        )

        # Build command
        entry_point = self.root / svc.entry_point
        if not entry_point.exists():
            inst.status = "error"
            inst.error = f"Entry point not found: {entry_point}"
            self._instances[instance_id] = inst
            return inst

        cmd = [sys.executable, str(entry_point)] + svc.args
        if inst.port:
            cmd += ["--port", str(inst.port)]

        # Build environment
        env = os.environ.copy()
        env["PLURIBUS_ROOT"] = str(self.root)
        env["PLURIBUS_BUS_DIR"] = str(self.pluribus_dir / "bus")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(svc.env)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
                cwd=str(self.root),
            )
            inst.pid = proc.pid
            inst.status = "running"
            self._procs[instance_id] = proc

            # Write PID file
            pid_file = self.pid_dir / f"{instance_id}.pid"
            pid_file.write_text(str(proc.pid))

        except Exception as e:
            inst.status = "error"
            inst.error = str(e)

        self._instances[instance_id] = inst

        # Persist instance
        rec = {
            "kind": "service_instance",
            "ts": time.time(),
            "iso": now_iso_utc(),
            **asdict(inst),
        }
        append_ndjson(self.instances_path, rec)

        return inst

    def stop_service(self, instance_id: str) -> bool:
        """Stop a service instance."""
        inst = self._instances.get(instance_id)
        if not inst:
            return False

        proc = self._procs.get(instance_id)
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
            del self._procs[instance_id]

        elif inst.pid:
            try:
                os.kill(inst.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                pass

        inst.status = "stopped"
        self._instances[instance_id] = inst

        # Remove PID file
        pid_file = self.pid_dir / f"{instance_id}.pid"
        if pid_file.exists():
            pid_file.unlink()

        # Persist state change
        rec = {
            "kind": "service_instance",
            "ts": time.time(),
            "iso": now_iso_utc(),
            "action": "stopped",
            **asdict(inst),
        }
        append_ndjson(self.instances_path, rec)

        return True

    def check_health(self, instance_id: str) -> str:
        """Check health of a service instance."""
        inst = self._instances.get(instance_id)
        if not inst:
            return "unknown"

        svc = self._services.get(inst.service_id)
        if not svc or not svc.health_check:
            # Check if process is running
            if inst.pid:
                try:
                    os.kill(inst.pid, 0)
                    return "healthy"
                except ProcessLookupError:
                    inst.status = "stopped"
                    return "stopped"
                except Exception:
                    return "unknown"
            return "unknown"

        # HTTP health check
        if svc.health_check.startswith("http"):
            try:
                import urllib.request
                req = urllib.request.urlopen(svc.health_check, timeout=5)
                return "healthy" if req.status == 200 else "unhealthy"
            except Exception:
                return "unhealthy"

        return "unknown"

    def refresh_instances(self) -> None:
        """Refresh status of all instances."""
        for instance_id, inst in list(self._instances.items()):
            if inst.status == "running":
                health = self.check_health(instance_id)
                inst.health = health
                inst.last_health_iso = now_iso_utc()
                if health == "stopped":
                    inst.status = "stopped"

    def get_by_tag(self, tag: str) -> list[ServiceDef]:
        """Get services by tag."""
        return [svc for svc in self._services.values() if tag in svc.tags]

    def get_by_kind(self, kind: str) -> list[ServiceDef]:
        """Get services by kind (port, composition, process)."""
        return [svc for svc in self._services.values() if svc.kind == kind]


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    reg = ServiceRegistry(root)
    reg.init()
    print(f"Initialized service registry at {reg.services_dir}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    reg = ServiceRegistry(root)
    reg.init()
    reg.load()

    services = reg.list_services()
    if args.kind:
        services = [s for s in services if s.kind == args.kind]
    if args.tag:
        services = [s for s in services if args.tag in s.tags]

    print(f"{'ID':<25} {'Name':<25} {'Lineage':<15} {'Gates':<6} {'Kind':<10} {'Tags'}")
    print("-" * 100)
    for svc in sorted(services, key=lambda s: s.name):
        port = str(svc.port) if svc.port else "-"
        tags = ",".join(svc.tags[:3])
        gates = "+" if svc.gates else "-"
        omega = "Ω" if svc.omega_motif else ""
        lineage = f"{svc.lineage} {omega}"
        print(f"{svc.id:<25} {svc.name:<25} {lineage:<15} {gates:<6} {svc.kind:<10} {tags}")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    reg = ServiceRegistry(root)
    reg.init()
    reg.load()
    reg.refresh_instances()

    instances = reg.list_instances()
    print(f"{'Instance':<10} {'Service':<25} {'Status':<10} {'PID':<8} {'Port':<6} {'Health'}")
    print("-" * 80)
    for inst in sorted(instances, key=lambda i: i.started_iso, reverse=True):
        svc = reg.get_service(inst.service_id)
        svc_name = svc.name if svc else inst.service_id
        port = str(inst.port) if inst.port else "-"
        pid = str(inst.pid) if inst.pid else "-"
        print(f"{inst.instance_id:<10} {svc_name:<25} {inst.status:<10} {pid:<8} {port:<6} {inst.health}")

    return 0


def cmd_start(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    reg = ServiceRegistry(root)
    reg.init()
    reg.load()

    inst = reg.start_service(args.service_id, port_override=args.port)
    if inst:
        if inst.status == "running":
            print(f"Started {args.service_id} (instance: {inst.instance_id}, pid: {inst.pid})")
            return 0
        else:
            print(f"Failed to start {args.service_id}: {inst.error}")
            return 1
    else:
        print(f"Service not found: {args.service_id}")
        return 1


def cmd_stop(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    reg = ServiceRegistry(root)
    reg.init()
    reg.load()

    if reg.stop_service(args.instance_id):
        print(f"Stopped instance {args.instance_id}")
        return 0
    else:
        print(f"Instance not found: {args.instance_id}")
        return 1


def cmd_register(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    reg = ServiceRegistry(root)
    reg.init()
    reg.load()

    svc = ServiceDef(
        id=args.id or "",
        name=args.name,
        kind=args.kind,
        entry_point=args.entry_point,
        description=args.description or "",
        port=args.port,
        tags=[t.strip() for t in (args.tags or "").split(",") if t.strip()],
    )

    svc_id = reg.register_service(svc)
    print(f"Registered service: {svc_id}")
    return 0


def cmd_sync_to_bus(args: argparse.Namespace) -> int:
    """Emit all services and instances to the bus for dashboard sync."""
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR") or str(root / ".pluribus" / "bus")
    try:
        import agent_bus  # type: ignore

        bus_events_path = Path(agent_bus.resolve_bus_paths(bus_dir).events_path)
    except Exception:
        bus_events_path = Path(bus_dir) / "events.ndjson"

    reg = ServiceRegistry(root)
    reg.init()
    reg.load()
    reg.refresh_instances()

    services = reg.list_services()
    instances = reg.list_instances()

    # Emit full registry sync event
    sync_event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": "service.registry.sync",
        "kind": "sync",
        "level": "info",
        "actor": default_actor(),
        "data": {
            "services": [asdict(svc) for svc in services],
            "instances": [asdict(inst) for inst in instances],
            "service_count": len(services),
            "instance_count": len(instances),
        },
    }
    append_ndjson(bus_events_path, sync_event)
    print(f"Synced {len(services)} services and {len(instances)} instances to bus")

    # Optionally emit individual service.register events for each
    if args.individual:
        for svc in services:
            event = {
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "iso": now_iso_utc(),
                "topic": "service.register",
                "kind": "register",
                "level": "info",
                "actor": default_actor(),
                "data": asdict(svc),
            }
            append_ndjson(bus_events_path, event)
        print(f"Emitted {len(services)} individual service.register events")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="service_registry.py", description="Service registry for Pluribus TUI")
    p.add_argument("--root", default=None, help="Rhizome root")
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="Initialize service registry")
    init_p.set_defaults(func=cmd_init)

    list_p = sub.add_parser("list", help="List registered services")
    list_p.add_argument("--kind", choices=["port", "composition", "process"])
    list_p.add_argument("--tag")
    list_p.set_defaults(func=cmd_list)

    status_p = sub.add_parser("status", help="Show running instances")
    status_p.set_defaults(func=cmd_status)

    start_p = sub.add_parser("start", help="Start a service")
    start_p.add_argument("service_id")
    start_p.add_argument("--port", type=int)
    start_p.set_defaults(func=cmd_start)

    stop_p = sub.add_parser("stop", help="Stop a service instance")
    stop_p.add_argument("instance_id")
    stop_p.set_defaults(func=cmd_stop)

    reg_p = sub.add_parser("register", help="Register a new service")
    reg_p.add_argument("--id")
    reg_p.add_argument("--name", required=True)
    reg_p.add_argument("--kind", required=True, choices=["port", "composition", "process"])
    reg_p.add_argument("--entry-point", required=True)
    reg_p.add_argument("--description")
    reg_p.add_argument("--port", type=int)
    reg_p.add_argument("--tags")
    reg_p.set_defaults(func=cmd_register)

    sync_p = sub.add_parser("sync-to-bus", help="Emit all services to bus for dashboard sync")
    sync_p.add_argument("--individual", action="store_true", help="Also emit individual service.register events")
    sync_p.set_defaults(func=cmd_sync_to_bus)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
