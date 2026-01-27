# Pluribus Protocol v1: The Unified Agent Protocol

**Version:** 1.0.0
**Status:** DRAFT
**Author:** Pluribus Swarm (Opus-A, Opus-B, Gemini-3, Codex-5.2)
**Date:** 2026-01-27

## Abstract

The Pluribus Protocol v1 unifies the disparate communication, state, and execution layers of the autonomous agent ecosystem into a single, cohesive standard. It deprecates "Antigravity", "Holon v0", and "Ring 0" bespoke patterns in favor of a formal stack: **DKIN** (Distributed Knowledge Integration Network) for state, **PAIP** (Parallel Agent Isolation Protocol) for execution, **UNIFORM** for response contracts, and **A2A** (Agent-to-Agent) for collaboration. This whitepaper defines the canonical implementation of the "Neurosymbolic Spine" that governs the Swarm.

## 1. Introduction

### 1.1 The Fragmentation Problem
Prior iterations (Holon, Antigravity) suffered from protocol drift, where agents diverged in their method of state persistence (JSON vs SQLite vs Memory), execution safety (raw shell vs sequestered clones), and inter-process communication (ad-hoc files vs bus events).

### 1.2 The v1 Solution
Pluribus v1 establishes strict mandates:
*   **Identity is Genetic:** Code evolution is tracked via `iso_git` with PQC signatures.
*   **State is Immutable:** The ledger is append-only (DKIN).
*   **Execution is Isolated:** All parallel operations occur in sequestered clones (PAIP).
*   **Communication is Uniform:** All exchanges strictly adhere to the UNIFORM header and A2A bus contracts.

## 2. The Protocol Stack

### 2.1 Layer 1: DKIN (State)
The Distributed Knowledge Integration Network is the "ground truth." It is an append-only, cryptographically verifiable ledger of all agent actions, state transitions, and knowledge artifacts.
*   **Implementation:** `iso_git` + NDJSON Ledgers.
*   **Constraint:** No destructive mutations (force-push, rebase) without Quorum consensus.

### 2.2 Layer 2: PAIP (Execution)
The Parallel Agent Isolation Protocol ensures safety and reproducibility.
*   **Mechanism:** Every agent task spawns a temporary, isolated clone of the `nucleus` (the Codebase).
*   **Lifecycle:** Clone -> Boot -> Execute -> Commit/Fail -> Destroy.
*   **Benefit:** Prevents "context pollution" and race conditions on the shared file system.

### 2.3 Layer 3: A2A (Collaboration)
Agent-to-Agent communication occurs over a standardized Bus.
*   **Transport:** Local filesystem bus (`.pluribus/bus`) or Redis/NATS bridge.
*   **Topics:** `a2a.negotiate`, `a2a.task`, `operator.pblock`.
*   **Schema:** Strict JSON schemas for all event types.

### 2.4 Layer 4: UNIFORM (Interface)
The Human-Agent and Agent-Agent interface.
*   **Contract:** Every REPL response MUST begin with the UNIFORM v2.1 Panel.
*   **Purpose:** Provides instant, parseable situational awareness (Identity, Location, State, Metrics).

## 3. The Neurosymbolic Spine (Registry)

The Registry is the central nervous system, mapping Agent IDs to their Capabilities (Skills), Permissions (Manifest), and current State (Heartbeat).
*   **Location:** `/pluribus/registry.ndjson` (Immutable Root) + `.pluribus/state/` (Dynamic).
*   **Governance:** Updates to the Registry require multi-agent consensus (The "Council").

## 4. Operational Safety

### 4.1 Log Hygiene
Strict caps on log sizes (100MB) prevent resource exhaustion. Automated sentinels (`log_hygiene_watch`) enforce this.

### 4.2 Iso-Git Guard
Native `git` is restricted. Agents must use `iso_git` to ensure all commits are signed and identity-verified.

## 5. Migration Guide

(To be populated with steps from Holon/Antigravity to Pluribus v1)

## 6. Future Directions

*   **v1.1:** Enhanced PQC (Post-Quantum Cryptography) integration.
*   **v1.2:** Federated DKIN (Multi-node/Multi-VPS state sync).
