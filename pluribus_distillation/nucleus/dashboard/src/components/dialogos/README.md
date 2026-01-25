# Dialogos Widget Documentation

**The Unified Epistemic Ingress**

The `DialogosWidget` is the primary interface for all Human-Agent interaction in the Pluribus system. It replaces the fragmented chat, task, and code entry points with a single, context-aware "Omni-Bar".

## Architecture

The widget is composed of four layers:

1.  **The Body (Physics Shell)**: `DialogosShell.tsx`
    *   Handles the "Growable" physics (Rest -> Active -> Full).
    *   Uses FLIP transitions and Glassmorphism (`glass-tokens.css`).
    *   Mounted in `root.tsx` at `z-[9000]`.

2.  **The Nervous System (Bridge)**: `PBTSOBridge.ts`
    *   Connects the UI to backend task ingress; control stays in PBTSO tmux orchestration.
    *   Dispatches `task.create` events for PBTSO task ingress.
    *   Listens for `pbtso.task.created` (and legacy `tbtso.task.created`) to update UI state.
    *   Requires `nucleus/tools/pbtso_task_daemon.py` to be running.

3.  **The Brain (Logic Core)**: `IntentRouter.ts` & `CommandParser.ts`
    *   Heuristically routes input to `query`, `mutation`, `task`, or `reflection`.
    *   Parses slash commands (`/task --lane=inbox`) into structured data.
    *   Uses `SnapshotService.ts` to capture browser/IPE context.

4.  **The Soul (Entelexis)**: `EntelexisTracker.ts`
    *   Visualizes the "Realization" of thought.
    *   Maps atom states (`potential` -> `actualizing` -> `actualized`) to UI pulses and colors.

## Usage

*   **Toggle**: Press `Cmd+K` (Mac) or `Ctrl+K` (Windows/Linux).
*   **Tasks**: Type `/task My Task` to create a card in the default lane.
*   **Refactor**: Type `/fix` or just "Refactor this function" (context aware).
*   **Research**: Type "Why..." or `/sota` to trigger a knowledge graph query.

## Development

*   **Store**: `use-dialogos-store.ts` manages the optimistic UI and persistence.
*   **Persistence**: `persistence.ts` uses IndexedDB to save history across reloads.
*   **Testing**: Run `npx vitest run nucleus/dashboard/src/components/dialogos/logic/logic.test.ts`.

## Integration Status (Jan 21, 2026)

*   ✅ **Foundation**: Types, Store, Persistence.
*   ✅ **Visuals**: NeonInput, SmartChips, Glass Shell.
*   ✅ **Logic**: Intent Router, Command Parser, Snapshotting.
*   ✅ **Bridge**: PBTSO (Tasks).
*   🚧 **Pending**: WebLLM (Inference), IPE (Code Mutation).
