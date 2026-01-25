# WHIPPERSNAPPER: UI Refactor & Menu Synthesis Plan

**Objective:** Modernize Pluribus Dashboard UI with advanced tech stack (Lit, Clarity, Hakim, Isotope), refactor menu into semantic domains (DevOps/EvoCode), and implement "Hidden App" concurrency.

## 1. Menu Refactor: The "Bicameral" Split

We will deprecate the flat `nav` list in favor of a two-pane or tabbed sidebar approach, splitting the dashboard into Operational vs. Evolutionary concerns.

### Domain A: DevOps & MLOps (The Machine)
*Focus: Runtime, Observability, Inference, Infrastructure.*
*   **Bus:** Consolidated synthesis of Events, Agents, Requests. This is the "Nerve Center".
    *   *Sub-views:* Live Pulse, Event Stream (Isotope), Agent Grid.
*   **Services:** Process management, PM2 status, Bridge health.
*   **Diagnostics:** Deep system introspection (MBAD).
*   **Terminal:** Direct CLI access.
*   **Inference:**
    *   **WebLLM:** Local edge inference.
    *   **Voice:** Speech I/O.
    *   **SOTA:** Model/Tool catalog (Isotope grid).
*   **SemOps:** Semantic Operator registry and editor.

### Domain B: Evolutionary Code (The Organism)
*Focus: Creation, Mutation, History, Knowledge.*
*   **Studio:** The "Home" workspace for active creation.
*   **Rhizome:** Codebase explorer and knowledge graph visualization.
*   **Git:** Evolutionary lineage (VGT/HGT), history, branches.
*   **Distill:** Knowledge crystallization, beam search logs.
*   **Generative:** Art surface, creative coding experiments.

### Synthesis Strategy
*   **'Bus' as Synthesis:** Instead of separate tabs for Events/Agents/Requests, the 'Bus' view will become a unified dashboard widget grid.
    *   *Widget 1:* **Agent Swarm** (Agent status bubbles).
    *   *Widget 2:* **Request Queue** (Kanban board of STRp requests).
    *   *Widget 3:* **Event Stream** (High-density log with Isotope filtering).

## 2. Tech Stack Integration (The "Fancy" Stuff)

*   **Metafizzy Isotope:**
    *   *Target:* **SOTA Catalog** & **Event Stream**.
    *   *Implementation:* Wrap Isotope in a Qwik `useVisibleTask$` hook. Use it to filter/sort SOTA tools by tag/provider and Events by kind/impact without page reloads.
*   **Hakim.se (Visual Flair):**
    *   *Target:* **Generative Background** & **Page Transitions**.
    *   *Implementation:* Integrate `reveal.js`-style 3D transforms for switching between the two main Menu Domains (DevOps <-> EvoCode). Use particle effects for the "Omega Heartbeat" visualizer.
    *   *Update (King Mode):* Replace static sidebar with an **Expandable Overlay** (fullscreen modal with blur) triggered by a "MENU" button. This mimics high-end editorial/portfolio sites (Hakim.se style).

*   **Visual Synthesis Layer:**
    *   Create `VisualSynthesis.tsx` to orchestrate Shaders (GenerativeBackground) + Particles + CSS 3D Transforms.
    *   Ensure z-index layering allows content interaction while maintaining depth.

*   **VMware Clarity (Design System):**
    *   *Target:* **DevOps Domain** (Services, Diagnostics, Terminal).
    *   *Implementation:* Adopt Clarity's Data Grid and Card patterns for high-density information display. Use their color system (clean grays/blues) for the "Machine" side of the UI to contrast with the "Organism" side.
*   **Lit (Web Components):**
    *   *Target:* **Portable Widgets** (WebLLM, Voice).
    *   *Implementation:* Refactor `WebLLMWidget` and `VoiceSpeechView` into standalone Lit components (`<pluribus-webllm>`, `<pluribus-voice>`). This allows them to be embedded in *any* view (or even external pages) via the Shadow DOM, ensuring style isolation and portability.

## 3. The "Hidden App" Architecture (Concurrency)

To achieve "extremely fast" UI with non-blocking execution:

### Architecture: `ShadowWorker`
*   **Concept:** A dedicated Web Worker (`shadow.worker.ts`) that runs a lightweight "Headless App".
*   **Responsibility:**
    *   Pre-fetching data for all views (SOTA, Git, Rhizome).
    *   Managing shared state (Agents, Requests) via `BroadcastChannel`.
    *   Running heavy logic (e.g., local RAG indexing, client-side filtering).
*   **UI interaction:** The main Qwik UI becomes a "Thin Client" view layer. It only renders data pushed to it by the `ShadowWorker` via signals.
*   **Lazy Loading:**
    *   Qwik handles DOM lazy-loading automatically.
    *   We will aggressively `server$` load heavy data only when requested, but pre-cache critical paths in the `ShadowWorker`.

## 4. Execution Plan (WHIPPERSNAPPER)

1.  **Scaffold `ShadowWorker`:** Create `src/workers/shadow.worker.ts` and wire it to `layout.tsx` (alongside `omega.worker.ts`).
2.  **Refactor Menu:** Create `src/components/nav/BicameralNav.tsx` with the new split structure.
3.  **Synthesize Bus:** Create `src/components/views/UnifiedBusView.tsx` aggregating Agents/Requests/Events using Isotope.
4.  **Lit Integration:** Port `WebLLMWidget` to a Lit element wrapper.
5.  **Visual Polish:** Apply Clarity styles to `Services` view and Hakim transitions to the main layout switch.

*Signed: WHIPPERSNAPPER (Subagent)*
