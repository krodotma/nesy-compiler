# ark-cli: Design & PLI Schema

**Version:** 1.0.0
**Status:** DRAFT
**Context:** The Command Line Interface and "Pli" (Plural Linearized Interface).

## CLI Architecture (Commander)
*   **Entry:** `ark`
*   **Commands:**
    *   `ark start` (Boot Nucleus/Spine/API).
    *   `ark git ...` (Wrapper for `ark-git`).
    *   `ark bus ...` (Interact with `ark-bus`).
    *   `ark doctor` (Health check).

## PLI: Plural Linearized Interface
**PLI** is a JSON schema for generating UIs (CLI Menus and React components) from a single definition.

### Schema Definition
```typescript
interface PliNode {
  id: string;
  type: "group" | "action" | "input" | "display";
  label: string;
  icon?: string;
  // Dynamic children
  children?: PliNode[];
  // Executable action
  exec?: {
    rpc: string;
    params: Record<string, any>;
  };
  // Data binding
  bind?: {
    store: "spine" | "graph";
    key: string;
  };
}
```

### Example: Main Menu
```json
{
  "id": "root",
  "type": "group",
  "label": "Ark Main Menu",
  "children": [
    {
      "id": "git_status",
      "type": "display",
      "label": "Git Status",
      "bind": { "store": "spine", "key": "git_status_summary" }
    },
    {
      "id": "start_agent",
      "type": "action",
      "label": "Start Agent",
      "exec": { "rpc": "agent.start", "params": {} }
    }
  ]
}
```
This schema allows the CLI to render a TUI (Text User Interface) and the Web Dashboard to render a Sidebar Menu from the **same source of truth**.
