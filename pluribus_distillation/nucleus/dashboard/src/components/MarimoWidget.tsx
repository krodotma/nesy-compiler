/**
 * MarimoWidget.tsx - Notebook surface (placeholder)
 *
 * Intended future state:
 * - Embed a Marimo-like reactive notebook via Pyodide/WASM.
 * - Persist notebooks as Rhizome artifacts.
 * - Allow agents to generate live reports backed by bus + rhizome queries.
 */

import { component$ } from '@builder.io/qwik';

// M3 Components - MarimoWidget
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/text-button.js';

export const MarimoWidget = component$(() => {
  return (
    <div class="rounded-lg border border-border bg-card p-4 space-y-3">
      <div class="flex items-center justify-between">
        <div class="font-semibold">🧪 Marimo (WASM) Notebook</div>
        <div class="text-xs text-muted-foreground mono">planned</div>
      </div>
      <div class="text-sm text-muted-foreground">
        This panel is a stable placeholder for a future in-browser notebook runtime (Pyodide/WASM). For now, use the
        SemOps editor + bus evidence to define “operators as interfaces”, then promote notebooks into Rhizome artifacts.
      </div>
      <div class="text-xs text-muted-foreground mono">
        TODO: pyodide bootstrap • marimo runtime • rhizome persistence • ui.render event bridge
      </div>
    </div>
  );
});

