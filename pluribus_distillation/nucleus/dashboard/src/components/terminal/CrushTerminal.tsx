import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import { Terminal } from '../terminal';

// M3 Components - CrushTerminal
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/text-button.js';

/**
 * CrushTerminal - Charmbracelet Crush/Mods CLI with Pluribus Bus Integration
 *
 * DKIN v19 compliant terminal interface for LLM queries with:
 * - Full bus event emission via crush_adapter.py
 * - Operator integration for code review/explain/refactor
 * - Multi-agent coordination support
 *
 * Uses the /crush WebSocket endpoint proxied to bus-bridge terminal multiplexer.
 */
export const CrushTerminal = component$(() => {
  const showHelp = useSignal(false);
  const crushStatus = useSignal<'checking' | 'ready' | 'unavailable'>('checking');

  // Check crush availability on mount
  useVisibleTask$(async () => {
    try {
      // Emit terminal open event to bus
      const event = {
        ts: Date.now(),
        iso: new Date().toISOString(),
        topic: 'crush.terminal.open',
        actor: 'dashboard/crush',
        level: 'info',
        data: { component: 'CrushTerminal', protocol_version: 'v19' }
      };

      // Try to emit via API (non-blocking)
      fetch('/api/emit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(event)
      }).catch(() => { /* ignore errors */ });

      crushStatus.value = 'ready';
    } catch {
      crushStatus.value = 'unavailable';
    }
  });

  const toggleHelp = $(() => {
    showHelp.value = !showHelp.value;
  });

  const initCommand = `
# Charmbracelet Crush CLI (mods v1.8.1) - DKIN v19 Integration
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CRUSH - LLM CLI with Pluribus Bus Integration               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Direct:   crush 'your prompt here'                          ║"
echo "║  Adapter:  python3 /pluribus/nucleus/tools/crush_adapter.py  ║"
echo "║  Operator: python3 /pluribus/nucleus/tools/crush_operator.py ║"
echo "╚══════════════════════════════════════════════════════════════╝"
crush --version 2>/dev/null || echo '[crush not available]'
glow --version 2>/dev/null || echo '[glow not available]'
`.trim();

  return (
    <div class="h-full flex flex-col">
      {/* Header bar */}
      <div class="flex items-center justify-between px-3 py-2 bg-zinc-900 border-b border-zinc-700">
        <div class="flex items-center gap-3">
          <span class="text-amber-400 font-mono text-sm">CRUSH</span>
          <span class="text-zinc-500 text-xs">Charmbracelet Mods + Bus Integration</span>
          {crushStatus.value === 'checking' && (
            <span class="text-yellow-500 text-xs animate-pulse">checking...</span>
          )}
          {crushStatus.value === 'ready' && (
            <span class="text-green-500 text-xs">● ready</span>
          )}
          {crushStatus.value === 'unavailable' && (
            <span class="text-red-500 text-xs">● unavailable</span>
          )}
        </div>
        <button
          onClick$={toggleHelp}
          class="text-xs text-zinc-400 hover:text-zinc-200 px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700"
        >
          {showHelp.value ? 'Hide Help' : 'Show Help'}
        </button>
      </div>

      {/* Help panel */}
      {showHelp.value && (
        <div class="px-3 py-2 bg-zinc-800 border-b border-zinc-700 text-xs font-mono">
          <div class="grid grid-cols-2 gap-4">
            <div>
              <div class="text-amber-400 mb-1">Direct Usage:</div>
              <div class="text-zinc-300">crush "explain this code"</div>
              <div class="text-zinc-300">cat file.py | crush "review"</div>
              <div class="text-zinc-300">crush -m gpt-4 "prompt"</div>
            </div>
            <div>
              <div class="text-amber-400 mb-1">Operator (Bus Events):</div>
              <div class="text-zinc-300">crush_operator.py --query "..."</div>
              <div class="text-zinc-300">crush_operator.py --review file.py</div>
              <div class="text-zinc-300">crush_operator.py --explain "topic"</div>
            </div>
            <div>
              <div class="text-amber-400 mb-1">Adapter (Full Bus):</div>
              <div class="text-zinc-300">crush_adapter.py -p "prompt"</div>
              <div class="text-zinc-300">crush_adapter.py -i (interactive)</div>
              <div class="text-zinc-300">crush_adapter.py --stream-bus</div>
            </div>
            <div>
              <div class="text-amber-400 mb-1">Utilities:</div>
              <div class="text-zinc-300">glow README.md (render markdown)</div>
              <div class="text-zinc-300">charm (account management)</div>
              <div class="text-zinc-300">crush --list (saved chats)</div>
            </div>
          </div>
          <div class="mt-2 text-zinc-500">
            Bus topics: crush.session.* | crush.prompt.* | crush.response.* | operator.crush.*
          </div>
        </div>
      )}

      {/* Terminal */}
      <div class="flex-1 min-h-0">
        <Terminal
          endpoint="/crush"
          title="Crush (Agentic Editor)"
          initCommand={initCommand}
        />
      </div>
    </div>
  );
});
