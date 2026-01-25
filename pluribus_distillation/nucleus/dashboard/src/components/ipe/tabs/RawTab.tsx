/**
 * RawTab.tsx
 *
 * Raw JSON viewer/editor tab for IPE.
 * Shows the full captured context as JSON.
 */

import {
  component$,
  useSignal,
  $,
} from '@builder.io/qwik';
import type { IPEContext } from '../../../lib/ipe';

interface RawTabProps {
  context: IPEContext;
}

export const RawTab = component$<RawTabProps>(({ context }) => {
  const copied = useSignal(false);

  const handleCopy = $(() => {
    navigator.clipboard.writeText(JSON.stringify(context, null, 2));
    copied.value = true;
    setTimeout(() => { copied.value = false; }, 2000);
  });

  return (
    <div class="space-y-3">
      {/* Header with copy button */}
      <div class="flex items-center justify-between">
        <div class="text-sm text-gray-400">Raw Context (JSON)</div>
        <button
          type="button"
          class={[
            'px-3 py-1 rounded text-xs font-medium transition-all',
            copied.value
              ? 'bg-green-600 text-white'
              : 'bg-white/10 hover:bg-white/20 text-gray-300',
          ]}
          onClick$={handleCopy}
        >
          {copied.value ? '✓ Copied!' : 'Copy JSON'}
        </button>
      </div>

      {/* JSON viewer */}
      <pre
        class={[
          'p-4 rounded-lg bg-black/50 overflow-auto',
          'text-xs font-mono text-gray-300 leading-relaxed',
          'max-h-[400px] border border-[var(--glass-border)]',
        ]}
      >
        <JSONHighlight data={context} />
      </pre>

      {/* Quick stats */}
      <div class="grid grid-cols-2 gap-2 text-xs">
        <div class="p-2 rounded bg-white/5">
          <div class="text-gray-500">Element Type</div>
          <div class="text-blue-400 font-mono">{context.elementType}</div>
        </div>
        <div class="p-2 rounded bg-white/5">
          <div class="text-gray-500">Instance ID</div>
          <div class="text-purple-400 font-mono truncate">{context.instanceId}</div>
        </div>
        <div class="p-2 rounded bg-white/5">
          <div class="text-gray-500">CSS Variables</div>
          <div class="text-green-400 font-mono">{Object.keys(context.cssVariables).length}</div>
        </div>
        <div class="p-2 rounded bg-white/5">
          <div class="text-gray-500">Tailwind Classes</div>
          <div class="text-orange-400 font-mono">{context.tailwindClasses.length}</div>
        </div>
      </div>
    </div>
  );
});

// Simple JSON syntax highlighting component
const JSONHighlight = component$<{ data: unknown }>(({ data }) => {
  const json = JSON.stringify(data, null, 2);

  // Simple regex-based highlighting
  const highlighted = json
    // Keys
    .replace(/"([^"]+)":/g, '<span class="text-blue-400">"$1"</span>:')
    // String values
    .replace(/: "([^"]*?)"/g, ': <span class="text-green-400">"$1"</span>')
    // Numbers
    .replace(/: (\d+\.?\d*)/g, ': <span class="text-yellow-400">$1</span>')
    // Booleans
    .replace(/: (true|false)/g, ': <span class="text-purple-400">$1</span>')
    // Null
    .replace(/: null/g, ': <span class="text-gray-500">null</span>');

  return <span dangerouslySetInnerHTML={highlighted} />;
});

export default RawTab;
