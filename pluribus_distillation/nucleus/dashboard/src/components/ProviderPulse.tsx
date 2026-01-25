import { component$ } from '@builder.io/qwik';

interface ProviderPulseProps {
  latency?: number;
  available: boolean;
}

export const ProviderPulse = component$<ProviderPulseProps>(({ latency = 0, available }) => {
  // Visual mapping: <200ms = Green, <1000ms = Cyan, >1000ms = Amber, Unavailable = Red
  const color = !available ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]' :
                latency < 200 ? 'bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.6)]' :
                latency < 1000 ? 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]' :
                'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)]';

  // Bar width represents "Energy" or inverse latency?
  // Let's make it a "Ping" bar.
  // Width logic: 100% = fast, 10% = slow.
  // normalized = 100 - (latency / 2000 * 100)
  const normalized = available ? Math.max(10, 100 - (latency / 2000 * 100)) : 5;

  return (
    <div class="flex items-center gap-2">
      <div class="h-1 w-12 glass-surface-subtle rounded-full overflow-hidden">
        <div
          class={`h-full ${color} glass-transition-all`}
          style={{ width: `${normalized}%` }}
        />
      </div>
      <span class={`text-[9px] font-mono ${available ? 'text-glass-text-muted' : 'text-red-400'}`}>
        {available ? `${latency}ms` : 'OFF'}
      </span>
    </div>
  );
});
