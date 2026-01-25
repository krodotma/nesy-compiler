import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { NeonTitle } from '../ui/NeonTitle';

interface BootStep {
  id: string;
  label: string;
  status: 'pending' | 'active' | 'complete';
  icon: string;
}

export const CognitiveBootLoader = component$(() => {
  const steps = useSignal<BootStep[]>([
    { id: 'synapse', label: 'Checking Synapses...', status: 'pending', icon: '⚡' },
    { id: 'memory', label: 'Mounting Hippocampus...', status: 'pending', icon: '🧠' },
    { id: 'persona', label: 'Ingesting Personas...', status: 'pending', icon: '🎭' },
    { id: 'skills', label: 'Hydrating Skills...', status: 'pending', icon: '🦾' },
    { id: 'awareness', label: 'Self-Awareness Check...', status: 'pending', icon: '👁️' },
  ]);

  useVisibleTask$(() => {
    const sequence = async () => {
      for (let i = 0; i < steps.value.length; i++) {
        // Set current to active
        steps.value = steps.value.map((s, idx) => 
          idx === i ? { ...s, status: 'active' } : s
        );
        
        // Wait random time (simulating load)
        await new Promise(r => setTimeout(r, 400 + Math.random() * 600));
        
        // Set current to complete
        steps.value = steps.value.map((s, idx) => 
          idx === i ? { ...s, status: 'complete' } : s
        );
      }
    };
    sequence();
  });

  return (
    <div class="w-full max-w-md mx-auto p-4 bg-black/40 backdrop-blur-md rounded-lg border border-cyan-500/20">
      <div class="mb-4 text-center">
        <NeonTitle level="h3" color="cyan" size="sm">Cognitive Boot Sequence</NeonTitle>
      </div>
      <div class="space-y-3">
        {steps.value.map((step) => (
          <div key={step.id} class="flex items-center gap-3">
            <div class={`w-6 h-6 flex items-center justify-center rounded-full transition-all duration-300 ${
              step.status === 'complete' ? 'bg-cyan-500/20 text-cyan-400' :
              step.status === 'active' ? 'bg-amber-500/20 text-amber-400 animate-pulse' :
              'bg-muted/10 text-muted-foreground'
            }`}>
              <span class="text-xs">{step.icon}</span>
            </div>
            <div class="flex-1">
              <div class={`text-xs font-mono transition-colors ${
                step.status === 'active' ? 'text-cyan-100' : 
                step.status === 'complete' ? 'text-muted-foreground line-through opacity-50' : 
                'text-muted-foreground/50'
              }`}>
                {step.label}
              </div>
              {step.status === 'active' && (
                <div class="h-0.5 w-full bg-cyan-900/30 mt-1 rounded-full overflow-hidden">
                  <div class="h-full bg-cyan-400/50 w-1/2 animate-progress-indeterminate" />
                </div>
              )}
            </div>
            {step.status === 'complete' && (
              <span class="text-[10px] text-emerald-400 font-mono">OK</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
});
