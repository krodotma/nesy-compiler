import { component$, useSignal, useVisibleTask$, useStore, $ } from '@builder.io/qwik';

export interface ArkEvent {
    topic: string;
    data: any;
    timestamp: number;
}

export const ArkBusListener = component$(() => {
    const events = useStore<{ list: ArkEvent[] }>({ list: [] });
    const connected = useSignal(false);

    // Poll for events (Simple implementation for now, ideally WS)
    useVisibleTask$(({ cleanup }) => {
        const poll = async () => {
            try {
                // In a real implementation this would fetch from an API endpoint
                // For now we mock the "read" or assume an API exists
                // This is a placeholder for the actual fetch logic
                // const res = await fetch('/api/bus/events?topic=ark.*');
                // const newEvents = await res.json();
                // events.list = [...newEvents, ...events.list].slice(0, 50);
                connected.value = true;
            } catch (e) {
                connected.value = false;
            }
        };

        const interval = setInterval(poll, 2000);
        cleanup(() => clearInterval(interval));
    });

    return (
        <div class="glass-panel p-4 h-full overflow-hidden flex flex-col">
            <div class="flex items-center justify-between mb-2">
                <h3 class="text-xs font-mono uppercase tracking-widest text-cyan-400">
                    Neural Bus Feed
                </h3>
                <div class={`w-2 h-2 rounded-full ${connected.value ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`} />
            </div>
            <div class="flex-1 overflow-y-auto space-y-2 font-mono text-xs">
                {events.list.length === 0 && (
                    <div class="text-white/30 italic text-center mt-10">Waiting for signal...</div>
                )}
                {events.list.map((ev, i) => (
                    <div key={i} class="border-l-2 border-cyan-500/30 pl-2 py-1 bg-black/20">
                        <div class="flex justify-between text-[10px] text-white/50">
                            <span>{ev.topic}</span>
                            <span>{new Date(ev.timestamp * 1000).toLocaleTimeString()}</span>
                        </div>
                        <div class="text-cyan-100 truncate">
                            {JSON.stringify(ev.data)}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
});
