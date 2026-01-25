import { component$ } from '@builder.io/qwik';
import { ArkBusListener } from '../../components/ArkBusListener';

export default component$(() => {
    return (
        <div class="h-full w-full p-8 flex flex-col gap-6">
            <header class="flex items-end justify-between border-b border-white/10 pb-6">
                <div>
                    <h1 class="text-4xl font-bold font-display text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
                        ARK Neural Kernel
                    </h1>
                    <div class="text-sm font-mono text-cyan-400/60 mt-2 flex gap-4">
                        <span>DNA-GATED EVOLUTION</span> |
                        <span>TRUNK B: OBSERVER</span>
                    </div>
                </div>
                <div class="flex gap-2 text-xs font-mono">
                    <div class="px-3 py-1 glass-panel bg-green-500/10 text-green-400 border-green-500/30">
                        SYSTEM HEALTHY
                    </div>
                    <div class="px-3 py-1 glass-panel bg-purple-500/10 text-purple-400 border-purple-500/30">
                        BRIDGE ACTIVE
                    </div>
                </div>
            </header>

            <div class="grid grid-cols-12 gap-6 flex-1 min-h-0">
                {/* Left Column: Cell Cycle & Metrics */}
                <div class="col-span-8 flex flex-col gap-6">

                    {/* Cell Cycle Visualization */}
                    <section class="glass-panel p-6 relative overflow-hidden">
                        <h2 class="text-sm font-mono uppercase text-white/50 mb-4">Cell Cycle Status</h2>
                        <div class="flex justify-between items-center relative z-10 px-10 py-8">
                            {['G1', 'S', 'G2', 'M'].map((phase, i) => (
                                <div key={phase} class="flex flex-col items-center gap-2 group">
                                    <div class={`w-16 h-16 rounded-full flex items-center justify-center text-xl font-bold border-2 transition-all duration-500
                     ${i === 0 ? 'border-cyan-400 bg-cyan-400/10 text-cyan-100 shadow-[0_0_20px_rgba(34,211,238,0.3)]' : 'border-white/10 text-white/30'}
                   `}>
                                        {phase}
                                    </div>
                                    <div class="text-[10px] uppercase tracking-widest text-white/40 group-hover:text-white/80 transition-colors">
                                        {['Ready', 'Synth', 'Verify', 'Commit'][i]}
                                    </div>
                                </div>
                            ))}
                            {/* Connector Lines */}
                            <div class="absolute top-1/2 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent -z-10" />
                        </div>
                    </section>

                    {/* Neural Gates & CMP */}
                    <div class="grid grid-cols-2 gap-6 flex-1">
                        <div class="glass-panel p-6">
                            <h2 class="text-sm font-mono uppercase text-white/50 mb-4">Neural Gates</h2>
                            <div class="space-y-4">
                                {['Inertia', 'Entelecheia', 'Homeostasis'].map(gate => (
                                    <div key={gate} class="flex items-center justify-between">
                                        <span class="text-sm text-cyan-100">{gate}</span>
                                        <div class="w-32 h-2 bg-white/10 rounded-full overflow-hidden">
                                            <div class="h-full bg-cyan-500 w-[85%]" />
                                        </div>
                                        <span class="text-xs font-mono text-cyan-400">0.85</span>
                                    </div>
                                ))}
                            </div>

                            <div class="mt-8 pt-6 border-t border-white/10">
                                <div class="flex justify-between items-end mb-2">
                                    <span class="text-xs text-red-300">Thrash Probability</span>
                                    <span class="text-xl font-bold text-red-400">12%</span>
                                </div>
                                <div class="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                                    <div class="h-full bg-red-500 w-[12%]" />
                                </div>
                            </div>
                        </div>

                        <div class="glass-panel p-6 flex flex-col">
                            <h2 class="text-sm font-mono uppercase text-white/50 mb-4">CMP Evolution</h2>
                            <div class="flex-1 flex items-end gap-1 pb-2 border-b border-white/10 relative">
                                {/* Mock Chart */}
                                {[40, 45, 42, 50, 55, 52, 60, 65, 62, 70].map((h, i) => (
                                    <div key={i} style={{ height: `${h}%` }} class="flex-1 bg-purple-500/50 hover:bg-purple-400 transition-colors rounded-t-sm" />
                                ))}
                            </div>
                            <div class="flex justify-between text-[10px] font-mono text-white/30 mt-2">
                                <span>T-10h</span>
                                <span>Now</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Column: Bus Feed */}
                <div class="col-span-4 h-full min-h-[500px]">
                    <ArkBusListener />
                </div>
            </div>
        </div>
    );
});
