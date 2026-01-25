import { component$ } from '@builder.io/qwik';
import { GenerativeBackground } from '../../components/art/GenerativeBackground';

export default component$(() => {
  return (
    <div class="h-screen w-screen relative overflow-hidden bg-transparent flex items-center justify-center">
      {/* Force Anxious/Glitch Mood for Error State */}
      <GenerativeBackground entropy={0.9} mood="anxious" />
      
      <div class="glass-card p-12 text-center max-w-lg mx-4 z-10 border-red-500/30 shadow-[0_0_50px_rgba(239,68,68,0.2)]">
        <h1 class="text-9xl font-bold text-transparent bg-clip-text bg-gradient-to-b from-red-500 to-purple-900 mb-4 animate-pulse">
          404
        </h1>
        <div class="text-2xl text-red-400 font-mono mb-8 tracking-widest uppercase">
          Signal Lost
        </div>
        <p class="text-muted-foreground mb-8">
          The requested trajectory has collapsed. 
          The Art Dept suggests you return to the <a href="/" class="text-cyan-400 hover:text-cyan-300 underline decoration-cyan-500/30">Substrate</a>.
        </p>
        
        <div class="text-xs font-mono text-white/20 border-t border-[var(--glass-border-subtle)] pt-4">
          ERROR_CODE: NULL_POINTER_EXCEPTION
          <br/>
          SECTOR: UNKNOWN
        </div>
      </div>
    </div>
  );
});
