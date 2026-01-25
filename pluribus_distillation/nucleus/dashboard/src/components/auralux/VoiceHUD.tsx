import { component$, useVisibleTask$, useSignal } from '@builder.io/qwik';
import { useVoice } from '../../lib/auralux/use-voice';

export const VoiceHUD = component$(() => {
  const { state, start, stop, toggleConsole } = useVoice();
  const canvasRef = useSignal<HTMLCanvasElement>();

  useVisibleTask$(({ cleanup }) => {
    let animationId: number;
    const canvas = canvasRef.value;
    const ctx = canvas?.getContext('2d');

    const render = () => {
      if (!state.isRunning || !state.orchestrator || !canvas || !ctx) {
        animationId = requestAnimationFrame(render);
        return;
      }

      // Safe access to analyser via exposed getter
      const analyser = state.orchestrator.getAnalyserNode();
      if (!analyser) {
         animationId = requestAnimationFrame(render);
         return;
      }

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);

      // Clear
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw Bars
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i] / 2; // Scale down
        
        // Gradient color based on height/loudness
        ctx.fillStyle = `rgba(34, 197, 94, ${barHeight / 100})`; // Green-ish
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
      }

      animationId = requestAnimationFrame(render);
    };

    render();

    cleanup(() => cancelAnimationFrame(animationId));
  });

  return (
    <div class="voice-hud flex items-center gap-2 p-1 rounded-full bg-background/50 backdrop-blur-md border border-white/10 shadow-sm ml-auto">
      <button
        onClick$={toggleConsole}
        disabled={!state.isReady}
        class={`
          relative flex items-center justify-center w-8 h-8 rounded-full transition-all
          ${!state.isReady ? 'opacity-50 cursor-not-allowed' : ''}
          ${state.isConsoleOpen 
            ? 'bg-primary/20 text-primary border-primary/30'
            : state.isRunning 
              ? 'bg-red-500/20 text-red-500 animate-pulse border-red-500/30' 
              : 'bg-muted/20 text-muted-foreground hover:bg-muted/40 hover:text-foreground border-transparent'}
          border
        `}
        title={state.error ? state.error : "Toggle Auralux Console"}
      >
        {/* Mic Icon */}
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
          <line x1="12" y1="19" x2="12" y2="23"></line>
          <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>
      </button>

      {/* Expanded Status (Only when active or error) */}
      {(state.isRunning || state.error) && (
        <div class="flex flex-col text-[9px] leading-tight px-2 font-mono border-l border-white/10">
          {state.error ? (
            <span class="text-red-400 font-bold">ERROR</span>
          ) : (
            <div class="flex items-center gap-2">
              <div class="flex flex-col">
                <div class="flex items-center gap-1.5">
                  <span class={`w-1.5 h-1.5 rounded-full ${state.vadState !== 'silence' ? 'bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.6)]' : 'bg-yellow-500/50'}`}></span>
                  <span class="text-foreground/80 tracking-wider">{state.vadState === 'silence' ? 'IDLE' : 'LISTENING'}</span>
                </div>
              </div>
              {/* Visualizer Canvas */}
              <canvas 
                ref={canvasRef} 
                width={60} 
                height={20} 
                class="opacity-80"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
});
