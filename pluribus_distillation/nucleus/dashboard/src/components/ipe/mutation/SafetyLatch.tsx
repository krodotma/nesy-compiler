/**
 * SafetyLatch.tsx
 * [Ultrathink Agent 2: UX]
 * 
 * A high-consequence UI interaction component.
 * Requires a deliberate "Slide to Arm" gesture to enable dangerous mutations.
 * 
 * Aesthetic: Industrial Sci-Fi / Nuclear Launch Control.
 */

import {
  component$,
  useSignal,
  $,
  useVisibleTask$,
} from '@builder.io/qwik';

interface SafetyLatchProps {
  onArm$: () => void;
  onDisarm$: () => void;
  isArmed: boolean;
}

export const SafetyLatch = component$<SafetyLatchProps>(({ onArm$, onDisarm$, isArmed }) => {
  const isDragging = useSignal(false);
  const dragX = useSignal(0);
  const containerRef = useSignal<HTMLDivElement>();
  const width = 200; // Fixed width for latch

  // Handle drag interaction
  const handleStart = $((e: MouseEvent | TouchEvent) => {
    if (isArmed) return; // Can't drag if already locked open
    isDragging.value = true;
  });

  const handleMove = $((e: MouseEvent | TouchEvent) => {
    if (!isDragging.value || isArmed) return;
    
    const clientX = 'touches' in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
    const rect = containerRef.value!.getBoundingClientRect();
    const offsetX = Math.max(0, Math.min(width - 40, clientX - rect.left));
    
    dragX.value = offsetX;

    // Threshold to arm (80% of width)
    if (offsetX > width * 0.8) {
      isDragging.value = false;
      dragX.value = width - 40; // Lock at end
      onArm$();
    }
  });

  const handleEnd = $(() => {
    if (isArmed) return;
    isDragging.value = false;
    dragX.value = 0; // Snap back
  });

  // Global event listeners for drag release outside component
  useVisibleTask$(({ cleanup }) => {
    const upHandler = () => {
      if (isDragging.value) {
        isDragging.value = false;
        dragX.value = 0;
      }
    };
    const moveHandler = (e: MouseEvent) => {
      if (isDragging.value) {
        // Replicate logic for global move if needed, 
        // strictly relying on bounded move for now for simplicity
      }
    };

    window.addEventListener('mouseup', upHandler);
    window.addEventListener('touchend', upHandler);
    
    cleanup(() => {
      window.removeEventListener('mouseup', upHandler);
      window.removeEventListener('touchend', upHandler);
    });
  });

  return (
    <div class="flex flex-col items-center gap-2 select-none">
      <div 
        ref={containerRef}
        class={`relative h-10 rounded-full border border-gray-600 bg-gray-900/80 overflow-hidden transition-all duration-300 ${
          isArmed ? 'shadow-[0_0_15px_rgba(239,68,68,0.5)] border-red-500/50' : ''
        }`}
        style={{ width: `${width}px` }}
        onMouseMove$={handleMove}
        onTouchMove$={handleMove}
        onMouseUp$={handleEnd}
        onTouchEnd$={handleEnd}
        onMouseLeave$={handleEnd}
      >
        {/* Track Track */}
        <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div class={`text-[10px] font-mono tracking-widest transition-opacity duration-300 ${isDragging.value || isArmed ? 'opacity-0' : 'opacity-50 text-gray-400'}`}>
            SLIDE TO ARM
          </div>
          <div class={`text-[10px] font-mono tracking-widest font-bold text-red-500 transition-opacity duration-300 ${isArmed ? 'opacity-100 animate-pulse' : 'opacity-0'}`}>
            SYSTEM ARMED
          </div>
        </div>

        {/* Hazard Striping Background (Visible when armed) */}
        <div 
          class={`absolute inset-0 opacity-10 bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,#000_10px,#000_20px)] transition-opacity ${isArmed ? 'opacity-30' : 'opacity-0'}`} 
        />

        {/* The Handle */}
        <div 
          class={`absolute top-1 bottom-1 w-8 rounded-full shadow-lg cursor-grab active:cursor-grabbing flex items-center justify-center transition-transform duration-75 ${
            isArmed ? 'bg-red-600' : 'bg-gray-400 hover:bg-white'
          }`}
          style={{ transform: `translateX(${dragX.value}px)` }}
          onMouseDown$={handleStart}
          onTouchStart$={handleStart}
        >
          {isArmed ? (
            <svg class="w-4 h-4 text-black" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          ) : (
            <svg class="w-4 h-4 text-gray-800" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
              <path d="M9 5l7 7-7 7" />
            </svg>
          )}
        </div>
      </div>

      {isArmed && (
        <button 
          onClick$={onDisarm$}
          class="text-[9px] text-gray-500 hover:text-gray-300 underline decoration-dotted"
        >
          Disengage Safety
        </button>
      )}
    </div>
  );
});
