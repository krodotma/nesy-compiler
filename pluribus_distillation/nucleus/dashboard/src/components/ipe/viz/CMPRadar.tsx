/**
 * CMPRadar.tsx
 * [Ultrathink Agent 2: Artist]
 * 
 * "Excellent work on the Jewel. Now, let's visualize the *metrics* 
 *  with the same level of fidelity. No libraries. Pure SVG math.
 *  It needs to glow. It needs to breathe."
 */

import { component$, useComputed$ } from '@builder.io/qwik';
import type { CMPVector } from '../../../lib/holon/cmp';

interface CMPRadarProps {
  current: CMPVector;
  cohort: CMPVector;
  size?: number;
}

export const CMPRadar = component$<CMPRadarProps>(({ current, cohort, size = 160 }) => {
  const center = size / 2;
  const radius = (size / 2) - 20; // Padding for labels

  const axes = ['Velocity', 'Quality', 'Stability', 'Longevity'];
  
  // Helper to map value (0-100) to coordinate
  const getPoint = (value: number, index: number, total: number) => {
    const angle = (Math.PI * 2 * index) / total - (Math.PI / 2); // Start at top
    const r = (value / 100) * radius;
    return {
      x: center + r * Math.cos(angle),
      y: center + r * Math.sin(angle)
    };
  };

  const pathData = useComputed$(() => {
    const values = [current.velocity, current.quality, current.stability, current.longevity];
    return values.map((v, i) => {
      const p = getPoint(v, i, 4);
      return `${i === 0 ? 'M' : 'L'} ${p.x},${p.y}`;
    }).join(' ') + ' Z';
  });

  const cohortPathData = useComputed$(() => {
    const values = [cohort.velocity, cohort.quality, cohort.stability, cohort.longevity];
    return values.map((v, i) => {
      const p = getPoint(v, i, 4);
      return `${i === 0 ? 'M' : 'L'} ${p.x},${p.y}`;
    }).join(' ') + ' Z';
  });

  return (
    <div class="relative flex items-center justify-center select-none">
      <svg width={size} height={size} class="overflow-visible">
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
          <linearGradient id="radarFill" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="rgba(34, 211, 238, 0.2)" /> {/* Cyan */}
            <stop offset="100%" stop-color="rgba(168, 85, 247, 0.2)" /> {/* Purple */}
          </linearGradient>
        </defs>

        {/* Grid Circles (Levels 25, 50, 75, 100) */}
        {[25, 50, 75, 100].map(level => (
          <circle
            key={level}
            cx={center}
            cy={center}
            r={(level / 100) * radius}
            fill="none"
            stroke="rgba(255,255,255,0.05)"
            stroke-dasharray="2 2"
          />
        ))}

        {/* Axes Lines */}
        {axes.map((_, i) => {
          const p = getPoint(100, i, 4);
          return (
            <line
              key={i}
              x1={center}
              y1={center}
              x2={p.x}
              y2={p.y}
              stroke="rgba(255,255,255,0.1)"
            />
          );
        })}

        {/* Axis Labels */}
        {axes.map((label, i) => {
          const p = getPoint(115, i, 4);
          return (
            <text
              key={label}
              x={p.x}
              y={p.y}
              text-anchor="middle"
              dominant-baseline="middle"
              class="text-[8px] fill-gray-500 font-mono uppercase tracking-widest"
            >
              {label}
            </text>
          );
        })}

        {/* Cohort (Ghost) Shape */}
        <path
          d={cohortPathData.value}
          fill="none"
          stroke="rgba(255,255,255,0.15)"
          stroke-width="1"
          stroke-dasharray="4 2"
        />

        {/* Active Data Shape */}
        <path
          d={pathData.value}
          fill="url(#radarFill)"
          stroke="cyan"
          stroke-width="2"
          filter="url(#glow)"
          class="drop-shadow-[0_0_8px_rgba(6,182,212,0.5)] transition-all duration-500 ease-out"
        />

        {/* Vertices */}
        {[current.velocity, current.quality, current.stability, current.longevity].map((v, i) => {
          const p = getPoint(v, i, 4);
          return (
            <circle
              key={i}
              cx={p.x}
              cy={p.y}
              r="2"
              fill="white"
              class="animate-pulse"
            />
          );
        })}
      </svg>
    </div>
  );
});
