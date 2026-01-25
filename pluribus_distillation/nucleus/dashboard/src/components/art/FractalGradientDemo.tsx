/**
 * FractalGradientDemo - Example Integration of Fractal Gradient Backgrounds
 * ==========================================================================
 *
 * Shows how to use FractalGradientCanvas in various contexts.
 */

import { component$ } from '@builder.io/qwik';
import { FractalGradientCanvas, FractalGradientSection, FractalGradientPresets } from './FractalGradientCanvas';

/**
 * Example: Card with fractal background
 */
export const FractalCard = component$<{ title: string; children?: any }>((props) => {
  return (
    <div class="relative overflow-hidden rounded-xl glass-surface-elevated">
      <FractalGradientCanvas
        {...FractalGradientPresets.card}
        height={200}
      />
      <div class="relative z-10 p-6">
        <h3 class="glass-title-neon text-sm mb-4">{props.title}</h3>
        {props.children}
      </div>
    </div>
  );
});

/**
 * Example: Panel with fractal background
 */
export const FractalPanel = component$<{ title: string; children?: any }>((props) => {
  return (
    <div class="relative overflow-hidden rounded-lg glass-surface">
      <FractalGradientCanvas
        {...FractalGradientPresets.panel}
        height="100%"
      />
      <div class="relative z-10 p-4">
        <div class="flex items-center gap-2 mb-3">
          <span class="glass-title-neon">{props.title}</span>
          <div class="flex-1 h-px bg-gradient-to-r from-[var(--glass-accent-cyan)]/40 to-transparent" />
        </div>
        {props.children}
      </div>
    </div>
  );
});

/**
 * Example: Hero section with fractal background
 */
export const FractalHero = component$<{ children?: any }>((props) => {
  return (
    <div class="relative overflow-hidden min-h-[400px]">
      <FractalGradientCanvas
        {...FractalGradientPresets.hero}
        interactive={true}
      />
      <div class="relative z-10 flex items-center justify-center min-h-[400px] p-8">
        {props.children}
      </div>
    </div>
  );
});

/**
 * Example: Modal backdrop with fractal
 */
export const FractalModalBackdrop = component$<{ children?: any }>((props) => {
  return (
    <div class="fixed inset-0 flex items-center justify-center">
      <div class="absolute inset-0 bg-[var(--glass-bg-dark)]/80 backdrop-blur-lg">
        <FractalGradientCanvas
          {...FractalGradientPresets.modal}
        />
      </div>
      <div class="relative z-10 glass-surface-elevated rounded-xl p-6 m-4 max-w-2xl w-full">
        {props.children}
      </div>
    </div>
  );
});

/**
 * Demo component showing all variants
 */
export const FractalGradientShowcase = component$(() => {
  return (
    <div class="space-y-8 p-8">
      <h1 class="glass-title-neon text-xl">Fractal Gradient Showcase</h1>

      {/* Card examples */}
      <section>
        <h2 class="glass-title-neon-magenta text-sm mb-4">Cards</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <FractalCard title="NEON CARD">
            <p class="text-[var(--glass-text-secondary)]">
              Beautiful fractal gradient with neon palette.
            </p>
          </FractalCard>

          <div class="relative overflow-hidden rounded-xl glass-surface-elevated">
            <FractalGradientCanvas
              numLayers={12}
              palette="cool"
              opacity={0.2}
              height={200}
            />
            <div class="relative z-10 p-6">
              <h3 class="glass-title-neon text-sm mb-4">COOL CARD</h3>
              <p class="text-[var(--glass-text-secondary)]">
                Cool blue palette variant.
              </p>
            </div>
          </div>

          <div class="relative overflow-hidden rounded-xl glass-surface-elevated">
            <FractalGradientCanvas
              numLayers={12}
              palette="warm"
              opacity={0.2}
              height={200}
            />
            <div class="relative z-10 p-6">
              <h3 class="glass-title-neon-amber text-sm mb-4">WARM CARD</h3>
              <p class="text-[var(--glass-text-secondary)]">
                Warm orange/gold palette.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Panel example */}
      <section>
        <h2 class="glass-title-neon-magenta text-sm mb-4">Panels</h2>
        <FractalPanel title="SYSTEM STATUS">
          <div class="grid grid-cols-2 gap-4">
            <div class="glass-surface-subtle rounded p-3">
              <span class="glass-text-muted text-xs">CPU</span>
              <div class="text-lg font-mono">24%</div>
            </div>
            <div class="glass-surface-subtle rounded p-3">
              <span class="glass-text-muted text-xs">MEMORY</span>
              <div class="text-lg font-mono">68%</div>
            </div>
          </div>
        </FractalPanel>
      </section>

      {/* Interactive hero */}
      <section>
        <h2 class="glass-title-neon-magenta text-sm mb-4">Interactive Hero (Click to Regenerate)</h2>
        <div class="relative overflow-hidden rounded-xl min-h-[200px]">
          <FractalGradientCanvas
            numLayers={25}
            palette="neon"
            opacity={0.4}
            interactive={true}
          />
          <div class="relative z-10 flex items-center justify-center min-h-[200px]">
            <div class="text-center">
              <h3 class="text-2xl font-bold glass-text-gradient">PLURIBUS</h3>
              <p class="glass-text-muted text-sm mt-2">Click background to regenerate</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
});

export default FractalGradientShowcase;
