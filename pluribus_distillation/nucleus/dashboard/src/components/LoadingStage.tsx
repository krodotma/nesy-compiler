/**
 * LoadingStage - Accessible loading wrapper with glassmorphism transitions
 *
 * Wraps components to provide:
 * - Loading state tracking via window.__verboseTracker
 * - Fade-in animation on mount
 * - ARIA live region for screen readers
 * - Cognitive boot loader integration
 *
 * @version 2.0.0 - Enhanced accessibility + transitions
 */

import { component$, Slot, useVisibleTask$, useSignal } from "@builder.io/qwik";
import { trackEntry } from "../lib/telemetry/use-tracking";
import { CognitiveBootLoader } from "./boot/CognitiveBootLoader";

interface LoadingStageProps {
  /** Unique identifier for tracking */
  id: string;
  /** Optional accessible label */
  ariaLabel?: string;
  /** Disable fade animation */
  noAnimation?: boolean;
}

export const LoadingStage = component$<LoadingStageProps>(({
  id,
  ariaLabel,
  noAnimation = false
}) => {
  const isLoaded = useSignal(false);

  useVisibleTask$(() => {
    const stopTracking = trackEntry(id);
    // Small delay to allow render, then mark complete
    requestAnimationFrame(() => {
      stopTracking();
      // Trigger fade-in after load
      isLoaded.value = true;
    });
  });

  const animationStyle = noAnimation ? {} : {
    opacity: isLoaded.value ? 1 : 0,
    transform: isLoaded.value ? 'translateY(0)' : 'translateY(8px)',
    transition: 'opacity 300ms cubic-bezier(0.05, 0.7, 0.1, 1), transform 300ms cubic-bezier(0.05, 0.7, 0.1, 1)',
  };

  return (
    <div
      style={animationStyle}
      role="region"
      aria-label={ariaLabel || `Loading stage: ${id}`}
      aria-busy={!isLoaded.value}
    >
      {id === 'comp:supermotd' && <CognitiveBootLoader />}
      <Slot />
    </div>
  );
});
