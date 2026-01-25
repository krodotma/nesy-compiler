/**
 * useTracking Hook - Bridges both tracking systems
 * 
 * Uses window globals to ensure singleton works across Qwik chunks
 */

import { useVisibleTask$ } from '@builder.io/qwik';

/**
 * Hook that registers component entry/exit with BOTH tracking systems
 * Uses window globals to ensure we're using the same instances across chunks
 */
export function useTracking(trackingId: string) {
  useVisibleTask$(({ cleanup }) => {
    // Use window globals for true singleton behavior
    const registry = (window as any).__loadingRegistry;
    const tracker = (window as any).__verboseTracker;
    
    if (registry) {
      registry.entry(trackingId);
    }
    if (tracker) {
      tracker.entry(trackingId);
    }
    
    console.log('[useTracking] entry:', trackingId);
    
    cleanup(() => {
      if (registry) {
        registry.exit(trackingId);
      }
      if (tracker) {
        tracker.exit(trackingId);
      }
    });
  });
}

/**
 * Inline tracking for non-component contexts
 */
export function trackEntry(trackingId: string): () => void {
  const registry = (window as any).__loadingRegistry;
  const tracker = (window as any).__verboseTracker;
  
  if (registry) registry.entry(trackingId);
  if (tracker) tracker.entry(trackingId);
  
  return () => {
    if (registry) registry.exit(trackingId);
    if (tracker) tracker.exit(trackingId);
  };
}

/**
 * Manual exit call
 */
export function trackExit(trackingId: string): void {
  const registry = (window as any).__loadingRegistry;
  const tracker = (window as any).__verboseTracker;
  
  if (registry) registry.exit(trackingId);
  if (tracker) tracker.exit(trackingId);
}
