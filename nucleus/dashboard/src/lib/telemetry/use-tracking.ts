/**
 * useTracking Hook - Auto-registers component lifecycle with verbose tracker
 * 
 * Usage in any component:
 * 
 * import { useTracking } from '../lib/telemetry/use-tracking';
 * 
 * export const MyComponent = component$(() => {
 *   useTracking('comp:my-component');
 *   // ... rest of component
 * });
 */

import { useVisibleTask$ } from '@builder.io/qwik';
import { tracker } from './verbose-tracker';

/**
 * Hook that registers component entry/exit with the loading tracker
 * @param trackingId - ID from TRACKING_MANIFEST (e.g., 'comp:header', 'sys:auth')
 */
export function useTracking(trackingId: string) {
    useVisibleTask$(({ cleanup }) => {
        // Entry - component is now visible/active
        tracker.entry(trackingId);

        // Exit - component unmounts or cleanup
        cleanup(() => {
            tracker.exit(trackingId);
        });
    });
}

/**
 * Inline tracking for non-component contexts (flows, subsystems)
 * Returns cleanup function
 */
export function trackEntry(trackingId: string): () => void {
    tracker.entry(trackingId);
    return () => tracker.exit(trackingId);
}

/**
 * Manual exit call for flows that complete asynchronously
 */
export function trackExit(trackingId: string): void {
    tracker.exit(trackingId);
}
