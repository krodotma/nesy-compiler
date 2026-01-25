import { createContextId, type QRL, type Signal } from '@builder.io/qwik';

export type ProviderStatusMap = Record<string, { available: boolean; error?: string }>;

export interface DashboardLayoutContextValue {
  authOverlayOpen: Signal<boolean>;
  providerStatus: Signal<ProviderStatusMap>;
  flowMode: Signal<'m' | 'A'>;
  setFlowMode$: QRL<(mode: 'm' | 'A') => void>;
}

export const DashboardLayoutContext = createContextId<DashboardLayoutContextValue>('dashboard-layout-context');
