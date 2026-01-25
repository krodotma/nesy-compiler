/**
 * LaneErrorBoundary - Error handling and recovery for lane components
 *
 * Phase 6, Iteration 46 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Graceful error handling
 * - Error recovery options
 * - Error reporting
 * - Fallback UI
 * - Error state persistence
 */

import {
  component$,
  useSignal,
  useStore,
  Slot,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface ErrorInfo {
  id: string;
  message: string;
  stack?: string;
  componentName?: string;
  timestamp: string;
  recovered: boolean;
  retryCount: number;
}

export interface LaneErrorBoundaryProps {
  /** Component name for error identification */
  componentName?: string;
  /** Callback when error occurs */
  onError$?: QRL<(error: ErrorInfo) => void>;
  /** Callback when recovery is attempted */
  onRecovery$?: QRL<(errorId: string, success: boolean) => void>;
  /** Max retry attempts before giving up */
  maxRetries?: number;
  /** Show detailed error info (dev mode) */
  showDetails?: boolean;
  /** Custom fallback message */
  fallbackMessage?: string;
}

// ============================================================================
// Component
// ============================================================================

export const LaneErrorBoundary = component$<LaneErrorBoundaryProps>(({
  componentName = 'Unknown',
  onError$,
  onRecovery$,
  maxRetries = 3,
  showDetails = false,
  fallbackMessage = 'Something went wrong',
}) => {
  // State
  const hasError = useSignal(false);
  const errorInfo = useStore<ErrorInfo>({
    id: '',
    message: '',
    stack: '',
    componentName: '',
    timestamp: '',
    recovered: false,
    retryCount: 0,
  });
  const isRetrying = useSignal(false);

  // Error handler
  const handleError = $(async (error: Error) => {
    const id = `err_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

    errorInfo.id = id;
    errorInfo.message = error.message;
    errorInfo.stack = error.stack;
    errorInfo.componentName = componentName;
    errorInfo.timestamp = new Date().toISOString();
    errorInfo.recovered = false;

    hasError.value = true;

    if (onError$) {
      await onError$({ ...errorInfo });
    }
  });

  // Retry handler
  const handleRetry = $(async () => {
    if (errorInfo.retryCount >= maxRetries) {
      return;
    }

    isRetrying.value = true;
    errorInfo.retryCount++;

    try {
      // Attempt recovery by resetting error state
      hasError.value = false;
      errorInfo.recovered = true;

      if (onRecovery$) {
        await onRecovery$(errorInfo.id, true);
      }
    } catch (e) {
      // Recovery failed
      hasError.value = true;
      errorInfo.recovered = false;

      if (onRecovery$) {
        await onRecovery$(errorInfo.id, false);
      }
    } finally {
      isRetrying.value = false;
    }
  });

  // Reset handler
  const handleReset = $(() => {
    hasError.value = false;
    errorInfo.id = '';
    errorInfo.message = '';
    errorInfo.stack = '';
    errorInfo.recovered = false;
    errorInfo.retryCount = 0;
  });

  // Copy error details
  const copyErrorDetails = $(() => {
    const details = JSON.stringify({
      component: errorInfo.componentName,
      message: errorInfo.message,
      stack: errorInfo.stack,
      timestamp: errorInfo.timestamp,
    }, null, 2);
    navigator.clipboard.writeText(details);
  });

  if (hasError.value) {
    return (
      <div class="rounded-lg border border-red-500/30 bg-red-500/10 p-4">
        {/* Error icon and message */}
        <div class="flex items-start gap-3">
          <div class="text-2xl text-red-400">⚠</div>
          <div class="flex-grow">
            <h3 class="text-sm font-medium text-red-400 mb-1">
              {fallbackMessage}
            </h3>
            <p class="text-[10px] text-red-300/70">
              Error in {componentName}
            </p>
          </div>
        </div>

        {/* Error details (dev mode) */}
        {showDetails && (
          <div class="mt-4 p-3 rounded bg-black/30 border border-red-500/20">
            <div class="flex items-center justify-between mb-2">
              <span class="text-[9px] font-semibold text-red-400">ERROR DETAILS</span>
              <button
                onClick$={copyErrorDetails}
                class="text-[9px] px-2 py-0.5 rounded bg-red-500/20 text-red-300 hover:bg-red-500/30"
              >
                Copy
              </button>
            </div>
            <div class="text-[10px] font-mono text-red-300 mb-2">
              {errorInfo.message}
            </div>
            {errorInfo.stack && (
              <pre class="text-[8px] font-mono text-red-300/50 max-h-24 overflow-y-auto whitespace-pre-wrap">
                {errorInfo.stack}
              </pre>
            )}
            <div class="mt-2 text-[8px] text-red-300/50">
              ID: {errorInfo.id} | Time: {errorInfo.timestamp}
            </div>
          </div>
        )}

        {/* Recovery options */}
        <div class="mt-4 flex items-center gap-2">
          <button
            onClick$={handleRetry}
            disabled={isRetrying.value || errorInfo.retryCount >= maxRetries}
            class="px-3 py-1.5 text-[10px] rounded bg-red-500/20 text-red-300 hover:bg-red-500/30 disabled:opacity-50 transition-colors"
          >
            {isRetrying.value ? 'Retrying...' : `Retry (${maxRetries - errorInfo.retryCount} left)`}
          </button>
          <button
            onClick$={handleReset}
            class="px-3 py-1.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
          >
            Reset
          </button>
        </div>

        {/* Retry exhausted message */}
        {errorInfo.retryCount >= maxRetries && (
          <div class="mt-3 text-[9px] text-red-300/70">
            Maximum retry attempts reached. Please refresh the page or contact support.
          </div>
        )}
      </div>
    );
  }

  return <Slot />;
});

// ============================================================================
// Error Fallback Component
// ============================================================================

export interface ErrorFallbackProps {
  error?: Error;
  componentName?: string;
  onRetry$?: QRL<() => void>;
}

export const ErrorFallback = component$<ErrorFallbackProps>(({
  error,
  componentName = 'Component',
  onRetry$,
}) => {
  return (
    <div class="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-center">
      <div class="text-2xl text-red-400 mb-2">⚠</div>
      <h3 class="text-sm font-medium text-red-400 mb-1">
        Failed to load {componentName}
      </h3>
      {error && (
        <p class="text-[10px] text-red-300/70 mb-3">
          {error.message}
        </p>
      )}
      {onRetry$ && (
        <button
          onClick$={onRetry$}
          class="px-3 py-1.5 text-[10px] rounded bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-colors"
        >
          Try Again
        </button>
      )}
    </div>
  );
});

// ============================================================================
// Loading Fallback Component
// ============================================================================

export interface LoadingFallbackProps {
  componentName?: string;
  message?: string;
}

export const LoadingFallback = component$<LoadingFallbackProps>(({
  componentName = 'Content',
  message,
}) => {
  return (
    <div class="rounded-lg border border-border/30 bg-muted/10 p-4 text-center animate-pulse">
      <div class="w-8 h-8 mx-auto mb-2 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
      <p class="text-[10px] text-muted-foreground">
        {message || `Loading ${componentName}...`}
      </p>
    </div>
  );
});

// ============================================================================
// Empty State Component
// ============================================================================

export interface EmptyStateProps {
  title?: string;
  message?: string;
  icon?: string;
  actionLabel?: string;
  onAction$?: QRL<() => void>;
}

export const EmptyState = component$<EmptyStateProps>(({
  title = 'No data',
  message = 'There is no data to display',
  icon = '📭',
  actionLabel,
  onAction$,
}) => {
  return (
    <div class="rounded-lg border border-border/30 bg-muted/5 p-8 text-center">
      <div class="text-3xl mb-3">{icon}</div>
      <h3 class="text-sm font-medium text-foreground mb-1">{title}</h3>
      <p class="text-[10px] text-muted-foreground mb-4">{message}</p>
      {actionLabel && onAction$ && (
        <button
          onClick$={onAction$}
          class="px-4 py-2 text-[10px] rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
});

export default LaneErrorBoundary;
