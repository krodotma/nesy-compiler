/**
 * Toast / Snackbar - Notification System
 * Phase 4, Step 98 - MW3 Component Integration
 *
 * Glass-styled toast notifications
 */

import { component$, useSignal, useVisibleTask$, type PropFunction, createContextId, useContext, useContextProvider } from '@builder.io/qwik';

// Note: M3 snackbar is not yet stable in @material/web, so we build a custom glass version

export type ToastVariant = 'info' | 'success' | 'warning' | 'error';
export type ToastPosition = 'top' | 'top-left' | 'top-right' | 'bottom' | 'bottom-left' | 'bottom-right';

export interface ToastProps {
  /** Toast message */
  message: string;
  /** Toast variant/severity */
  variant?: ToastVariant;
  /** Duration in ms (0 = permanent) */
  duration?: number;
  /** Action button text */
  action?: string;
  /** Action callback */
  onAction$?: PropFunction<() => void>;
  /** Close callback */
  onClose$?: PropFunction<() => void>;
  /** Whether to show close button */
  closable?: boolean;
  /** Icon override */
  icon?: string;
  /** Additional class names */
  class?: string;
}

export interface ToastItem extends ToastProps {
  id: string;
  timestamp: number;
}

export interface ToastContextType {
  toasts: ToastItem[];
  show: (toast: Omit<ToastItem, 'id' | 'timestamp'>) => string;
  dismiss: (id: string) => void;
  dismissAll: () => void;
}

export const ToastContext = createContextId<ToastContextType>('toast-context');

const VARIANT_STYLES: Record<ToastVariant, { bg: string; border: string; icon: string; iconColor: string }> = {
  info: {
    bg: 'rgba(0, 255, 255, 0.1)',
    border: 'rgba(0, 255, 255, 0.3)',
    icon: 'info',
    iconColor: 'var(--glass-accent-cyan)',
  },
  success: {
    bg: 'rgba(46, 213, 115, 0.1)',
    border: 'rgba(46, 213, 115, 0.3)',
    icon: 'check_circle',
    iconColor: 'var(--glass-status-success)',
  },
  warning: {
    bg: 'rgba(255, 193, 7, 0.1)',
    border: 'rgba(255, 193, 7, 0.3)',
    icon: 'warning',
    iconColor: 'var(--glass-status-warning)',
  },
  error: {
    bg: 'rgba(255, 71, 87, 0.1)',
    border: 'rgba(255, 71, 87, 0.3)',
    icon: 'error',
    iconColor: 'var(--glass-status-error)',
  },
};

const ICON_MAP: Record<ToastVariant, string> = {
  info: 'ℹ️',
  success: '✓',
  warning: '⚠️',
  error: '✕',
};

const POSITION_CLASSES: Record<ToastPosition, string> = {
  'top': 'top-4 left-1/2 -translate-x-1/2',
  'top-left': 'top-4 left-4',
  'top-right': 'top-4 right-4',
  'bottom': 'bottom-4 left-1/2 -translate-x-1/2',
  'bottom-left': 'bottom-4 left-4',
  'bottom-right': 'bottom-4 right-4',
};

/**
 * Single Toast component
 */
export const Toast = component$<ToastProps & { onDismiss$?: PropFunction<() => void> }>((props) => {
  const variant = props.variant || 'info';
  const styles = VARIANT_STYLES[variant];
  const icon = props.icon || ICON_MAP[variant];
  const visible = useSignal(true);
  const exiting = useSignal(false);

  useVisibleTask$(({ cleanup }) => {
    if (props.duration && props.duration > 0) {
      const timer = setTimeout(() => {
        exiting.value = true;
        setTimeout(() => {
          visible.value = false;
          props.onClose$?.();
        }, 300);
      }, props.duration);

      cleanup(() => clearTimeout(timer));
    }
  });

  const handleClose = $(() => {
    exiting.value = true;
    setTimeout(() => {
      visible.value = false;
      props.onClose$?.();
    }, 300);
  });

  if (!visible.value) return null;

  return (
    <div
      class={`
        flex items-center gap-3
        px-4 py-3
        min-w-[280px] max-w-[480px]
        rounded-xl
        border
        backdrop-blur-[var(--glass-blur-lg)]
        shadow-lg
        glass-transition-all
        ${exiting.value ? 'opacity-0 translate-y-2' : 'opacity-100 translate-y-0'}
        ${props.class || ''}
      `}
      style={{
        backgroundColor: styles.bg,
        borderColor: styles.border,
        boxShadow: `0 0 20px ${styles.border}, var(--glass-shadow)`,
      }}
      role="alert"
    >
      {/* Icon */}
      <div
        class="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-full text-sm font-bold"
        style={{ color: styles.iconColor }}
      >
        {icon}
      </div>

      {/* Message */}
      <div class="flex-grow text-sm text-[var(--glass-text-primary)]">
        {props.message}
      </div>

      {/* Action button */}
      {props.action && (
        <button
          class="flex-shrink-0 text-xs font-medium uppercase tracking-wide px-2 py-1 rounded hover:bg-white/10 transition-colors"
          style={{ color: styles.iconColor }}
          onClick$={props.onAction$}
        >
          {props.action}
        </button>
      )}

      {/* Close button */}
      {props.closable !== false && (
        <button
          class="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-full hover:bg-white/10 transition-colors text-[var(--glass-text-tertiary)]"
          onClick$={handleClose}
          aria-label="Close"
        >
          ✕
        </button>
      )}
    </div>
  );
});

/**
 * Toast Container - Renders all active toasts
 */
export const ToastContainer = component$<{
  position?: ToastPosition;
  maxToasts?: number;
}>((props) => {
  const position = props.position || 'bottom-right';
  const maxToasts = props.maxToasts || 5;
  const ctx = useContext(ToastContext);

  const visibleToasts = ctx.toasts.slice(-maxToasts);
  const isTop = position.startsWith('top');

  return (
    <div
      class={`
        fixed z-[9999]
        flex flex-col gap-2
        pointer-events-none
        ${POSITION_CLASSES[position]}
        ${isTop ? 'flex-col' : 'flex-col-reverse'}
      `}
    >
      {visibleToasts.map((toast) => (
        <div key={toast.id} class="pointer-events-auto glass-animate-enter">
          <Toast
            message={toast.message}
            variant={toast.variant}
            duration={toast.duration}
            action={toast.action}
            onAction$={toast.onAction$}
            closable={toast.closable}
            icon={toast.icon}
            onClose$={() => ctx.dismiss(toast.id)}
          />
        </div>
      ))}
    </div>
  );
});

/**
 * Toast Provider - Provides toast context to children
 *
 * Usage:
 * ```tsx
 * // In layout or root
 * <ToastProvider>
 *   <App />
 * </ToastProvider>
 *
 * // In any child component
 * const toast = useToast();
 * toast.show({ message: 'Hello!', variant: 'success' });
 * ```
 */
export const ToastProvider = component$<{
  position?: ToastPosition;
  defaultDuration?: number;
}>((props) => {
  const toasts = useSignal<ToastItem[]>([]);

  const show = $((toast: Omit<ToastItem, 'id' | 'timestamp'>) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const newToast: ToastItem = {
      ...toast,
      id,
      timestamp: Date.now(),
      duration: toast.duration ?? props.defaultDuration ?? 5000,
    };
    toasts.value = [...toasts.value, newToast];
    return id;
  });

  const dismiss = $((id: string) => {
    toasts.value = toasts.value.filter(t => t.id !== id);
  });

  const dismissAll = $(() => {
    toasts.value = [];
  });

  const contextValue: ToastContextType = {
    toasts: toasts.value,
    show,
    dismiss,
    dismissAll,
  };

  useContextProvider(ToastContext, contextValue);

  return (
    <>
      <ToastContainer position={props.position} />
    </>
  );
});

/**
 * useToast hook - Access toast functions
 */
export const useToast = () => {
  return useContext(ToastContext);
};

/**
 * Standalone toast function (for non-context usage)
 */
export const showToast = (props: ToastProps) => {
  // This would need a global toast container to work
  // For now, emit a custom event that a container can listen to
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('pluribus:toast', { detail: props }));
  }
};

export default Toast;
