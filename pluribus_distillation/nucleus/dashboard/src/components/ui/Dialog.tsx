/**
 * Dialog - Material Web 3 Dialog Wrapper
 * Phase 4, Step 92 - MW3 Component Integration
 *
 * Glass-styled dialog with M3 semantics
 */

import { component$, Slot, useSignal, useVisibleTask$, $, type PropFunction } from '@builder.io/qwik';

// M3 Dialog component
import '@material/web/dialog/dialog.js';

export interface DialogProps {
  /** Whether the dialog is open */
  open?: boolean;
  /** Dialog title */
  title?: string;
  /** Optional icon for the headline */
  icon?: string;
  /** Whether the dialog can be dismissed by clicking outside */
  dismissible?: boolean;
  /** Callback when dialog is closed */
  onClose$?: PropFunction<() => void>;
  /** Additional class names */
  class?: string;
  /** Dialog type - alert, confirm, or full-screen */
  type?: 'alert' | 'confirm' | 'fullscreen';
}

/**
 * Material Web 3 Dialog with glassmorphism styling
 *
 * Usage:
 * ```tsx
 * <Dialog open={showDialog.value} title="Confirm Action" onClose$={() => showDialog.value = false}>
 *   <p>Are you sure you want to proceed?</p>
 *   <div q:slot="actions">
 *     <Button variant="text" onClick$={() => showDialog.value = false}>Cancel</Button>
 *     <Button onClick$={handleConfirm}>Confirm</Button>
 *   </div>
 * </Dialog>
 * ```
 */
export const Dialog = component$<DialogProps>((props) => {
  const dialogRef = useSignal<HTMLElement>();

  // Sync open state with the dialog element
  useVisibleTask$(({ track }) => {
    track(() => props.open);
    const dialog = dialogRef.value as any;
    if (dialog) {
      if (props.open) {
        dialog.show?.();
      } else {
        dialog.close?.();
      }
    }
  });

  const handleClosed = $(() => {
    props.onClose$?.();
  });

  const glassClasses = `
    glass-surface-overlay
    [&::part(dialog)]:bg-[var(--glass-bg-overlay)]
    [&::part(dialog)]:backdrop-blur-[var(--glass-blur-xl)]
    [&::part(dialog)]:border
    [&::part(dialog)]:border-[var(--glass-border)]
    [&::part(dialog)]:rounded-[var(--glass-radius-xl)]
    [&::part(dialog)]:shadow-[var(--glass-shadow-xl)]
    [&::part(headline)]:text-[var(--glass-text-primary)]
    [&::part(headline)]:font-semibold
    [&::part(content)]:text-[var(--glass-text-secondary)]
    [&::part(scrim)]:bg-black/60
    [&::part(scrim)]:backdrop-blur-sm
  `;

  return (
    <md-dialog
      ref={dialogRef}
      class={`${glassClasses} ${props.class || ''}`}
      open={props.open}
      type={props.type || 'alert'}
      onClosed$={handleClosed}
    >
      {props.title && (
        <div slot="headline" class="flex items-center gap-3 glass-title-neon">
          {props.icon && <span class="text-xl">{props.icon}</span>}
          <span>{props.title}</span>
        </div>
      )}

      <div slot="content" class="py-4">
        <Slot />
      </div>

      <div slot="actions" class="flex items-center gap-2 justify-end">
        <Slot name="actions" />
      </div>
    </md-dialog>
  );
});

/**
 * Confirm Dialog - Convenience wrapper for confirmation dialogs
 */
export const ConfirmDialog = component$<{
  open?: boolean;
  title?: string;
  message?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm$?: PropFunction<() => void>;
  onCancel$?: PropFunction<() => void>;
  variant?: 'default' | 'danger';
}>((props) => {
  return (
    <Dialog
      open={props.open}
      title={props.title || 'Confirm'}
      icon={props.variant === 'danger' ? '⚠️' : '❓'}
      onClose$={props.onCancel$}
    >
      <p class="text-sm">{props.message || 'Are you sure you want to proceed?'}</p>

      <div q:slot="actions" class="flex gap-2">
        <button
          onClick$={props.onCancel$}
          class="glass-interactive px-4 py-2 text-sm rounded-lg"
        >
          {props.cancelLabel || 'Cancel'}
        </button>
        <button
          onClick$={props.onConfirm$}
          class={`px-4 py-2 text-sm rounded-lg font-medium ${
            props.variant === 'danger'
              ? 'bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30'
              : 'bg-[var(--glass-accent-cyan)]/20 text-[var(--glass-accent-cyan)] border border-[var(--glass-accent-cyan)]/30 hover:bg-[var(--glass-accent-cyan)]/30'
          }`}
        >
          {props.confirmLabel || 'Confirm'}
        </button>
      </div>
    </Dialog>
  );
});

export default Dialog;
