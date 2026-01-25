/**
 * BottomSheet - Mobile Bottom Sheet Component
 * Phase 4, Step 117 - Menu & Navigation Refinement
 *
 * Slide-up panel for mobile contexts with drag-to-dismiss
 */

import { component$, Slot, useSignal, useVisibleTask$, type PropFunction } from '@builder.io/qwik';

export interface BottomSheetProps {
  /** Whether the sheet is open */
  open?: boolean;
  /** Callback when sheet closes */
  onClose$?: PropFunction<() => void>;
  /** Sheet title */
  title?: string;
  /** Whether to show drag handle */
  showHandle?: boolean;
  /** Snap points (percentages of viewport height) */
  snapPoints?: number[];
  /** Default snap point index */
  defaultSnapIndex?: number;
  /** Whether sheet can be dismissed by dragging */
  dismissible?: boolean;
  /** Whether to show backdrop */
  backdrop?: boolean;
  /** Additional class names */
  class?: string;
}

/**
 * Bottom Sheet with glass styling
 *
 * Usage:
 * ```tsx
 * <BottomSheet
 *   open={sheetOpen.value}
 *   title="Options"
 *   showHandle
 *   dismissible
 *   onClose$={() => sheetOpen.value = false}
 * >
 *   <div class="p-4">
 *     Sheet content here...
 *   </div>
 * </BottomSheet>
 * ```
 */
export const BottomSheet = component$<BottomSheetProps>((props) => {
  const sheetRef = useSignal<HTMLDivElement>();
  const isDragging = useSignal(false);
  const dragStartY = useSignal(0);
  const currentTranslateY = useSignal(0);
  const snapPoints = props.snapPoints || [50, 100];
  const currentSnapIndex = useSignal(props.defaultSnapIndex ?? 0);

  // Calculate current height based on snap point
  const currentHeight = snapPoints[currentSnapIndex.value] || 50;

  // Handle drag interactions
  useVisibleTask$(({ track, cleanup }) => {
    track(() => props.open);

    if (!props.open || !props.dismissible) return;

    const sheet = sheetRef.value;
    if (!sheet) return;

    let startY = 0;
    let startTranslate = 0;

    const handleTouchStart = (e: TouchEvent) => {
      if (!e.target || !(e.target as HTMLElement).closest('.bottom-sheet-handle')) {
        // Only drag from handle area
        const target = e.target as HTMLElement;
        if (!target.closest('.bottom-sheet-header')) return;
      }
      isDragging.value = true;
      startY = e.touches[0].clientY;
      startTranslate = currentTranslateY.value;
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (!isDragging.value) return;
      const deltaY = e.touches[0].clientY - startY;
      // Only allow dragging down (positive deltaY)
      currentTranslateY.value = Math.max(0, startTranslate + deltaY);
    };

    const handleTouchEnd = () => {
      if (!isDragging.value) return;
      isDragging.value = false;

      const threshold = 100;
      if (currentTranslateY.value > threshold) {
        // Dismiss
        props.onClose$?.();
        currentTranslateY.value = 0;
      } else {
        // Snap back
        currentTranslateY.value = 0;
      }
    };

    sheet.addEventListener('touchstart', handleTouchStart, { passive: true });
    sheet.addEventListener('touchmove', handleTouchMove, { passive: true });
    sheet.addEventListener('touchend', handleTouchEnd, { passive: true });

    cleanup(() => {
      sheet.removeEventListener('touchstart', handleTouchStart);
      sheet.removeEventListener('touchmove', handleTouchMove);
      sheet.removeEventListener('touchend', handleTouchEnd);
    });
  });

  // Reset translate when opened
  useVisibleTask$(({ track }) => {
    track(() => props.open);
    if (props.open) {
      currentTranslateY.value = 0;
    }
  });

  if (!props.open) return null;

  return (
    <>
      {/* Backdrop */}
      {props.backdrop !== false && (
        <div
          class={`
            fixed inset-0
            bg-black/50
            backdrop-blur-sm
            z-[90]
            glass-animate-enter
          `}
          onClick$={props.onClose$}
        />
      )}

      {/* Sheet */}
      <div
        ref={sheetRef}
        class={`
          fixed
          bottom-0
          left-0
          right-0
          z-[91]
          glass-surface-overlay
          rounded-t-3xl
          overflow-hidden
          glass-transition-transform
          ${isDragging.value ? '' : 'transition-transform duration-300'}
          ${props.class || ''}
        `}
        style={{
          height: `${currentHeight}vh`,
          transform: props.open
            ? `translateY(${currentTranslateY.value}px)`
            : 'translateY(100%)',
        }}
      >
        {/* Header */}
        <div class="bottom-sheet-header flex flex-col items-center pt-3 pb-2">
          {/* Drag handle */}
          {props.showHandle !== false && (
            <div class="bottom-sheet-handle w-12 h-1 rounded-full bg-[var(--glass-text-tertiary)] mb-3 cursor-grab active:cursor-grabbing" />
          )}

          {/* Title */}
          {props.title && (
            <div class="w-full px-4 pb-2 border-b border-[var(--glass-border)]">
              <h2 class="text-sm font-semibold text-[var(--glass-text-primary)] text-center">
                {props.title}
              </h2>
            </div>
          )}
        </div>

        {/* Content */}
        <div class="flex-grow overflow-y-auto">
          <Slot />
        </div>

        {/* Footer slot */}
        <Slot name="footer" />
      </div>
    </>
  );
});

/**
 * Bottom Sheet Action List - Common pattern for action sheets
 */
export const BottomSheetActions = component$<{
  open?: boolean;
  title?: string;
  actions: Array<{
    id: string;
    label: string;
    icon?: string;
    danger?: boolean;
    disabled?: boolean;
  }>;
  onSelect$?: PropFunction<(actionId: string) => void>;
  onClose$?: PropFunction<() => void>;
}>((props) => {
  return (
    <BottomSheet
      open={props.open}
      title={props.title}
      showHandle
      dismissible
      backdrop
      onClose$={props.onClose$}
    >
      <div class="py-2">
        {props.actions.map((action) => (
          <button
            key={action.id}
            class={`
              w-full
              flex items-center gap-4
              px-6 py-4
              text-left
              glass-transition-colors
              ${action.disabled
                ? 'opacity-50 cursor-not-allowed'
                : 'hover:bg-[var(--glass-state-layer-hover)] active:bg-[var(--glass-state-layer-pressed)]'
              }
              ${action.danger ? 'text-[var(--glass-status-error)]' : 'text-[var(--glass-text-primary)]'}
            `}
            disabled={action.disabled}
            onClick$={() => {
              if (action.disabled) return;
              props.onSelect$?.(action.id);
              props.onClose$?.();
            }}
          >
            {action.icon && (
              <span class="text-xl w-8 text-center">{action.icon}</span>
            )}
            <span class="text-base font-medium">{action.label}</span>
          </button>
        ))}
      </div>

      {/* Cancel button */}
      <div q:slot="footer" class="p-4 border-t border-[var(--glass-border)]">
        <button
          class="w-full py-3 text-center text-sm font-medium text-[var(--glass-text-secondary)] rounded-xl bg-[var(--glass-bg-card)] hover:bg-[var(--glass-bg-hover)] glass-transition-colors"
          onClick$={props.onClose$}
        >
          Cancel
        </button>
      </div>
    </BottomSheet>
  );
});

export default BottomSheet;
