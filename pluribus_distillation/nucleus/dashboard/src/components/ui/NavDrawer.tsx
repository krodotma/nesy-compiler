/**
 * NavDrawer - Navigation Drawer Component
 * Phase 4, Step 105 - Progressive Disclosure Implementation
 *
 * Glass-styled navigation drawer with nested navigation support
 */

import { component$, Slot, useSignal, type PropFunction } from '@builder.io/qwik';

// M3 Components
import '@material/web/list/list.js';
import '@material/web/list/list-item.js';
import '@material/web/divider/divider.js';
import '@material/web/ripple/ripple.js';

export interface NavItem {
  id: string;
  label: string;
  icon?: string;
  badge?: string | number;
  badgeColor?: 'cyan' | 'magenta' | 'amber' | 'error';
  disabled?: boolean;
  children?: NavItem[];
  dividerAfter?: boolean;
  href?: string;
}

export interface NavDrawerProps {
  /** Navigation items */
  items: NavItem[];
  /** Currently active item ID */
  activeId?: string;
  /** Callback when item is selected */
  onSelect$?: PropFunction<(itemId: string) => void>;
  /** Whether the drawer is open (for mobile/overlay mode) */
  open?: boolean;
  /** Callback when drawer should close */
  onClose$?: PropFunction<() => void>;
  /** Drawer mode */
  mode?: 'permanent' | 'dismissible' | 'modal';
  /** Drawer position */
  position?: 'left' | 'right';
  /** Header content */
  header?: string;
  /** Footer content */
  footer?: string;
  /** Additional class names */
  class?: string;
}

const BADGE_COLORS: Record<string, string> = {
  cyan: 'bg-[var(--glass-accent-cyan)]/20 text-[var(--glass-accent-cyan)]',
  magenta: 'bg-[var(--glass-accent-magenta)]/20 text-[var(--glass-accent-magenta)]',
  amber: 'bg-[var(--glass-accent-amber)]/20 text-[var(--glass-accent-amber)]',
  error: 'bg-[var(--glass-status-error)]/20 text-[var(--glass-status-error)]',
};

/**
 * Navigation Drawer with glass styling
 *
 * Usage:
 * ```tsx
 * <NavDrawer
 *   items={[
 *     { id: 'home', label: 'Home', icon: '🏠' },
 *     { id: 'settings', label: 'Settings', icon: '⚙️', children: [
 *       { id: 'account', label: 'Account' },
 *       { id: 'privacy', label: 'Privacy' },
 *     ]},
 *   ]}
 *   activeId={currentView.value}
 *   onSelect$={(id) => currentView.value = id}
 * />
 * ```
 */
export const NavDrawer = component$<NavDrawerProps>((props) => {
  const mode = props.mode || 'permanent';
  const position = props.position || 'left';
  const expandedGroups = useSignal<Set<string>>(new Set());

  const toggleGroup = $((id: string) => {
    const newSet = new Set(expandedGroups.value);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    expandedGroups.value = newSet;
  });

  const handleSelect = $((id: string) => {
    props.onSelect$?.(id);
    if (mode === 'modal' && props.onClose$) {
      props.onClose$();
    }
  });

  const renderItem = (item: NavItem, depth = 0) => {
    const isActive = props.activeId === item.id;
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedGroups.value.has(item.id);
    const indent = depth * 12;

    return (
      <div key={item.id}>
        <button
          class={`
            relative
            w-full
            flex items-center gap-3
            px-4 py-3
            text-left
            rounded-xl
            glass-transition-all
            ${isActive
              ? 'bg-[var(--glass-accent-cyan)]/15 text-[var(--glass-accent-cyan)]'
              : 'hover:bg-[var(--glass-state-layer-hover)] text-[var(--glass-text-secondary)]'
            }
            ${item.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
          style={{ paddingLeft: `${16 + indent}px` }}
          onClick$={() => {
            if (item.disabled) return;
            if (hasChildren) {
              toggleGroup(item.id);
            } else {
              handleSelect(item.id);
            }
          }}
          disabled={item.disabled}
        >
          <md-ripple class="rounded-xl" />

          {/* Icon */}
          {item.icon && (
            <span class="text-lg flex-shrink-0 w-6 text-center">{item.icon}</span>
          )}

          {/* Label */}
          <span class="flex-grow text-sm font-medium truncate">{item.label}</span>

          {/* Badge */}
          {item.badge !== undefined && (
            <span
              class={`
                px-1.5 py-0.5
                text-[10px]
                rounded-full
                ${BADGE_COLORS[item.badgeColor || 'cyan']}
              `}
            >
              {item.badge}
            </span>
          )}

          {/* Expand indicator for groups */}
          {hasChildren && (
            <span
              class={`
                text-[var(--glass-text-tertiary)]
                glass-transition-transform
                ${isExpanded ? 'rotate-90' : ''}
              `}
            >
              ▸
            </span>
          )}
        </button>

        {/* Children */}
        {hasChildren && (
          <div
            class={`
              grid
              glass-transition-all
              ${isExpanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}
            `}
          >
            <div class="overflow-hidden">
              {item.children!.map(child => renderItem(child, depth + 1))}
            </div>
          </div>
        )}

        {/* Divider */}
        {item.dividerAfter && (
          <div class="h-px bg-[var(--glass-border)] my-2 mx-4" />
        )}
      </div>
    );
  };

  const drawerContent = (
    <div
      class={`
        h-full
        flex flex-col
        glass-surface-elevated
        ${props.class || ''}
      `}
    >
      {/* Header */}
      {props.header && (
        <div class="flex-shrink-0 p-4 border-b border-[var(--glass-border)]">
          <h2 class="text-sm font-bold text-[var(--glass-text-primary)] uppercase tracking-wider">
            {props.header}
          </h2>
        </div>
      )}
      <Slot name="header" />

      {/* Navigation items */}
      <nav class="flex-grow overflow-y-auto p-2 space-y-1">
        {props.items.map(item => renderItem(item))}
      </nav>

      {/* Footer */}
      {props.footer && (
        <div class="flex-shrink-0 p-4 border-t border-[var(--glass-border)]">
          <p class="text-xs text-[var(--glass-text-tertiary)]">{props.footer}</p>
        </div>
      )}
      <Slot name="footer" />
    </div>
  );

  // Modal mode with backdrop
  if (mode === 'modal') {
    return (
      <>
        {/* Backdrop */}
        {props.open && (
          <div
            class="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 glass-animate-enter"
            onClick$={props.onClose$}
          />
        )}

        {/* Drawer */}
        <div
          class={`
            fixed top-0 bottom-0
            ${position === 'left' ? 'left-0' : 'right-0'}
            w-72
            z-50
            glass-transition-transform
            ${props.open
              ? 'translate-x-0'
              : position === 'left' ? '-translate-x-full' : 'translate-x-full'
            }
          `}
        >
          {drawerContent}
        </div>
      </>
    );
  }

  // Dismissible mode (can be toggled)
  if (mode === 'dismissible') {
    return (
      <div
        class={`
          w-72
          flex-shrink-0
          glass-transition-all
          ${props.open ? 'ml-0' : position === 'left' ? '-ml-72' : '-mr-72'}
        `}
      >
        {drawerContent}
      </div>
    );
  }

  // Permanent mode (always visible)
  return (
    <div class="w-72 flex-shrink-0">
      {drawerContent}
    </div>
  );
});

/**
 * Navigation Rail - Compact vertical navigation
 */
export const NavRail = component$<{
  items: Array<{
    id: string;
    icon: string;
    label: string;
    badge?: number;
  }>;
  activeId?: string;
  onSelect$?: PropFunction<(id: string) => void>;
  class?: string;
}>((props) => {
  return (
    <div
      class={`
        w-20
        flex-shrink-0
        glass-surface-elevated
        flex flex-col items-center
        py-4
        gap-2
        ${props.class || ''}
      `}
    >
      {props.items.map(item => {
        const isActive = props.activeId === item.id;
        return (
          <button
            key={item.id}
            class={`
              relative
              w-16 h-14
              flex flex-col items-center justify-center
              gap-1
              rounded-2xl
              glass-transition-all
              ${isActive
                ? 'bg-[var(--glass-accent-cyan)]/15'
                : 'hover:bg-[var(--glass-state-layer-hover)]'
              }
            `}
            onClick$={() => props.onSelect$?.(item.id)}
          >
            <md-ripple class="rounded-2xl" />

            <span class="text-xl relative">
              {item.icon}
              {item.badge !== undefined && item.badge > 0 && (
                <span class="absolute -top-1 -right-2 w-4 h-4 flex items-center justify-center text-[8px] rounded-full bg-[var(--glass-status-error)] text-white">
                  {item.badge > 99 ? '99+' : item.badge}
                </span>
              )}
            </span>

            <span
              class={`
                text-[10px]
                ${isActive ? 'text-[var(--glass-accent-cyan)]' : 'text-[var(--glass-text-tertiary)]'}
              `}
            >
              {item.label}
            </span>
          </button>
        );
      })}
    </div>
  );
});

export default NavDrawer;
