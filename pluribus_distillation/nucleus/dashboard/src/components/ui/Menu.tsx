/**
 * Menu - Material Web 3 Menu Wrapper
 * Phase 4, Step 94 - MW3 Component Integration
 *
 * Glass-styled menus with M3 semantics
 */

import { component$, useSignal, Slot, type PropFunction, useVisibleTask$ } from '@builder.io/qwik';

// M3 Menu components
import '@material/web/menu/menu.js';
import '@material/web/menu/menu-item.js';
import '@material/web/menu/sub-menu.js';
import '@material/web/divider/divider.js';

export interface MenuItem {
  id: string;
  label: string;
  icon?: string;
  shortcut?: string;
  disabled?: boolean;
  danger?: boolean;
  children?: MenuItem[];
  dividerAfter?: boolean;
}

export interface MenuProps {
  /** Menu items */
  items: MenuItem[];
  /** Whether the menu is open */
  open?: boolean;
  /** Anchor element ID */
  anchor?: string;
  /** Callback when menu closes */
  onClose$?: PropFunction<() => void>;
  /** Callback when item is selected */
  onSelect$?: PropFunction<(itemId: string) => void>;
  /** Positioning - can be 'popover' or 'fixed' */
  positioning?: 'popover' | 'fixed';
  /** Additional class names */
  class?: string;
}

/**
 * Material Web 3 Menu with glassmorphism styling
 *
 * Usage:
 * ```tsx
 * <button id="menu-anchor">Open Menu</button>
 * <Menu
 *   anchor="menu-anchor"
 *   open={menuOpen.value}
 *   items={[
 *     { id: 'edit', label: 'Edit', icon: '✏️', shortcut: '⌘E' },
 *     { id: 'delete', label: 'Delete', icon: '🗑️', danger: true },
 *   ]}
 *   onSelect$={(id) => handleAction(id)}
 *   onClose$={() => menuOpen.value = false}
 * />
 * ```
 */
export const Menu = component$<MenuProps>((props) => {
  const menuRef = useSignal<HTMLElement>();

  const glassClasses = `
    [&::part(surface)]:bg-[var(--glass-bg-overlay)]
    [&::part(surface)]:backdrop-blur-[var(--glass-blur-xl)]
    [&::part(surface)]:border
    [&::part(surface)]:border-[var(--glass-border)]
    [&::part(surface)]:rounded-[var(--glass-radius-lg)]
    [&::part(surface)]:shadow-[var(--glass-shadow-xl)]
    [&::part(surface)]:min-w-[180px]
  `;

  const handleClose = $(() => {
    props.onClose$?.();
  });

  const handleSelect = $((itemId: string) => {
    props.onSelect$?.(itemId);
    props.onClose$?.();
  });

  return (
    <md-menu
      ref={menuRef}
      class={`${glassClasses} ${props.class || ''}`}
      anchor={props.anchor}
      open={props.open}
      positioning={props.positioning || 'popover'}
      onClosed$={handleClose}
    >
      {props.items.map((item) => (
        <div key={item.id}>
          {item.children ? (
            <md-sub-menu
              class="glass-dropdown-item"
            >
              <md-menu-item slot="item" class="glass-dropdown-item">
                {item.icon && <span slot="start" class="text-base">{item.icon}</span>}
                <span slot="headline">{item.label}</span>
                <span slot="end" class="text-muted-foreground">▸</span>
              </md-menu-item>
              <md-menu slot="menu" class={glassClasses}>
                {item.children.map((child) => (
                  <md-menu-item
                    key={child.id}
                    class={`glass-dropdown-item ${child.danger ? 'text-red-400' : ''}`}
                    disabled={child.disabled}
                    onClick$={() => handleSelect(child.id)}
                  >
                    {child.icon && <span slot="start" class="text-base">{child.icon}</span>}
                    <span slot="headline">{child.label}</span>
                    {child.shortcut && (
                      <span slot="end" class="text-[10px] text-muted-foreground font-mono">
                        {child.shortcut}
                      </span>
                    )}
                  </md-menu-item>
                ))}
              </md-menu>
            </md-sub-menu>
          ) : (
            <md-menu-item
              class={`
                glass-dropdown-item
                [&:hover]:bg-[var(--glass-state-layer-hover)]
                [&:focus]:bg-[var(--glass-state-layer-focused)]
                [&:active]:bg-[var(--glass-state-layer-pressed)]
                ${item.danger ? 'text-red-400 [&:hover]:bg-red-500/10' : ''}
              `}
              disabled={item.disabled}
              onClick$={() => handleSelect(item.id)}
            >
              {item.icon && <span slot="start" class="text-base">{item.icon}</span>}
              <span slot="headline" class={item.danger ? 'text-red-400' : ''}>{item.label}</span>
              {item.shortcut && (
                <span slot="end" class="text-[10px] text-muted-foreground font-mono">
                  {item.shortcut}
                </span>
              )}
            </md-menu-item>
          )}
          {item.dividerAfter && (
            <md-divider class="my-1 bg-[var(--glass-border)]" />
          )}
        </div>
      ))}
    </md-menu>
  );
});

/**
 * ContextMenu - Right-click context menu
 */
export const ContextMenu = component$<{
  items: MenuItem[];
  onSelect$?: PropFunction<(itemId: string) => void>;
}>((props) => {
  const open = useSignal(false);
  const position = useSignal({ x: 0, y: 0 });
  const anchorId = `context-menu-anchor-${Math.random().toString(36).slice(2)}`;

  useVisibleTask$(() => {
    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault();
      position.value = { x: e.clientX, y: e.clientY };
      open.value = true;
    };

    const handleClick = () => {
      open.value = false;
    };

    document.addEventListener('contextmenu', handleContextMenu);
    document.addEventListener('click', handleClick);

    return () => {
      document.removeEventListener('contextmenu', handleContextMenu);
      document.removeEventListener('click', handleClick);
    };
  });

  return (
    <>
      <div
        id={anchorId}
        style={{
          position: 'fixed',
          left: `${position.value.x}px`,
          top: `${position.value.y}px`,
          width: '1px',
          height: '1px',
          pointerEvents: 'none',
        }}
      />
      <Menu
        anchor={anchorId}
        items={props.items}
        open={open.value}
        onClose$={() => open.value = false}
        onSelect$={props.onSelect$}
        positioning="fixed"
      />
    </>
  );
});

export default Menu;
