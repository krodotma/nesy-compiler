/**
 * Breadcrumb - Navigation Breadcrumbs
 * Phase 4, Step 114 - Navigation Refinement
 *
 * Glass-styled breadcrumb navigation with M3 semantics
 */

import { component$, $, type PropFunction } from '@builder.io/qwik';

// M3 Ripple for interactive items
import '@material/web/ripple/ripple.js';

export interface BreadcrumbItem {
  /** Unique identifier */
  id: string;
  /** Display label */
  label: string;
  /** Optional icon */
  icon?: string;
  /** Navigation href (if using links) */
  href?: string;
  /** Whether this is the current/active item */
  current?: boolean;
}

export interface BreadcrumbProps {
  /** Breadcrumb items from root to current */
  items: BreadcrumbItem[];
  /** Callback when item is clicked */
  onNavigate$?: PropFunction<(itemId: string, item: BreadcrumbItem) => void>;
  /** Separator character or element */
  separator?: string;
  /** Maximum items to show (will truncate middle with ellipsis) */
  maxItems?: number;
  /** Show home icon for first item */
  showHomeIcon?: boolean;
  /** Additional class names */
  class?: string;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
}

const SIZE_CLASSES = {
  sm: 'text-[10px] gap-1',
  md: 'text-xs gap-2',
  lg: 'text-sm gap-2',
};

const ITEM_SIZE_CLASSES = {
  sm: 'px-1.5 py-0.5',
  md: 'px-2 py-1',
  lg: 'px-3 py-1.5',
};

/**
 * Breadcrumb Navigation
 *
 * Usage:
 * ```tsx
 * <Breadcrumb
 *   items={[
 *     { id: 'home', label: 'Home', icon: 'home' },
 *     { id: 'settings', label: 'Settings' },
 *     { id: 'account', label: 'Account', current: true },
 *   ]}
 *   onNavigate$={(id) => navigate(id)}
 * />
 * ```
 */
export const Breadcrumb = component$<BreadcrumbProps>((props) => {
  const separator = props.separator || '/';
  const size = props.size || 'md';
  const maxItems = props.maxItems || 0; // 0 = no limit

  // Truncate items if needed
  let displayItems = props.items;
  let truncated = false;

  if (maxItems > 2 && props.items.length > maxItems) {
    const firstItems = props.items.slice(0, 1);
    const lastItems = props.items.slice(-(maxItems - 2));
    displayItems = [...firstItems, { id: '__ellipsis__', label: '...' }, ...lastItems];
    truncated = true;
  }

  const handleClick = $((item: BreadcrumbItem, e: Event) => {
    if (item.id === '__ellipsis__' || item.current) return;

    if (!item.href) {
      e.preventDefault();
    }
    props.onNavigate$?.(item.id, item);
  });

  return (
    <nav
      aria-label="Breadcrumb"
      class={`
        flex items-center flex-wrap
        ${SIZE_CLASSES[size]}
        ${props.class || ''}
      `}
    >
      <ol class="flex items-center gap-1">
        {displayItems.map((item, index) => {
          const isLast = index === displayItems.length - 1;
          const isEllipsis = item.id === '__ellipsis__';
          const isFirst = index === 0;

          return (
            <li key={item.id} class="flex items-center">
              {/* Separator (not before first item) */}
              {index > 0 && (
                <span class="mx-1 text-[var(--glass-text-disabled)]" aria-hidden="true">
                  {separator}
                </span>
              )}

              {/* Breadcrumb item */}
              {isEllipsis ? (
                <span
                  class={`
                    ${ITEM_SIZE_CLASSES[size]}
                    text-[var(--glass-text-disabled)]
                    cursor-default
                  `}
                  aria-hidden="true"
                >
                  ...
                </span>
              ) : item.current || isLast ? (
                <span
                  class={`
                    ${ITEM_SIZE_CLASSES[size]}
                    font-medium
                    text-[var(--glass-accent-cyan)]
                    flex items-center gap-1
                  `}
                  aria-current="page"
                >
                  {item.icon && <span>{item.icon}</span>}
                  {isFirst && props.showHomeIcon && !item.icon && <span>home</span>}
                  {item.label}
                </span>
              ) : (
                <button
                  class={`
                    relative
                    ${ITEM_SIZE_CLASSES[size]}
                    rounded-md
                    text-[var(--glass-text-secondary)]
                    hover:text-[var(--glass-text-primary)]
                    hover:bg-[var(--glass-state-layer-hover)]
                    glass-transition-colors
                    flex items-center gap-1
                  `}
                  onClick$={(e) => handleClick(item, e)}
                >
                  <md-ripple class="rounded-md" />
                  {item.icon && <span>{item.icon}</span>}
                  {isFirst && props.showHomeIcon && !item.icon && <span>home</span>}
                  {item.label}
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
});

/**
 * Breadcrumb with dropdown for truncated items
 */
export const BreadcrumbWithOverflow = component$<BreadcrumbProps & {
  /** Show dropdown menu for hidden items */
  showOverflowMenu?: boolean;
}>((props) => {
  // For now, delegate to standard Breadcrumb
  // Could enhance with dropdown for truncated items
  return <Breadcrumb {...props} />;
});

/**
 * Page Header with Breadcrumb
 */
export const PageHeader = component$<{
  title: string;
  subtitle?: string;
  breadcrumbs?: BreadcrumbItem[];
  onNavigate$?: PropFunction<(id: string, item: BreadcrumbItem) => void>;
  class?: string;
}>((props) => {
  return (
    <div class={`space-y-2 ${props.class || ''}`}>
      {props.breadcrumbs && props.breadcrumbs.length > 0 && (
        <Breadcrumb
          items={props.breadcrumbs}
          onNavigate$={props.onNavigate$}
          size="sm"
        />
      )}
      <div>
        <h1 class="text-xl font-bold text-[var(--glass-text-primary)]">
          {props.title}
        </h1>
        {props.subtitle && (
          <p class="text-sm text-[var(--glass-text-tertiary)] mt-1">
            {props.subtitle}
          </p>
        )}
      </div>
    </div>
  );
});

export default Breadcrumb;
