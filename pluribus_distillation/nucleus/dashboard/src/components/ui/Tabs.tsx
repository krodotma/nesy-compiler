/**
 * Tabs - Material Web 3 Tabs Wrapper
 * Phase 4, Step 93 - MW3 Component Integration
 *
 * Glass-styled tabs with M3 semantics
 */

import { component$, useSignal, Slot, type PropFunction } from '@builder.io/qwik';

// M3 Tabs components
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';
import '@material/web/tabs/secondary-tab.js';

export interface TabItem {
  id: string;
  label: string;
  icon?: string;
  badge?: string | number;
  disabled?: boolean;
}

export interface TabsProps {
  /** Tab items */
  tabs: TabItem[];
  /** Currently active tab ID */
  activeTab?: string;
  /** Callback when tab changes */
  onTabChange$?: PropFunction<(tabId: string) => void>;
  /** Tab variant - primary (underline) or secondary (pill) */
  variant?: 'primary' | 'secondary';
  /** Additional class names */
  class?: string;
  /** Full width tabs */
  fullWidth?: boolean;
}

/**
 * Material Web 3 Tabs with glassmorphism styling
 *
 * Usage:
 * ```tsx
 * <Tabs
 *   tabs={[
 *     { id: 'overview', label: 'Overview', icon: 'home' },
 *     { id: 'details', label: 'Details', icon: 'info' },
 *     { id: 'history', label: 'History', icon: 'history' }
 *   ]}
 *   activeTab={activeTab.value}
 *   onTabChange$={(id) => activeTab.value = id}
 * />
 * ```
 */
export const Tabs = component$<TabsProps>((props) => {
  const variant = props.variant || 'primary';

  const glassClasses = `
    glass-surface
    rounded-xl
    p-1
    [&::part(divider)]:bg-[var(--glass-border)]
    [&::part(divider)]:h-[1px]
  `;

  const handleTabChange = $((e: Event) => {
    const target = e.target as HTMLElement;
    const tabs = target.closest('md-tabs');
    if (tabs) {
      const activeIndex = (tabs as any).activeTabIndex;
      const tab = props.tabs[activeIndex];
      if (tab && props.onTabChange$) {
        props.onTabChange$(tab.id);
      }
    }
  });

  const activeIndex = props.tabs.findIndex(t => t.id === props.activeTab);

  return (
    <md-tabs
      class={`${glassClasses} ${props.fullWidth ? 'w-full' : ''} ${props.class || ''}`}
      active-tab-index={activeIndex >= 0 ? activeIndex : 0}
      onChange$={handleTabChange}
    >
      {props.tabs.map((tab) => (
        variant === 'primary' ? (
          <md-primary-tab
            key={tab.id}
            class={`
              glass-transition-colors
              [&::part(content)]:text-[var(--glass-text-secondary)]
              [&[active]::part(content)]:text-[var(--glass-accent-cyan)]
              [&::part(indicator)]:bg-[var(--glass-accent-cyan)]
              [&::part(indicator)]:h-[2px]
              [&::part(indicator)]:rounded-full
              [&:hover::part(content)]:text-[var(--glass-text-primary)]
            `}
            aria-controls={`panel-${tab.id}`}
            disabled={tab.disabled}
          >
            {tab.icon && (
              <span slot="icon" class="text-lg">{tab.icon}</span>
            )}
            <span class="flex items-center gap-2">
              {tab.label}
              {tab.badge !== undefined && (
                <span class="px-1.5 py-0.5 text-[10px] rounded-full bg-[var(--glass-accent-cyan)]/20 text-[var(--glass-accent-cyan)]">
                  {tab.badge}
                </span>
              )}
            </span>
          </md-primary-tab>
        ) : (
          <md-secondary-tab
            key={tab.id}
            class={`
              glass-interactive
              rounded-lg
              mx-0.5
              [&::part(content)]:text-[var(--glass-text-secondary)]
              [&[active]::part(content)]:text-[var(--glass-text-primary)]
              [&[active]]:bg-[var(--glass-bg-active)]
            `}
            aria-controls={`panel-${tab.id}`}
            disabled={tab.disabled}
          >
            {tab.icon && (
              <span slot="icon" class="text-lg">{tab.icon}</span>
            )}
            <span class="flex items-center gap-2">
              {tab.label}
              {tab.badge !== undefined && (
                <span class="px-1.5 py-0.5 text-[10px] rounded-full bg-[var(--glass-accent-cyan)]/20 text-[var(--glass-accent-cyan)]">
                  {tab.badge}
                </span>
              )}
            </span>
          </md-secondary-tab>
        )
      ))}
    </md-tabs>
  );
});

/**
 * TabPanel - Content container for a tab
 */
export const TabPanel = component$<{
  id: string;
  activeTab?: string;
  class?: string;
}>((props) => {
  const isActive = props.id === props.activeTab;

  return (
    <div
      id={`panel-${props.id}`}
      role="tabpanel"
      aria-labelledby={props.id}
      hidden={!isActive}
      class={`
        glass-animate-enter
        ${isActive ? 'block' : 'hidden'}
        ${props.class || ''}
      `}
    >
      {isActive && <Slot />}
    </div>
  );
});

/**
 * TabGroup - Combines Tabs and TabPanels
 */
export const TabGroup = component$<{
  tabs: TabItem[];
  defaultTab?: string;
  variant?: 'primary' | 'secondary';
  class?: string;
  panelClass?: string;
}>((props) => {
  const activeTab = useSignal(props.defaultTab || props.tabs[0]?.id);

  return (
    <div class={props.class}>
      <Tabs
        tabs={props.tabs}
        activeTab={activeTab.value}
        onTabChange$={(id) => activeTab.value = id}
        variant={props.variant}
      />
      <div class={`mt-4 ${props.panelClass || ''}`}>
        <Slot />
      </div>
    </div>
  );
});

export default Tabs;
