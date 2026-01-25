/**
 * UI Components - Material Web 3 + Glass Styling
 *
 * Phase 4 MW3 Component Library for Pluribus Dashboard
 *
 * Features:
 * - Material Web 3 integration with glassmorphism styling
 * - Progressive disclosure patterns
 * - Keyboard navigation support
 * - Neon accent colors
 *
 * Based on Qwik with M3 web components.
 */

// === Legacy Exports (Shadcn-style) ===
export * from './button';
export * from './card';
export * from './badge';
export * from './status';
export * from './cn';

// === MW3 Component Wrappers (Phase 4 Steps 91-100) ===
export { Dialog, ConfirmDialog } from './Dialog';
export { Tabs, TabPanel, TabGroup } from './Tabs';
export { Menu, ContextMenu } from './Menu';
export { ChipSet, Chip } from './ChipSet';
export { LinearProgress, CircularProgress, LoadingSpinner, ProgressWithLabel } from './Progress';
export { Switch, ToggleGroup } from './Switch';

// === Progressive Disclosure (Phase 4 Steps 101-110) ===
export { CollapsibleSection, Accordion, GlassDetails } from './CollapsibleSection';
export { ExpandableCard, QuickStatsCard, RevealOnHover } from './ExpandableCard';
export { NavDrawer, NavRail } from './NavDrawer';
export { InfoTooltip, InfoIcon, HelpText, LabelWithInfo } from './InfoTooltip';

// === Navigation Refinement (Phase 4 Steps 111-120) ===
export { HamburgerMenu, MenuButton, KebabMenu } from './HamburgerMenu';
export { Breadcrumb, BreadcrumbWithOverflow, PageHeader } from './Breadcrumb';
export { FAB, SpeedDialFAB } from './FAB';
export { CommandPalette } from './CommandPalette';
export { Toast, ToastContainer, ToastProvider, useToast, showToast, ToastContext } from './Toast';

// === Additional UI Components ===
export { NeonTitle } from './NeonTitle';
export { BottomSheet } from './BottomSheet';
export { Button } from './Button';
export { Card } from './Card';
export { Input } from './Input';

// === Utility Components ===
export { CodeViewer } from './CodeViewer';
export { VirtualList } from './VirtualList';

// === Type Exports ===
export type { DialogProps } from './Dialog';
export type { TabsProps, TabItem } from './Tabs';
export type { MenuProps, MenuItem } from './Menu';
export type { ChipSetProps, ChipItem } from './ChipSet';
export type { LinearProgressProps, CircularProgressProps } from './Progress';
export type { SwitchProps } from './Switch';
export type { CollapsibleSectionProps } from './CollapsibleSection';
export type { ExpandableCardProps } from './ExpandableCard';
export type { NavDrawerProps, NavItem } from './NavDrawer';
export type { InfoTooltipProps, TooltipPosition } from './InfoTooltip';
export type { HamburgerMenuProps } from './HamburgerMenu';
export type { BreadcrumbProps, BreadcrumbItem } from './Breadcrumb';
export type { FABProps } from './FAB';
export type { CommandPaletteProps, CommandItem } from './CommandPalette';
export type { ToastProps, ToastVariant, ToastPosition, ToastItem, ToastContextType } from './Toast';
