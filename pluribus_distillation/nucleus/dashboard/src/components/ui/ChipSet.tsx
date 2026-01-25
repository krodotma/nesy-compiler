/**
 * ChipSet - Material Web 3 Chips Wrapper
 * Phase 4, Step 95 - MW3 Component Integration
 *
 * Glass-styled chips with M3 semantics
 */

import { component$, type PropFunction } from '@builder.io/qwik';

// M3 Chip components
import '@material/web/chips/chip-set.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/chips/input-chip.js';
import '@material/web/chips/assist-chip.js';
import '@material/web/chips/suggestion-chip.js';

export interface ChipItem {
  id: string;
  label: string;
  icon?: string;
  avatar?: string;
  selected?: boolean;
  disabled?: boolean;
  removable?: boolean;
  color?: 'cyan' | 'magenta' | 'amber' | 'emerald' | 'purple' | 'default';
}

export interface ChipSetProps {
  /** Chip items */
  chips: ChipItem[];
  /** Chip variant */
  variant?: 'filter' | 'input' | 'assist' | 'suggestion';
  /** Callback when chip selection changes */
  onSelectionChange$?: PropFunction<(selectedIds: string[]) => void>;
  /** Callback when chip is removed */
  onRemove$?: PropFunction<(chipId: string) => void>;
  /** Callback when chip is clicked */
  onClick$?: PropFunction<(chipId: string) => void>;
  /** Additional class names */
  class?: string;
  /** Wrap chips or scroll horizontally */
  wrap?: boolean;
}

const CHIP_COLORS: Record<string, { bg: string; border: string; text: string; glow: string }> = {
  cyan: {
    bg: 'rgba(0, 255, 255, 0.15)',
    border: 'rgba(0, 255, 255, 0.4)',
    text: 'var(--glass-accent-cyan)',
    glow: '0 0 8px rgba(0, 255, 255, 0.3)',
  },
  magenta: {
    bg: 'rgba(255, 0, 255, 0.15)',
    border: 'rgba(255, 0, 255, 0.4)',
    text: 'var(--glass-accent-magenta)',
    glow: '0 0 8px rgba(255, 0, 255, 0.3)',
  },
  amber: {
    bg: 'rgba(255, 191, 0, 0.15)',
    border: 'rgba(255, 191, 0, 0.4)',
    text: 'var(--glass-accent-amber)',
    glow: '0 0 8px rgba(255, 191, 0, 0.3)',
  },
  emerald: {
    bg: 'rgba(16, 185, 129, 0.15)',
    border: 'rgba(16, 185, 129, 0.4)',
    text: 'var(--glass-accent-emerald)',
    glow: '0 0 8px rgba(16, 185, 129, 0.3)',
  },
  purple: {
    bg: 'rgba(138, 43, 226, 0.15)',
    border: 'rgba(138, 43, 226, 0.4)',
    text: 'var(--glass-accent-purple)',
    glow: '0 0 8px rgba(138, 43, 226, 0.3)',
  },
  default: {
    bg: 'var(--glass-bg-card)',
    border: 'var(--glass-border)',
    text: 'var(--glass-text-secondary)',
    glow: 'none',
  },
};

/**
 * Material Web 3 ChipSet with glassmorphism styling
 *
 * Usage:
 * ```tsx
 * <ChipSet
 *   variant="filter"
 *   chips={[
 *     { id: 'all', label: 'All', selected: true },
 *     { id: 'active', label: 'Active', color: 'cyan' },
 *     { id: 'error', label: 'Errors', color: 'magenta' },
 *   ]}
 *   onSelectionChange$={(ids) => setFilters(ids)}
 * />
 * ```
 */
export const ChipSet = component$<ChipSetProps>((props) => {
  const variant = props.variant || 'filter';

  const handleClick = $((chipId: string, chip: ChipItem) => {
    props.onClick$?.(chipId);

    if (variant === 'filter') {
      // Toggle selection and report
      const currentSelected = props.chips.filter(c => c.selected).map(c => c.id);
      const newSelected = chip.selected
        ? currentSelected.filter(id => id !== chipId)
        : [...currentSelected, chipId];
      props.onSelectionChange$?.(newSelected);
    }
  });

  const handleRemove = $((chipId: string) => {
    props.onRemove$?.(chipId);
  });

  const renderChip = (chip: ChipItem) => {
    const colors = CHIP_COLORS[chip.color || 'default'];
    const selectedColors = chip.selected ? CHIP_COLORS[chip.color || 'cyan'] : colors;

    const chipStyle = chip.selected
      ? {
          '--md-filter-chip-selected-container-color': selectedColors.bg,
          '--md-filter-chip-selected-label-text-color': selectedColors.text,
          '--md-filter-chip-selected-outline-color': selectedColors.border,
          boxShadow: selectedColors.glow,
        }
      : {
          '--md-filter-chip-container-color': colors.bg,
          '--md-filter-chip-label-text-color': colors.text,
          '--md-filter-chip-outline-color': colors.border,
        };

    const glassChipClass = `
      glass-transition-all
      [&::part(container)]:rounded-full
      [&:hover]:scale-[1.02]
      [&:active]:scale-[0.98]
    `;

    switch (variant) {
      case 'filter':
        return (
          <md-filter-chip
            key={chip.id}
            class={glassChipClass}
            style={chipStyle as any}
            label={chip.label}
            selected={chip.selected}
            disabled={chip.disabled}
            onClick$={() => handleClick(chip.id, chip)}
          >
            {chip.icon && <span slot="icon">{chip.icon}</span>}
          </md-filter-chip>
        );

      case 'input':
        return (
          <md-input-chip
            key={chip.id}
            class={glassChipClass}
            style={chipStyle as any}
            label={chip.label}
            disabled={chip.disabled}
            remove-only={chip.removable}
            onClick$={() => handleClick(chip.id, chip)}
            onRemove$={() => handleRemove(chip.id)}
          >
            {chip.icon && <span slot="icon">{chip.icon}</span>}
            {chip.avatar && <img slot="icon" src={chip.avatar} alt="" class="w-6 h-6 rounded-full" />}
          </md-input-chip>
        );

      case 'assist':
        return (
          <md-assist-chip
            key={chip.id}
            class={glassChipClass}
            style={chipStyle as any}
            label={chip.label}
            disabled={chip.disabled}
            onClick$={() => handleClick(chip.id, chip)}
          >
            {chip.icon && <span slot="icon">{chip.icon}</span>}
          </md-assist-chip>
        );

      case 'suggestion':
        return (
          <md-suggestion-chip
            key={chip.id}
            class={glassChipClass}
            style={chipStyle as any}
            label={chip.label}
            disabled={chip.disabled}
            onClick$={() => handleClick(chip.id, chip)}
          >
            {chip.icon && <span slot="icon">{chip.icon}</span>}
          </md-suggestion-chip>
        );

      default:
        return null;
    }
  };

  return (
    <md-chip-set
      class={`
        ${props.wrap ? 'flex flex-wrap gap-2' : 'flex gap-2 overflow-x-auto'}
        ${props.class || ''}
      `}
    >
      {props.chips.map(renderChip)}
    </md-chip-set>
  );
});

/**
 * Single Chip component for standalone use
 */
export const Chip = component$<{
  label: string;
  icon?: string;
  color?: ChipItem['color'];
  selected?: boolean;
  removable?: boolean;
  onClick$?: PropFunction<() => void>;
  onRemove$?: PropFunction<() => void>;
  class?: string;
}>((props) => {
  const colors = props.selected
    ? CHIP_COLORS[props.color || 'cyan']
    : CHIP_COLORS[props.color || 'default'];

  return (
    <button
      class={`
        inline-flex items-center gap-1.5
        px-3 py-1.5
        text-xs font-medium
        rounded-full
        border
        glass-transition-all
        hover:scale-[1.02]
        active:scale-[0.98]
        ${props.class || ''}
      `}
      style={{
        backgroundColor: colors.bg,
        borderColor: colors.border,
        color: colors.text,
        boxShadow: props.selected ? colors.glow : 'none',
      }}
      onClick$={props.onClick$}
    >
      {props.icon && <span>{props.icon}</span>}
      <span>{props.label}</span>
      {props.removable && (
        <button
          class="ml-1 hover:opacity-70"
          onClick$={(e) => {
            e.stopPropagation();
            props.onRemove$?.();
          }}
        >
          ✕
        </button>
      )}
    </button>
  );
});

export default ChipSet;
