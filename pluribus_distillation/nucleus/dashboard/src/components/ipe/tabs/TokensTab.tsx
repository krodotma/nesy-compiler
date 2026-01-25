/**
 * TokensTab.tsx
 *
 * Design token editor tab for IPE.
 * Qwik-native implementation inspired by tweakcn's design.
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';
import type { IPEContext, IPEScope } from '../../../lib/ipe';
import {
  contextToThemeProps,
  parseHSL,
  formatHSL,
  meetsWCAGAA,
  type ThemeStyleProps,
} from '../../../lib/ipe';

interface TokensTabProps {
  context: IPEContext;
  scope: IPEScope;
  onTokenChange$?: QRL<(token: string, value: string) => void>;
}

interface TokenGroup {
  label: string;
  tokens: { key: keyof ThemeStyleProps; label: string; pair?: keyof ThemeStyleProps }[];
}

const TOKEN_GROUPS: TokenGroup[] = [
  {
    label: 'Core Colors',
    tokens: [
      { key: 'background', label: 'Background', pair: 'foreground' },
      { key: 'foreground', label: 'Foreground', pair: 'background' },
      { key: 'primary', label: 'Primary', pair: 'primary-foreground' },
      { key: 'primary-foreground', label: 'Primary FG', pair: 'primary' },
      { key: 'secondary', label: 'Secondary', pair: 'secondary-foreground' },
      { key: 'secondary-foreground', label: 'Secondary FG', pair: 'secondary' },
    ],
  },
  {
    label: 'UI Colors',
    tokens: [
      { key: 'muted', label: 'Muted', pair: 'muted-foreground' },
      { key: 'muted-foreground', label: 'Muted FG', pair: 'muted' },
      { key: 'accent', label: 'Accent', pair: 'accent-foreground' },
      { key: 'accent-foreground', label: 'Accent FG', pair: 'accent' },
      { key: 'destructive', label: 'Destructive', pair: 'destructive-foreground' },
      { key: 'border', label: 'Border' },
      { key: 'ring', label: 'Ring' },
    ],
  },
  {
    label: 'Card & Popover',
    tokens: [
      { key: 'card', label: 'Card', pair: 'card-foreground' },
      { key: 'card-foreground', label: 'Card FG', pair: 'card' },
      { key: 'popover', label: 'Popover', pair: 'popover-foreground' },
      { key: 'popover-foreground', label: 'Popover FG', pair: 'popover' },
    ],
  },
  {
    label: 'Dimensions',
    tokens: [
      { key: 'radius', label: 'Border Radius' },
      { key: 'letter-spacing', label: 'Letter Spacing' },
    ],
  },
];

export const TokensTab = component$<TokensTabProps>(({
  context,
  scope,
  onTokenChange$,
}) => {
  const tokens = useStore<Partial<ThemeStyleProps>>(() =>
    contextToThemeProps(context)
  );

  // Re-initialize when context changes
  useVisibleTask$(({ track }) => {
    track(() => context.instanceId);
    const newTokens = contextToThemeProps(context);
    // Reset tokens store
    for (const key in tokens) delete tokens[key as keyof ThemeStyleProps];
    Object.assign(tokens, newTokens);
  });

  const expandedGroups = useStore<Record<string, boolean>>({
    'Core Colors': true,
    'UI Colors': false,
    'Card & Popover': false,
    'Dimensions': false,
  });

  const activeToken = useSignal<keyof ThemeStyleProps | null>(null);

  // Toggle group expansion
  const toggleGroup = $((label: string) => {
    expandedGroups[label] = !expandedGroups[label];
  });

  // Handle token value change
  const handleTokenChange = $((key: keyof ThemeStyleProps, value: string) => {
    tokens[key] = value;
    onTokenChange$?.(key, value);
  });

  return (
    <div class="space-y-4">
      {/* Scope indicator */}
      <div
        class={[
          'px-3 py-2 rounded-lg text-sm',
          scope === 'global'
            ? 'bg-orange-500/10 border border-orange-500/30 text-orange-300'
            : 'bg-blue-500/10 border border-blue-500/30 text-blue-300',
        ]}
      >
        {scope === 'global'
          ? '🌐 Editing global design tokens'
          : `📍 Editing instance: ${context.instanceId}`}
      </div>

      {/* Token groups */}
      {TOKEN_GROUPS.map(group => (
        <div
          key={group.label}
          class="rounded-lg bg-white/5 overflow-hidden"
        >
          {/* Group header */}
          <button
            type="button"
            class="w-full flex items-center justify-between px-3 py-2 hover:bg-white/5 transition-colors"
            onClick$={() => toggleGroup(group.label)}
          >
            <span class="text-sm font-medium text-gray-300">
              {group.label}
            </span>
            <span class="text-gray-500">
              {expandedGroups[group.label] ? '▼' : '▶'}
            </span>
          </button>

          {/* Group content */}
          {expandedGroups[group.label] && (
            <div class="px-3 pb-3 space-y-2">
              {group.tokens.map(token => {
                const value = tokens[token.key] || '';
                const isColor = !['radius', 'letter-spacing', 'spacing'].includes(token.key);
                const pairValue = token.pair ? tokens[token.pair] : undefined;
                const hasContrastIssue = isColor && token.pair && pairValue
                  ? !meetsWCAGAA(value, pairValue)
                  : false;

                return (
                  <div
                    key={token.key}
                    class={[
                      'rounded-lg p-2 transition-colors',
                      activeToken.value === token.key
                        ? 'bg-blue-500/20 ring-1 ring-blue-500/50'
                        : 'bg-white/5 hover:bg-white/10',
                    ]}
                    onClick$={() => { activeToken.value = token.key; }}
                  >
                    <div class="flex items-center justify-between mb-1">
                      <span class="text-xs text-gray-400">{token.label}</span>
                      {hasContrastIssue && (
                        <span
                          class="text-xs text-yellow-400"
                          title="Low contrast ratio with paired color"
                        >
                          ⚠️ Low contrast
                        </span>
                      )}
                    </div>

                    {isColor ? (
                      <ColorEditor
                        value={value}
                        onChange$={(v) => handleTokenChange(token.key, v)}
                      />
                    ) : (
                      <DimensionEditor
                        value={value}
                        onChange$={(v) => handleTokenChange(token.key, v)}
                        tokenKey={token.key}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ))}

      {/* No tokens message */}
      {Object.keys(tokens).length === 0 && (
        <div class="text-center py-8 text-gray-500">
          No design tokens found for this element.
          <br />
          <span class="text-xs">
            Try selecting an element that uses CSS variables.
          </span>
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Sub-components
// ============================================================================

interface ColorEditorProps {
  value: string;
  onChange$: QRL<(value: string) => void>;
}

const ColorEditor = component$<ColorEditorProps>(({ value, onChange$ }) => {
  const hsl = parseHSL(value);
  const h = useSignal(hsl?.h ?? 0);
  const s = useSignal(hsl?.s ?? 50);
  const l = useSignal(hsl?.l ?? 50);

  // Update from external value
  useVisibleTask$(({ track }) => {
    const v = track(() => value);
    const parsed = parseHSL(v);
    if (parsed) {
      h.value = parsed.h;
      s.value = parsed.s;
      l.value = parsed.l;
    }
  });

  // Emit change
  const emitChange = $(() => {
    const newValue = formatHSL(h.value, s.value, l.value);
    onChange$(newValue);
  });

  // Preview color as CSS
  const previewColor = `hsl(${h.value} ${s.value}% ${l.value}%)`;

  return (
    <div class="space-y-2">
      {/* Color preview + raw value */}
      <div class="flex items-center gap-2">
        <div
          class="w-8 h-8 rounded border border-[var(--glass-border-active)]"
          style={{ backgroundColor: previewColor }}
        />
        <input
          type="text"
          value={value}
          class={[
            'flex-1 px-2 py-1 rounded text-xs font-mono',
            'bg-black/30 border border-[var(--glass-border)] text-gray-300',
            'focus:outline-none focus:border-blue-500/50',
          ]}
          onInput$={(e) => onChange$((e.target as HTMLInputElement).value)}
        />
      </div>

      {/* HSL sliders */}
      <div class="space-y-1">
        {/* Hue */}
        <div class="flex items-center gap-2">
          <span class="w-4 text-xs text-gray-500">H</span>
          <input
            type="range"
            min="0"
            max="360"
            value={h.value}
            class="flex-1 h-1.5 rounded appearance-none bg-gradient-to-r from-red-500 via-yellow-500 via-green-500 via-cyan-500 via-blue-500 via-purple-500 to-red-500"
            onInput$={(e) => {
              h.value = parseInt((e.target as HTMLInputElement).value);
              emitChange();
            }}
          />
          <span class="w-8 text-xs text-gray-400 text-right">{Math.round(h.value)}°</span>
        </div>

        {/* Saturation */}
        <div class="flex items-center gap-2">
          <span class="w-4 text-xs text-gray-500">S</span>
          <input
            type="range"
            min="0"
            max="100"
            value={s.value}
            class="flex-1 h-1.5 rounded appearance-none bg-gradient-to-r from-gray-500 to-blue-500"
            onInput$={(e) => {
              s.value = parseInt((e.target as HTMLInputElement).value);
              emitChange();
            }}
          />
          <span class="w-8 text-xs text-gray-400 text-right">{Math.round(s.value)}%</span>
        </div>

        {/* Lightness */}
        <div class="flex items-center gap-2">
          <span class="w-4 text-xs text-gray-500">L</span>
          <input
            type="range"
            min="0"
            max="100"
            value={l.value}
            class="flex-1 h-1.5 rounded appearance-none bg-gradient-to-r from-black via-gray-500 to-white"
            onInput$={(e) => {
              l.value = parseInt((e.target as HTMLInputElement).value);
              emitChange();
            }}
          />
          <span class="w-8 text-xs text-gray-400 text-right">{Math.round(l.value)}%</span>
        </div>
      </div>
    </div>
  );
});

interface DimensionEditorProps {
  value: string;
  onChange$: QRL<(value: string) => void>;
  tokenKey: string;
}

const DimensionEditor = component$<DimensionEditorProps>(({ value, onChange$, tokenKey }) => {
  // Parse numeric value from string like "0.5rem" or "0"
  const match = value.match(/^([\d.]+)(.*)/);
  const numValue = useSignal(match ? parseFloat(match[1]) : 0);
  const unit = useSignal(match?.[2] || (tokenKey === 'radius' ? 'rem' : ''));

  const config = tokenKey === 'radius'
    ? { min: 0, max: 2, step: 0.125, defaultUnit: 'rem' }
    : { min: -0.1, max: 0.2, step: 0.01, defaultUnit: 'em' };

  const emitChange = $(() => {
    const newValue = `${numValue.value}${unit.value}`;
    onChange$(newValue);
  });

  return (
    <div class="space-y-2">
      <div class="flex items-center gap-2">
        <input
          type="number"
          value={numValue.value}
          step={config.step}
          class={[
            'w-20 px-2 py-1 rounded text-xs font-mono',
            'bg-black/30 border border-[var(--glass-border)] text-gray-300',
            'focus:outline-none focus:border-blue-500/50',
          ]}
          onInput$={(e) => {
            numValue.value = parseFloat((e.target as HTMLInputElement).value) || 0;
            emitChange();
          }}
        />
        <select
          value={unit.value}
          class={[
            'px-2 py-1 rounded text-xs',
            'bg-black/30 border border-[var(--glass-border)] text-gray-300',
            'focus:outline-none focus:border-blue-500/50',
          ]}
          onChange$={(e) => {
            unit.value = (e.target as HTMLSelectElement).value;
            emitChange();
          }}
        >
          <option value="rem">rem</option>
          <option value="px">px</option>
          <option value="em">em</option>
          <option value="">none</option>
        </select>
      </div>

      <input
        type="range"
        min={config.min}
        max={config.max}
        step={config.step}
        value={numValue.value}
        class="w-full h-1.5 rounded appearance-none bg-white/20"
        onInput$={(e) => {
          numValue.value = parseFloat((e.target as HTMLInputElement).value);
          emitChange();
        }}
      />

      {/* Preview for radius */}
      {tokenKey === 'radius' && (
        <div class="flex justify-center">
          <div
            class="w-12 h-12 bg-blue-500"
            style={{ borderRadius: `${numValue.value}${unit.value}` }}
          />
        </div>
      )}
    </div>
  );
});

export default TokensTab;
