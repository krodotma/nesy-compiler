import { useStore, useVisibleTask$ } from '@builder.io/qwik';

const DEFAULT_TOKENS = [
  '--sys-color-primary',
  '--sys-color-secondary',
  '--sys-color-tertiary',
  '--sys-color-background',
  '--sys-color-surface',
  '--sys-color-surface-variant',
  '--sys-color-on-surface',
  '--sys-color-on-surface-variant',
  '--sys-glass-bg-card',
  '--sys-glass-border',
] as const;

type ThemeTokenMap = Record<string, string>;

export const useThemeTokens = (keys: readonly string[] = DEFAULT_TOKENS) => {
  const tokens = useStore<ThemeTokenMap>({});

  useVisibleTask$(() => {
    const root = document.documentElement;
    const computed = getComputedStyle(root);
    keys.forEach((key) => {
      tokens[key] = computed.getPropertyValue(key).trim();
    });
  });

  return tokens;
};
