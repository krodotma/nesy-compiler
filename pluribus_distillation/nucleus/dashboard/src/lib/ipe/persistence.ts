/**
 * IPE Persistence Layer
 *
 * Handles saving and loading IPE data:
 * - localStorage for instant access
 * - Optional rhizome sync for cross-device persistence
 */

import type { IPEContext, IPELocalStorage, IPEPreferences } from './context-schema';
import type { InstanceOverride } from './instance-manager';
import type { ThemeStyleProps } from './token-bridge';

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY = 'pluribus.ipe';
const VERSION = 1;

const DEFAULT_STORAGE: IPELocalStorage = {
  version: VERSION,
  instances: {},
  undoStack: [],
  preferences: {
    panelPosition: { x: -1, y: -1 },
    panelSize: { width: 380, height: 520 },
    activeTab: 'tokens',
    togglePosition: { x: -1, y: -1 },
  },
};

// ============================================================================
// LocalStorage Operations
// ============================================================================

/**
 * Load IPE data from localStorage
 */
export function loadFromStorage(): IPELocalStorage {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_STORAGE };

    const data = JSON.parse(raw) as IPELocalStorage;

    // Version migration if needed
    if (data.version !== VERSION) {
      return migrateStorage(data);
    }

    return data;
  } catch (error) {
    console.error('[IPE] Failed to load from storage:', error);
    return { ...DEFAULT_STORAGE };
  }
}

/**
 * Save IPE data to localStorage
 */
export function saveToStorage(data: Partial<IPELocalStorage>): void {
  try {
    const current = loadFromStorage();
    const merged: IPELocalStorage = {
      ...current,
      ...data,
      version: VERSION,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
  } catch (error) {
    console.error('[IPE] Failed to save to storage:', error);
  }
}

/**
 * Clear all IPE data from localStorage
 */
export function clearStorage(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error('[IPE] Failed to clear storage:', error);
  }
}

/**
 * Migrate storage from older versions
 */
function migrateStorage(data: IPELocalStorage): IPELocalStorage {
  // Add migration logic as needed for future versions
  return {
    ...DEFAULT_STORAGE,
    ...data,
    version: VERSION,
  };
}

// ============================================================================
// Instance Persistence
// ============================================================================

/**
 * Save an instance override
 */
export function saveInstance(instance: InstanceOverride): void {
  const storage = loadFromStorage();
  storage.instances[instance.instanceId] = {
    instanceId: instance.instanceId,
    styles: instance.styles as Record<string, string>,
    shaderOverride: instance.shaderOverride,
    purpose: instance.purpose,
    updatedAt: instance.updatedAt,
  };
  saveToStorage({ instances: storage.instances });
}

/**
 * Load all saved instances
 */
export function loadInstances(): Record<string, {
  instanceId: string;
  styles: Record<string, string>;
  shaderOverride?: string;
  purpose?: string;
  updatedAt: string;
}> {
  return loadFromStorage().instances;
}

/**
 * Remove a saved instance
 */
export function removeInstance(instanceId: string): void {
  const storage = loadFromStorage();
  delete storage.instances[instanceId];
  saveToStorage({ instances: storage.instances });
}

// ============================================================================
// Undo Stack
// ============================================================================

/**
 * Push context to undo stack
 */
export function pushUndo(context: IPEContext): void {
  const storage = loadFromStorage();
  storage.undoStack.push(context);

  // Limit stack size
  if (storage.undoStack.length > 50) {
    storage.undoStack.shift();
  }

  saveToStorage({ undoStack: storage.undoStack });
}

/**
 * Pop context from undo stack
 */
export function popUndo(): IPEContext | null {
  const storage = loadFromStorage();
  const context = storage.undoStack.pop() || null;
  saveToStorage({ undoStack: storage.undoStack });
  return context;
}

/**
 * Clear undo stack
 */
export function clearUndo(): void {
  saveToStorage({ undoStack: [] });
}

// ============================================================================
// Preferences
// ============================================================================

/**
 * Save preferences
 */
export function savePreferences(prefs: Partial<IPEPreferences>): void {
  const storage = loadFromStorage();
  storage.preferences = { ...storage.preferences, ...prefs };
  saveToStorage({ preferences: storage.preferences });
}

/**
 * Load preferences
 */
export function loadPreferences(): IPEPreferences {
  return loadFromStorage().preferences;
}

// ============================================================================
// Rhizome Sync (Optional)
// ============================================================================

interface RhizomeSyncOptions {
  actor?: string;
  baseUrl?: string;
}

/**
 * Sync instances to rhizome (append-only bus)
 */
export async function syncToRhizome(options: RhizomeSyncOptions = {}): Promise<boolean> {
  const { actor = 'ipe', baseUrl = '/api/rhizome' } = options;

  try {
    const instances = loadInstances();

    const response = await fetch(baseUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'ingest',
        params: {
          object_type: 'ipe_instances',
          content: JSON.stringify(instances),
          metadata: {
            actor,
            timestamp: new Date().toISOString(),
            version: VERSION,
          },
        },
      }),
    });

    return response.ok;
  } catch (error) {
    console.error('[IPE] Rhizome sync failed:', error);
    return false;
  }
}

/**
 * Load instances from rhizome
 */
export async function loadFromRhizome(options: RhizomeSyncOptions = {}): Promise<Record<string, {
  instanceId: string;
  styles: Record<string, string>;
  shaderOverride?: string;
  purpose?: string;
  updatedAt: string;
}> | null> {
  const { baseUrl = '/api/rhizome' } = options;

  try {
    const response = await fetch(baseUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'query',
        params: {
          object_type: 'ipe_instances',
          limit: 1,
          sort: 'desc',
        },
      }),
    });

    if (!response.ok) return null;

    const data = await response.json();
    if (data.result?.content) {
      return JSON.parse(data.result.content);
    }
    return null;
  } catch (error) {
    console.error('[IPE] Failed to load from rhizome:', error);
    return null;
  }
}

/**
 * Merge local and remote instances (remote wins on conflict)
 */
export async function mergeWithRhizome(options: RhizomeSyncOptions = {}): Promise<void> {
  const local = loadInstances();
  const remote = await loadFromRhizome(options);

  if (!remote) return;

  // Merge: remote wins on conflict
  const merged = { ...local };
  for (const [id, instance] of Object.entries(remote)) {
    const localInstance = merged[id];
    if (!localInstance || new Date(instance.updatedAt) > new Date(localInstance.updatedAt)) {
      merged[id] = instance;
    }
  }

  saveToStorage({ instances: merged });
}

// ============================================================================
// Global Token Persistence
// ============================================================================

/**
 * Save global tokens to theme.ts (requires API)
 */
export async function saveGlobalTokens(
  tokens: Partial<ThemeStyleProps>,
  options: { apiUrl?: string } = {}
): Promise<{ success: boolean; error?: string }> {
  const { apiUrl = '/api/fs/write' } = options;

  try {
    // Generate theme.ts content
    const themeContent = generateThemeFile(tokens);

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        path: 'nucleus/dashboard/src/lib/theme.ts',
        content: themeContent,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return { success: false, error };
    }

    // Emit bus event for hot reload
    window.dispatchEvent(new CustomEvent('ipe:global:update', {
      detail: { tokens }
    }));

    return { success: true };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Generate theme.ts file content
 */
function generateThemeFile(tokens: Partial<ThemeStyleProps>): string {
  const lines = [
    '/**',
    ' * Theme Configuration',
    ' * Generated by IPE (In-Place Editor)',
    ` * ${new Date().toISOString()}`,
    ' */',
    '',
    'export const theme = {',
  ];

  for (const [key, value] of Object.entries(tokens)) {
    lines.push(`  '${key}': '${value}',`);
  }

  lines.push('} as const;');
  lines.push('');
  lines.push('export type ThemeKey = keyof typeof theme;');
  lines.push('');

  return lines.join('\n');
}

// ============================================================================
// Export
// ============================================================================

export default {
  // Storage
  loadFromStorage,
  saveToStorage,
  clearStorage,

  // Instances
  saveInstance,
  loadInstances,
  removeInstance,

  // Undo
  pushUndo,
  popUndo,
  clearUndo,

  // Preferences
  savePreferences,
  loadPreferences,

  // Rhizome
  syncToRhizome,
  loadFromRhizome,
  mergeWithRhizome,

  // Global
  saveGlobalTokens,
};
