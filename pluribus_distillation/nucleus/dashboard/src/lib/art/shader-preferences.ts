/**
 * Shader Preference System - A/B Testing for Art Department
 *
 * Tracks user preferences for shaders via localStorage.
 * Enables weighted random selection based on likes/dislikes.
 */

const STORAGE_KEY = 'pluribus_shader_prefs';

export interface ShaderPref {
  id: string;
  name: string;
  likes: number;
  dislikes: number;
  seen: number;
  lastSeen?: number; // timestamp
}

export interface ShaderPrefs {
  version: number;
  shaders: Record<string, ShaderPref>;
  currentId?: string;
  currentName?: string;
  blacklist: string[]; // Never show these again
}

const DEFAULT_PREFS: ShaderPrefs = {
  version: 1,
  shaders: {},
  blacklist: [],
};

function loadPrefs(): ShaderPrefs {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_PREFS };
    const prefs = JSON.parse(raw) as ShaderPrefs;
    return { ...DEFAULT_PREFS, ...prefs };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

function savePrefs(prefs: ShaderPrefs): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // ignore quota errors
  }
}

/** Get current preferences */
export function getShaderPrefs(): ShaderPrefs {
  return loadPrefs();
}

/** Record that a shader was displayed */
export function recordShaderSeen(id: string, name: string): void {
  const prefs = loadPrefs();
  if (!prefs.shaders[id]) {
    prefs.shaders[id] = { id, name, likes: 0, dislikes: 0, seen: 0 };
  }
  prefs.shaders[id].seen++;
  prefs.shaders[id].lastSeen = Date.now();
  prefs.shaders[id].name = name; // Update name in case it changed
  prefs.currentId = id;
  prefs.currentName = name;
  savePrefs(prefs);

  // Emit event for UI components
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('pluribus:shader:current', {
      detail: { id, name, pref: prefs.shaders[id] }
    }));
  }
}

/** Like current shader */
export function likeShader(id: string): ShaderPref | null {
  const prefs = loadPrefs();
  if (!prefs.shaders[id]) return null;
  prefs.shaders[id].likes++;
  // Remove from blacklist if it was there
  prefs.blacklist = prefs.blacklist.filter(b => b !== id);
  savePrefs(prefs);
  return prefs.shaders[id];
}

/** Dislike current shader */
export function dislikeShader(id: string): ShaderPref | null {
  const prefs = loadPrefs();
  if (!prefs.shaders[id]) return null;
  prefs.shaders[id].dislikes++;
  savePrefs(prefs);
  return prefs.shaders[id];
}

/** Blacklist a shader (never show again) */
export function blacklistShader(id: string): void {
  const prefs = loadPrefs();
  if (!prefs.blacklist.includes(id)) {
    prefs.blacklist.push(id);
  }
  savePrefs(prefs);
}

/** Remove from blacklist */
export function unblacklistShader(id: string): void {
  const prefs = loadPrefs();
  prefs.blacklist = prefs.blacklist.filter(b => b !== id);
  savePrefs(prefs);
}

/** Calculate preference score for weighted random selection */
export function getPreferenceScore(pref: ShaderPref): number {
  // Score = (likes * 2 - dislikes * 3) / max(seen, 1)
  // Higher = more liked, novelty bonus for unseen
  const base = pref.likes * 2 - pref.dislikes * 3;
  const novelty = Math.max(3 - pref.seen, 0) * 0.5; // Novelty bonus decays
  return base + novelty + 1; // +1 so everything starts positive
}

/** Weighted random selection from shader list */
export function selectWeightedRandom<T extends { id: string }>(
  shaders: T[]
): T | null {
  if (!shaders.length) return null;
  if (shaders.length === 1) return shaders[0];

  const prefs = loadPrefs();

  // Filter out blacklisted
  const available = shaders.filter(s => !prefs.blacklist.includes(s.id));
  if (!available.length) return shaders[0]; // All blacklisted, just pick first

  // Calculate weights
  const weights = available.map(s => {
    const pref = prefs.shaders[s.id];
    if (!pref) return 2; // Novelty bonus for never-seen
    return Math.max(0.1, getPreferenceScore(pref));
  });

  const totalWeight = weights.reduce((a, b) => a + b, 0);
  let random = Math.random() * totalWeight;

  for (let i = 0; i < available.length; i++) {
    random -= weights[i];
    if (random <= 0) return available[i];
  }

  return available[available.length - 1];
}

/** Get top-rated shaders */
export function getTopShaders(limit = 10): ShaderPref[] {
  const prefs = loadPrefs();
  return Object.values(prefs.shaders)
    .filter(p => p.likes > 0)
    .sort((a, b) => getPreferenceScore(b) - getPreferenceScore(a))
    .slice(0, limit);
}

/** Get blacklisted shaders */
export function getBlacklist(): string[] {
  return loadPrefs().blacklist;
}

/** Export preferences (for backup) */
export function exportPrefs(): string {
  return JSON.stringify(loadPrefs(), null, 2);
}

/** Import preferences */
export function importPrefs(json: string): boolean {
  try {
    const prefs = JSON.parse(json) as ShaderPrefs;
    if (typeof prefs.version === 'number' && typeof prefs.shaders === 'object') {
      savePrefs(prefs);
      return true;
    }
    return false;
  } catch {
    return false;
  }
}
