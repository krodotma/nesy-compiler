export interface WebLLMModelSpec {
  id: string;
  name?: string;
  vram?: string;
}

const DEFAULT_SEED_WORDS = [
  'fractal',
  'entropy',
  'agency',
  'memory',
  'signal',
  'latency',
  'vector',
  'gradient',
  'cache',
  'runtime',
  'edge',
  'causality',
  'emergence',
  'alignment',
  'feedback',
  'synthesis',
  'context',
  'resonance',
  'topology',
  'coherence',
];

const DEFAULT_SEED_TEMPLATES = [
  'Discuss {a} and {b} with a concrete example.',
  'Compare {a} vs {b}, then propose a synthesis.',
  'Draft a research question connecting {a}, {b}, and {c}.',
  'Start a short dialogue about {a}, {b}, and {c}.',
  'Sketch a system design for {a} under {b}.',
];

export function selectAutoModelIds(
  models: WebLLMModelSpec[],
  cachedIds: string[],
  desiredCount = 2
): string[] {
  if (desiredCount <= 0 || models.length === 0) return [];
  const target = Math.min(desiredCount, models.length);
  const cachedSet = new Set(cachedIds);
  const picked: string[] = [];

  const push = (id: string) => {
    if (picked.length >= target) return;
    if (picked.includes(id)) return;
    picked.push(id);
  };

  for (const model of models) {
    if (cachedSet.has(model.id)) push(model.id);
  }

  for (const model of models) {
    push(model.id);
  }

  return picked;
}

export function getRandomSeed(deterministic: boolean): number {
  if (deterministic) return 1337;
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const buf = new Uint32Array(1);
    crypto.getRandomValues(buf);
    return buf[0] || 1;
  }
  return Math.floor(Math.random() * 0xffffffff) >>> 0;
}

export function createSeededRng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function pickSeedWords(random: () => number, count: number): string[] {
  const pool = [...DEFAULT_SEED_WORDS];
  const target = Math.max(2, Math.min(count, pool.length));
  const out: string[] = [];
  for (let i = 0; i < target; i += 1) {
    const idx = Math.floor(random() * pool.length);
    out.push(pool.splice(idx, 1)[0]);
  }
  return out;
}

export function buildSeedPhrase(
  random: () => number,
  wordCount = 3
): { phrase: string; words: string[] } {
  const words = pickSeedWords(random, wordCount);
  const template = DEFAULT_SEED_TEMPLATES[Math.floor(random() * DEFAULT_SEED_TEMPLATES.length)];
  const [a, b, c] = [words[0], words[1], words[2] || words[1]];
  const phrase = template.replace('{a}', a).replace('{b}', b).replace('{c}', c);
  return { phrase, words };
}
