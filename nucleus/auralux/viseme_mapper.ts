/**
 * viseme_mapper.ts
 * Maps phonemes (CMU Arpabet) to visemes (Oculus standard).
 * Produces timed viseme frames for lip-sync animation.
 */

// ============================================================================
// Types
// ============================================================================

export interface VisemeFrame {
    viseme: string;
    weight: number;      // 0-1 blend weight
    startMs: number;     // Start time in ms
    durationMs: number;  // Duration in ms
}

export interface PhonemeInput {
    phoneme: string;
    startMs: number;
    durationMs: number;
}

// ============================================================================
// CMU Arpabet → Oculus Viseme Mapping
// Based on Oculus Lipsync standard (15 visemes)
// ============================================================================

/**
 * Oculus Viseme Set:
 * sil - silence
 * PP  - bilabial plosives (p, b, m)
 * FF  - labiodental fricatives (f, v)
 * TH  - dental fricatives (th)
 * DD  - alveolar plosives (t, d, n)
 * kk  - velar plosives (k, g, ng)
 * CH  - postalveolar affricates (ch, j, sh, zh)
 * SS  - alveolar fricatives (s, z)
 * nn  - alveolar nasals (n)
 * RR  - alveolar approximants (r)
 * aa  - open vowels (a, ah)
 * E   - front mid vowels (e, eh)
 * ih  - front close vowels (i, ih)
 * oh  - back mid rounded (o, aw)
 * ou  - close back rounded (u, oo)
 */

const ARPABET_TO_VISEME: Record<string, string> = {
    // Silence
    'SIL': 'sil',
    'SP': 'sil',

    // Bilabial plosives (PP)
    'P': 'PP',
    'B': 'PP',
    'M': 'PP',

    // Labiodental fricatives (FF)
    'F': 'FF',
    'V': 'FF',

    // Dental fricatives (TH)
    'TH': 'TH',
    'DH': 'TH',

    // Alveolar plosives/nasals (DD)
    'T': 'DD',
    'D': 'DD',
    'N': 'DD',

    // Velar plosives (kk)
    'K': 'kk',
    'G': 'kk',
    'NG': 'kk',

    // Postalveolar affricates/fricatives (CH)
    'CH': 'CH',
    'JH': 'CH',
    'SH': 'CH',
    'ZH': 'CH',

    // Alveolar fricatives (SS)
    'S': 'SS',
    'Z': 'SS',

    // Approximants
    'R': 'RR',
    'L': 'DD',  // Lateral approximant → DD
    'W': 'ou',  // Labial approximant → ou
    'Y': 'ih',  // Palatal approximant → ih
    'HH': 'sil', // Glottal → minimal

    // Vowels - Open/Low (aa)
    'AA': 'aa',
    'AE': 'aa',
    'AH': 'aa',
    'AO': 'oh',
    'AW': 'oh',
    'AY': 'aa',

    // Vowels - Front/Mid (E)
    'EH': 'E',
    'ER': 'E',
    'EY': 'E',

    // Vowels - Front/Close (ih)
    'IH': 'ih',
    'IY': 'ih',

    // Vowels - Back/Rounded (oh/ou)
    'OW': 'oh',
    'OY': 'oh',
    'UH': 'ou',
    'UW': 'ou',
};

// Stress markers are suffixes (0, 1, 2) - strip them
function normalizeArpabet(phoneme: string): string {
    return phoneme.replace(/[012]$/, '').toUpperCase();
}

// ============================================================================
// Mapper Functions
// ============================================================================

/**
 * Convert a single phoneme to its viseme.
 */
export function phonemeToViseme(phoneme: string): string {
    const normalized = normalizeArpabet(phoneme);
    return ARPABET_TO_VISEME[normalized] || 'sil';
}

/**
 * Convert a sequence of timed phonemes to viseme frames.
 * Includes coarticulation blending at phoneme boundaries.
 */
export function mapPhonemeSequence(phonemes: PhonemeInput[]): VisemeFrame[] {
    if (phonemes.length === 0) return [];

    const frames: VisemeFrame[] = [];
    const BLEND_MS = 20; // Coarticulation blend window

    for (let i = 0; i < phonemes.length; i++) {
        const { phoneme, startMs, durationMs } = phonemes[i];
        const viseme = phonemeToViseme(phoneme);

        // Main frame
        frames.push({
            viseme,
            weight: 1.0,
            startMs,
            durationMs: durationMs - BLEND_MS,
        });

        // Blend frame (transition to next)
        if (i < phonemes.length - 1) {
            const nextViseme = phonemeToViseme(phonemes[i + 1].phoneme);
            if (nextViseme !== viseme) {
                frames.push({
                    viseme: nextViseme,
                    weight: 0.5, // Blend
                    startMs: startMs + durationMs - BLEND_MS,
                    durationMs: BLEND_MS * 2,
                });
            }
        }
    }

    return frames;
}

/**
 * Get viseme blend weights for a specific time.
 * Returns weights for all active visemes at the given timestamp.
 */
export function getVisemeWeightsAtTime(
    frames: VisemeFrame[],
    timeMs: number
): Record<string, number> {
    const weights: Record<string, number> = {};

    for (const frame of frames) {
        const frameEnd = frame.startMs + frame.durationMs;
        if (timeMs >= frame.startMs && timeMs < frameEnd) {
            // Smooth weight curve (ease in/out)
            const progress = (timeMs - frame.startMs) / frame.durationMs;
            const smoothWeight = frame.weight * (1 - Math.cos(progress * Math.PI)) / 2;

            if (!weights[frame.viseme] || weights[frame.viseme] < smoothWeight) {
                weights[frame.viseme] = smoothWeight;
            }
        }
    }

    return weights;
}

/**
 * All available viseme names (for morph target initialization).
 */
export const VISEME_NAMES = [
    'sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS', 'nn', 'RR',
    'aa', 'E', 'ih', 'oh', 'ou'
] as const;

export type VisemeName = typeof VISEME_NAMES[number];
