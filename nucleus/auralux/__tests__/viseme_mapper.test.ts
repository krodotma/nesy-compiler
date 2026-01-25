/**
 * viseme_mapper.test.ts
 * Unit tests for CMU Arpabet to Oculus viseme mapping.
 */
import { describe, it, expect } from 'vitest';
import {
    phonemeToViseme,
    mapPhonemeSequence,
    getVisemeWeightsAtTime,
    VISEME_NAMES,
    PhonemeInput,
    VisemeFrame,
} from '../viseme_mapper';

describe('viseme_mapper', () => {
    describe('phonemeToViseme', () => {
        it('maps bilabial plosives to PP', () => {
            expect(phonemeToViseme('P')).toBe('PP');
            expect(phonemeToViseme('B')).toBe('PP');
            expect(phonemeToViseme('M')).toBe('PP');
        });

        it('maps labiodental fricatives to FF', () => {
            expect(phonemeToViseme('F')).toBe('FF');
            expect(phonemeToViseme('V')).toBe('FF');
        });

        it('maps dental fricatives to TH', () => {
            expect(phonemeToViseme('TH')).toBe('TH');
            expect(phonemeToViseme('DH')).toBe('TH');
        });

        it('maps alveolar plosives to DD', () => {
            expect(phonemeToViseme('T')).toBe('DD');
            expect(phonemeToViseme('D')).toBe('DD');
            expect(phonemeToViseme('N')).toBe('DD');
        });

        it('maps velar plosives to kk', () => {
            expect(phonemeToViseme('K')).toBe('kk');
            expect(phonemeToViseme('G')).toBe('kk');
            expect(phonemeToViseme('NG')).toBe('kk');
        });

        it('maps postalveolar affricates to CH', () => {
            expect(phonemeToViseme('CH')).toBe('CH');
            expect(phonemeToViseme('JH')).toBe('CH');
            expect(phonemeToViseme('SH')).toBe('CH');
            expect(phonemeToViseme('ZH')).toBe('CH');
        });

        it('maps alveolar fricatives to SS', () => {
            expect(phonemeToViseme('S')).toBe('SS');
            expect(phonemeToViseme('Z')).toBe('SS');
        });

        it('maps open vowels to aa', () => {
            expect(phonemeToViseme('AA')).toBe('aa');
            expect(phonemeToViseme('AE')).toBe('aa');
            expect(phonemeToViseme('AH')).toBe('aa');
            expect(phonemeToViseme('AY')).toBe('aa');
        });

        it('maps front mid vowels to E', () => {
            expect(phonemeToViseme('EH')).toBe('E');
            expect(phonemeToViseme('ER')).toBe('E');
            expect(phonemeToViseme('EY')).toBe('E');
        });

        it('maps close front vowels to ih', () => {
            expect(phonemeToViseme('IH')).toBe('ih');
            expect(phonemeToViseme('IY')).toBe('ih');
        });

        it('maps back rounded vowels to oh/ou', () => {
            expect(phonemeToViseme('OW')).toBe('oh');
            expect(phonemeToViseme('OY')).toBe('oh');
            expect(phonemeToViseme('AO')).toBe('oh');
            expect(phonemeToViseme('UH')).toBe('ou');
            expect(phonemeToViseme('UW')).toBe('ou');
        });

        it('strips stress markers', () => {
            expect(phonemeToViseme('AA0')).toBe('aa');
            expect(phonemeToViseme('AA1')).toBe('aa');
            expect(phonemeToViseme('AA2')).toBe('aa');
            expect(phonemeToViseme('EY1')).toBe('E');
        });

        it('handles case insensitivity', () => {
            expect(phonemeToViseme('aa')).toBe('aa');
            expect(phonemeToViseme('Aa')).toBe('aa');
        });

        it('returns sil for unknown phonemes', () => {
            expect(phonemeToViseme('UNKNOWN')).toBe('sil');
            expect(phonemeToViseme('')).toBe('sil');
        });

        it('returns sil for silence markers', () => {
            expect(phonemeToViseme('SIL')).toBe('sil');
            expect(phonemeToViseme('SP')).toBe('sil');
        });
    });

    describe('mapPhonemeSequence', () => {
        it('returns empty array for empty input', () => {
            expect(mapPhonemeSequence([])).toEqual([]);
        });

        it('maps single phoneme to single frame', () => {
            const input: PhonemeInput[] = [
                { phoneme: 'AA', startMs: 0, durationMs: 100 },
            ];
            const result = mapPhonemeSequence(input);

            expect(result.length).toBeGreaterThan(0);
            expect(result[0].viseme).toBe('aa');
            expect(result[0].weight).toBe(1.0);
        });

        it('creates blend frames at transitions', () => {
            const input: PhonemeInput[] = [
                { phoneme: 'P', startMs: 0, durationMs: 100 },
                { phoneme: 'AA', startMs: 100, durationMs: 100 },
            ];
            const result = mapPhonemeSequence(input);

            // Should have main frames plus blend frames
            expect(result.length).toBeGreaterThan(2);

            // Check for PP and aa visemes
            const visemes = result.map(f => f.viseme);
            expect(visemes).toContain('PP');
            expect(visemes).toContain('aa');
        });

        it('does not create blend for same viseme', () => {
            const input: PhonemeInput[] = [
                { phoneme: 'P', startMs: 0, durationMs: 100 },
                { phoneme: 'B', startMs: 100, durationMs: 100 }, // Both map to PP
            ];
            const result = mapPhonemeSequence(input);

            // Only main frames, no blend needed
            expect(result.every(f => f.viseme === 'PP')).toBe(true);
        });
    });

    describe('getVisemeWeightsAtTime', () => {
        const frames: VisemeFrame[] = [
            { viseme: 'PP', weight: 1.0, startMs: 0, durationMs: 100 },
            { viseme: 'aa', weight: 1.0, startMs: 100, durationMs: 100 },
        ];

        it('returns correct weight during frame', () => {
            const weights = getVisemeWeightsAtTime(frames, 50);
            expect(weights['PP']).toBeGreaterThan(0);
        });

        it('returns empty weights before all frames', () => {
            const weights = getVisemeWeightsAtTime(frames, -10);
            expect(Object.keys(weights).length).toBe(0);
        });

        it('returns empty weights after all frames', () => {
            const weights = getVisemeWeightsAtTime(frames, 300);
            expect(Object.keys(weights).length).toBe(0);
        });

        it('applies smooth weight curve', () => {
            const weights1 = getVisemeWeightsAtTime(frames, 10);
            const weights2 = getVisemeWeightsAtTime(frames, 50);
            const weights3 = getVisemeWeightsAtTime(frames, 90);

            // Weight should increase toward middle then decrease
            expect(weights2['PP']).toBeGreaterThan(weights1['PP']);
        });
    });

    describe('VISEME_NAMES', () => {
        it('contains all 15 Oculus visemes', () => {
            expect(VISEME_NAMES.length).toBe(15);
            expect(VISEME_NAMES).toContain('sil');
            expect(VISEME_NAMES).toContain('PP');
            expect(VISEME_NAMES).toContain('FF');
            expect(VISEME_NAMES).toContain('TH');
            expect(VISEME_NAMES).toContain('DD');
            expect(VISEME_NAMES).toContain('kk');
            expect(VISEME_NAMES).toContain('CH');
            expect(VISEME_NAMES).toContain('SS');
            expect(VISEME_NAMES).toContain('nn');
            expect(VISEME_NAMES).toContain('RR');
            expect(VISEME_NAMES).toContain('aa');
            expect(VISEME_NAMES).toContain('E');
            expect(VISEME_NAMES).toContain('ih');
            expect(VISEME_NAMES).toContain('oh');
            expect(VISEME_NAMES).toContain('ou');
        });
    });
});
