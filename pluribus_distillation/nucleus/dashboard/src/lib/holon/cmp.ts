/**
 * CMP Metrics & Math
 * [Ultrathink Agent 1: Architect]
 * 
 * "Team, the previous phase was surgical. Now we need *insight*. 
 *  We break the monolith of CMP into four orthogonal vectors."
 * 
 * 1. Velocity: Development speed / churn rate.
 * 2. Quality: Test coverage / lint compliance.
 * 3. Stability: Runtime uptime / error rate inverse.
 * 4. Longevity: Motif recurrence / architectural fit.
 */

export interface CMPVector {
  velocity: number;  // 0-100
  quality: number;   // 0-100
  stability: number; // 0-100
  longevity: number; // 0-100
}

/**
 * Calculate aggregate CMP from vectors using weighted harmonic mean.
 * (Harmonic mean punishes low values in any single dimension).
 */
export function calculateAggregateCMP(v: CMPVector): number {
  const values = [v.velocity, v.quality, v.stability, v.longevity];
  const zeroCheck = values.some(x => x === 0);
  if (zeroCheck) return 0;
  
  return 4 / (
    (1 / v.velocity) + 
    (1 / v.quality) + 
    (1 / v.stability) + 
    (1 / v.longevity)
  );
}

export const COHORT_AVERAGE: CMPVector = {
  velocity: 65,
  quality: 70,
  stability: 85,
  longevity: 60
};
