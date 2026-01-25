/**
 * Entelexis Tracker (The Realization Engine)
 * Author: opus_algo_1
 * Context: Phase 4 - State Machine Integration
 * 
 * Maps Dialogos interactions to Entelexis states:
 * Potential -> Actualizing -> Actualized
 */

import type { DialogosAtom, AtomState } from '../types/dialogos';

export class EntelexisTracker {
  
  static mapAtomToState(atom: DialogosAtom): string {
    // 1. Initial State
    if (atom.state === 'potential') {
      return 'ENTELEXIS_POTENTIAL_DETECTED';
    }

    // 2. Processing
    if (atom.state === 'actualizing') {
      if (atom.intent === 'mutation') return 'ENTELEXIS_MUTATING_CODE';
      if (atom.intent === 'task') return 'ENTELEXIS_PLANNING_TASK';
      return 'ENTELEXIS_INFERRING';
    }

    // 3. Completion
    if (atom.state === 'actualized') {
      return 'ENTELEXIS_ACTUALIZED';
    }

    return 'ENTELEXIS_IDLE';
  }

  static getVisualCue(entelexisState: string): string {
    switch (entelexisState) {
      case 'ENTELEXIS_POTENTIAL_DETECTED':
        return 'animate-pulse border-white/30';
      case 'ENTELEXIS_MUTATING_CODE':
        return 'animate-glitch border-[var(--glass-accent-magenta)]';
      case 'ENTELEXIS_PLANNING_TASK':
        return 'animate-shimmer border-[var(--glass-accent-cyan)]';
      case 'ENTELEXIS_ACTUALIZED':
        return 'border-[var(--chroma-success)] shadow-lg';
      default:
        return '';
    }
  }
}
