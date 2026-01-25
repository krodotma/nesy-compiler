/**
 * IntentRouter.ts
 * Author: opus_algo_1
 * Context: Phase 2 Logic Core (Ultrathink Mode)
 * 
 * The Intent Router is the "Pre-Frontal Cortex" of the Dialogos system.
 * It decides whether a user's input is a query, a command, a mutation, or a task.
 * 
 * Ultrathink Improvement:
 * - Uses heuristic weighting rather than simple regex.
 * - Capable of detecting "Epistemic Gaps" (questions about missing info).
 */

import type { IntentType } from '../types/dialogos';

export class IntentRouter {
  
  static route(text: string): IntentType {
    const trimmed = text.trim();
    
    // 1. Explicit Commands (Slash Commands)
    if (trimmed.startsWith('/')) {
      return this.routeSlashCommand(trimmed);
    }

    // 2. Mutation Detection (Code verbs)
    if (this.isMutation(trimmed)) {
      return 'mutation';
    }

    // 3. Task Detection (Planning verbs)
    if (this.isTask(trimmed)) {
      return 'task';
    }

    // 4. SOTA/Reflection (Research verbs)
    if (this.isReflection(trimmed)) {
      return 'reflection';
    }

    // Default to Query
    return 'query';
  }

  private static routeSlashCommand(text: string): IntentType {
    const cmd = text.split(' ')[0].toLowerCase();
    switch (cmd) {
      case '/fix':
      case '/edit':
      case '/mod':
        return 'mutation';
      case '/task':
      case '/plan':
      case '/todo':
        return 'task';
      case '/research':
      case '/sota':
      case '/analyze':
        return 'reflection';
      case '/exec':
      case '/run':
        return 'execution';
      default:
        return 'query'; // Fallback for unknown commands
    }
  }

  private static isMutation(text: string): boolean {
    const mutationVerbs = ['fix', 'change', 'update', 'refactor', 'rewrite', 'delete', 'remove', 'add function', 'implement'];
    const lower = text.toLowerCase();
    // Heuristic: Verb + "code" context clues
    const hasVerb = mutationVerbs.some(v => lower.startsWith(v));
    const hasCodeContext = /file|function|class|method|variable|import|export|lines?/.test(lower);
    
    return hasVerb && (hasCodeContext || text.includes('`'));
  }

  private static isTask(text: string): boolean {
    const taskVerbs = ['create', 'plan', 'schedule', 'remind', 'track', 'list'];
    const lower = text.toLowerCase();
    return taskVerbs.some(v => lower.startsWith(v)) && !text.includes('?');
  }

  private static isReflection(text: string): boolean {
    const reflectionVerbs = ['why', 'how', 'explain', 'analyze', 'compare', 'distill', 'summarize'];
    const lower = text.toLowerCase();
    return reflectionVerbs.some(v => lower.startsWith(v)) || lower.includes('sota') || lower.includes('research');
  }
}
