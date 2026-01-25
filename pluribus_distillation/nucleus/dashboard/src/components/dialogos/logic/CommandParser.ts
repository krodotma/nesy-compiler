/**
 * CommandParser.ts
 * Author: opus_algo_1
 * Context: Phase 2 Logic Core
 * 
 * Parses raw text into structured intent payloads.
 * Handles the extraction of parameters, flags, and contexts from natural language.
 */

import type { IntentType, DialogosContent } from '../types/dialogos';

export interface ParsedCommand {
  intent: IntentType;
  args: string[];
  flags: Record<string, boolean | string>;
  raw: string;
  structuredContent: DialogosContent;
}

export class CommandParser {
  
  static parse(text: string, intent: IntentType): ParsedCommand {
    const args: string[] = [];
    const flags: Record<string, boolean | string> = {};
    
    // Basic tokenizer (respecting quotes would be the next "Ultrathink" step, simplistic for now)
    const tokens = text.split(' ');
    
    tokens.forEach(token => {
      if (token.startsWith('--')) {
        const parts = token.substring(2).split('=');
        flags[parts[0]] = parts.length > 1 ? parts[1] : true;
      } else if (token.startsWith('-')) {
        flags[token.substring(1)] = true;
      } else {
        args.push(token);
      }
    });

    return {
      intent,
      args,
      flags,
      raw: text,
      structuredContent: this.generateContent(text, intent, flags)
    };
  }

  private static generateContent(text: string, intent: IntentType, flags: Record<string, any>): DialogosContent {
    switch (intent) {
      case 'task':
        return {
          type: 'task',
          title: text.replace(/^\/(task|plan|todo)\s*/i, ''),
          status: 'todo',
          laneId: (flags['lane'] as string) || 'inbox'
        };
      case 'mutation':
        // Detect if it's a code block
        if (text.includes('```')) {
           const match = text.match(/```(\w*)\n([\s\S]*?)```/);
           if (match) {
             return {
               type: 'code',
               language: match[1] || 'typescript',
               value: match[2]
             };
           }
        }
        return { type: 'text', value: text };
      case 'reflection':
        return {
          type: 'sota',
          title: 'Research Request',
          url: (flags['url'] as string) || '',
          summary: text
        };
      default:
        return { type: 'text', value: text };
    }
  }
}
