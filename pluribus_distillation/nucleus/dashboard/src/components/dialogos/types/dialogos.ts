/**
 * Dialogos Data Models (Schema)
 * Author: opus_architect_1
 * Context: Foundation Layer
 */

import type { QRL } from '@builder.io/qwik';

// 1. The Atom - Fundamental Unit of Dialogos
export interface DialogosAtom {
  id: string;
  timestamp: number;
  author: AgentIdentity;
  intent: IntentType;
  content: DialogosContent;
  context: ContextSnapshot;
  state: AtomState;
  // Rheomode links
  causes: string[]; // IDs of atoms that caused this one
  effects: string[]; // IDs of atoms caused by this one
}

export interface AgentIdentity {
  id: string;
  name: string;
  role: 'human' | 'agent' | 'system';
  avatar?: string; // URL or generated hash
  color?: string; // M3 Token
}

export type IntentType = 
  | 'query'       // "How do I..."
  | 'mutation'    // "Fix this..."
  | 'task'        // "Create a plan..."
  | 'reflection'  // "Analyze..."
  | 'clarification' // "Did you mean..."
  | 'execution';  // "Running command..."

export type DialogosContent = 
  | { type: 'text'; value: string }
  | { type: 'code'; language: string; value: string; diff?: string }
  | { type: 'task'; title: string; status: 'todo' | 'doing' | 'done'; laneId?: string }
  | { type: 'sota'; title: string; url: string; summary: string }
  | { type: 'plan'; steps: string[]; progress: number };

export interface ContextSnapshot {
  url: string;
  selection?: string; // Selected text
  activeFile?: string; // IPE context
  viewport?: { width: number; height: number };
  systemLoad?: number; // CPU/Memory pressure
}

export type AtomState = 
  | 'potential'   // Draft / Thinking
  | 'actualizing' // Streaming / Executing
  | 'actualized'  // Done / Sent
  | 'rejected'    // Cancelled / Error
  | 'archived';   // History

// 2. The Store State
export interface DialogosState {
  isOpen: boolean;
  mode: 'rest' | 'active' | 'full'; // Physics states
  activeSessionId: string | null;
  atoms: Record<string, DialogosAtom>; // Normalized store
  timeline: string[]; // Ordered IDs
  inputDraft: string;
  pendingAttachments: File[];
  isThinking: boolean;
}

// 3. Actions
export interface DialogosActions {
  submit$: QRL<(text: string, attachments?: File[]) => void>;
  cancel$: QRL<() => void>;
  retry$: QRL<(atomId: string) => void>;
  branch$: QRL<(atomId: string) => void>; // Create parallel timeline
}
