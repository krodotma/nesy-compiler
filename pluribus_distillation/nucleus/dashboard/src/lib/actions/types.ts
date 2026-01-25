/**
 * Action System Types
 *
 * Defines the protocol for server actions with notebook-style output.
 */

import type { BusEvent } from '../state/types';

export type ActionStatus = 'idle' | 'pending' | 'streaming' | 'success' | 'error';

export type ActionCellTab = 'outputs' | 'request' | 'trace' | 'artifacts';

export interface ActionRequest {
  id: string;
  type: string;
  payload: Record<string, unknown>;
  timestamp: number;
}

export interface ActionOutput {
  type: 'text' | 'code' | 'json' | 'table' | 'progress' | 'error';
  content: string | Record<string, unknown> | unknown[];
  timestamp: number;
  metadata?: {
    language?: string;
    title?: string;
    progress?: number;
    total?: number;
  };
}

export interface ActionResult {
  id: string;
  requestId: string;
  status: ActionStatus;
  outputs: ActionOutput[];
  events: BusEvent[]; // Raw correlated bus events (for trace view)
  startedAt: number;
  completedAt?: number;
  error?: string;
}

export interface ActionCell {
  id: string;
  request: ActionRequest;
  result: ActionResult | null;
  collapsed: boolean;
  activeTab: ActionCellTab;
}

// Service-specific action types
export type ServiceActionType =
  | 'service.start'
  | 'service.stop'
  | 'service.restart'
  | 'service.logs'
  | 'composition.run';

export interface ServiceActionPayload {
  serviceId: string;
  instanceId?: string;
  args?: string[];
  env?: Record<string, string>;
}

// Generic action types
export type SystemActionType =
  | 'curation.trigger'
  | 'worker.spawn'
  | 'verify.run'
  | 'command.send';

export interface CommandActionPayload {
  topic: string;
  kind: string;
  data: Record<string, unknown>;
}

// InferCell action types
export type InferCellActionType =
  | 'infercell.verify'
  | 'infercell.inspect'
  | 'infercell.fork'
  | 'infercell.test'
  | 'infercell.emit'
  | 'infercell.merge'
  | 'infercell.pause'
  | 'infercell.resume';

export interface InferCellActionPayload {
  moduleName: string;
  moduleFile: string;
  goldenId?: string;
  traceId?: string;
  action: string;
  params?: Record<string, unknown>;
}

export interface InferCellEvent {
  id: string;
  traceId: string;
  cellId: string;
  parentTraceId?: string;
  state: 'pending' | 'active' | 'paused' | 'complete' | 'merged';
  moduleName: string;
  moduleFile: string;
  action: string;
  timestamp: string;
  result?: {
    success: boolean;
    data?: unknown;
    error?: string;
  };
}
