/**
 * Action Executor - Robust action execution engine
 *
 * Features:
 * - Input validation with Zod-like schemas
 * - Configurable timeouts
 * - Retry logic with exponential backoff
 * - Request/response correlation via request_id
 * - Cancellation support
 * - Health monitoring
 * - Graceful error handling with context
 */

import type { ActionRequest, ActionOutput, ActionResult, ActionStatus } from './types';

// ============================================================================
// Validation
// ============================================================================

export interface ValidationError {
  field: string;
  message: string;
  value?: unknown;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

export type Validator<T> = (value: unknown) => ValidationResult & { data?: T };

// Schema validators
export const validators = {
  string: (opts?: { minLength?: number; maxLength?: number; pattern?: RegExp }) =>
    (value: unknown): ValidationResult & { data?: string } => {
      if (typeof value !== 'string') {
        return { valid: false, errors: [{ field: '', message: 'Expected string', value }] };
      }
      const errors: ValidationError[] = [];
      if (opts?.minLength !== undefined && value.length < opts.minLength) {
        errors.push({ field: '', message: `Minimum length is ${opts.minLength}`, value });
      }
      if (opts?.maxLength !== undefined && value.length > opts.maxLength) {
        errors.push({ field: '', message: `Maximum length is ${opts.maxLength}`, value });
      }
      if (opts?.pattern && !opts.pattern.test(value)) {
        errors.push({ field: '', message: 'Does not match required pattern', value });
      }
      return errors.length > 0 ? { valid: false, errors } : { valid: true, errors: [], data: value };
    },

  number: (opts?: { min?: number; max?: number; integer?: boolean }) =>
    (value: unknown): ValidationResult & { data?: number } => {
      if (typeof value !== 'number' || Number.isNaN(value)) {
        return { valid: false, errors: [{ field: '', message: 'Expected number', value }] };
      }
      const errors: ValidationError[] = [];
      if (opts?.min !== undefined && value < opts.min) {
        errors.push({ field: '', message: `Minimum value is ${opts.min}`, value });
      }
      if (opts?.max !== undefined && value > opts.max) {
        errors.push({ field: '', message: `Maximum value is ${opts.max}`, value });
      }
      if (opts?.integer && !Number.isInteger(value)) {
        errors.push({ field: '', message: 'Must be an integer', value });
      }
      return errors.length > 0 ? { valid: false, errors } : { valid: true, errors: [], data: value };
    },

  array: <T>(itemValidator?: Validator<T>) =>
    (value: unknown): ValidationResult & { data?: T[] } => {
      if (!Array.isArray(value)) {
        return { valid: false, errors: [{ field: '', message: 'Expected array', value }] };
      }
      if (!itemValidator) {
        return { valid: true, errors: [], data: value as T[] };
      }
      const errors: ValidationError[] = [];
      const data: T[] = [];
      for (let i = 0; i < value.length; i++) {
        const result = itemValidator(value[i]);
        if (!result.valid) {
          result.errors.forEach(e => errors.push({ ...e, field: `[${i}]${e.field}` }));
        } else if (result.data !== undefined) {
          data.push(result.data);
        }
      }
      return errors.length > 0 ? { valid: false, errors } : { valid: true, errors: [], data };
    },

  object: <T extends Record<string, unknown>>(schema: Record<keyof T, Validator<unknown>>) =>
    (value: unknown): ValidationResult & { data?: T } => {
      if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        return { valid: false, errors: [{ field: '', message: 'Expected object', value }] };
      }
      const obj = value as Record<string, unknown>;
      const errors: ValidationError[] = [];
      const data: Record<string, unknown> = {};

      for (const [key, validator] of Object.entries(schema)) {
        const result = validator(obj[key]);
        if (!result.valid) {
          result.errors.forEach(e => errors.push({ ...e, field: `${key}${e.field ? '.' + e.field : ''}` }));
        } else if (result.data !== undefined) {
          data[key] = result.data;
        }
      }
      return errors.length > 0 ? { valid: false, errors } : { valid: true, errors: [], data: data as T };
    },

  optional: <T>(validator: Validator<T>) =>
    (value: unknown): ValidationResult & { data?: T | undefined } => {
      if (value === undefined || value === null) {
        return { valid: true, errors: [], data: undefined };
      }
      return validator(value);
    },

  oneOf: <T extends string | number | boolean>(values: readonly T[]) =>
    (value: unknown): ValidationResult & { data?: T } => {
      if (!values.includes(value as T)) {
        return {
          valid: false,
          errors: [{ field: '', message: `Must be one of: ${values.map(v => String(v)).join(', ')}`, value }],
        };
      }
      return { valid: true, errors: [], data: value as T };
    },
};

// ============================================================================
// Action Schemas
// ============================================================================

export interface ActionSchema {
  type: string;
  description: string;
  payloadValidator: Validator<unknown>;
  timeoutMs: number;
  retryable: boolean;
  maxRetries: number;
}

export const ACTION_SCHEMAS: Record<string, ActionSchema> = {
  'service.start': {
    type: 'service.start',
    description: 'Start a service',
    payloadValidator: validators.object({
      serviceId: validators.string({ minLength: 1, maxLength: 128 }),
      args: validators.optional(validators.array(validators.string())),
      env: validators.optional(validators.object({})),
    }),
    timeoutMs: 30000,
    retryable: true,
    maxRetries: 2,
  },
  'service.stop': {
    type: 'service.stop',
    description: 'Stop a service',
    payloadValidator: validators.object({
      serviceId: validators.string({ minLength: 1, maxLength: 128 }),
      force: validators.optional(validators.oneOf([true, false] as const)),
      gracePeriodMs: validators.optional(validators.number({ min: 0, max: 60000, integer: true })),
    }),
    timeoutMs: 15000,
    retryable: false,
    maxRetries: 0,
  },
  'service.restart': {
    type: 'service.restart',
    description: 'Restart a service',
    payloadValidator: validators.object({
      serviceId: validators.string({ minLength: 1, maxLength: 128 }),
    }),
    timeoutMs: 45000,
    retryable: true,
    maxRetries: 1,
  },
  'service.logs': {
    type: 'service.logs',
    description: 'Fetch service logs',
    payloadValidator: validators.object({
      serviceId: validators.string({ minLength: 1, maxLength: 128 }),
      lines: validators.optional(validators.number({ min: 1, max: 10000, integer: true })),
      follow: validators.optional(validators.oneOf([true, false] as const)),
    }),
    timeoutMs: 10000,
    retryable: true,
    maxRetries: 2,
  },
  'composition.run': {
    type: 'composition.run',
    description: 'Run a composition/pipeline',
    payloadValidator: validators.object({
      compositionId: validators.string({ minLength: 1, maxLength: 128 }),
      args: validators.optional(validators.array(validators.string())),
      dryRun: validators.optional(validators.oneOf([true, false] as const)),
    }),
    timeoutMs: 300000, // 5 minutes
    retryable: false,
    maxRetries: 0,
  },
  'curation.trigger': {
    type: 'curation.trigger',
    description: 'Trigger curation workflow',
    payloadValidator: validators.object({
      source: validators.optional(validators.string({ maxLength: 64 })),
      force: validators.optional(validators.oneOf([true, false] as const)),
    }),
    timeoutMs: 60000,
    retryable: true,
    maxRetries: 2,
  },
  'worker.spawn': {
    type: 'worker.spawn',
    description: 'Spawn a worker process',
    payloadValidator: validators.object({
      provider: validators.optional(validators.string({ maxLength: 64 })),
      topic: validators.optional(validators.string({ maxLength: 128 })),
      count: validators.optional(validators.number({ min: 1, max: 10, integer: true })),
    }),
    timeoutMs: 30000,
    retryable: true,
    maxRetries: 2,
  },
  'verify.run': {
    type: 'verify.run',
    description: 'Run verification',
    payloadValidator: validators.object({
      target: validators.optional(validators.string({ maxLength: 256 })),
      quick: validators.optional(validators.oneOf([true, false] as const)),
    }),
    timeoutMs: 120000,
    retryable: true,
    maxRetries: 1,
  },
  'command.send': {
    type: 'command.send',
    description: 'Send custom command to bus',
    payloadValidator: validators.object({
      topic: validators.string({ minLength: 1, maxLength: 128 }),
      kind: validators.string({ minLength: 1, maxLength: 64 }),
      data: validators.object({}),
    }),
    timeoutMs: 30000,
    retryable: false,
    maxRetries: 0,
  },
};

// ============================================================================
// Error Types
// ============================================================================

export class ActionError extends Error {
  constructor(
    message: string,
    public readonly code: ActionErrorCode,
    public readonly context?: Record<string, unknown>,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = 'ActionError';
  }

  toOutput(): ActionOutput {
    return {
      type: 'error',
      content: this.message,
      timestamp: Date.now(),
      metadata: {
        title: this.code,
      },
    };
  }
}

export type ActionErrorCode =
  | 'VALIDATION_ERROR'
  | 'TIMEOUT_ERROR'
  | 'BUS_ERROR'
  | 'CANCELLED'
  | 'UNKNOWN_ACTION'
  | 'EXECUTION_ERROR'
  | 'RETRY_EXHAUSTED'
  | 'CONNECTION_ERROR';

// ============================================================================
// Bus Interface
// ============================================================================

export interface BusEvent {
  id?: string;
  topic: string;
  kind: string;
  level?: string;
  actor: string;
  ts: number;
  iso: string;
  data: unknown;
}

export interface BusConnection {
  emit: (event: BusEvent) => Promise<void>;
  subscribe: (pattern: string, callback: (event: BusEvent) => void) => () => void;
  isConnected: () => boolean;
  reconnect: () => Promise<void>;
}

// ============================================================================
// Action Executor
// ============================================================================

export interface ExecutionCallbacks {
  onOutput: (output: ActionOutput) => void;
  onComplete: (error?: ActionError) => void;
  onStatusChange: (status: ActionStatus) => void;
  onProgress: (current: number, total: number, message?: string) => void;
}

export interface ExecutionOptions {
  timeoutMs?: number;
  retryable?: boolean;
  maxRetries?: number;
  retryDelayMs?: number;
  signal?: AbortSignal;
}

// Required options with signal remaining optional (provided via abortController)
export interface ResolvedExecutionOptions {
  timeoutMs: number;
  retryable: boolean;
  maxRetries: number;
  retryDelayMs: number;
  signal?: AbortSignal;
}

export interface ExecutionContext {
  request: ActionRequest;
  schema: ActionSchema;
  callbacks: ExecutionCallbacks;
  options: ResolvedExecutionOptions;
  attempt: number;
  startedAt: number;
  abortController: AbortController;
}

export class ActionExecutor {
  private bus: BusConnection;
  private actor: string;
  private pendingResponses = new Map<string, {
    resolve: (event: BusEvent) => void;
    reject: (error: ActionError) => void;
    timeout: ReturnType<typeof setTimeout>;
  }>();
  private unsubscribe?: () => void;

  constructor(bus: BusConnection, actor = 'dashboard') {
    this.bus = bus;
    this.actor = actor;
    this.setupResponseListener();
  }

  private setupResponseListener(): void {
    // Subscribe to response events
    this.unsubscribe = this.bus.subscribe('*.response', (event) => {
      const requestId = (event.data as Record<string, unknown>)?.request_id as string;
      if (requestId && this.pendingResponses.has(requestId)) {
        const pending = this.pendingResponses.get(requestId)!;
        clearTimeout(pending.timeout);
        this.pendingResponses.delete(requestId);
        pending.resolve(event);
      }
    });

    // Also listen for error responses
    this.bus.subscribe('*.error', (event) => {
      const requestId = (event.data as Record<string, unknown>)?.request_id as string;
      if (requestId && this.pendingResponses.has(requestId)) {
        const pending = this.pendingResponses.get(requestId)!;
        clearTimeout(pending.timeout);
        this.pendingResponses.delete(requestId);
        pending.reject(new ActionError(
          (event.data as Record<string, unknown>)?.error as string || 'Unknown error',
          'EXECUTION_ERROR',
          { event }
        ));
      }
    });
  }

  destroy(): void {
    this.unsubscribe?.();
    for (const [, pending] of this.pendingResponses) {
      clearTimeout(pending.timeout);
      pending.reject(new ActionError('Executor destroyed', 'CANCELLED'));
    }
    this.pendingResponses.clear();
  }

  async execute(
    request: ActionRequest,
    callbacks: ExecutionCallbacks,
    options: ExecutionOptions = {}
  ): Promise<void> {
    // Get schema
    const schema = ACTION_SCHEMAS[request.type];
    if (!schema) {
      const error = new ActionError(
        `Unknown action type: ${request.type}`,
        'UNKNOWN_ACTION',
        { type: request.type, availableTypes: Object.keys(ACTION_SCHEMAS) }
      );
      callbacks.onOutput(error.toOutput());
      callbacks.onComplete(error);
      return;
    }

    // Validate payload
    const validation = schema.payloadValidator(request.payload);
    if (!validation.valid) {
      const error = new ActionError(
        `Validation failed: ${validation.errors.map(e => `${e.field}: ${e.message}`).join(', ')}`,
        'VALIDATION_ERROR',
        { errors: validation.errors }
      );
      callbacks.onOutput(error.toOutput());
      callbacks.onComplete(error);
      return;
    }

    // Setup execution context
    const ctx: ExecutionContext = {
      request,
      schema,
      callbacks,
      options: {
        timeoutMs: options.timeoutMs ?? schema.timeoutMs,
        retryable: options.retryable ?? schema.retryable,
        maxRetries: options.maxRetries ?? schema.maxRetries,
        retryDelayMs: options.retryDelayMs ?? 1000,
        signal: options.signal,
      },
      attempt: 0,
      startedAt: Date.now(),
      abortController: new AbortController(),
    };

    // Link external abort signal
    if (options.signal) {
      options.signal.addEventListener('abort', () => {
        ctx.abortController.abort();
      });
    }

    // Execute with retries
    await this.executeWithRetry(ctx);
  }

  private async executeWithRetry(ctx: ExecutionContext): Promise<void> {
    const { request, schema, callbacks, options } = ctx;

    while (ctx.attempt <= options.maxRetries) {
      ctx.attempt++;

      // Check for cancellation
      if (ctx.abortController.signal.aborted) {
        const error = new ActionError('Action cancelled', 'CANCELLED');
        callbacks.onComplete(error);
        return;
      }

      // Log attempt
      if (ctx.attempt > 1) {
        callbacks.onOutput({
          type: 'text',
          content: `Retry attempt ${ctx.attempt - 1}/${options.maxRetries}...`,
          timestamp: Date.now(),
        });
        // Exponential backoff
        const delay = options.retryDelayMs * Math.pow(2, ctx.attempt - 2);
        await this.sleep(delay);
      }

      try {
        await this.executeOnce(ctx);
        return; // Success
      } catch (err) {
        const actionError = err instanceof ActionError ? err : new ActionError(
          String(err),
          'EXECUTION_ERROR',
          undefined,
          err instanceof Error ? err : undefined
        );

        // Check if retryable
        const isRetryable = options.retryable &&
          ctx.attempt <= options.maxRetries &&
          actionError.code !== 'CANCELLED' &&
          actionError.code !== 'VALIDATION_ERROR';

        if (!isRetryable) {
          callbacks.onOutput(actionError.toOutput());
          callbacks.onComplete(actionError);
          return;
        }

        callbacks.onOutput({
          type: 'text',
          content: `Attempt ${ctx.attempt} failed: ${actionError.message}`,
          timestamp: Date.now(),
        });
      }
    }

    // Exhausted retries
    const error = new ActionError(
      `Action failed after ${options.maxRetries} retries`,
      'RETRY_EXHAUSTED',
      { attempts: ctx.attempt }
    );
    callbacks.onOutput(error.toOutput());
    callbacks.onComplete(error);
  }

  private async executeOnce(ctx: ExecutionContext): Promise<void> {
    const { request, schema, callbacks, options } = ctx;

    callbacks.onStatusChange('streaming');
    callbacks.onOutput({
      type: 'text',
      content: `Executing ${schema.description}...`,
      timestamp: Date.now(),
    });

    // Check bus connection
    if (!this.bus.isConnected()) {
      callbacks.onOutput({
        type: 'text',
        content: 'Bus disconnected, attempting reconnect...',
        timestamp: Date.now(),
      });
      try {
        await this.bus.reconnect();
      } catch (err) {
        throw new ActionError('Failed to connect to bus', 'CONNECTION_ERROR', undefined, err instanceof Error ? err : undefined);
      }
    }

    // Create bus event
    const busEvent: BusEvent = {
      id: request.id,
      topic: this.getTopicForAction(request.type),
      kind: this.getKindForAction(request.type),
      level: 'info',
      actor: this.actor,
      ts: Date.now(),
      iso: new Date().toISOString(),
      data: {
        request_id: request.id,
        ...request.payload,
      },
    };

    // Emit to bus
    callbacks.onOutput({
      type: 'json',
      content: { topic: busEvent.topic, kind: busEvent.kind, payload: request.payload },
      timestamp: Date.now(),
      metadata: { title: 'Request' },
    });

    await this.bus.emit(busEvent);

    // Wait for response with timeout
    const response = await this.waitForResponse(request.id, options.timeoutMs, ctx.abortController.signal);

    // Process response
    callbacks.onOutput({
      type: 'json',
      content: response.data as Record<string, unknown>,
      timestamp: Date.now(),
      metadata: { title: 'Response' },
    });

    callbacks.onStatusChange('success');
    callbacks.onComplete();
  }

  private waitForResponse(requestId: string, timeoutMs: number, signal: AbortSignal): Promise<BusEvent> {
    return new Promise((resolve, reject) => {
      // Check for immediate cancellation
      if (signal.aborted) {
        reject(new ActionError('Action cancelled', 'CANCELLED'));
        return;
      }

      // Setup timeout
      const timeout = setTimeout(() => {
        this.pendingResponses.delete(requestId);
        reject(new ActionError(
          `Action timed out after ${timeoutMs}ms`,
          'TIMEOUT_ERROR',
          { requestId, timeoutMs }
        ));
      }, timeoutMs);

      // Register pending response
      this.pendingResponses.set(requestId, { resolve, reject, timeout });

      // Listen for cancellation
      signal.addEventListener('abort', () => {
        clearTimeout(timeout);
        this.pendingResponses.delete(requestId);
        reject(new ActionError('Action cancelled', 'CANCELLED'));
      });
    });
  }

  private getTopicForAction(actionType: string): string {
    const [category, action] = actionType.split('.');
    return `${category}.control`;
  }

  private getKindForAction(actionType: string): string {
    const [, action] = actionType.split('.');
    return action || 'request';
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// File-based Bus Connection (for Node.js environments)
// ============================================================================

export interface FileBusOptions {
  busDir: string;
  eventsFile?: string;
}

export function createFileBusConnection(options: FileBusOptions): BusConnection {
  const { busDir, eventsFile = 'events.ndjson' } = options;
  const eventsPath = `${busDir}/${eventsFile}`;
  const subscribers = new Map<string, Set<(event: BusEvent) => void>>();
  let connected = false;
  let fileHandle: { appendFile: (data: string) => Promise<void> } | null = null;

  return {
    async emit(event: BusEvent): Promise<void> {
      if (!connected) {
        throw new ActionError('Bus not connected', 'CONNECTION_ERROR');
      }
      const line = JSON.stringify(event) + '\n';
      // In browser, this would go through WebSocket
      // In Node.js, write to file
      if (typeof window === 'undefined') {
        const fs = await import('fs/promises');
        await fs.appendFile(eventsPath, line, 'utf-8');
      } else {
        // Browser - send via WebSocket (handled by dashboard's WS connection)
        console.log('[FileBus] Would emit:', event);
      }
    },

    subscribe(pattern: string, callback: (event: BusEvent) => void): () => void {
      if (!subscribers.has(pattern)) {
        subscribers.set(pattern, new Set());
      }
      subscribers.get(pattern)!.add(callback);

      return () => {
        subscribers.get(pattern)?.delete(callback);
      };
    },

    isConnected(): boolean {
      return connected;
    },

    async reconnect(): Promise<void> {
      // In browser, reconnect WebSocket
      // In Node.js, verify file access
      if (typeof window === 'undefined') {
        const fs = await import('fs/promises');
        try {
          await fs.access(eventsPath);
          connected = true;
        } catch {
          // Create directory and file if needed
          await fs.mkdir(busDir, { recursive: true });
          await fs.writeFile(eventsPath, '', 'utf-8');
          connected = true;
        }
      } else {
        // Browser - assume WS will handle
        connected = true;
      }
    },
  };
}

// ============================================================================
// WebSocket Bus Connection (for browser environments)
// ============================================================================

export interface WebSocketBusOptions {
  url: string;
  reconnectDelayMs?: number;
  maxReconnectAttempts?: number;
}

export function createWebSocketBusConnection(options: WebSocketBusOptions): BusConnection & {
  onMessage: (handler: (event: BusEvent) => void) => void;
  close: () => void;
} {
  const { url, reconnectDelayMs = 1000, maxReconnectAttempts = 10 } = options;
  const subscribers = new Map<string, Set<(event: BusEvent) => void>>();
  let ws: WebSocket | null = null;
  let reconnectAttempts = 0;
  let messageHandler: ((event: BusEvent) => void) | null = null;

  function matchPattern(pattern: string, topic: string): boolean {
    if (pattern === '*') return true;
    if (pattern.endsWith('*')) {
      return topic.startsWith(pattern.slice(0, -1));
    }
    return pattern === topic;
  }

  function notifySubscribers(event: BusEvent): void {
    for (const [pattern, callbacks] of subscribers) {
      if (matchPattern(pattern, event.topic)) {
        for (const callback of callbacks) {
          try {
            callback(event);
          } catch (err) {
            console.error('[WebSocketBus] Subscriber error:', err);
          }
        }
      }
    }
    messageHandler?.(event);
  }

  function connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        ws = new WebSocket(url);

        ws.onopen = () => {
          reconnectAttempts = 0;
          resolve();
        };

        ws.onmessage = (msg) => {
          try {
            const data = JSON.parse(msg.data);
            if (data.type === 'event' && data.event) {
              notifySubscribers(data.event);
            } else if (data.type === 'sync' && data.events) {
              for (const event of data.events) {
                notifySubscribers(event);
              }
            }
          } catch {
            // Ignore malformed messages
          }
        };

        ws.onclose = () => {
          ws = null;
        };

        ws.onerror = (err) => {
          reject(new ActionError('WebSocket error', 'CONNECTION_ERROR'));
        };
      } catch (err) {
        reject(new ActionError('Failed to create WebSocket', 'CONNECTION_ERROR'));
      }
    });
  }

  return {
    async emit(event: BusEvent): Promise<void> {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        throw new ActionError('WebSocket not connected', 'CONNECTION_ERROR');
      }
      ws.send(JSON.stringify({ type: 'publish', event }));
    },

    subscribe(pattern: string, callback: (event: BusEvent) => void): () => void {
      if (!subscribers.has(pattern)) {
        subscribers.set(pattern, new Set());
      }
      subscribers.get(pattern)!.add(callback);

      // Send subscribe message to server
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'subscribe', topic: pattern }));
      }

      return () => {
        subscribers.get(pattern)?.delete(callback);
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'unsubscribe', topic: pattern }));
        }
      };
    },

    isConnected(): boolean {
      return ws?.readyState === WebSocket.OPEN;
    },

    async reconnect(): Promise<void> {
      if (ws?.readyState === WebSocket.OPEN) return;

      if (reconnectAttempts >= maxReconnectAttempts) {
        throw new ActionError(
          `Max reconnect attempts (${maxReconnectAttempts}) exceeded`,
          'CONNECTION_ERROR'
        );
      }

      reconnectAttempts++;
      const delay = reconnectDelayMs * Math.pow(2, reconnectAttempts - 1);

      await new Promise(resolve => setTimeout(resolve, delay));
      await connect();
    },

    onMessage(handler: (event: BusEvent) => void): void {
      messageHandler = handler;
    },

    close(): void {
      ws?.close();
      ws = null;
    },
  };
}

// ============================================================================
// Health Monitor
// ============================================================================

export interface HealthStatus {
  healthy: boolean;
  busConnected: boolean;
  pendingActions: number;
  lastCheck: string;
  errors: string[];
}

export class ActionHealthMonitor {
  private executor: ActionExecutor;
  private bus: BusConnection;
  private checkIntervalMs: number;
  private intervalId?: ReturnType<typeof setInterval>;
  private errors: string[] = [];
  private maxErrors = 10;

  constructor(executor: ActionExecutor, bus: BusConnection, checkIntervalMs = 30000) {
    this.executor = executor;
    this.bus = bus;
    this.checkIntervalMs = checkIntervalMs;
  }

  start(): void {
    this.intervalId = setInterval(() => this.check(), this.checkIntervalMs);
    this.check();
  }

  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }

  recordError(error: string): void {
    this.errors.push(`${new Date().toISOString()}: ${error}`);
    if (this.errors.length > this.maxErrors) {
      this.errors.shift();
    }
  }

  getStatus(): HealthStatus {
    return {
      healthy: this.bus.isConnected() && this.errors.length < 5,
      busConnected: this.bus.isConnected(),
      pendingActions: 0, // Would track from executor
      lastCheck: new Date().toISOString(),
      errors: [...this.errors],
    };
  }

  private async check(): Promise<void> {
    if (!this.bus.isConnected()) {
      this.recordError('Bus disconnected');
      try {
        await this.bus.reconnect();
      } catch (err) {
        this.recordError(`Reconnect failed: ${err}`);
      }
    }
  }
}
