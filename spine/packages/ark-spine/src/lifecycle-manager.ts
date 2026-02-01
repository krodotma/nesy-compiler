/**
 * @ark/spine/lifecycle-manager - Module Lifecycle Management
 *
 * Manages module lifecycle transitions with hooks, timeouts, and retries.
 *
 * @module
 */

import type {
  ModuleInstance,
  ModuleState,
  LifecycleHook,
  LifecycleHandler,
  LifecycleContext,
  LifecycleConfig,
  OperationResult,
} from './types.js';

/**
 * Valid state transitions
 */
const STATE_TRANSITIONS: Record<ModuleState, ModuleState[]> = {
  unloaded: ['loading'],
  loading: ['loaded', 'error'],
  loaded: ['initializing', 'unloaded'],
  initializing: ['ready', 'error'],
  ready: ['starting', 'disposed'],
  starting: ['running', 'error'],
  running: ['stopping', 'error'],
  stopping: ['stopped', 'error'],
  stopped: ['starting', 'disposed'],
  error: ['unloaded', 'disposed'],
  disposed: [],
};

/**
 * Lifecycle Manager - Manages module lifecycle transitions
 */
export class LifecycleManager {
  private hooks: Map<LifecycleHook, Set<LifecycleHandler>> = new Map();
  private config: Required<LifecycleConfig>;
  private contextProvider?: () => LifecycleContext;

  constructor(config: LifecycleConfig = {}) {
    this.config = {
      defaultTimeout: config.defaultTimeout ?? 30000, // 30 seconds
      retryCount: config.retryCount ?? 3,
      retryDelay: config.retryDelay ?? 1000, // 1 second
      parallelOps: config.parallelOps ?? false,
    };
  }

  /**
   * Set context provider for lifecycle hooks
   */
  setContextProvider(provider: () => LifecycleContext): void {
    this.contextProvider = provider;
  }

  /**
   * Register a lifecycle hook handler
   */
  onHook(hook: LifecycleHook, handler: LifecycleHandler): () => void {
    if (!this.hooks.has(hook)) {
      this.hooks.set(hook, new Set());
    }

    this.hooks.get(hook)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.hooks.get(hook);
      if (handlers) {
        handlers.delete(handler);
      }
    };
  }

  /**
   * Remove a lifecycle hook handler
   */
  offHook(hook: LifecycleHook, handler: LifecycleHandler): boolean {
    const handlers = this.hooks.get(hook);
    if (!handlers) {
      return false;
    }
    return handlers.delete(handler);
  }

  /**
   * Clear all handlers for a hook
   */
  clearHook(hook: LifecycleHook): void {
    this.hooks.delete(hook);
  }

  /**
   * Clear all hooks
   */
  clearAllHooks(): void {
    this.hooks.clear();
  }

  /**
   * Trigger a lifecycle hook
   */
  async triggerHook(
    hook: LifecycleHook,
    instance: ModuleInstance,
    context?: Partial<LifecycleContext>
  ): Promise<void> {
    const handlers = this.hooks.get(hook);
    if (!handlers || handlers.size === 0) {
      return;
    }

    const fullContext = this.buildContext(context);

    const errors: Error[] = [];

    for (const handler of handlers) {
      try {
        await handler(instance, fullContext);
      } catch (error) {
        errors.push(error instanceof Error ? error : new Error(String(error)));
      }
    }

    if (errors.length > 0) {
      throw new AggregateError(
        errors,
        `${errors.length} hook handler(s) failed for ${hook}`
      );
    }
  }

  /**
   * Check if a state transition is valid
   */
  canTransition(from: ModuleState, to: ModuleState): boolean {
    const allowed = STATE_TRANSITIONS[from];
    return allowed.includes(to);
  }

  /**
   * Get allowed transitions from a state
   */
  getAllowedTransitions(from: ModuleState): ModuleState[] {
    return [...STATE_TRANSITIONS[from]];
  }

  /**
   * Perform a state transition with hooks
   */
  async transition(
    instance: ModuleInstance,
    to: ModuleState,
    updateState: (state: ModuleState) => void,
    options?: {
      timeout?: number;
      context?: Partial<LifecycleContext>;
    }
  ): Promise<OperationResult> {
    const startTime = Date.now();
    const from = instance.state;

    // Validate transition
    if (!this.canTransition(from, to)) {
      return {
        success: false,
        error: `Invalid transition: ${from} -> ${to}`,
        duration: Date.now() - startTime,
      };
    }

    const timeout = options?.timeout ?? this.config.defaultTimeout;
    const context = options?.context;

    try {
      // Execute transition with timeout
      await this.withTimeout(
        async () => {
          // Trigger before hook
          const beforeHook = this.getBeforeHook(to);
          if (beforeHook) {
            await this.triggerHook(beforeHook, instance, context);
          }

          // Update state
          updateState(to);

          // Trigger after hook
          const afterHook = this.getAfterHook(to);
          if (afterHook) {
            await this.triggerHook(afterHook, { ...instance, state: to }, context);
          }
        },
        timeout
      );

      return {
        success: true,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);

      // Trigger error hook
      try {
        await this.triggerHook(
          'onError',
          { ...instance, state: 'error', error: errorMessage },
          context
        );
      } catch {
        // Ignore errors in error hook
      }

      return {
        success: false,
        error: errorMessage,
        duration: Date.now() - startTime,
      };
    }
  }

  /**
   * Load a module (unloaded -> loading -> loaded)
   */
  async load(
    instance: ModuleInstance,
    loadFn: () => Promise<void>,
    updateState: (state: ModuleState) => void,
    options?: { timeout?: number; context?: Partial<LifecycleContext> }
  ): Promise<OperationResult> {
    const startTime = Date.now();

    // Transition to loading
    let result = await this.transition(instance, 'loading', updateState, options);
    if (!result.success) {
      return result;
    }

    try {
      // Execute load function
      await this.withTimeout(
        loadFn,
        options?.timeout ?? this.config.defaultTimeout
      );

      // Transition to loaded
      result = await this.transition(
        { ...instance, state: 'loading' },
        'loaded',
        updateState,
        options
      );

      return {
        success: result.success,
        error: result.error,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      // Transition to error
      updateState('error');
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime,
      };
    }
  }

  /**
   * Initialize a module (loaded -> initializing -> ready)
   */
  async initialize(
    instance: ModuleInstance,
    initFn: () => Promise<void>,
    updateState: (state: ModuleState) => void,
    options?: { timeout?: number; context?: Partial<LifecycleContext> }
  ): Promise<OperationResult> {
    const startTime = Date.now();

    // Transition to initializing
    let result = await this.transition(
      instance,
      'initializing',
      updateState,
      options
    );
    if (!result.success) {
      return result;
    }

    try {
      // Execute init function
      await this.withTimeout(
        initFn,
        options?.timeout ?? this.config.defaultTimeout
      );

      // Transition to ready
      result = await this.transition(
        { ...instance, state: 'initializing' },
        'ready',
        updateState,
        options
      );

      return {
        success: result.success,
        error: result.error,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      updateState('error');
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime,
      };
    }
  }

  /**
   * Start a module (ready/stopped -> starting -> running)
   */
  async start(
    instance: ModuleInstance,
    startFn: () => Promise<void>,
    updateState: (state: ModuleState) => void,
    options?: { timeout?: number; context?: Partial<LifecycleContext> }
  ): Promise<OperationResult> {
    const startTime = Date.now();

    // Transition to starting
    let result = await this.transition(
      instance,
      'starting',
      updateState,
      options
    );
    if (!result.success) {
      return result;
    }

    try {
      // Execute start function
      await this.withTimeout(
        startFn,
        options?.timeout ?? this.config.defaultTimeout
      );

      // Transition to running
      result = await this.transition(
        { ...instance, state: 'starting' },
        'running',
        updateState,
        options
      );

      return {
        success: result.success,
        error: result.error,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      updateState('error');
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime,
      };
    }
  }

  /**
   * Stop a module (running -> stopping -> stopped)
   */
  async stop(
    instance: ModuleInstance,
    stopFn: () => Promise<void>,
    updateState: (state: ModuleState) => void,
    options?: { timeout?: number; context?: Partial<LifecycleContext> }
  ): Promise<OperationResult> {
    const startTime = Date.now();

    // Transition to stopping
    let result = await this.transition(
      instance,
      'stopping',
      updateState,
      options
    );
    if (!result.success) {
      return result;
    }

    try {
      // Execute stop function with retry
      await this.withRetry(
        () =>
          this.withTimeout(stopFn, options?.timeout ?? this.config.defaultTimeout),
        this.config.retryCount,
        this.config.retryDelay
      );

      // Transition to stopped
      result = await this.transition(
        { ...instance, state: 'stopping' },
        'stopped',
        updateState,
        options
      );

      return {
        success: result.success,
        error: result.error,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      updateState('error');
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime,
      };
    }
  }

  /**
   * Dispose a module (stopped/ready/error -> disposed)
   */
  async dispose(
    instance: ModuleInstance,
    disposeFn: () => Promise<void>,
    updateState: (state: ModuleState) => void,
    options?: { timeout?: number; context?: Partial<LifecycleContext> }
  ): Promise<OperationResult> {
    const startTime = Date.now();

    // Check if already disposed
    if (instance.state === 'disposed') {
      return {
        success: true,
        duration: 0,
      };
    }

    // Trigger before dispose hook
    const context = options?.context;
    try {
      await this.triggerHook('beforeDispose', instance, context);
    } catch {
      // Continue even if hook fails
    }

    try {
      // Execute dispose function
      await this.withTimeout(
        disposeFn,
        options?.timeout ?? this.config.defaultTimeout
      );

      // Update state
      updateState('disposed');

      // Trigger after dispose hook
      try {
        await this.triggerHook(
          'afterDispose',
          { ...instance, state: 'disposed' },
          context
        );
      } catch {
        // Ignore errors in after hook
      }

      return {
        success: true,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime,
      };
    }
  }

  /**
   * Perform health check
   */
  async healthCheck(
    instance: ModuleInstance,
    checkFn: () => Promise<boolean>,
    options?: { timeout?: number; context?: Partial<LifecycleContext> }
  ): Promise<OperationResult<boolean>> {
    const startTime = Date.now();

    try {
      // Trigger health check hook
      await this.triggerHook('onHealthCheck', instance, options?.context);

      // Execute check function
      const healthy = await this.withTimeout(
        checkFn,
        options?.timeout ?? 5000 // Shorter timeout for health checks
      );

      return {
        success: true,
        data: healthy,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        data: false,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime,
      };
    }
  }

  // ==================== Helper Methods ====================

  /**
   * Get before hook for a target state
   */
  private getBeforeHook(state: ModuleState): LifecycleHook | null {
    switch (state) {
      case 'loading':
        return 'beforeLoad';
      case 'initializing':
        return 'beforeInit';
      case 'starting':
        return 'beforeStart';
      case 'stopping':
        return 'beforeStop';
      case 'disposed':
        return 'beforeDispose';
      default:
        return null;
    }
  }

  /**
   * Get after hook for a target state
   */
  private getAfterHook(state: ModuleState): LifecycleHook | null {
    switch (state) {
      case 'loaded':
        return 'afterLoad';
      case 'ready':
        return 'afterInit';
      case 'running':
        return 'afterStart';
      case 'stopped':
        return 'afterStop';
      case 'disposed':
        return 'afterDispose';
      default:
        return null;
    }
  }

  /**
   * Build lifecycle context
   */
  private buildContext(partial?: Partial<LifecycleContext>): LifecycleContext {
    const base: LifecycleContext = this.contextProvider
      ? this.contextProvider()
      : {
          coordinator: null,
          registry: null,
          timestamp: Date.now(),
        };

    return {
      ...base,
      ...partial,
      timestamp: Date.now(),
    };
  }

  /**
   * Execute with timeout
   */
  private async withTimeout<T>(
    fn: () => Promise<T>,
    timeout: number
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Operation timed out after ${timeout}ms`));
      }, timeout);

      fn()
        .then((result) => {
          clearTimeout(timer);
          resolve(result);
        })
        .catch((error) => {
          clearTimeout(timer);
          reject(error);
        });
    });
  }

  /**
   * Execute with retry
   */
  private async withRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number,
    delay: number
  ): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (attempt < maxRetries) {
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError;
  }

  /**
   * Get configuration
   */
  getConfig(): Required<LifecycleConfig> {
    return { ...this.config };
  }

  /**
   * Get registered hooks
   */
  getRegisteredHooks(): LifecycleHook[] {
    return Array.from(this.hooks.keys());
  }

  /**
   * Get handler count for a hook
   */
  getHandlerCount(hook: LifecycleHook): number {
    const handlers = this.hooks.get(hook);
    return handlers ? handlers.size : 0;
  }
}

/**
 * Create a new lifecycle manager
 */
export function createLifecycleManager(
  config?: LifecycleConfig
): LifecycleManager {
  return new LifecycleManager(config);
}
