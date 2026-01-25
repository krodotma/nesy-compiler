/**
 * Automation API for Lanes
 *
 * Phase 8, Iteration 68 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Scripting API for lane operations
 * - Event triggers
 * - Scheduled tasks
 * - Webhook integration
 * - Rule-based automation
 */

import type { Lane, LaneAction, LanesState } from './store';
import type { CascadeEngine } from './cascade';

// ============================================================================
// Types
// ============================================================================

export interface AutomationRule {
  id: string;
  name: string;
  description?: string;
  enabled: boolean;
  trigger: AutomationTrigger;
  conditions: AutomationCondition[];
  actions: AutomationAction[];
  priority: number;
  cooldownMs?: number;
  lastTriggered?: number;
}

export type AutomationTrigger =
  | { type: 'lane.status_changed'; laneId?: string; fromStatus?: string; toStatus?: string }
  | { type: 'lane.wip_changed'; laneId?: string; threshold?: number; direction?: 'above' | 'below' }
  | { type: 'lane.blocker_added'; laneId?: string }
  | { type: 'lane.blocker_removed'; laneId?: string }
  | { type: 'lane.owner_changed'; laneId?: string }
  | { type: 'schedule'; cron: string }
  | { type: 'webhook'; path: string }
  | { type: 'manual' };

export interface AutomationCondition {
  type: 'and' | 'or' | 'lane_status' | 'lane_wip' | 'lane_owner' | 'lane_has_blocker' | 'time_of_day' | 'day_of_week';
  params: Record<string, unknown>;
}

export interface AutomationAction {
  type: 'set_status' | 'set_owner' | 'add_blocker' | 'remove_blocker' | 'add_tag' | 'remove_tag' | 'notify' | 'webhook' | 'cascade' | 'custom';
  params: Record<string, unknown>;
  delay?: number;
}

export interface AutomationContext {
  lanes: LanesState;
  triggerEvent: unknown;
  triggeredAt: number;
  rule: AutomationRule;
}

export interface AutomationResult {
  success: boolean;
  rule: AutomationRule;
  triggeredAt: number;
  executedActions: number;
  skippedActions: number;
  errors: string[];
  changes: LaneAction[];
}

export interface AutomationEngineConfig {
  /** Maximum rules to evaluate per event */
  maxRulesPerEvent: number;
  /** Maximum actions per rule execution */
  maxActionsPerRule: number;
  /** Enable logging */
  debug: boolean;
  /** Cascade engine for cascade actions */
  cascadeEngine?: CascadeEngine;
  /** Custom action handlers */
  customHandlers?: Record<string, CustomActionHandler>;
}

export type CustomActionHandler = (
  action: AutomationAction,
  context: AutomationContext
) => Promise<LaneAction[]>;

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: AutomationEngineConfig = {
  maxRulesPerEvent: 10,
  maxActionsPerRule: 20,
  debug: false,
};

// ============================================================================
// Automation Engine
// ============================================================================

export class AutomationEngine {
  private config: AutomationEngineConfig;
  private rules: AutomationRule[] = [];
  private scheduledTasks: Map<string, NodeJS.Timeout> = new Map();
  private webhookHandlers: Map<string, AutomationRule[]> = new Map();

  constructor(config: Partial<AutomationEngineConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ============================================================================
  // Rule Management
  // ============================================================================

  /**
   * Register an automation rule
   */
  registerRule(rule: AutomationRule): void {
    this.rules.push(rule);
    this.rules.sort((a, b) => b.priority - a.priority);

    // Set up scheduled triggers
    if (rule.trigger.type === 'schedule' && rule.enabled) {
      this.setupScheduledTrigger(rule);
    }

    // Register webhook handlers
    if (rule.trigger.type === 'webhook' && rule.enabled) {
      const handlers = this.webhookHandlers.get(rule.trigger.path) || [];
      handlers.push(rule);
      this.webhookHandlers.set(rule.trigger.path, handlers);
    }
  }

  /**
   * Unregister a rule
   */
  unregisterRule(ruleId: string): void {
    const index = this.rules.findIndex(r => r.id === ruleId);
    if (index >= 0) {
      const rule = this.rules[index];

      // Clear scheduled task
      const task = this.scheduledTasks.get(rule.id);
      if (task) {
        clearTimeout(task);
        this.scheduledTasks.delete(rule.id);
      }

      // Remove webhook handler
      if (rule.trigger.type === 'webhook') {
        const handlers = this.webhookHandlers.get(rule.trigger.path) || [];
        this.webhookHandlers.set(
          rule.trigger.path,
          handlers.filter(h => h.id !== ruleId)
        );
      }

      this.rules.splice(index, 1);
    }
  }

  /**
   * Enable/disable a rule
   */
  setRuleEnabled(ruleId: string, enabled: boolean): void {
    const rule = this.rules.find(r => r.id === ruleId);
    if (rule) {
      rule.enabled = enabled;
    }
  }

  /**
   * Get all rules
   */
  getRules(): AutomationRule[] {
    return [...this.rules];
  }

  // ============================================================================
  // Event Processing
  // ============================================================================

  /**
   * Process an event and execute matching rules
   */
  async processEvent(
    eventType: string,
    eventData: unknown,
    state: LanesState
  ): Promise<AutomationResult[]> {
    const results: AutomationResult[] = [];
    let processedCount = 0;

    for (const rule of this.rules) {
      if (processedCount >= this.config.maxRulesPerEvent) break;
      if (!rule.enabled) continue;
      if (!this.matchesTrigger(rule.trigger, eventType, eventData)) continue;

      // Check cooldown
      if (rule.cooldownMs && rule.lastTriggered) {
        if (Date.now() - rule.lastTriggered < rule.cooldownMs) continue;
      }

      const context: AutomationContext = {
        lanes: state,
        triggerEvent: eventData,
        triggeredAt: Date.now(),
        rule,
      };

      // Check conditions
      if (!this.evaluateConditions(rule.conditions, context)) continue;

      // Execute actions
      const result = await this.executeRule(rule, context);
      results.push(result);

      rule.lastTriggered = Date.now();
      processedCount++;
    }

    return results;
  }

  /**
   * Handle webhook request
   */
  async handleWebhook(
    path: string,
    payload: unknown,
    state: LanesState
  ): Promise<AutomationResult[]> {
    const handlers = this.webhookHandlers.get(path) || [];
    const results: AutomationResult[] = [];

    for (const rule of handlers) {
      if (!rule.enabled) continue;

      const context: AutomationContext = {
        lanes: state,
        triggerEvent: payload,
        triggeredAt: Date.now(),
        rule,
      };

      if (this.evaluateConditions(rule.conditions, context)) {
        const result = await this.executeRule(rule, context);
        results.push(result);
      }
    }

    return results;
  }

  // ============================================================================
  // Condition Evaluation
  // ============================================================================

  private matchesTrigger(
    trigger: AutomationTrigger,
    eventType: string,
    eventData: unknown
  ): boolean {
    if (trigger.type !== eventType) return false;

    // Additional trigger-specific matching
    switch (trigger.type) {
      case 'lane.status_changed': {
        const data = eventData as { laneId: string; fromStatus: string; toStatus: string };
        if (trigger.laneId && trigger.laneId !== data.laneId) return false;
        if (trigger.fromStatus && trigger.fromStatus !== data.fromStatus) return false;
        if (trigger.toStatus && trigger.toStatus !== data.toStatus) return false;
        return true;
      }

      case 'lane.wip_changed': {
        const data = eventData as { laneId: string; wip: number };
        if (trigger.laneId && trigger.laneId !== data.laneId) return false;
        if (trigger.threshold !== undefined) {
          if (trigger.direction === 'above' && data.wip <= trigger.threshold) return false;
          if (trigger.direction === 'below' && data.wip >= trigger.threshold) return false;
        }
        return true;
      }

      default:
        return true;
    }
  }

  private evaluateConditions(
    conditions: AutomationCondition[],
    context: AutomationContext
  ): boolean {
    if (conditions.length === 0) return true;

    for (const condition of conditions) {
      if (!this.evaluateCondition(condition, context)) {
        return false;
      }
    }

    return true;
  }

  private evaluateCondition(
    condition: AutomationCondition,
    context: AutomationContext
  ): boolean {
    switch (condition.type) {
      case 'and': {
        const subConditions = condition.params.conditions as AutomationCondition[];
        return subConditions.every(c => this.evaluateCondition(c, context));
      }

      case 'or': {
        const subConditions = condition.params.conditions as AutomationCondition[];
        return subConditions.some(c => this.evaluateCondition(c, context));
      }

      case 'lane_status': {
        const laneId = condition.params.laneId as string;
        const status = condition.params.status as string;
        const lane = context.lanes.lanes.find(l => l.id === laneId);
        return lane?.status === status;
      }

      case 'lane_wip': {
        const laneId = condition.params.laneId as string;
        const min = condition.params.min as number | undefined;
        const max = condition.params.max as number | undefined;
        const lane = context.lanes.lanes.find(l => l.id === laneId);
        if (!lane) return false;
        if (min !== undefined && lane.wip_pct < min) return false;
        if (max !== undefined && lane.wip_pct > max) return false;
        return true;
      }

      case 'lane_has_blocker': {
        const laneId = condition.params.laneId as string;
        const lane = context.lanes.lanes.find(l => l.id === laneId);
        return (lane?.blockers.length || 0) > 0;
      }

      case 'time_of_day': {
        const minHour = condition.params.minHour as number;
        const maxHour = condition.params.maxHour as number;
        const hour = new Date().getHours();
        return hour >= minHour && hour <= maxHour;
      }

      case 'day_of_week': {
        const days = condition.params.days as number[];
        const today = new Date().getDay();
        return days.includes(today);
      }

      default:
        return true;
    }
  }

  // ============================================================================
  // Action Execution
  // ============================================================================

  private async executeRule(
    rule: AutomationRule,
    context: AutomationContext
  ): Promise<AutomationResult> {
    const result: AutomationResult = {
      success: true,
      rule,
      triggeredAt: context.triggeredAt,
      executedActions: 0,
      skippedActions: 0,
      errors: [],
      changes: [],
    };

    for (const action of rule.actions) {
      if (result.executedActions >= this.config.maxActionsPerRule) {
        result.skippedActions++;
        continue;
      }

      try {
        // Delay if specified
        if (action.delay) {
          await new Promise(resolve => setTimeout(resolve, action.delay));
        }

        const changes = await this.executeAction(action, context);
        result.changes.push(...changes);
        result.executedActions++;
      } catch (err: any) {
        result.errors.push(`Action ${action.type}: ${err?.message || 'Unknown error'}`);
        result.success = false;
      }
    }

    this.log(`Rule "${rule.name}" executed: ${result.executedActions} actions, ${result.errors.length} errors`);

    return result;
  }

  private async executeAction(
    action: AutomationAction,
    context: AutomationContext
  ): Promise<LaneAction[]> {
    switch (action.type) {
      case 'set_status': {
        const laneId = action.params.laneId as string;
        const status = action.params.status as 'green' | 'yellow' | 'red';
        return [{
          type: 'UPDATE_LANE',
          payload: { id: laneId, changes: { status } },
        }];
      }

      case 'set_owner': {
        const laneId = action.params.laneId as string;
        const owner = action.params.owner as string;
        return [{
          type: 'UPDATE_LANE',
          payload: { id: laneId, changes: { owner } },
        }];
      }

      case 'add_blocker': {
        const laneId = action.params.laneId as string;
        const blocker = action.params.blocker as string;
        const lane = context.lanes.lanes.find(l => l.id === laneId);
        if (!lane) return [];
        return [{
          type: 'UPDATE_LANE',
          payload: {
            id: laneId,
            changes: { blockers: [...lane.blockers, blocker] },
          },
        }];
      }

      case 'remove_blocker': {
        const laneId = action.params.laneId as string;
        const blocker = action.params.blocker as string;
        const lane = context.lanes.lanes.find(l => l.id === laneId);
        if (!lane) return [];
        return [{
          type: 'UPDATE_LANE',
          payload: {
            id: laneId,
            changes: { blockers: lane.blockers.filter(b => b !== blocker) },
          },
        }];
      }

      case 'notify': {
        // Would emit notification event
        this.log(`Notification: ${action.params.message}`);
        return [];
      }

      case 'webhook': {
        // Would call external webhook
        this.log(`Webhook: ${action.params.url}`);
        return [];
      }

      case 'custom': {
        const handlerName = action.params.handler as string;
        const handler = this.config.customHandlers?.[handlerName];
        if (handler) {
          return await handler(action, context);
        }
        return [];
      }

      default:
        return [];
    }
  }

  // ============================================================================
  // Scheduled Triggers
  // ============================================================================

  private setupScheduledTrigger(rule: AutomationRule): void {
    if (rule.trigger.type !== 'schedule') return;

    // Simple interval-based scheduling (for demo)
    // In production, use proper cron parsing
    const intervalMs = this.parseCron(rule.trigger.cron);
    if (intervalMs > 0) {
      const task = setInterval(() => {
        // Would need to call processEvent with current state
        this.log(`Scheduled rule "${rule.name}" triggered`);
      }, intervalMs);
      this.scheduledTasks.set(rule.id, task);
    }
  }

  private parseCron(cron: string): number {
    // Simple cron parsing (very basic)
    if (cron === '*/5 * * * *') return 5 * 60 * 1000; // Every 5 minutes
    if (cron === '0 * * * *') return 60 * 60 * 1000; // Every hour
    if (cron === '0 0 * * *') return 24 * 60 * 60 * 1000; // Daily
    return 0;
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  private log(message: string): void {
    if (this.config.debug) {
      console.log(`[AutomationEngine] ${message}`);
    }
  }

  /**
   * Dispose the engine
   */
  dispose(): void {
    for (const task of this.scheduledTasks.values()) {
      clearTimeout(task);
    }
    this.scheduledTasks.clear();
    this.rules = [];
    this.webhookHandlers.clear();
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalEngine: AutomationEngine | null = null;

export function getGlobalAutomationEngine(config?: Partial<AutomationEngineConfig>): AutomationEngine {
  if (!globalEngine) {
    globalEngine = new AutomationEngine(config);
  }
  return globalEngine;
}

export function resetGlobalAutomationEngine(): void {
  if (globalEngine) {
    globalEngine.dispose();
  }
  globalEngine = null;
}

export default AutomationEngine;
