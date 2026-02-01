/**
 * @ark/core/types - Core Type Definitions
 *
 * Foundational types used across the ARK system.
 *
 * @module
 */

/**
 * Unique identifier (typically UUID or content-addressed hash)
 */
export type Id = string;

/**
 * Unix timestamp in milliseconds
 */
export type Timestamp = number;

/**
 * Duration in milliseconds
 */
export type Duration = number;

/**
 * Semantic version string (major.minor.patch)
 */
export type SemVer = `${number}.${number}.${number}`;

/**
 * JSON-serializable value
 */
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/**
 * JSON object
 */
export type JsonObject = { [key: string]: JsonValue };

/**
 * Generic metadata
 */
export type Metadata = Record<string, unknown>;

/**
 * Task status
 */
export type TaskStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'blocked';

/**
 * Agent state
 */
export type AgentState =
  | 'idle'
  | 'thinking'
  | 'acting'
  | 'waiting'
  | 'error'
  | 'terminated';

/**
 * Priority levels
 */
export type Priority = 'critical' | 'high' | 'normal' | 'low' | 'background';

/**
 * Log levels
 */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'fatal';

/**
 * Base entity interface
 */
export interface Entity {
  /** Unique identifier */
  id: Id;
  /** Creation timestamp */
  createdAt: Timestamp;
  /** Last update timestamp */
  updatedAt: Timestamp;
}

/**
 * Named entity
 */
export interface NamedEntity extends Entity {
  /** Human-readable name */
  name: string;
  /** Description */
  description?: string;
}

/**
 * Tagged entity
 */
export interface TaggedEntity extends Entity {
  /** Tags for categorization */
  tags: string[];
}

/**
 * Versioned entity
 */
export interface VersionedEntity extends Entity {
  /** Version string */
  version: SemVer;
  /** Previous version ID (for lineage) */
  previousVersion?: Id;
}

/**
 * Task definition
 */
export interface Task extends NamedEntity, TaggedEntity {
  /** Task status */
  status: TaskStatus;
  /** Priority */
  priority: Priority;
  /** Assigned agent ID */
  assignedTo?: Id;
  /** Parent task ID */
  parentTask?: Id;
  /** Child task IDs */
  subtasks: Id[];
  /** Dependencies (task IDs that must complete first) */
  dependencies: Id[];
  /** Task-specific data */
  data: Metadata;
  /** Error message if failed */
  error?: string;
  /** Completion timestamp */
  completedAt?: Timestamp;
}

/**
 * Agent definition
 */
export interface Agent extends NamedEntity {
  /** Current state */
  state: AgentState;
  /** Agent type/role */
  role: string;
  /** Capabilities */
  capabilities: string[];
  /** Current task ID */
  currentTask?: Id;
  /** Configuration */
  config: Metadata;
  /** Runtime metrics */
  metrics: AgentMetrics;
}

/**
 * Agent metrics
 */
export interface AgentMetrics {
  /** Tasks completed */
  tasksCompleted: number;
  /** Tasks failed */
  tasksFailed: number;
  /** Total actions taken */
  totalActions: number;
  /** Average response time (ms) */
  avgResponseTime: number;
  /** Uptime (ms) */
  uptime: Duration;
  /** Last active timestamp */
  lastActive: Timestamp;
}

/**
 * Configuration interface
 */
export interface Config<T = Metadata> {
  /** Configuration data */
  data: T;
  /** Schema version */
  schemaVersion: SemVer;
  /** Source (file path, environment, etc.) */
  source?: string;
  /** Validation status */
  validated: boolean;
}

/**
 * Resource with lifecycle
 */
export interface Resource extends NamedEntity {
  /** Resource type */
  type: string;
  /** Resource state */
  state: 'creating' | 'ready' | 'busy' | 'error' | 'destroying' | 'destroyed';
  /** Owner ID */
  owner?: Id;
  /** Resource-specific data */
  data: Metadata;
}

/**
 * Health check result
 */
export interface HealthCheck {
  /** Is healthy? */
  healthy: boolean;
  /** Component name */
  component: string;
  /** Status message */
  message: string;
  /** Check timestamp */
  timestamp: Timestamp;
  /** Duration of check (ms) */
  duration: Duration;
  /** Additional details */
  details?: Metadata;
}

/**
 * System status
 */
export interface SystemStatus {
  /** Overall health */
  healthy: boolean;
  /** System version */
  version: SemVer;
  /** Uptime */
  uptime: Duration;
  /** Component health checks */
  components: HealthCheck[];
  /** Active agent count */
  activeAgents: number;
  /** Pending task count */
  pendingTasks: number;
  /** Memory usage (bytes) */
  memoryUsage?: number;
  /** CPU usage (0-1) */
  cpuUsage?: number;
}

/**
 * Type guard for Entity
 */
export function isEntity(value: unknown): value is Entity {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'createdAt' in value &&
    'updatedAt' in value
  );
}

/**
 * Type guard for Task
 */
export function isTask(value: unknown): value is Task {
  return (
    isEntity(value) &&
    'status' in value &&
    'priority' in value &&
    'subtasks' in value
  );
}

/**
 * Create a new entity with ID and timestamps
 */
export function createEntity(id?: Id): Entity {
  const now = Date.now();
  return {
    id: id ?? crypto.randomUUID(),
    createdAt: now,
    updatedAt: now,
  };
}

/**
 * Update an entity's timestamp
 */
export function touchEntity<T extends Entity>(entity: T): T {
  return {
    ...entity,
    updatedAt: Date.now(),
  };
}
