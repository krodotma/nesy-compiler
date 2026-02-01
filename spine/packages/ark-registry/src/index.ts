/**
 * @ark/registry - Service Registry for Neo Pluribus
 *
 * Manages the "DNA" of the system - services, agents, and pipelines -
 * treating them as evolving genotypes within the CMP-LARGE framework.
 *
 * Concepts:
 * - ServiceDef: The "Genotype" (static code, plan, DNA)
 * - ServiceInstance: The "Phenotype" (runtime behavior, process)
 * - Clade: A lineage of services evolving together
 * - Gates: P/E/L/R/Q/Omega checks for acceptance
 *
 * Ported from: nucleus/tools/service_registry.py
 *
 * @module
 * @example
 * ```typescript
 * import { ServiceRegistry, createRegistry } from '@ark/registry';
 *
 * const registry = createRegistry('/path/to/root');
 * await registry.init();
 * await registry.load();
 *
 * // Register a new service
 * const serviceId = registry.registerService({
 *   name: 'my-service',
 *   kind: 'process',
 *   entryPoint: 'scripts/my-service.py',
 * });
 *
 * // List services
 * const services = registry.listServices();
 * ```
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';
import { Ring } from '@ark/core';

/**
 * Service kind types
 */
export type ServiceKind = 'port' | 'composition' | 'process';

/**
 * Restart policy types
 */
export type RestartPolicy = 'never' | 'on_failure' | 'always';

/**
 * Health status types
 */
export type HealthStatus = 'unknown' | 'healthy' | 'unhealthy' | 'stopped';

/**
 * Instance status types
 */
export type InstanceStatus = 'stopped' | 'starting' | 'running' | 'error';

/**
 * Gate types for P/E/L/R/Q/Omega validation
 */
export type GateType = 'P' | 'E' | 'L' | 'R' | 'Q' | 'Omega' | 'omega';

/**
 * Gate definitions mapping gate types to validation identifiers
 */
export type Gates = Partial<Record<GateType, string>>;

/**
 * Provenance metadata
 */
export interface Provenance {
  /** Who added this service */
  added_by?: string;
  /** Source of the service definition */
  source?: string;
  /** Additional metadata */
  [key: string]: unknown;
}

/**
 * Service Definition (Genotype)
 *
 * In CMP-LARGE terms, this is the 'Genotype' - the static code/plan.
 */
export interface ServiceDef {
  /** Unique service ID */
  id: string;
  /** Human-readable name */
  name: string;
  /** Service kind: port, composition, or process */
  kind: ServiceKind;
  /** Entry point (Python script or command) */
  entryPoint: string;
  /** Service description */
  description: string;
  /** Port number for port-based services */
  port?: number;
  /** Service IDs this depends on */
  dependsOn: string[];
  /** Environment variables */
  env: Record<string, string>;
  /** Command-line arguments */
  args: string[];
  /** Tags for categorization */
  tags: string[];
  /** Auto-start on registry load */
  autoStart: boolean;
  /** Restart policy */
  restartPolicy: RestartPolicy;
  /** Health check URL or command */
  healthCheck?: string;
  /** Creation timestamp (ISO format) */
  createdIso: string;
  /** Provenance metadata */
  provenance: Provenance;
  /** Evolutionary lineage (e.g., "core.tui", "mcp.host") */
  lineage: string;
  /** P/E/L/R/Q + Omega/omega gate requirements */
  gates: Gates;
  /** Is this a recurring, stable pattern (evolutionary attractor)? */
  omegaMotif: boolean;
  /** Clade-Metaproductivity score (descendant promise) */
  cmpScore: number;
  /** Ring level for access control */
  ring: Ring;
}

/**
 * Service Instance (Phenotype)
 *
 * In CMP-LARGE terms, this is the 'Phenotype' - the runtime behavior.
 */
export interface ServiceInstance {
  /** ID of the parent service definition */
  serviceId: string;
  /** Unique instance ID */
  instanceId: string;
  /** Process ID if running */
  pid?: number;
  /** Port number if bound */
  port?: number;
  /** Current status */
  status: InstanceStatus;
  /** Start timestamp (ISO format) */
  startedIso: string;
  /** Last health check timestamp (ISO format) */
  lastHealthIso: string;
  /** Health status */
  health: HealthStatus;
  /** Error message if in error state */
  error?: string;
}

/**
 * A lineage of related services (Genes)
 */
export interface Clade {
  /** Unique clade ID */
  id: string;
  /** Description of this clade */
  description: string;
  /** Service IDs in this clade */
  services: string[];
  /** Aggregate CMP score */
  cmpAggregate: number;
}

/**
 * Options for creating a new service
 */
export interface CreateServiceOptions {
  /** Service ID (auto-generated if not provided) */
  id?: string;
  /** Human-readable name */
  name: string;
  /** Service kind */
  kind: ServiceKind;
  /** Entry point */
  entryPoint: string;
  /** Description */
  description?: string;
  /** Port number */
  port?: number;
  /** Dependencies */
  dependsOn?: string[];
  /** Environment variables */
  env?: Record<string, string>;
  /** Arguments */
  args?: string[];
  /** Tags */
  tags?: string[];
  /** Auto-start */
  autoStart?: boolean;
  /** Restart policy */
  restartPolicy?: RestartPolicy;
  /** Health check */
  healthCheck?: string;
  /** Lineage */
  lineage?: string;
  /** Gates */
  gates?: Gates;
  /** Omega motif */
  omegaMotif?: boolean;
  /** CMP score */
  cmpScore?: number;
  /** Ring level */
  ring?: Ring;
}

/**
 * Registry configuration
 */
export interface RegistryConfig {
  /** Root directory (rhizome root) */
  root: string;
  /** Actor name for bus events */
  actor?: string;
  /** TTL for health checks in milliseconds */
  healthTtlMs?: number;
}

/**
 * NDJSON record for service definition
 */
interface ServiceDefRecord {
  kind: 'service_def';
  ts: number;
  iso: string;
  service_kind: ServiceKind;
  id: string;
  name: string;
  entry_point: string;
  description: string;
  port?: number;
  depends_on: string[];
  env: Record<string, string>;
  args: string[];
  tags: string[];
  auto_start: boolean;
  restart_policy: RestartPolicy;
  health_check?: string;
  created_iso: string;
  provenance: Provenance;
  lineage: string;
  gates: Gates;
  omega_motif: boolean;
  cmp_score: number;
  ring: Ring;
}

/**
 * NDJSON record for service instance
 */
interface ServiceInstanceRecord {
  kind: 'service_instance';
  ts: number;
  iso: string;
  action?: string;
  service_id: string;
  instance_id: string;
  pid?: number;
  port?: number;
  status: InstanceStatus;
  started_iso: string;
  last_health_iso: string;
  health: HealthStatus;
  error?: string;
}

/**
 * Get current ISO timestamp in UTC
 */
export function nowIsoUtc(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}

/**
 * Get default actor from environment
 */
export function defaultActor(): string {
  return process.env.PLURIBUS_ACTOR ?? process.env.USER ?? 'unknown';
}

/**
 * Ensure a directory exists
 */
export function ensureDir(p: string): void {
  if (!fs.existsSync(p)) {
    fs.mkdirSync(p, { recursive: true });
  }
}

/**
 * Append a JSON object to an NDJSON file
 */
export function appendNdjson(filePath: string, obj: Record<string, unknown>): void {
  ensureDir(path.dirname(filePath));
  const line = JSON.stringify(obj) + '\n';
  fs.appendFileSync(filePath, line, 'utf-8');
}

/**
 * Iterate over NDJSON file lines
 */
export function* iterNdjson(filePath: string): Generator<Record<string, unknown>> {
  if (!fs.existsSync(filePath)) {
    return;
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    try {
      yield JSON.parse(trimmed);
    } catch {
      // Skip malformed lines
      continue;
    }
  }
}

/**
 * Find the rhizome root by looking for .pluribus/rhizome.json
 */
export function findRhizomeRoot(start: string): string | null {
  let current = path.resolve(start);

  while (true) {
    const rhizomePath = path.join(current, '.pluribus', 'rhizome.json');
    if (fs.existsSync(rhizomePath)) {
      return current;
    }

    const parent = path.dirname(current);
    if (parent === current) {
      // Reached root
      return null;
    }
    current = parent;
  }
}

/**
 * Service Registry
 *
 * Manages service definitions and instances with NDJSON-backed persistence.
 */
export class ServiceRegistry {
  private root: string;
  private pluribusDir: string;
  private servicesDir: string;
  private registryPath: string;
  private instancesPath: string;
  private pidDir: string;
  private actor: string;
  private healthTtlMs: number;

  private services: Map<string, ServiceDef> = new Map();
  private instances: Map<string, ServiceInstance> = new Map();

  constructor(config: RegistryConfig) {
    this.root = path.resolve(config.root);
    this.pluribusDir = path.join(this.root, '.pluribus');
    this.servicesDir = path.join(this.pluribusDir, 'services');
    this.registryPath = path.join(this.servicesDir, 'registry.ndjson');
    this.instancesPath = path.join(this.servicesDir, 'instances.ndjson');
    this.pidDir = path.join(this.servicesDir, 'pids');
    this.actor = config.actor ?? defaultActor();
    this.healthTtlMs = config.healthTtlMs ?? 30000;
  }

  /**
   * Initialize registry directories
   */
  async init(): Promise<void> {
    ensureDir(this.servicesDir);
    ensureDir(this.pidDir);

    // Touch registry files
    if (!fs.existsSync(this.registryPath)) {
      fs.writeFileSync(this.registryPath, '', 'utf-8');
    }
    if (!fs.existsSync(this.instancesPath)) {
      fs.writeFileSync(this.instancesPath, '', 'utf-8');
    }
  }

  /**
   * Load services and instances from disk
   */
  async load(): Promise<void> {
    this.services.clear();
    this.instances.clear();

    // Load registered services from NDJSON
    for (const obj of iterNdjson(this.registryPath)) {
      if (obj.kind === 'service_def') {
        const record = obj as unknown as ServiceDefRecord;
        const svc = this.recordToServiceDef(record);
        this.services.set(svc.id, svc);
      }
    }

    // Load running instances from NDJSON
    for (const obj of iterNdjson(this.instancesPath)) {
      if (obj.kind === 'service_instance') {
        const record = obj as unknown as ServiceInstanceRecord;
        const inst = this.recordToServiceInstance(record);
        this.instances.set(inst.instanceId, inst);
      }
    }
  }

  /**
   * Register a new service definition
   */
  registerService(options: CreateServiceOptions): string {
    const id = options.id ?? crypto.randomUUID();
    const now = nowIsoUtc();

    const svc: ServiceDef = {
      id,
      name: options.name,
      kind: options.kind,
      entryPoint: options.entryPoint,
      description: options.description ?? '',
      port: options.port,
      dependsOn: options.dependsOn ?? [],
      env: options.env ?? {},
      args: options.args ?? [],
      tags: options.tags ?? [],
      autoStart: options.autoStart ?? false,
      restartPolicy: options.restartPolicy ?? 'never',
      healthCheck: options.healthCheck,
      createdIso: now,
      provenance: { added_by: this.actor },
      lineage: options.lineage ?? 'orphan',
      gates: options.gates ?? {},
      omegaMotif: options.omegaMotif ?? false,
      cmpScore: options.cmpScore ?? 0.0,
      ring: options.ring ?? Ring.User,
    };

    this.services.set(id, svc);

    // Persist to NDJSON
    const record: ServiceDefRecord = {
      kind: 'service_def',
      ts: Date.now(),
      iso: now,
      service_kind: svc.kind,
      id: svc.id,
      name: svc.name,
      entry_point: svc.entryPoint,
      description: svc.description,
      port: svc.port,
      depends_on: svc.dependsOn,
      env: svc.env,
      args: svc.args,
      tags: svc.tags,
      auto_start: svc.autoStart,
      restart_policy: svc.restartPolicy,
      health_check: svc.healthCheck,
      created_iso: svc.createdIso,
      provenance: svc.provenance,
      lineage: svc.lineage,
      gates: svc.gates,
      omega_motif: svc.omegaMotif,
      cmp_score: svc.cmpScore,
      ring: svc.ring,
    };

    appendNdjson(this.registryPath, record as unknown as Record<string, unknown>);

    return id;
  }

  /**
   * Unregister a service (marks it as removed, does not delete history)
   */
  unregisterService(serviceId: string): boolean {
    const svc = this.services.get(serviceId);
    if (!svc) {
      return false;
    }

    this.services.delete(serviceId);

    // Append removal record
    const record = {
      kind: 'service_unregister',
      ts: Date.now(),
      iso: nowIsoUtc(),
      service_id: serviceId,
      actor: this.actor,
    };
    appendNdjson(this.registryPath, record);

    return true;
  }

  /**
   * List all registered services
   */
  listServices(): ServiceDef[] {
    return Array.from(this.services.values());
  }

  /**
   * Get a service by ID
   */
  getService(serviceId: string): ServiceDef | undefined {
    return this.services.get(serviceId);
  }

  /**
   * List all service instances
   */
  listInstances(): ServiceInstance[] {
    return Array.from(this.instances.values());
  }

  /**
   * Get an instance by ID
   */
  getInstance(instanceId: string): ServiceInstance | undefined {
    return this.instances.get(instanceId);
  }

  /**
   * Get services by tag
   */
  getByTag(tag: string): ServiceDef[] {
    return Array.from(this.services.values()).filter((svc) =>
      svc.tags.includes(tag)
    );
  }

  /**
   * Get services by kind
   */
  getByKind(kind: ServiceKind): ServiceDef[] {
    return Array.from(this.services.values()).filter((svc) => svc.kind === kind);
  }

  /**
   * Get services by lineage
   */
  getByLineage(lineage: string): ServiceDef[] {
    return Array.from(this.services.values()).filter((svc) =>
      svc.lineage.startsWith(lineage)
    );
  }

  /**
   * Get services with omega motif
   */
  getOmegaMotifs(): ServiceDef[] {
    return Array.from(this.services.values()).filter((svc) => svc.omegaMotif);
  }

  /**
   * Get services by ring level
   */
  getByRing(ring: Ring): ServiceDef[] {
    return Array.from(this.services.values()).filter((svc) => svc.ring === ring);
  }

  /**
   * Check if a ring level can access a service
   */
  canAccessService(serviceId: string, actorRing: Ring): boolean {
    const svc = this.services.get(serviceId);
    if (!svc) {
      return false;
    }

    // Lower ring number = higher privilege
    // Actor can access services at same or higher ring number
    return actorRing <= svc.ring;
  }

  /**
   * Start a service instance
   */
  async startService(
    serviceId: string,
    portOverride?: number
  ): Promise<ServiceInstance | null> {
    const svc = this.services.get(serviceId);
    if (!svc) {
      return null;
    }

    // Check dependencies
    for (const depId of svc.dependsOn) {
      const depRunning = Array.from(this.instances.values()).some(
        (inst) => inst.serviceId === depId && inst.status === 'running'
      );

      if (!depRunning) {
        // Try to start dependency first
        const depInst = await this.startService(depId);
        if (!depInst || depInst.status !== 'running') {
          return null;
        }
      }
    }

    // Create instance
    const instanceId = crypto.randomUUID().slice(0, 8);
    const now = nowIsoUtc();

    const inst: ServiceInstance = {
      serviceId,
      instanceId,
      port: portOverride ?? svc.port,
      status: 'starting',
      startedIso: now,
      lastHealthIso: '',
      health: 'unknown',
    };

    // Validate entry point exists
    const entryPoint = path.join(this.root, svc.entryPoint);
    if (!fs.existsSync(entryPoint)) {
      inst.status = 'error';
      inst.error = `Entry point not found: ${entryPoint}`;
      this.instances.set(instanceId, inst);
      return inst;
    }

    // NOTE: In the TypeScript version, we don't actually spawn processes
    // This is a pure registry - process management is handled externally
    // We just track the instance state
    inst.status = 'running';
    inst.pid = undefined; // Would be set by external process manager

    this.instances.set(instanceId, inst);

    // Persist instance
    const record: ServiceInstanceRecord = {
      kind: 'service_instance',
      ts: Date.now(),
      iso: now,
      service_id: inst.serviceId,
      instance_id: inst.instanceId,
      pid: inst.pid,
      port: inst.port,
      status: inst.status,
      started_iso: inst.startedIso,
      last_health_iso: inst.lastHealthIso,
      health: inst.health,
      error: inst.error,
    };

    appendNdjson(this.instancesPath, record as unknown as Record<string, unknown>);

    return inst;
  }

  /**
   * Stop a service instance
   */
  async stopService(instanceId: string): Promise<boolean> {
    const inst = this.instances.get(instanceId);
    if (!inst) {
      return false;
    }

    // NOTE: Actual process termination would be handled externally
    // We just update the registry state
    inst.status = 'stopped';
    inst.health = 'stopped';

    // Persist state change
    const record: ServiceInstanceRecord = {
      kind: 'service_instance',
      ts: Date.now(),
      iso: nowIsoUtc(),
      action: 'stopped',
      service_id: inst.serviceId,
      instance_id: inst.instanceId,
      pid: inst.pid,
      port: inst.port,
      status: inst.status,
      started_iso: inst.startedIso,
      last_health_iso: inst.lastHealthIso,
      health: inst.health,
      error: inst.error,
    };

    appendNdjson(this.instancesPath, record as unknown as Record<string, unknown>);

    return true;
  }

  /**
   * Update health status for an instance
   */
  updateHealth(instanceId: string, health: HealthStatus, error?: string): boolean {
    const inst = this.instances.get(instanceId);
    if (!inst) {
      return false;
    }

    inst.health = health;
    inst.lastHealthIso = nowIsoUtc();
    if (error) {
      inst.error = error;
    }

    if (health === 'stopped') {
      inst.status = 'stopped';
    } else if (health === 'unhealthy') {
      inst.status = 'error';
    }

    return true;
  }

  /**
   * Check health of a service instance
   */
  async checkHealth(instanceId: string): Promise<HealthStatus> {
    const inst = this.instances.get(instanceId);
    if (!inst) {
      return 'unknown';
    }

    const svc = this.services.get(inst.serviceId);
    if (!svc) {
      return 'unknown';
    }

    // If no health check configured, check process liveness
    if (!svc.healthCheck) {
      // NOTE: Would check if PID is alive in real implementation
      return inst.status === 'running' ? 'healthy' : 'stopped';
    }

    // HTTP health check
    if (svc.healthCheck.startsWith('http')) {
      try {
        const response = await fetch(svc.healthCheck, {
          signal: AbortSignal.timeout(5000),
        });
        const health = response.ok ? 'healthy' : 'unhealthy';
        this.updateHealth(instanceId, health);
        return health;
      } catch {
        this.updateHealth(instanceId, 'unhealthy');
        return 'unhealthy';
      }
    }

    return 'unknown';
  }

  /**
   * Refresh health status of all running instances
   */
  async refreshInstances(): Promise<void> {
    for (const [instanceId, inst] of this.instances) {
      if (inst.status === 'running') {
        await this.checkHealth(instanceId);
      }
    }
  }

  /**
   * Get instances that are past their health TTL
   */
  getStaleInstances(): ServiceInstance[] {
    const now = Date.now();
    return Array.from(this.instances.values()).filter((inst) => {
      if (inst.status !== 'running') {
        return false;
      }
      if (!inst.lastHealthIso) {
        return true;
      }
      const lastHealth = new Date(inst.lastHealthIso).getTime();
      return now - lastHealth > this.healthTtlMs;
    });
  }

  /**
   * Get running instances for a service
   */
  getRunningInstances(serviceId: string): ServiceInstance[] {
    return Array.from(this.instances.values()).filter(
      (inst) => inst.serviceId === serviceId && inst.status === 'running'
    );
  }

  /**
   * Get registry statistics
   */
  getStats(): {
    serviceCount: number;
    instanceCount: number;
    runningCount: number;
    byKind: Record<ServiceKind, number>;
    byLineage: Record<string, number>;
    omegaMotifCount: number;
  } {
    const services = this.listServices();
    const instances = this.listInstances();

    const byKind: Record<ServiceKind, number> = {
      port: 0,
      composition: 0,
      process: 0,
    };

    const byLineage: Record<string, number> = {};

    let omegaMotifCount = 0;

    for (const svc of services) {
      byKind[svc.kind]++;

      const lineageRoot = svc.lineage.split('.')[0];
      byLineage[lineageRoot] = (byLineage[lineageRoot] ?? 0) + 1;

      if (svc.omegaMotif) {
        omegaMotifCount++;
      }
    }

    return {
      serviceCount: services.length,
      instanceCount: instances.length,
      runningCount: instances.filter((i) => i.status === 'running').length,
      byKind,
      byLineage,
      omegaMotifCount,
    };
  }

  /**
   * Get the root directory
   */
  getRoot(): string {
    return this.root;
  }

  /**
   * Get the registry path
   */
  getRegistryPath(): string {
    return this.registryPath;
  }

  /**
   * Get the instances path
   */
  getInstancesPath(): string {
    return this.instancesPath;
  }

  /**
   * Convert NDJSON record to ServiceDef
   */
  private recordToServiceDef(record: ServiceDefRecord): ServiceDef {
    return {
      id: record.id,
      name: record.name,
      kind: record.service_kind,
      entryPoint: record.entry_point,
      description: record.description ?? '',
      port: record.port,
      dependsOn: record.depends_on ?? [],
      env: record.env ?? {},
      args: record.args ?? [],
      tags: record.tags ?? [],
      autoStart: record.auto_start ?? false,
      restartPolicy: record.restart_policy ?? 'never',
      healthCheck: record.health_check,
      createdIso: record.created_iso ?? '',
      provenance: record.provenance ?? {},
      lineage: record.lineage ?? 'orphan',
      gates: record.gates ?? {},
      omegaMotif: record.omega_motif ?? false,
      cmpScore: record.cmp_score ?? 0.0,
      ring: record.ring ?? Ring.User,
    };
  }

  /**
   * Convert NDJSON record to ServiceInstance
   */
  private recordToServiceInstance(record: ServiceInstanceRecord): ServiceInstance {
    return {
      serviceId: record.service_id,
      instanceId: record.instance_id,
      pid: record.pid,
      port: record.port,
      status: record.status ?? 'stopped',
      startedIso: record.started_iso ?? '',
      lastHealthIso: record.last_health_iso ?? '',
      health: record.health ?? 'unknown',
      error: record.error,
    };
  }
}

/**
 * Create a new service registry
 */
export function createRegistry(config: RegistryConfig | string): ServiceRegistry {
  if (typeof config === 'string') {
    config = { root: config };
  }
  return new ServiceRegistry(config);
}

// Export version
export const VERSION = '0.1.0';
