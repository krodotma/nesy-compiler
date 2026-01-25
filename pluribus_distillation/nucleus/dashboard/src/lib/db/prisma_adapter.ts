/**
 * prisma_adapter.ts - Type-Safe Database Adapter for Pluribus Entities
 *
 * Provides Prisma-style type-safe database operations without requiring
 * a traditional database server. Uses NDJSON files as the backing store
 * (aligned with Pluribus bus/rhizome architecture).
 *
 * Features:
 * - Type-safe queries with TypeScript
 * - CRUD operations for all Pluribus entities
 * - Transaction support via append-only NDJSON
 * - Query builder pattern (Prisma-compatible API)
 *
 * Specification from SOTA catalog:
 * - TypeScript ORM for type-safe DB access
 * - Schema file for Pluribus entities
 * - Type-safe queries from dashboard
 */

// ---------------------------------------------------------------------------
// Schema Definitions (Pluribus Entities)
// ---------------------------------------------------------------------------

/**
 * Service definition entity
 */
export interface Service {
  id: string;
  name: string;
  kind: 'port' | 'composition' | 'process';
  entryPoint: string;
  description: string;
  port: number | null;
  dependsOn: string[];
  env: Record<string, string>;
  args: string[];
  tags: string[];
  autoStart: boolean;
  restartPolicy: 'never' | 'on_failure' | 'always';
  healthCheck: string | null;
  createdAt: Date;
  updatedAt: Date;
  lineage: string;
  omegaMotif: boolean;
  cmpScore: number;
}

/**
 * Service instance (running process)
 */
export interface ServiceInstance {
  id: string;
  serviceId: string;
  pid: number | null;
  port: number | null;
  status: 'stopped' | 'starting' | 'running' | 'error';
  startedAt: Date;
  lastHealthAt: Date | null;
  health: 'unknown' | 'healthy' | 'unhealthy';
  error: string | null;
}

/**
 * Bus event entity
 */
export interface BusEvent {
  id: string;
  topic: string;
  kind: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  actor: string;
  timestamp: Date;
  iso: string;
  data: Record<string, unknown>;
  traceId: string | null;
  parentId: string | null;
}

/**
 * Agent status entity
 */
export interface Agent {
  id: string;
  actor: string;
  status: string;
  health: string;
  queueDepth: number;
  currentTask: string | null;
  blockers: string[];
  vorCdi: number | null;
  vorPassed: number;
  vorFailed: number;
  lastSeenAt: Date;
}

/**
 * Rhizome artifact entity
 */
export interface Artifact {
  id: string;
  type: 'document' | 'code' | 'data' | 'model' | 'config';
  path: string;
  hash: string;
  size: number;
  mimeType: string;
  metadata: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
  lineageId: string | null;
  parentId: string | null;
}

/**
 * SOTA tool entry
 */
export interface SotaTool {
  id: string;
  name: string;
  category: string;
  viability: number;
  priority: 'P1' | 'P2' | 'P3' | 'P4';
  pluribusMapping: string;
  notes: string;
  integrationStatus: 'planned' | 'in_progress' | 'complete' | 'blocked';
  lastEvalAt: Date | null;
}

// ---------------------------------------------------------------------------
// Query Builder Types
// ---------------------------------------------------------------------------

type WhereClause<T> = Partial<{
  [K in keyof T]: T[K] | { equals?: T[K]; not?: T[K]; in?: T[K][]; contains?: string; startsWith?: string; endsWith?: string; gt?: number; gte?: number; lt?: number; lte?: number };
}>;

type OrderByClause<T> = Partial<{
  [K in keyof T]: 'asc' | 'desc';
}>;

type SelectClause<T> = Partial<{
  [K in keyof T]: boolean;
}>;

interface FindManyArgs<T> {
  where?: WhereClause<T>;
  orderBy?: OrderByClause<T> | OrderByClause<T>[];
  take?: number;
  skip?: number;
  select?: SelectClause<T>;
}

interface FindFirstArgs<T> extends FindManyArgs<T> {}

interface FindUniqueArgs<T> {
  where: { id: string };
  select?: SelectClause<T>;
}

interface CreateArgs<T> {
  data: Omit<T, 'id' | 'createdAt' | 'updatedAt'> & { id?: string };
}

interface UpdateArgs<T> {
  where: { id: string };
  data: Partial<Omit<T, 'id' | 'createdAt'>>;
}

interface DeleteArgs {
  where: { id: string };
}

interface CountArgs<T> {
  where?: WhereClause<T>;
}

// ---------------------------------------------------------------------------
// NDJSON Storage Backend
// ---------------------------------------------------------------------------

interface StorageBackend {
  read<T>(collection: string): Promise<T[]>;
  write<T>(collection: string, records: T[]): Promise<void>;
  append<T>(collection: string, record: T): Promise<void>;
}

class NdjsonStorage implements StorageBackend {
  private baseUrl: string;
  private cache: Map<string, { data: unknown[]; ts: number }> = new Map();
  private cacheTtl = 5000; // 5 seconds

  constructor(baseUrl = '/api/fs/.pluribus/db') {
    this.baseUrl = baseUrl;
  }

  async read<T>(collection: string): Promise<T[]> {
    const cached = this.cache.get(collection);
    if (cached && Date.now() - cached.ts < this.cacheTtl) {
      return cached.data as T[];
    }

    try {
      const response = await fetch(`${this.baseUrl}/${collection}.ndjson`);
      if (!response.ok) {
        if (response.status === 404) return [];
        throw new Error(`Failed to read ${collection}: ${response.statusText}`);
      }

      const text = await response.text();
      const records = text
        .split('\n')
        .filter(line => line.trim())
        .map(line => JSON.parse(line) as T);

      this.cache.set(collection, { data: records, ts: Date.now() });
      return records;
    } catch (err) {
      console.error(`[prisma_adapter] Read error for ${collection}:`, err);
      return [];
    }
  }

  async write<T>(collection: string, records: T[]): Promise<void> {
    const content = records.map(r => JSON.stringify(r)).join('\n') + '\n';

    await fetch(`${this.baseUrl}/${collection}.ndjson`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/x-ndjson' },
      body: content,
    });

    this.cache.set(collection, { data: records, ts: Date.now() });
  }

  async append<T>(collection: string, record: T): Promise<void> {
    const line = JSON.stringify(record) + '\n';

    await fetch(`${this.baseUrl}/${collection}.ndjson`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-ndjson' },
      body: line,
    });

    // Invalidate cache
    this.cache.delete(collection);
  }
}

// ---------------------------------------------------------------------------
// Query Executor
// ---------------------------------------------------------------------------

function matchesWhere<T>(record: T, where?: WhereClause<T>): boolean {
  if (!where) return true;

  for (const [key, condition] of Object.entries(where)) {
    const value = (record as any)[key];

    if (condition === undefined) continue;

    // Simple equality
    if (typeof condition !== 'object' || condition === null || condition instanceof Date) {
      if (value !== condition) return false;
      continue;
    }

    // Complex conditions
    const cond = condition as any;

    if ('equals' in cond && value !== cond.equals) return false;
    if ('not' in cond && value === cond.not) return false;
    if ('in' in cond && !cond.in.includes(value)) return false;
    if ('contains' in cond) {
      if (typeof value !== 'string') return false;
      if (!value.includes(cond.contains)) return false;
    }
    if ('startsWith' in cond) {
      if (typeof value !== 'string') return false;
      if (!value.startsWith(cond.startsWith)) return false;
    }
    if ('endsWith' in cond) {
      if (typeof value !== 'string') return false;
      if (!value.endsWith(cond.endsWith)) return false;
    }
    if ('gt' in cond) {
      if (typeof value !== 'number') return false;
      if (value <= cond.gt) return false;
    }
    if ('gte' in cond) {
      if (typeof value !== 'number') return false;
      if (value < cond.gte) return false;
    }
    if ('lt' in cond) {
      if (typeof value !== 'number') return false;
      if (value >= cond.lt) return false;
    }
    if ('lte' in cond) {
      if (typeof value !== 'number') return false;
      if (value > cond.lte) return false;
    }
  }

  return true;
}

function applyOrderBy<T>(records: T[], orderBy?: OrderByClause<T> | OrderByClause<T>[]): T[] {
  if (!orderBy) return records;

  const orders = Array.isArray(orderBy) ? orderBy : [orderBy];

  return [...records].sort((a, b) => {
    for (const order of orders) {
      for (const [key, direction] of Object.entries(order)) {
        const aVal = (a as any)[key];
        const bVal = (b as any)[key];

        let cmp = 0;
        if (typeof aVal === 'string' && typeof bVal === 'string') {
          cmp = aVal.localeCompare(bVal);
        } else if (aVal instanceof Date && bVal instanceof Date) {
          cmp = aVal.getTime() - bVal.getTime();
        } else {
          cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        }

        if (cmp !== 0) {
          return direction === 'desc' ? -cmp : cmp;
        }
      }
    }
    return 0;
  });
}

function applySelect<T>(record: T, select?: SelectClause<T>): Partial<T> {
  if (!select) return record;

  const result: Partial<T> = {};
  for (const [key, include] of Object.entries(select)) {
    if (include) {
      (result as any)[key] = (record as any)[key];
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Model Delegate (Prisma-style API)
// ---------------------------------------------------------------------------

class ModelDelegate<T extends { id: string }> {
  constructor(
    private storage: StorageBackend,
    private collection: string,
    private idGenerator: () => string = () => crypto.randomUUID()
  ) {}

  async findMany(args?: FindManyArgs<T>): Promise<T[]> {
    let records = await this.storage.read<T>(this.collection);

    // Filter
    if (args?.where) {
      records = records.filter(r => matchesWhere(r, args.where));
    }

    // Sort
    if (args?.orderBy) {
      records = applyOrderBy(records, args.orderBy);
    }

    // Pagination
    if (args?.skip) {
      records = records.slice(args.skip);
    }
    if (args?.take) {
      records = records.slice(0, args.take);
    }

    // Select
    if (args?.select) {
      return records.map(r => applySelect(r, args.select) as T);
    }

    return records;
  }

  async findFirst(args?: FindFirstArgs<T>): Promise<T | null> {
    const results = await this.findMany({ ...args, take: 1 });
    return results[0] || null;
  }

  async findUnique(args: FindUniqueArgs<T>): Promise<T | null> {
    const records = await this.storage.read<T>(this.collection);
    const record = records.find(r => r.id === args.where.id);

    if (!record) return null;

    if (args.select) {
      return applySelect(record, args.select) as T;
    }

    return record;
  }

  async create(args: CreateArgs<T>): Promise<T> {
    const now = new Date();
    const record = {
      ...args.data,
      id: args.data.id || this.idGenerator(),
      createdAt: now,
      updatedAt: now,
    } as unknown as T;

    await this.storage.append(this.collection, record);
    return record;
  }

  async update(args: UpdateArgs<T>): Promise<T | null> {
    const records = await this.storage.read<T>(this.collection);
    const index = records.findIndex(r => r.id === args.where.id);

    if (index === -1) return null;

    const updated = {
      ...records[index],
      ...args.data,
      updatedAt: new Date(),
    } as unknown as T;

    records[index] = updated;
    await this.storage.write(this.collection, records);

    return updated;
  }

  async delete(args: DeleteArgs): Promise<T | null> {
    const records = await this.storage.read<T>(this.collection);
    const index = records.findIndex(r => r.id === args.where.id);

    if (index === -1) return null;

    const deleted = records[index];
    records.splice(index, 1);
    await this.storage.write(this.collection, records);

    return deleted;
  }

  async count(args?: CountArgs<T>): Promise<number> {
    let records = await this.storage.read<T>(this.collection);

    if (args?.where) {
      records = records.filter(r => matchesWhere(r, args.where));
    }

    return records.length;
  }

  async upsert(args: { where: { id: string }; create: CreateArgs<T>['data']; update: UpdateArgs<T>['data'] }): Promise<T> {
    const existing = await this.findUnique({ where: args.where });

    if (existing) {
      return (await this.update({ where: args.where, data: args.update }))!;
    }

    return this.create({ data: { ...args.create, id: args.where.id } });
  }

  async createMany(args: { data: CreateArgs<T>['data'][] }): Promise<{ count: number }> {
    let count = 0;
    for (const data of args.data) {
      await this.create({ data });
      count++;
    }
    return { count };
  }

  async deleteMany(args?: { where?: WhereClause<T> }): Promise<{ count: number }> {
    const records = await this.storage.read<T>(this.collection);
    const filtered = args?.where
      ? records.filter(r => !matchesWhere(r, args.where))
      : [];

    const count = records.length - filtered.length;
    await this.storage.write(this.collection, filtered);

    return { count };
  }
}

// ---------------------------------------------------------------------------
// Prisma Client
// ---------------------------------------------------------------------------

export class PrismaClient {
  private storage: StorageBackend;

  service: ModelDelegate<Service>;
  serviceInstance: ModelDelegate<ServiceInstance>;
  busEvent: ModelDelegate<BusEvent>;
  agent: ModelDelegate<Agent>;
  artifact: ModelDelegate<Artifact>;
  sotaTool: ModelDelegate<SotaTool>;

  constructor(options?: { baseUrl?: string }) {
    this.storage = new NdjsonStorage(options?.baseUrl);

    this.service = new ModelDelegate<Service>(this.storage, 'services');
    this.serviceInstance = new ModelDelegate<ServiceInstance>(this.storage, 'service_instances');
    this.busEvent = new ModelDelegate<BusEvent>(this.storage, 'bus_events');
    this.agent = new ModelDelegate<Agent>(this.storage, 'agents');
    this.artifact = new ModelDelegate<Artifact>(this.storage, 'artifacts');
    this.sotaTool = new ModelDelegate<SotaTool>(this.storage, 'sota_tools');
  }

  async $connect(): Promise<void> {
    // No-op for NDJSON backend
    console.log('[prisma_adapter] Connected to NDJSON storage');
  }

  async $disconnect(): Promise<void> {
    // No-op for NDJSON backend
    console.log('[prisma_adapter] Disconnected from NDJSON storage');
  }

  async $transaction<T>(fn: (prisma: PrismaClient) => Promise<T>): Promise<T> {
    // For NDJSON, transactions are not truly atomic, but we provide the API
    return fn(this);
  }

  async $queryRaw<T>(query: string): Promise<T[]> {
    // Parse simple SQL-like queries (limited support)
    const match = query.match(/SELECT \* FROM (\w+)/i);
    if (match) {
      const collection = match[1].toLowerCase();
      return this.storage.read<T>(collection);
    }
    throw new Error('Raw queries not fully supported');
  }
}

// ---------------------------------------------------------------------------
// Singleton Instance
// ---------------------------------------------------------------------------

let prismaInstance: PrismaClient | null = null;

export function getPrismaClient(): PrismaClient {
  if (!prismaInstance) {
    prismaInstance = new PrismaClient();
  }
  return prismaInstance;
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/**
 * Execute a type-safe query and return results
 */
export async function query<T>(
  model: ModelDelegate<T & { id: string }>,
  args?: FindManyArgs<T & { id: string }>
): Promise<T[]> {
  return model.findMany(args);
}

/**
 * Sync bus events from NDJSON file to typed storage
 */
export async function syncBusEvents(eventsPath: string): Promise<number> {
  const prisma = getPrismaClient();

  try {
    const response = await fetch(eventsPath);
    const text = await response.text();
    const events = text
      .split('\n')
      .filter(line => line.trim())
      .map(line => {
        const raw = JSON.parse(line);
        return {
          id: raw.id || crypto.randomUUID(),
          topic: raw.topic,
          kind: raw.kind,
          level: raw.level,
          actor: raw.actor,
          timestamp: new Date(raw.ts),
          iso: raw.iso,
          data: raw.data || {},
          traceId: raw.trace_id || null,
          parentId: raw.parent_id || null,
        } as BusEvent;
      });

    // Batch insert (upsert to avoid duplicates)
    let count = 0;
    for (const event of events.slice(-500)) { // Last 500 events
      try {
        await prisma.busEvent.upsert({
          where: { id: event.id },
          create: event,
          update: event,
        });
        count++;
      } catch {
        // Skip duplicates
      }
    }

    return count;
  } catch (err) {
    console.error('[prisma_adapter] Sync error:', err);
    return 0;
  }
}

// ---------------------------------------------------------------------------
// Example Queries (for documentation)
// ---------------------------------------------------------------------------

/**
 * Example: Find all running services
 *
 * ```typescript
 * const prisma = getPrismaClient();
 * const running = await prisma.serviceInstance.findMany({
 *   where: { status: 'running' },
 *   orderBy: { startedAt: 'desc' },
 * });
 * ```
 */

/**
 * Example: Find agents with high queue depth
 *
 * ```typescript
 * const prisma = getPrismaClient();
 * const busy = await prisma.agent.findMany({
 *   where: { queueDepth: { gt: 5 } },
 *   orderBy: { queueDepth: 'desc' },
 *   take: 10,
 * });
 * ```
 */

/**
 * Example: Count errors in the last hour
 *
 * ```typescript
 * const prisma = getPrismaClient();
 * const hourAgo = new Date(Date.now() - 3600000);
 * const errorCount = await prisma.busEvent.count({
 *   where: {
 *     level: 'error',
 *     timestamp: { gte: hourAgo },
 *   },
 * });
 * ```
 */

export default PrismaClient;
