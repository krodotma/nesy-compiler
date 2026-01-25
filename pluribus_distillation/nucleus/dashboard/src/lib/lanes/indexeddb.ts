/**
 * IndexedDB Adapter for Lanes
 *
 * Phase 7, Iteration 54 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - IndexedDB persistence for lanes state
 * - Efficient querying with indexes
 * - Schema migrations
 * - Offline-first support
 * - Auto-cleanup of old data
 */

import type { Lane, LanesState, LaneHistory, Agent } from './store';

// ============================================================================
// Types
// ============================================================================

export interface DBConfig {
  /** Database name */
  dbName: string;
  /** Current schema version */
  version: number;
  /** Maximum history entries per lane */
  maxHistoryEntries: number;
  /** Maximum age for history in days */
  maxHistoryAgeDays: number;
}

export interface QueryOptions {
  limit?: number;
  offset?: number;
  orderBy?: keyof Lane;
  orderDirection?: 'asc' | 'desc';
  filter?: Partial<Lane>;
}

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: DBConfig = {
  dbName: 'pluribus-lanes',
  version: 1,
  maxHistoryEntries: 100,
  maxHistoryAgeDays: 30,
};

// ============================================================================
// IndexedDB Adapter
// ============================================================================

export class LanesIndexedDB {
  private config: DBConfig;
  private db: IDBDatabase | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(config: Partial<DBConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ============================================================================
  // Initialization
  // ============================================================================

  async init(): Promise<void> {
    if (this.db) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise((resolve, reject) => {
      if (typeof indexedDB === 'undefined') {
        reject(new Error('IndexedDB not available'));
        return;
      }

      const request = indexedDB.open(this.config.dbName, this.config.version);

      request.onerror = () => reject(request.error);

      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        this.runMigrations(db, event.oldVersion);
      };
    });

    return this.initPromise;
  }

  private runMigrations(db: IDBDatabase, oldVersion: number): void {
    // Version 1: Initial schema
    if (oldVersion < 1) {
      // Lanes store
      const lanesStore = db.createObjectStore('lanes', { keyPath: 'id' });
      lanesStore.createIndex('status', 'status', { unique: false });
      lanesStore.createIndex('owner', 'owner', { unique: false });
      lanesStore.createIndex('wip_pct', 'wip_pct', { unique: false });
      lanesStore.createIndex('updated', 'updated', { unique: false });

      // Agents store
      const agentsStore = db.createObjectStore('agents', { keyPath: 'id' });
      agentsStore.createIndex('status', 'status', { unique: false });
      agentsStore.createIndex('lane', 'lane', { unique: false });

      // History store (separate for efficient cleanup)
      const historyStore = db.createObjectStore('history', { keyPath: ['laneId', 'ts'] });
      historyStore.createIndex('laneId', 'laneId', { unique: false });
      historyStore.createIndex('ts', 'ts', { unique: false });

      // Metadata store
      db.createObjectStore('metadata', { keyPath: 'key' });
    }

    // Future migrations go here
    // if (oldVersion < 2) { ... }
  }

  // ============================================================================
  // Lanes Operations
  // ============================================================================

  async saveLane(lane: Lane): Promise<void> {
    await this.init();
    return this.transaction('lanes', 'readwrite', store => {
      store.put(lane);
    });
  }

  async saveLanes(lanes: Lane[]): Promise<void> {
    await this.init();
    return this.transaction('lanes', 'readwrite', store => {
      for (const lane of lanes) {
        store.put(lane);
      }
    });
  }

  async getLane(id: string): Promise<Lane | undefined> {
    await this.init();
    return this.transactionAsync<Lane>('lanes', 'readonly', store => {
      return store.get(id);
    });
  }

  async getAllLanes(): Promise<Lane[]> {
    await this.init();
    return this.transactionAsync<Lane[]>('lanes', 'readonly', store => {
      return store.getAll();
    }) || [];
  }

  async queryLanes(options: QueryOptions = {}): Promise<Lane[]> {
    await this.init();

    let lanes = await this.getAllLanes();

    // Apply filter
    if (options.filter) {
      lanes = lanes.filter(lane => {
        for (const [key, value] of Object.entries(options.filter!)) {
          if (lane[key as keyof Lane] !== value) return false;
        }
        return true;
      });
    }

    // Apply sort
    if (options.orderBy) {
      const key = options.orderBy;
      const dir = options.orderDirection === 'desc' ? -1 : 1;
      lanes.sort((a, b) => {
        const aVal = a[key];
        const bVal = b[key];
        if (aVal === undefined) return 1;
        if (bVal === undefined) return -1;
        if (aVal < bVal) return -1 * dir;
        if (aVal > bVal) return 1 * dir;
        return 0;
      });
    }

    // Apply pagination
    if (options.offset) {
      lanes = lanes.slice(options.offset);
    }
    if (options.limit) {
      lanes = lanes.slice(0, options.limit);
    }

    return lanes;
  }

  async deleteLane(id: string): Promise<void> {
    await this.init();
    return this.transaction('lanes', 'readwrite', store => {
      store.delete(id);
    });
  }

  async getLanesByStatus(status: 'green' | 'yellow' | 'red'): Promise<Lane[]> {
    await this.init();
    return this.transactionAsync<Lane[]>('lanes', 'readonly', store => {
      const index = store.index('status');
      return index.getAll(status);
    }) || [];
  }

  async getLanesByOwner(owner: string): Promise<Lane[]> {
    await this.init();
    return this.transactionAsync<Lane[]>('lanes', 'readonly', store => {
      const index = store.index('owner');
      return index.getAll(owner);
    }) || [];
  }

  // ============================================================================
  // Agents Operations
  // ============================================================================

  async saveAgent(agent: Agent): Promise<void> {
    await this.init();
    return this.transaction('agents', 'readwrite', store => {
      store.put(agent);
    });
  }

  async saveAgents(agents: Agent[]): Promise<void> {
    await this.init();
    return this.transaction('agents', 'readwrite', store => {
      for (const agent of agents) {
        store.put(agent);
      }
    });
  }

  async getAgent(id: string): Promise<Agent | undefined> {
    await this.init();
    return this.transactionAsync<Agent>('agents', 'readonly', store => {
      return store.get(id);
    });
  }

  async getAllAgents(): Promise<Agent[]> {
    await this.init();
    return this.transactionAsync<Agent[]>('agents', 'readonly', store => {
      return store.getAll();
    }) || [];
  }

  async getAgentsByLane(laneId: string): Promise<Agent[]> {
    await this.init();
    return this.transactionAsync<Agent[]>('agents', 'readonly', store => {
      const index = store.index('lane');
      return index.getAll(laneId);
    }) || [];
  }

  // ============================================================================
  // History Operations
  // ============================================================================

  async saveHistory(laneId: string, entry: LaneHistory): Promise<void> {
    await this.init();
    return this.transaction('history', 'readwrite', store => {
      store.put({ ...entry, laneId });
    });
  }

  async getHistory(laneId: string): Promise<LaneHistory[]> {
    await this.init();
    const entries = await this.transactionAsync<Array<LaneHistory & { laneId: string }>>(
      'history',
      'readonly',
      store => {
        const index = store.index('laneId');
        return index.getAll(laneId);
      }
    ) || [];

    return entries
      .map(({ laneId: _, ...rest }) => rest)
      .sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime());
  }

  async cleanupHistory(): Promise<number> {
    await this.init();

    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.maxHistoryAgeDays);
    const cutoffTs = cutoffDate.toISOString();

    let deletedCount = 0;

    await this.transaction('history', 'readwrite', store => {
      const index = store.index('ts');
      const range = IDBKeyRange.upperBound(cutoffTs);
      const request = index.openCursor(range);

      request.onsuccess = () => {
        const cursor = request.result;
        if (cursor) {
          cursor.delete();
          deletedCount++;
          cursor.continue();
        }
      };
    });

    return deletedCount;
  }

  // ============================================================================
  // Full State Operations
  // ============================================================================

  async saveState(state: LanesState): Promise<void> {
    await this.init();

    // Save lanes
    await this.saveLanes(state.lanes);

    // Save agents
    await this.saveAgents(state.agents);

    // Save metadata
    await this.saveMetadata('version', state.version);
    await this.saveMetadata('generated', state.generated);
    await this.saveMetadata('updated', state.updated);
    await this.saveMetadata('lastSaved', new Date().toISOString());
  }

  async loadState(): Promise<LanesState | null> {
    await this.init();

    const lanes = await this.getAllLanes();
    const agents = await this.getAllAgents();
    const version = await this.getMetadata('version') as string;
    const generated = await this.getMetadata('generated') as string;
    const updated = await this.getMetadata('updated') as string;

    if (!version) return null;

    return {
      version,
      generated,
      updated,
      lanes,
      agents,
    };
  }

  // ============================================================================
  // Metadata Operations
  // ============================================================================

  async saveMetadata(key: string, value: unknown): Promise<void> {
    await this.init();
    return this.transaction('metadata', 'readwrite', store => {
      store.put({ key, value });
    });
  }

  async getMetadata(key: string): Promise<unknown> {
    await this.init();
    const result = await this.transactionAsync<{ key: string; value: unknown }>(
      'metadata',
      'readonly',
      store => store.get(key)
    );
    return result?.value;
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  async clear(): Promise<void> {
    await this.init();
    const stores = ['lanes', 'agents', 'history', 'metadata'];
    for (const storeName of stores) {
      await this.transaction(storeName, 'readwrite', store => {
        store.clear();
      });
    }
  }

  async getStats(): Promise<{
    laneCount: number;
    agentCount: number;
    historyCount: number;
    dbSize: number;
  }> {
    await this.init();

    const laneCount = await this.transactionAsync<number>('lanes', 'readonly', store => {
      return store.count();
    }) || 0;

    const agentCount = await this.transactionAsync<number>('agents', 'readonly', store => {
      return store.count();
    }) || 0;

    const historyCount = await this.transactionAsync<number>('history', 'readonly', store => {
      return store.count();
    }) || 0;

    return {
      laneCount,
      agentCount,
      historyCount,
      dbSize: 0, // Would need navigator.storage API
    };
  }

  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    this.initPromise = null;
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private transaction(
    storeName: string,
    mode: IDBTransactionMode,
    callback: (store: IDBObjectStore) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(storeName, mode);
      const store = transaction.objectStore(storeName);

      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);

      callback(store);
    });
  }

  private transactionAsync<T>(
    storeName: string,
    mode: IDBTransactionMode,
    callback: (store: IDBObjectStore) => IDBRequest
  ): Promise<T | undefined> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(storeName, mode);
      const store = transaction.objectStore(storeName);
      const request = callback(store);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalDB: LanesIndexedDB | null = null;

export function getGlobalDB(config?: Partial<DBConfig>): LanesIndexedDB {
  if (!globalDB) {
    globalDB = new LanesIndexedDB(config);
  }
  return globalDB;
}

export function resetGlobalDB(): void {
  if (globalDB) {
    globalDB.close();
  }
  globalDB = null;
}

export default LanesIndexedDB;
