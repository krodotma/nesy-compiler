import type { BusEvent } from '../state/types';
import { createBusClient } from '../bus/bus-client';

export type PortalAssetStatus = 'staged' | 'uploading' | 'cached' | 'expired' | 'error';

export interface PortalAssetMeta {
  id: string;
  name: string;
  mime: string;
  byte_size: number;
  created_iso: string;
  expires_at: string;
  status: PortalAssetStatus;
  storage: 'indexeddb';
  session_id?: string;
  source_type?: 'file';
  source_uri?: string;
}

export interface PortalAssetIndex {
  assets: PortalAssetMeta[];
  updated_at: string;
}

const ASSET_INDEX_KEY = 'pluribus.portal.assets.v1';
const SESSION_KEY = 'pluribus.portal.session.v1';
const DEFAULT_TTL_MS = 24 * 60 * 60 * 1000;
const ASSET_INDEX_LIMIT = 200;

const DB_NAME = 'pluribus_portal_assets';
const DB_VERSION = 1;
const DB_STORE = 'assets';

const BRIDGE_KEY = '__portalAssetBridge__';

interface PortalAssetRecord {
  id: string;
  blob: Blob;
  mime: string;
  byte_size: number;
  created_ms: number;
  expires_ms: number;
}

const isBrowser = (): boolean => typeof window !== 'undefined';

const nowIso = (): string => new Date().toISOString();

const generateId = (): string => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `asset_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
};

export function getPortalSessionId(): string {
  if (!isBrowser() || typeof localStorage === 'undefined') {
    return generateId();
  }
  try {
    const existing = localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
    const fresh = generateId();
    localStorage.setItem(SESSION_KEY, fresh);
    return fresh;
  } catch {
    return generateId();
  }
}

export function loadPortalAssetIndex(): PortalAssetIndex {
  const fallback: PortalAssetIndex = { assets: [], updated_at: nowIso() };
  if (!isBrowser() || typeof localStorage === 'undefined') return fallback;
  try {
    const raw = localStorage.getItem(ASSET_INDEX_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as PortalAssetIndex | PortalAssetMeta[];
    if (Array.isArray(parsed)) {
      return { assets: parsed, updated_at: nowIso() };
    }
    if (parsed && Array.isArray(parsed.assets)) {
      return { assets: parsed.assets, updated_at: parsed.updated_at || nowIso() };
    }
    return fallback;
  } catch {
    return fallback;
  }
}

export function savePortalAssetIndex(index: PortalAssetIndex): void {
  if (!isBrowser() || typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(ASSET_INDEX_KEY, JSON.stringify(index));
  } catch {
    // ignore storage failures
  }
}

export function upsertPortalAssetMeta(meta: PortalAssetMeta): PortalAssetIndex {
  const index = loadPortalAssetIndex();
  const next = index.assets.filter((asset) => asset.id !== meta.id);
  next.push(meta);
  next.sort((a, b) => Date.parse(a.created_iso) - Date.parse(b.created_iso));
  if (next.length > ASSET_INDEX_LIMIT) {
    next.splice(0, next.length - ASSET_INDEX_LIMIT);
  }
  const updated = { assets: next, updated_at: nowIso() };
  savePortalAssetIndex(updated);
  return updated;
}

export function updatePortalAssetMeta(assetId: string, patch: Partial<PortalAssetMeta>): PortalAssetIndex {
  const index = loadPortalAssetIndex();
  const next = index.assets.map((asset) => (asset.id === assetId ? { ...asset, ...patch } : asset));
  const updated = { assets: next, updated_at: nowIso() };
  savePortalAssetIndex(updated);
  return updated;
}

const openPortalAssetDb = async (): Promise<IDBDatabase> => {
  if (!isBrowser() || typeof indexedDB === 'undefined') {
    throw new Error('IndexedDB not available');
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(DB_STORE)) {
        db.createObjectStore(DB_STORE, { keyPath: 'id' });
      }
    };

    request.onsuccess = () => resolve(request.result);
  });
};

const storePortalAssetRecord = async (record: PortalAssetRecord): Promise<void> => {
  const db = await openPortalAssetDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(DB_STORE).put(record);
  });
};

const loadPortalAssetRecord = async (assetId: string): Promise<PortalAssetRecord | null> => {
  const db = await openPortalAssetDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readonly');
    tx.onerror = () => reject(tx.error);
    const req = tx.objectStore(DB_STORE).get(assetId);
    req.onsuccess = () => resolve((req.result as PortalAssetRecord) || null);
    req.onerror = () => reject(req.error);
  });
};

const deletePortalAssetRecord = async (assetId: string): Promise<void> => {
  const db = await openPortalAssetDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(DB_STORE).delete(assetId);
  });
};

export async function purgeExpiredPortalAssets(nowMs = Date.now()): Promise<PortalAssetIndex> {
  const index = loadPortalAssetIndex();
  const keep: PortalAssetMeta[] = [];
  const expired: PortalAssetMeta[] = [];

  for (const asset of index.assets) {
    const expiresMs = Date.parse(asset.expires_at);
    if (!Number.isFinite(expiresMs) || expiresMs <= nowMs) {
      expired.push(asset);
    } else {
      keep.push(asset);
    }
  }

  if (expired.length > 0) {
    for (const asset of expired) {
      try {
        await deletePortalAssetRecord(asset.id);
      } catch {
        // ignore missing records
      }
    }
  }

  const updated = { assets: keep, updated_at: nowIso() };
  savePortalAssetIndex(updated);
  return updated;
}

export async function emitPortalAssetEvent(topic: string, data: Record<string, unknown>): Promise<void> {
  if (typeof fetch === 'undefined') return;
  try {
    await fetch('/api/emit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic,
        kind: 'artifact',
        level: 'info',
        actor: 'portal-ingest',
        data,
      }),
    });
  } catch {
    // best-effort
  }
}

export async function stagePortalAsset(
  file: File,
  opts: { sessionId?: string; ttlMs?: number; emitBus?: boolean } = {},
): Promise<PortalAssetMeta> {
  const nowMs = Date.now();
  const ttlMs = opts.ttlMs ?? DEFAULT_TTL_MS;
  const sessionId = opts.sessionId || getPortalSessionId();
  const assetId = generateId();
  const expiresAt = new Date(nowMs + ttlMs).toISOString();

  const meta: PortalAssetMeta = {
    id: assetId,
    name: file.name || `asset-${assetId}`,
    mime: file.type || 'application/octet-stream',
    byte_size: file.size || 0,
    created_iso: new Date(nowMs).toISOString(),
    expires_at: expiresAt,
    status: 'staged',
    storage: 'indexeddb',
    session_id: sessionId,
    source_type: 'file',
    source_uri: file.name || undefined,
  };

  await storePortalAssetRecord({
    id: assetId,
    blob: file,
    mime: meta.mime,
    byte_size: meta.byte_size,
    created_ms: nowMs,
    expires_ms: nowMs + ttlMs,
  });

  upsertPortalAssetMeta(meta);

  if (opts.emitBus !== false) {
    await emitPortalAssetEvent('portal.ingest.asset_staged', {
      asset_id: meta.id,
      name: meta.name,
      mime: meta.mime,
      byte_size: meta.byte_size,
      created_iso: meta.created_iso,
      expires_at: meta.expires_at,
      session_id: meta.session_id,
      source_type: meta.source_type,
      source_uri: meta.source_uri,
      storage: meta.storage,
    });
  }

  return meta;
}

export async function uploadPortalAsset(assetId: string, uploadUrl: string): Promise<boolean> {
  const meta = loadPortalAssetIndex().assets.find((asset) => asset.id === assetId);
  if (!meta) return false;

  const record = await loadPortalAssetRecord(assetId).catch(() => null);
  if (!record) {
    updatePortalAssetMeta(assetId, { status: 'error' });
    await emitPortalAssetEvent('portal.ingest.asset_upload_failed', { asset_id: assetId, reason: 'missing_blob' });
    return false;
  }

  updatePortalAssetMeta(assetId, { status: 'uploading' });

  const headers: Record<string, string> = {
    'Content-Type': record.mime || 'application/octet-stream',
    'x-portal-asset-id': assetId,
    'x-portal-asset-size': String(record.byte_size),
  };

  if (meta.name) headers['x-portal-asset-name'] = meta.name;
  if (meta.session_id) headers['x-portal-asset-session'] = meta.session_id;
  if (meta.created_iso) headers['x-portal-asset-created-at'] = meta.created_iso;
  if (meta.expires_at) headers['x-portal-asset-expires-at'] = meta.expires_at;

  try {
    const res = await fetch(uploadUrl, {
      method: 'POST',
      headers,
      body: record.blob,
    });

    if (!res.ok) {
      updatePortalAssetMeta(assetId, { status: 'error' });
      await emitPortalAssetEvent('portal.ingest.asset_upload_failed', {
        asset_id: assetId,
        http_status: res.status,
      });
      return false;
    }

    updatePortalAssetMeta(assetId, { status: 'cached' });
    return true;
  } catch (err) {
    updatePortalAssetMeta(assetId, { status: 'error' });
    await emitPortalAssetEvent('portal.ingest.asset_upload_failed', {
      asset_id: assetId,
      error: String(err),
    });
    return false;
  }
}

export function startPortalAssetBridge(
  opts: { sessionId?: string; uploadUrl?: string; onIndexUpdate?: (index: PortalAssetIndex) => void } = {},
): () => void {
  if (!isBrowser()) return () => {};
  const existing = (window as any)[BRIDGE_KEY];
  if (existing?.stop) return existing.stop;

  const sessionId = opts.sessionId || getPortalSessionId();
  const uploadUrl = opts.uploadUrl || '/api/portal/assets';
  const client = createBusClient({ platform: 'browser' });
  let active = true;
  const notifyIndex = () => {
    if (opts.onIndexUpdate) opts.onIndexUpdate(loadPortalAssetIndex());
  };

  const handlePull = async (event: BusEvent) => {
    if (!active) return;
    if (event.topic !== 'portal.ingest.asset_pull') return;
    const data = (event.data || {}) as Record<string, unknown>;
    const assetId = typeof data.asset_id === 'string' ? data.asset_id : '';
    if (!assetId) return;
    const targetSession = typeof data.session_id === 'string' ? data.session_id : null;
    if (targetSession && targetSession !== sessionId) return;
    const targetUrl = typeof data.upload_url === 'string' ? data.upload_url : uploadUrl;
    await uploadPortalAsset(assetId, targetUrl);
    notifyIndex();
  };

  const handleCached = (event: BusEvent) => {
    if (event.topic !== 'portal.ingest.asset_cached') return;
    const data = (event.data || {}) as Record<string, unknown>;
    const assetId = typeof data.asset_id === 'string' ? data.asset_id : '';
    if (!assetId) return;
    updatePortalAssetMeta(assetId, { status: 'cached' });
    notifyIndex();
  };

  client.connect()
    .then(() => {
      client.subscribe('portal.ingest.asset_pull', handlePull);
      client.subscribe('portal.ingest.asset_cached', handleCached);
    })
    .catch(() => {
      // connection failures are handled by bus client retries
    });

  const stop = () => {
    active = false;
    client.disconnect();
  };

  (window as any)[BRIDGE_KEY] = { stop, client };
  return stop;
}
