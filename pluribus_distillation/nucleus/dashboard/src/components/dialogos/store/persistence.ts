/**
 * Persistence Layer (IndexedDB Adapter)
 * Author: opus_backend_1
 * Context: Phase 1/2 Foundation
 * 
 * Stores the Dialogos timeline locally so it survives reloads.
 * Uses raw IndexedDB to avoid dependencies.
 */

import type { DialogosAtom } from '../types/dialogos';

const DB_NAME = 'Pluribus_Dialogos_DB';
const STORE_NAME = 'atoms';
const VERSION = 1;

export class Persistence {
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    if (typeof window === 'undefined' || !('indexedDB' in window)) return;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, VERSION);

      request.onerror = () => reject('DB Open failed');
      
      request.onsuccess = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        }
      };
    });
  }

  async saveAtom(atom: DialogosAtom): Promise<void> {
    if (!this.db) await this.init();
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction([STORE_NAME], 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(atom);
      
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  async loadTimeline(limit = 100): Promise<DialogosAtom[]> {
    if (!this.db) await this.init();
    if (!this.db) return [];

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction([STORE_NAME], 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll(); // Naive get all for now, optimize with cursor later

      req.onsuccess = () => {
        const res = req.result as DialogosAtom[];
        // Sort by timestamp desc and limit
        resolve(res.sort((a, b) => b.timestamp - a.timestamp).slice(0, limit).reverse());
      };
      req.onerror = () => reject(req.error);
    });
  }
  
  async clear(): Promise<void> {
      if (!this.db) await this.init();
      if (!this.db) return;
      
      return new Promise((resolve, reject) => {
          const tx = this.db!.transaction([STORE_NAME], 'readwrite');
          const store = tx.objectStore(STORE_NAME);
          const req = store.clear();
          req.onsuccess = () => resolve();
          req.onerror = () => reject(req.error);
      });
  }
}

export const persistence = new Persistence();
