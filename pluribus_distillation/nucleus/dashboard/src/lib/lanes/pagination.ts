/**
 * Lanes Pagination Utilities
 *
 * Phase 2, Iteration 9 of OITERATE lanes-widget-enhancement
 * Provides pagination support for lanes and history data
 */

import type { Lane, LaneHistory } from '../../components/LaneHistoryTimeline';

export interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

export interface LaneFilterOptions {
  status?: 'green' | 'yellow' | 'red' | 'all';
  owner?: string;
  hasBlockers?: boolean;
  minWip?: number;
  maxWip?: number;
}

export interface LaneSortOptions {
  field: 'wip_pct' | 'name' | 'status' | 'owner';
  direction: 'asc' | 'desc';
}

/**
 * Paginate an array of lanes with filtering and sorting
 */
export function paginateLanes(
  lanes: Lane[],
  page: number = 1,
  pageSize: number = 20,
  filter?: LaneFilterOptions,
  sort?: LaneSortOptions
): PaginatedResult<Lane> {
  let filtered = [...lanes];

  // Apply filters
  if (filter) {
    if (filter.status && filter.status !== 'all') {
      filtered = filtered.filter(l => l.status === filter.status);
    }
    if (filter.owner) {
      filtered = filtered.filter(l => l.owner === filter.owner);
    }
    if (filter.hasBlockers !== undefined) {
      filtered = filtered.filter(l =>
        filter.hasBlockers ? l.blockers.length > 0 : l.blockers.length === 0
      );
    }
    if (filter.minWip !== undefined) {
      filtered = filtered.filter(l => l.wip_pct >= filter.minWip!);
    }
    if (filter.maxWip !== undefined) {
      filtered = filtered.filter(l => l.wip_pct <= filter.maxWip!);
    }
  }

  // Apply sorting
  if (sort) {
    const direction = sort.direction === 'asc' ? 1 : -1;
    filtered.sort((a, b) => {
      let comparison = 0;
      switch (sort.field) {
        case 'wip_pct':
          comparison = a.wip_pct - b.wip_pct;
          break;
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'status':
          const statusOrder = { red: 0, yellow: 1, green: 2 };
          comparison = statusOrder[a.status] - statusOrder[b.status];
          break;
        case 'owner':
          comparison = a.owner.localeCompare(b.owner);
          break;
      }
      return comparison * direction;
    });
  }

  // Calculate pagination
  const total = filtered.length;
  const totalPages = Math.ceil(total / pageSize);
  const clampedPage = Math.max(1, Math.min(page, totalPages || 1));
  const startIndex = (clampedPage - 1) * pageSize;
  const items = filtered.slice(startIndex, startIndex + pageSize);

  return {
    items,
    total,
    page: clampedPage,
    pageSize,
    totalPages,
    hasNext: clampedPage < totalPages,
    hasPrev: clampedPage > 1,
  };
}

/**
 * Paginate history entries for a single lane
 */
export function paginateHistory(
  history: LaneHistory[],
  page: number = 1,
  pageSize: number = 50
): PaginatedResult<LaneHistory> {
  const total = history.length;
  const totalPages = Math.ceil(total / pageSize);
  const clampedPage = Math.max(1, Math.min(page, totalPages || 1));
  const startIndex = (clampedPage - 1) * pageSize;
  const items = history.slice(startIndex, startIndex + pageSize);

  return {
    items,
    total,
    page: clampedPage,
    pageSize,
    totalPages,
    hasNext: clampedPage < totalPages,
    hasPrev: clampedPage > 1,
  };
}

/**
 * Get history for a specific lane by ID
 */
export function getLaneHistory(
  lanes: Lane[],
  laneId: string,
  page: number = 1,
  pageSize: number = 50
): PaginatedResult<LaneHistory> | null {
  const lane = lanes.find(l => l.id === laneId);
  if (!lane) return null;
  return paginateHistory(lane.history || [], page, pageSize);
}

/**
 * Get aggregated history across all lanes (for cross-lane timeline view)
 */
export interface AggregatedHistoryEntry {
  ts: string;
  laneId: string;
  laneName: string;
  wip_pct: number;
  note: string;
}

export function getAggregatedHistory(
  lanes: Lane[],
  page: number = 1,
  pageSize: number = 50,
  dateRange?: { start?: string; end?: string }
): PaginatedResult<AggregatedHistoryEntry> {
  // Collect all history entries with lane info
  const allEntries: AggregatedHistoryEntry[] = [];

  for (const lane of lanes) {
    for (const entry of lane.history || []) {
      allEntries.push({
        ts: entry.ts,
        laneId: lane.id,
        laneName: lane.name,
        wip_pct: entry.wip_pct,
        note: entry.note,
      });
    }
  }

  // Filter by date range if provided
  let filtered = allEntries;
  if (dateRange) {
    if (dateRange.start) {
      filtered = filtered.filter(e => e.ts >= dateRange.start!);
    }
    if (dateRange.end) {
      filtered = filtered.filter(e => e.ts <= dateRange.end!);
    }
  }

  // Sort by timestamp (most recent first)
  filtered.sort((a, b) => b.ts.localeCompare(a.ts));

  // Paginate
  const total = filtered.length;
  const totalPages = Math.ceil(total / pageSize);
  const clampedPage = Math.max(1, Math.min(page, totalPages || 1));
  const startIndex = (clampedPage - 1) * pageSize;
  const items = filtered.slice(startIndex, startIndex + pageSize);

  return {
    items,
    total,
    page: clampedPage,
    pageSize,
    totalPages,
    hasNext: clampedPage < totalPages,
    hasPrev: clampedPage > 1,
  };
}

/**
 * Get unique owners from lanes
 */
export function getUniqueOwners(lanes: Lane[]): string[] {
  return [...new Set(lanes.map(l => l.owner))].sort();
}

/**
 * Get lane statistics
 */
export interface LaneStats {
  total: number;
  byStatus: { green: number; yellow: number; red: number };
  avgWip: number;
  blocked: number;
  complete: number;
  active: number;
}

export function getLaneStats(lanes: Lane[]): LaneStats {
  const total = lanes.length;
  const byStatus = {
    green: lanes.filter(l => l.status === 'green').length,
    yellow: lanes.filter(l => l.status === 'yellow').length,
    red: lanes.filter(l => l.status === 'red').length,
  };
  const avgWip = total > 0
    ? Math.round(lanes.reduce((sum, l) => sum + l.wip_pct, 0) / total)
    : 0;
  const blocked = lanes.filter(l => l.blockers.length > 0).length;
  const complete = lanes.filter(l => l.wip_pct === 100).length;
  const active = lanes.filter(l => l.wip_pct > 0 && l.wip_pct < 100).length;

  return { total, byStatus, avgWip, blocked, complete, active };
}
