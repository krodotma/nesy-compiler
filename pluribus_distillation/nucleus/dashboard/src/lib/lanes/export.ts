/**
 * History Export Utilities
 *
 * Phase 2, Iteration 13 of OITERATE lanes-widget-enhancement
 * Export lane history in various formats
 */

import type { Lane, LaneHistory } from '../../components/LaneHistoryTimeline';

// ============================================================================
// Types
// ============================================================================

export interface ExportOptions {
  format: 'json' | 'csv' | 'markdown';
  includeMetadata?: boolean;
  dateFormat?: 'iso' | 'human';
}

export interface AggregatedHistoryEntry {
  ts: string;
  laneId: string;
  laneName: string;
  wip_pct: number;
  note: string;
}

// ============================================================================
// JSON Export
// ============================================================================

/**
 * Export lane history as JSON
 */
export function exportLaneHistoryJSON(
  lane: Lane,
  options: { includeMetadata?: boolean; pretty?: boolean } = {}
): string {
  const { includeMetadata = true, pretty = true } = options;

  const data = includeMetadata
    ? {
        exported_at: new Date().toISOString(),
        lane: {
          id: lane.id,
          name: lane.name,
          status: lane.status,
          owner: lane.owner,
          current_wip: lane.wip_pct,
          description: lane.description,
        },
        history: lane.history,
        total_entries: lane.history.length,
      }
    : lane.history;

  return pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);
}

/**
 * Export all lanes history as JSON
 */
export function exportAllHistoryJSON(
  lanes: Lane[],
  options: { includeMetadata?: boolean; pretty?: boolean } = {}
): string {
  const { includeMetadata = true, pretty = true } = options;

  const allHistory: AggregatedHistoryEntry[] = [];
  for (const lane of lanes) {
    for (const h of lane.history || []) {
      allHistory.push({
        ts: h.ts,
        laneId: lane.id,
        laneName: lane.name,
        wip_pct: h.wip_pct,
        note: h.note,
      });
    }
  }

  // Sort by timestamp (most recent first)
  allHistory.sort((a, b) => b.ts.localeCompare(a.ts));

  const data = includeMetadata
    ? {
        exported_at: new Date().toISOString(),
        total_lanes: lanes.length,
        total_entries: allHistory.length,
        history: allHistory,
      }
    : allHistory;

  return pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);
}

// ============================================================================
// CSV Export
// ============================================================================

/**
 * Escape a value for CSV
 */
function escapeCSV(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return '';
  const str = String(value);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Export lane history as CSV
 */
export function exportLaneHistoryCSV(lane: Lane): string {
  const headers = ['timestamp', 'wip_pct', 'note'];
  const rows = lane.history.map(h => [
    escapeCSV(h.ts),
    escapeCSV(h.wip_pct),
    escapeCSV(h.note),
  ].join(','));

  return [
    `# Lane: ${lane.name} (${lane.id})`,
    `# Owner: ${lane.owner}`,
    `# Status: ${lane.status}`,
    `# Current WIP: ${lane.wip_pct}%`,
    `# Exported: ${new Date().toISOString()}`,
    '',
    headers.join(','),
    ...rows,
  ].join('\n');
}

/**
 * Export all lanes history as CSV
 */
export function exportAllHistoryCSV(lanes: Lane[]): string {
  const headers = ['timestamp', 'lane_id', 'lane_name', 'owner', 'wip_pct', 'note'];

  const rows: string[] = [];
  for (const lane of lanes) {
    for (const h of lane.history || []) {
      rows.push([
        escapeCSV(h.ts),
        escapeCSV(lane.id),
        escapeCSV(lane.name),
        escapeCSV(lane.owner),
        escapeCSV(h.wip_pct),
        escapeCSV(h.note),
      ].join(','));
    }
  }

  // Sort by timestamp
  rows.sort((a, b) => b.localeCompare(a));

  return [
    `# Lanes History Export`,
    `# Total Lanes: ${lanes.length}`,
    `# Total Entries: ${rows.length}`,
    `# Exported: ${new Date().toISOString()}`,
    '',
    headers.join(','),
    ...rows,
  ].join('\n');
}

// ============================================================================
// Markdown Export
// ============================================================================

/**
 * Export lane history as Markdown
 */
export function exportLaneHistoryMarkdown(lane: Lane): string {
  const lines: string[] = [
    `# ${lane.name}`,
    '',
    `| Property | Value |`,
    `|----------|-------|`,
    `| ID | \`${lane.id}\` |`,
    `| Owner | @${lane.owner} |`,
    `| Status | ${lane.status} |`,
    `| Current WIP | ${lane.wip_pct}% |`,
    '',
    `## History (${lane.history.length} entries)`,
    '',
  ];

  if (lane.history.length > 0) {
    lines.push('| Date | WIP | Note |');
    lines.push('|------|-----|------|');

    for (const h of lane.history) {
      const date = h.ts.slice(0, 10);
      const note = h.note ? h.note.replace(/\|/g, '\\|').replace(/\n/g, ' ') : '-';
      lines.push(`| ${date} | ${h.wip_pct}% | ${note} |`);
    }
  } else {
    lines.push('*No history entries*');
  }

  lines.push('');
  lines.push(`---`);
  lines.push(`*Exported: ${new Date().toISOString()}*`);

  return lines.join('\n');
}

/**
 * Export all lanes history as Markdown
 */
export function exportAllHistoryMarkdown(lanes: Lane[]): string {
  const lines: string[] = [
    `# Lanes History Export`,
    '',
    `**Total Lanes:** ${lanes.length}`,
    `**Exported:** ${new Date().toISOString()}`,
    '',
    '---',
    '',
  ];

  for (const lane of lanes) {
    lines.push(`## ${lane.name}`);
    lines.push('');
    lines.push(`- **ID:** \`${lane.id}\``);
    lines.push(`- **Owner:** @${lane.owner}`);
    lines.push(`- **Status:** ${lane.status}`);
    lines.push(`- **WIP:** ${lane.wip_pct}%`);
    lines.push('');

    if (lane.history.length > 0) {
      lines.push('| Date | WIP | Note |');
      lines.push('|------|-----|------|');

      for (const h of lane.history.slice(0, 10)) {
        const date = h.ts.slice(0, 10);
        const note = h.note ? h.note.replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 50) : '-';
        lines.push(`| ${date} | ${h.wip_pct}% | ${note}${h.note && h.note.length > 50 ? '...' : ''} |`);
      }

      if (lane.history.length > 10) {
        lines.push(`| ... | ... | *${lane.history.length - 10} more entries* |`);
      }
    } else {
      lines.push('*No history*');
    }

    lines.push('');
  }

  return lines.join('\n');
}

// ============================================================================
// Clipboard / Download Helpers
// ============================================================================

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const result = document.execCommand('copy');
    document.body.removeChild(textarea);
    return result;
  } catch {
    return false;
  }
}

/**
 * Download text as a file
 */
export function downloadFile(
  content: string,
  filename: string,
  mimeType: string = 'text/plain'
): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export with format detection
 */
export function exportHistory(
  lanes: Lane | Lane[],
  options: ExportOptions
): string {
  const isArray = Array.isArray(lanes);
  const { format, includeMetadata = true } = options;

  switch (format) {
    case 'json':
      return isArray
        ? exportAllHistoryJSON(lanes, { includeMetadata })
        : exportLaneHistoryJSON(lanes, { includeMetadata });
    case 'csv':
      return isArray
        ? exportAllHistoryCSV(lanes)
        : exportLaneHistoryCSV(lanes);
    case 'markdown':
      return isArray
        ? exportAllHistoryMarkdown(lanes)
        : exportLaneHistoryMarkdown(lanes);
    default:
      throw new Error(`Unknown export format: ${format}`);
  }
}

/**
 * Get suggested filename for export
 */
export function getSuggestedFilename(
  lanes: Lane | Lane[],
  format: 'json' | 'csv' | 'markdown'
): string {
  const isArray = Array.isArray(lanes);
  const date = new Date().toISOString().slice(0, 10);
  const ext = format === 'markdown' ? 'md' : format;

  if (isArray) {
    return `lanes-history-${date}.${ext}`;
  } else {
    const safeName = lanes.name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    return `lane-${safeName}-history-${date}.${ext}`;
  }
}
