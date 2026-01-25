/**
 * LanePrintReport - Printer-friendly view and report generation
 *
 * Phase 4, Iteration 35 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Printer-friendly view with clean layout
 * - PDF export (via print dialog)
 * - Configurable report sections
 * - Executive summary generation
 * - Scheduled report settings
 * - Multiple export formats (JSON, CSV, Markdown)
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - LanePrintReport
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/checkbox/checkbox.js';
import '@material/web/select/outlined-select.js';
import '@material/web/select/select-option.js';

// ============================================================================
// Types
// ============================================================================

export interface ReportLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  blockers: number;
  actions: number;
  description?: string;
  startDate?: string;
  targetDate?: string;
  history: { date: string; wip_pct: number; note?: string }[];
}

export interface ReportConfig {
  title: string;
  subtitle?: string;
  includeExecutiveSummary: boolean;
  includeLaneDetails: boolean;
  includeCharts: boolean;
  includeHistory: boolean;
  includeBlockers: boolean;
  includeActions: boolean;
  groupBy?: 'status' | 'owner' | 'none';
  sortBy: 'name' | 'wip_pct' | 'status' | 'owner';
  sortDirection: 'asc' | 'desc';
  dateRange?: { start: string; end: string };
  footer?: string;
}

export interface ScheduleConfig {
  enabled: boolean;
  frequency: 'daily' | 'weekly' | 'monthly';
  dayOfWeek?: number;
  dayOfMonth?: number;
  time: string;
  recipients: string[];
  format: 'pdf' | 'email' | 'slack';
}

export interface LanePrintReportProps {
  /** Lanes to include in report */
  lanes: ReportLane[];
  /** Report configuration */
  config?: Partial<ReportConfig>;
  /** Callback when report is generated */
  onReportGenerate$?: QRL<(format: string) => void>;
  /** Callback when schedule is saved */
  onScheduleSave$?: QRL<(schedule: ScheduleConfig) => void>;
  /** Current user for footer */
  generatedBy?: string;
}

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: ReportConfig = {
  title: 'Lanes Status Report',
  includeExecutiveSummary: true,
  includeLaneDetails: true,
  includeCharts: true,
  includeHistory: false,
  includeBlockers: true,
  includeActions: true,
  groupBy: 'none',
  sortBy: 'wip_pct',
  sortDirection: 'desc',
};

// ============================================================================
// Helpers
// ============================================================================

function formatDate(dateStr?: string): string {
  if (!dateStr) return 'N/A';
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  } catch {
    return dateStr;
  }
}

function getStatusLabel(status: string): string {
  switch (status) {
    case 'green': return 'On Track';
    case 'yellow': return 'At Risk';
    case 'red': return 'Blocked';
    default: return 'Unknown';
  }
}

function generateCSV(lanes: ReportLane[]): string {
  const headers = ['Name', 'Owner', 'Status', 'WIP %', 'Blockers', 'Actions', 'Target Date'];
  const rows = lanes.map(l => [
    l.name,
    l.owner,
    l.status,
    l.wip_pct.toString(),
    l.blockers.toString(),
    l.actions.toString(),
    l.targetDate || '',
  ]);
  return [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
}

function generateMarkdown(lanes: ReportLane[], config: ReportConfig): string {
  let md = `# ${config.title}\n\n`;

  if (config.subtitle) {
    md += `*${config.subtitle}*\n\n`;
  }

  md += `**Generated:** ${new Date().toLocaleString()}\n\n`;

  // Summary
  if (config.includeExecutiveSummary) {
    const total = lanes.length;
    const green = lanes.filter(l => l.status === 'green').length;
    const yellow = lanes.filter(l => l.status === 'yellow').length;
    const red = lanes.filter(l => l.status === 'red').length;
    const avgWip = Math.round(lanes.reduce((sum, l) => sum + l.wip_pct, 0) / total);

    md += `## Executive Summary\n\n`;
    md += `- **Total Lanes:** ${total}\n`;
    md += `- **On Track:** ${green} (${Math.round(green/total*100)}%)\n`;
    md += `- **At Risk:** ${yellow} (${Math.round(yellow/total*100)}%)\n`;
    md += `- **Blocked:** ${red} (${Math.round(red/total*100)}%)\n`;
    md += `- **Average WIP:** ${avgWip}%\n\n`;
  }

  // Lanes table
  if (config.includeLaneDetails) {
    md += `## Lane Details\n\n`;
    md += `| Lane | Owner | Status | WIP % | Blockers |\n`;
    md += `|------|-------|--------|-------|----------|\n`;
    lanes.forEach(l => {
      md += `| ${l.name} | @${l.owner} | ${getStatusLabel(l.status)} | ${l.wip_pct}% | ${l.blockers} |\n`;
    });
    md += '\n';
  }

  return md;
}

function generateJSON(lanes: ReportLane[], config: ReportConfig): string {
  return JSON.stringify({
    title: config.title,
    generatedAt: new Date().toISOString(),
    summary: {
      total: lanes.length,
      byStatus: {
        green: lanes.filter(l => l.status === 'green').length,
        yellow: lanes.filter(l => l.status === 'yellow').length,
        red: lanes.filter(l => l.status === 'red').length,
      },
      averageWip: Math.round(lanes.reduce((sum, l) => sum + l.wip_pct, 0) / lanes.length),
    },
    lanes: lanes.map(l => ({
      id: l.id,
      name: l.name,
      owner: l.owner,
      status: l.status,
      wip_pct: l.wip_pct,
      blockers: l.blockers,
      actions: l.actions,
    })),
  }, null, 2);
}

// ============================================================================
// Component
// ============================================================================

export const LanePrintReport = component$<LanePrintReportProps>(({
  lanes,
  config: initialConfig,
  onReportGenerate$,
  onScheduleSave$,
  generatedBy = 'System',
}) => {
  // Merge config with defaults
  const config = { ...DEFAULT_CONFIG, ...initialConfig };

  // State
  const showPrintPreview = useSignal(false);
  const showScheduleModal = useSignal(false);
  const showConfigPanel = useSignal(false);

  const reportConfig = useSignal<ReportConfig>({ ...config });

  const scheduleConfig = useSignal<ScheduleConfig>({
    enabled: false,
    frequency: 'weekly',
    dayOfWeek: 1,
    time: '09:00',
    recipients: [],
    format: 'email',
  });

  // Sorted/grouped lanes
  const processedLanes = useComputed$(() => {
    let result = [...lanes];

    // Sort
    result.sort((a, b) => {
      const field = reportConfig.value.sortBy;
      let aVal = a[field];
      let bVal = b[field];

      if (field === 'status') {
        const order = { green: 0, yellow: 1, red: 2 };
        aVal = order[a.status];
        bVal = order[b.status];
      }

      if (aVal < bVal) return reportConfig.value.sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return reportConfig.value.sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return result;
  });

  // Summary stats
  const stats = useComputed$(() => ({
    total: lanes.length,
    green: lanes.filter(l => l.status === 'green').length,
    yellow: lanes.filter(l => l.status === 'yellow').length,
    red: lanes.filter(l => l.status === 'red').length,
    avgWip: lanes.length > 0
      ? Math.round(lanes.reduce((sum, l) => sum + l.wip_pct, 0) / lanes.length)
      : 0,
    totalBlockers: lanes.reduce((sum, l) => sum + l.blockers, 0),
    totalActions: lanes.reduce((sum, l) => sum + l.actions, 0),
  }));

  // Export functions
  const exportPDF = $(() => {
    showPrintPreview.value = true;
    setTimeout(() => {
      window.print();
    }, 100);
    if (onReportGenerate$) onReportGenerate$('pdf');
  });

  const exportCSV = $(() => {
    const csv = generateCSV(processedLanes.value);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lanes_report_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    if (onReportGenerate$) onReportGenerate$('csv');
  });

  const exportMarkdown = $(() => {
    const md = generateMarkdown(processedLanes.value, reportConfig.value);
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lanes_report_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
    if (onReportGenerate$) onReportGenerate$('markdown');
  });

  const exportJSON = $(() => {
    const json = generateJSON(processedLanes.value, reportConfig.value);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lanes_report_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    if (onReportGenerate$) onReportGenerate$('json');
  });

  const saveSchedule = $(async () => {
    if (onScheduleSave$) {
      await onScheduleSave$(scheduleConfig.value);
    }
    showScheduleModal.value = false;
  });

  return (
    <>
      {/* Control Panel (screen only) */}
      <div class="rounded-lg border border-border bg-card print:hidden">
        {/* Header */}
        <div class="flex items-center justify-between p-3 border-b border-border/50">
          <div class="flex items-center gap-2">
            <span class="text-xs font-semibold text-muted-foreground">REPORT GENERATOR</span>
          </div>
          <div class="flex items-center gap-2">
            <button
              onClick$={() => { showConfigPanel.value = !showConfigPanel.value; }}
              class="text-[10px] px-2 py-1 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
            >
              ⚙ Configure
            </button>
            <button
              onClick$={() => { showScheduleModal.value = true; }}
              class="text-[10px] px-2 py-1 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
            >
              ⏰ Schedule
            </button>
          </div>
        </div>

        {/* Config panel */}
        {showConfigPanel.value && (
          <div class="p-3 border-b border-border/30 bg-muted/5">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
              <label class="flex items-center gap-2 text-[10px]">
                <input
                  type="checkbox"
                  checked={reportConfig.value.includeExecutiveSummary}
                  onChange$={(e) => {
                    reportConfig.value = {
                      ...reportConfig.value,
                      includeExecutiveSummary: (e.target as HTMLInputElement).checked,
                    };
                  }}
                />
                Executive Summary
              </label>
              <label class="flex items-center gap-2 text-[10px]">
                <input
                  type="checkbox"
                  checked={reportConfig.value.includeLaneDetails}
                  onChange$={(e) => {
                    reportConfig.value = {
                      ...reportConfig.value,
                      includeLaneDetails: (e.target as HTMLInputElement).checked,
                    };
                  }}
                />
                Lane Details
              </label>
              <label class="flex items-center gap-2 text-[10px]">
                <input
                  type="checkbox"
                  checked={reportConfig.value.includeBlockers}
                  onChange$={(e) => {
                    reportConfig.value = {
                      ...reportConfig.value,
                      includeBlockers: (e.target as HTMLInputElement).checked,
                    };
                  }}
                />
                Blockers
              </label>
              <label class="flex items-center gap-2 text-[10px]">
                <input
                  type="checkbox"
                  checked={reportConfig.value.includeHistory}
                  onChange$={(e) => {
                    reportConfig.value = {
                      ...reportConfig.value,
                      includeHistory: (e.target as HTMLInputElement).checked,
                    };
                  }}
                />
                History
              </label>
            </div>

            <div class="mt-3 flex items-center gap-4">
              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Sort By</label>
                <select
                  value={reportConfig.value.sortBy}
                  onChange$={(e) => {
                    reportConfig.value = {
                      ...reportConfig.value,
                      sortBy: (e.target as HTMLSelectElement).value as ReportConfig['sortBy'],
                    };
                  }}
                  class="px-2 py-1 text-[10px] rounded bg-card border border-border/50"
                >
                  <option value="name">Name</option>
                  <option value="wip_pct">WIP %</option>
                  <option value="status">Status</option>
                  <option value="owner">Owner</option>
                </select>
              </div>
              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Direction</label>
                <select
                  value={reportConfig.value.sortDirection}
                  onChange$={(e) => {
                    reportConfig.value = {
                      ...reportConfig.value,
                      sortDirection: (e.target as HTMLSelectElement).value as 'asc' | 'desc',
                    };
                  }}
                  class="px-2 py-1 text-[10px] rounded bg-card border border-border/50"
                >
                  <option value="asc">Ascending</option>
                  <option value="desc">Descending</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Export buttons */}
        <div class="p-3">
          <div class="text-[9px] text-muted-foreground mb-2">EXPORT OPTIONS</div>
          <div class="flex flex-wrap gap-2">
            <button
              onClick$={exportPDF}
              class="px-3 py-2 text-xs rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors flex items-center gap-2"
            >
              <span>📄</span> Print / PDF
            </button>
            <button
              onClick$={exportCSV}
              class="px-3 py-2 text-xs rounded bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-colors flex items-center gap-2"
            >
              <span>📊</span> CSV
            </button>
            <button
              onClick$={exportMarkdown}
              class="px-3 py-2 text-xs rounded bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-colors flex items-center gap-2"
            >
              <span>📝</span> Markdown
            </button>
            <button
              onClick$={exportJSON}
              class="px-3 py-2 text-xs rounded bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 transition-colors flex items-center gap-2"
            >
              <span>{ }</span> JSON
            </button>
          </div>
        </div>

        {/* Preview summary */}
        <div class="p-3 border-t border-border/30 bg-muted/5">
          <div class="text-[9px] text-muted-foreground mb-2">REPORT PREVIEW</div>
          <div class="grid grid-cols-4 gap-3 text-center">
            <div class="p-2 rounded bg-card border border-border/30">
              <div class="text-lg font-bold text-foreground">{stats.value.total}</div>
              <div class="text-[9px] text-muted-foreground">Total Lanes</div>
            </div>
            <div class="p-2 rounded bg-emerald-500/10 border border-emerald-500/30">
              <div class="text-lg font-bold text-emerald-400">{stats.value.green}</div>
              <div class="text-[9px] text-muted-foreground">On Track</div>
            </div>
            <div class="p-2 rounded bg-amber-500/10 border border-amber-500/30">
              <div class="text-lg font-bold text-amber-400">{stats.value.yellow}</div>
              <div class="text-[9px] text-muted-foreground">At Risk</div>
            </div>
            <div class="p-2 rounded bg-red-500/10 border border-red-500/30">
              <div class="text-lg font-bold text-red-400">{stats.value.red}</div>
              <div class="text-[9px] text-muted-foreground">Blocked</div>
            </div>
          </div>
        </div>
      </div>

      {/* Print Preview / Printable Report */}
      <div class={`${showPrintPreview.value ? 'block' : 'hidden print:block'} bg-white text-black p-8 print:p-0`}>
        {/* Report Header */}
        <div class="border-b-2 border-black pb-4 mb-6">
          <h1 class="text-2xl font-bold">{reportConfig.value.title}</h1>
          {reportConfig.value.subtitle && (
            <p class="text-gray-600 mt-1">{reportConfig.value.subtitle}</p>
          )}
          <div class="text-sm text-gray-500 mt-2">
            Generated: {new Date().toLocaleString()} | By: {generatedBy}
          </div>
        </div>

        {/* Executive Summary */}
        {reportConfig.value.includeExecutiveSummary && (
          <div class="mb-6">
            <h2 class="text-lg font-bold border-b border-gray-300 pb-1 mb-3">Executive Summary</h2>
            <div class="grid grid-cols-4 gap-4">
              <div class="text-center p-3 border rounded">
                <div class="text-2xl font-bold">{stats.value.total}</div>
                <div class="text-sm text-gray-600">Total Lanes</div>
              </div>
              <div class="text-center p-3 border rounded bg-green-50">
                <div class="text-2xl font-bold text-green-700">{stats.value.green}</div>
                <div class="text-sm text-gray-600">On Track ({Math.round(stats.value.green/stats.value.total*100)}%)</div>
              </div>
              <div class="text-center p-3 border rounded bg-yellow-50">
                <div class="text-2xl font-bold text-yellow-700">{stats.value.yellow}</div>
                <div class="text-sm text-gray-600">At Risk ({Math.round(stats.value.yellow/stats.value.total*100)}%)</div>
              </div>
              <div class="text-center p-3 border rounded bg-red-50">
                <div class="text-2xl font-bold text-red-700">{stats.value.red}</div>
                <div class="text-sm text-gray-600">Blocked ({Math.round(stats.value.red/stats.value.total*100)}%)</div>
              </div>
            </div>
            <div class="mt-4 grid grid-cols-3 gap-4 text-sm">
              <div><strong>Average WIP:</strong> {stats.value.avgWip}%</div>
              <div><strong>Total Blockers:</strong> {stats.value.totalBlockers}</div>
              <div><strong>Pending Actions:</strong> {stats.value.totalActions}</div>
            </div>
          </div>
        )}

        {/* Lane Details Table */}
        {reportConfig.value.includeLaneDetails && (
          <div class="mb-6">
            <h2 class="text-lg font-bold border-b border-gray-300 pb-1 mb-3">Lane Details</h2>
            <table class="w-full text-sm border-collapse">
              <thead>
                <tr class="bg-gray-100">
                  <th class="border p-2 text-left">Lane Name</th>
                  <th class="border p-2 text-left">Owner</th>
                  <th class="border p-2 text-center">Status</th>
                  <th class="border p-2 text-center">WIP %</th>
                  {reportConfig.value.includeBlockers && (
                    <th class="border p-2 text-center">Blockers</th>
                  )}
                  {reportConfig.value.includeActions && (
                    <th class="border p-2 text-center">Actions</th>
                  )}
                  <th class="border p-2 text-center">Target</th>
                </tr>
              </thead>
              <tbody>
                {processedLanes.value.map(lane => (
                  <tr key={lane.id} class="hover:bg-gray-50">
                    <td class="border p-2 font-medium">{lane.name}</td>
                    <td class="border p-2">@{lane.owner}</td>
                    <td class={`border p-2 text-center ${
                      lane.status === 'green' ? 'bg-green-100 text-green-800' :
                      lane.status === 'yellow' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {getStatusLabel(lane.status)}
                    </td>
                    <td class="border p-2 text-center font-bold">{lane.wip_pct}%</td>
                    {reportConfig.value.includeBlockers && (
                      <td class={`border p-2 text-center ${lane.blockers > 0 ? 'text-red-600 font-bold' : ''}`}>
                        {lane.blockers}
                      </td>
                    )}
                    {reportConfig.value.includeActions && (
                      <td class="border p-2 text-center">{lane.actions}</td>
                    )}
                    <td class="border p-2 text-center">{formatDate(lane.targetDate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* History */}
        {reportConfig.value.includeHistory && (
          <div class="mb-6">
            <h2 class="text-lg font-bold border-b border-gray-300 pb-1 mb-3">Recent History</h2>
            {processedLanes.value.slice(0, 5).map(lane => (
              <div key={lane.id} class="mb-4">
                <h3 class="font-medium">{lane.name}</h3>
                <div class="text-sm text-gray-600 mt-1">
                  {lane.history.slice(0, 5).map((h, i) => (
                    <span key={i} class="inline-block mr-4">
                      {h.date.slice(5, 10)}: {h.wip_pct}%
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Footer */}
        <div class="border-t border-gray-300 pt-4 mt-8 text-sm text-gray-500 flex justify-between">
          <div>{reportConfig.value.footer || 'Pluribus Lanes Report'}</div>
          <div>Page 1 of 1</div>
        </div>
      </div>

      {/* Schedule Modal */}
      {showScheduleModal.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 print:hidden">
          <div class="bg-card rounded-lg border border-border p-4 w-96">
            <div class="text-xs font-semibold text-foreground mb-4">Schedule Report</div>

            <div class="space-y-4">
              <label class="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={scheduleConfig.value.enabled}
                  onChange$={(e) => {
                    scheduleConfig.value = {
                      ...scheduleConfig.value,
                      enabled: (e.target as HTMLInputElement).checked,
                    };
                  }}
                />
                <span class="text-xs">Enable scheduled reports</span>
              </label>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Frequency</label>
                <select
                  value={scheduleConfig.value.frequency}
                  onChange$={(e) => {
                    scheduleConfig.value = {
                      ...scheduleConfig.value,
                      frequency: (e.target as HTMLSelectElement).value as ScheduleConfig['frequency'],
                    };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                >
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>

              {scheduleConfig.value.frequency === 'weekly' && (
                <div>
                  <label class="text-[9px] text-muted-foreground block mb-1">Day of Week</label>
                  <select
                    value={scheduleConfig.value.dayOfWeek}
                    onChange$={(e) => {
                      scheduleConfig.value = {
                        ...scheduleConfig.value,
                        dayOfWeek: parseInt((e.target as HTMLSelectElement).value),
                      };
                    }}
                    class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                  >
                    <option value="1">Monday</option>
                    <option value="2">Tuesday</option>
                    <option value="3">Wednesday</option>
                    <option value="4">Thursday</option>
                    <option value="5">Friday</option>
                  </select>
                </div>
              )}

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Time</label>
                <input
                  type="time"
                  value={scheduleConfig.value.time}
                  onChange$={(e) => {
                    scheduleConfig.value = {
                      ...scheduleConfig.value,
                      time: (e.target as HTMLInputElement).value,
                    };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                />
              </div>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Delivery Format</label>
                <select
                  value={scheduleConfig.value.format}
                  onChange$={(e) => {
                    scheduleConfig.value = {
                      ...scheduleConfig.value,
                      format: (e.target as HTMLSelectElement).value as ScheduleConfig['format'],
                    };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                >
                  <option value="email">Email</option>
                  <option value="slack">Slack</option>
                  <option value="pdf">PDF Download</option>
                </select>
              </div>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Recipients (comma-separated)</label>
                <input
                  type="text"
                  value={scheduleConfig.value.recipients.join(', ')}
                  onInput$={(e) => {
                    scheduleConfig.value = {
                      ...scheduleConfig.value,
                      recipients: (e.target as HTMLInputElement).value.split(',').map(s => s.trim()).filter(Boolean),
                    };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                  placeholder="user@example.com, team@example.com"
                />
              </div>
            </div>

            <div class="flex items-center gap-2 mt-4">
              <button
                onClick$={() => { showScheduleModal.value = false; }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={saveSchedule}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground"
              >
                Save Schedule
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Print styles */}
      <style>
        {`
          @media print {
            body * {
              visibility: hidden;
            }
            .print\\:block, .print\\:block * {
              visibility: visible;
            }
            .print\\:block {
              position: absolute;
              left: 0;
              top: 0;
              width: 100%;
            }
            .print\\:hidden {
              display: none !important;
            }
          }
        `}
      </style>
    </>
  );
});

export default LanePrintReport;
