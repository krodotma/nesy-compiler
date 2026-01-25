import { component$, type Signal, useComputed$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import type { BusEvent } from '../../lib/state/types';
import { PBDeepGraph } from '../pbdeep/PBDeepGraph';
import { PBDeepCodeGraph } from '../pbdeep/PBDeepCodeGraph';
import { PBDeepPulse } from '../pbdeep/PBDeepPulse';

interface PBDeepViewProps {
  events: Signal<BusEvent[]>;
}

const asData = (event?: BusEvent | null): Record<string, unknown> => {
  if (!event || !event.data || typeof event.data !== 'object') return {};
  return event.data as Record<string, unknown>;
};

const normalizeFsPath = (raw: string): string => {
  if (!raw) return '';
  let path = raw.trim();
  if (path.startsWith('/pluribus/')) path = path.slice('/pluribus/'.length);
  if (path.startsWith('pluribus/')) path = path.slice('pluribus/'.length);
  if (path.startsWith('/')) path = path.slice(1);
  return path;
};

const encodeFsPath = (path: string): string => encodeURIComponent(path).replaceAll('%2F', '/');

const shortIso = (iso: string | undefined, ts: number | undefined): string => {
  if (iso) return iso.slice(0, 19);
  if (!ts) return '';
  return new Date(ts).toISOString().slice(0, 19);
};

export const PBDeepView = component$<PBDeepViewProps>((props) => {
  const reportData = useSignal<any | null>(null);
  const reportError = useSignal<string | null>(null);

  const pbdeepEvents = useComputed$(() =>
    props.events.value.filter((event) => event.topic.startsWith('operator.pbdeep.'))
  );

  const latestReport = useComputed$(() => {
    const reports = pbdeepEvents.value.filter((event) => event.topic === 'operator.pbdeep.report');
    return reports.sort((a, b) => (a.ts || 0) - (b.ts || 0))[reports.length - 1] || null;
  });

  const latestProgress = useComputed$(() => {
    const progress = pbdeepEvents.value.filter((event) => event.topic === 'operator.pbdeep.progress');
    return progress.sort((a, b) => (a.ts || 0) - (b.ts || 0))[progress.length - 1] || null;
  });

  const latestRequest = useComputed$(() => {
    const requests = pbdeepEvents.value.filter((event) => event.topic === 'operator.pbdeep.request');
    return requests.sort((a, b) => (a.ts || 0) - (b.ts || 0))[requests.length - 1] || null;
  });

  const activeReqId = useComputed$(() => {
    const report = asData(latestReport.value);
    const progress = asData(latestProgress.value);
    const request = asData(latestRequest.value);
    return String(report.req_id || progress.req_id || request.req_id || '') || null;
  });

  const progressEvents = useComputed$(() => {
    const reqId = activeReqId.value;
    return pbdeepEvents.value
      .filter((event) => event.topic === 'operator.pbdeep.progress')
      .filter((event) => !reqId || String(asData(event).req_id || '') === reqId)
      .sort((a, b) => (a.ts || 0) - (b.ts || 0));
  });

  const timelineEvents = useComputed$(() =>
    pbdeepEvents.value
      .filter((event) =>
        event.topic === 'operator.pbdeep.request' ||
        event.topic === 'operator.pbdeep.progress' ||
        event.topic === 'operator.pbdeep.report'
      )
      .sort((a, b) => (a.ts || 0) - (b.ts || 0))
      .slice(-16)
  );

  const progressState = useComputed$(() => {
    const last = progressEvents.value[progressEvents.value.length - 1];
    const data = asData(last);
    return {
      percent: Number(data.percent ?? 0),
      stage: String(data.stage || 'idle'),
      status: String(data.status || 'idle'),
      iso: String(data.iso || ''),
    };
  });

  const eventRate = useComputed$(() => {
    const events = progressEvents.value;
    if (events.length < 2) return 0;
    const first = events[0];
    const last = events[events.length - 1];
    const minutes = Math.max(0.1, ((last.ts || 0) - (first.ts || 0)) / 60000);
    return events.length / minutes;
  });

  const summary = useComputed$(() => {
    const report = asData(latestReport.value);
    return (report.summary as Record<string, unknown>) || (reportData.value?.summary ?? {});
  });

  const graph = useComputed$(() => {
    const report = asData(latestReport.value);
    return (report.graph as any) || reportData.value?.graph || { nodes: [], edges: [] };
  });

  const reportPath = useComputed$(() => {
    const report = asData(latestReport.value);
    return normalizeFsPath(String(report.report_path || report.reportPath || ''));
  });

  const reportMdPath = useComputed$(() => {
    const report = asData(latestReport.value);
    return normalizeFsPath(String(report.report_md_path || report.reportMdPath || ''));
  });

  useVisibleTask$(async ({ track }) => {
    const path = track(() => reportPath.value);
    reportError.value = null;
    reportData.value = null;
    if (!path) return;

    try {
      const res = await fetch(`/api/fs/${encodeFsPath(path)}`, { cache: 'no-store' });
      if (!res.ok) {
        throw new Error(`fs ${res.status}`);
      }
      const raw = await res.text();
      reportData.value = JSON.parse(raw);
    } catch (err) {
      reportError.value = err instanceof Error ? err.message : 'failed to load report';
    }
  });

  const lostPaths = useComputed$(() => {
    const paths = reportData.value?.lost_and_found?.paths;
    return Array.isArray(paths) ? paths.slice(0, 8) : [];
  });

  const untrackedPaths = useComputed$(() => {
    const paths = reportData.value?.untracked?.paths;
    return Array.isArray(paths) ? paths.slice(0, 8) : [];
  });

  const docMissing = useComputed$(() => {
    const paths = reportData.value?.doc_drift?.docs_missing;
    return Array.isArray(paths) ? paths.slice(0, 6) : [];
  });

  const codeMissing = useComputed$(() => {
    const paths = reportData.value?.doc_drift?.code_missing;
    return Array.isArray(paths) ? paths.slice(0, 6) : [];
  });

  return (
    <div class="space-y-6 glass-aurora-bg p-4 rounded-2xl">
      <div class="glass-surface glass-gradient-border p-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div class="text-2xl font-bold glass-chromatic-subtle">PBDEEP Forensics Console</div>
          <div class="text-sm text-muted-foreground">
            Deep audit of branches, lost work, untracked drift, and doc mismatches.
          </div>
        </div>
        <div class="text-xs text-muted-foreground mono">
          req:{activeReqId.value ? activeReqId.value.slice(0, 10) : '-'} | last:{shortIso(progressState.value.iso, latestProgress.value?.ts)}
        </div>
      </div>

      <div class="glass-surface glass-surface-1 p-3">
        <div class="flex items-center justify-between pb-2">
          <div class="text-xs uppercase tracking-[0.2em] text-muted-foreground">Progress</div>
          <div class="text-[10px] text-muted-foreground">
            {progressState.value.stage} | {progressState.value.status} | {Math.round(progressState.value.percent)}%
          </div>
        </div>
        <div class="h-2 rounded-full bg-muted/40 overflow-hidden">
          <div
            class="h-full bg-cyan-400/80 transition-all"
            style={{ width: `${Math.min(100, Math.max(0, progressState.value.percent))}%` }}
          />
        </div>
      </div>

      <div class="grid grid-cols-12 gap-6">
        <div class="col-span-12 xl:col-span-7 space-y-6">
          <PBDeepGraph graph={graph.value} height={360} />
          <PBDeepCodeGraph graph={graph.value} height={260} />
        </div>

        <div class="col-span-12 xl:col-span-5 space-y-6">
          <PBDeepPulse
            percent={progressState.value.percent}
            stage={progressState.value.stage}
            status={progressState.value.status}
            eventRate={eventRate.value}
            pulseCount={progressEvents.value.length}
            activeReqId={activeReqId.value}
            width={520}
            height={220}
          />

          <div class="glass-surface glass-surface-1 p-4 space-y-3">
            <div class="flex items-center justify-between">
              <div class="text-xs uppercase tracking-[0.2em] text-muted-foreground">Summary</div>
              <div class="text-[10px] text-muted-foreground">mode {(asData(latestReport.value).mode as string) || '-'}</div>
            </div>
            <div class="grid grid-cols-2 gap-3 text-sm">
              <div class="glass-surface glass-surface-interactive p-2">
                <div class="text-xs text-muted-foreground uppercase">Branches</div>
                <div class="text-lg font-mono">{String(summary.value?.branches_total ?? '-')}</div>
              </div>
              <div class="glass-surface glass-surface-interactive p-2">
                <div class="text-xs text-muted-foreground uppercase">Final</div>
                <div class="text-lg font-mono">{String(summary.value?.final_branches ?? '-')}</div>
              </div>
              <div class="glass-surface glass-surface-interactive p-2">
                <div class="text-xs text-muted-foreground uppercase">Lost/Found</div>
                <div class="text-lg font-mono">{String(summary.value?.lost_and_found_count ?? '-')}</div>
              </div>
              <div class="glass-surface glass-surface-interactive p-2">
                <div class="text-xs text-muted-foreground uppercase">Untracked</div>
                <div class="text-lg font-mono">{String(summary.value?.untracked_count ?? '-')}</div>
              </div>
              <div class="glass-surface glass-surface-interactive p-2">
                <div class="text-xs text-muted-foreground uppercase">Doc Missing</div>
                <div class="text-lg font-mono">{String(summary.value?.doc_missing_count ?? '-')}</div>
              </div>
              <div class="glass-surface glass-surface-interactive p-2">
                <div class="text-xs text-muted-foreground uppercase">Code Missing</div>
                <div class="text-lg font-mono">{String(summary.value?.code_missing_docs ?? '-')}</div>
              </div>
            </div>
            <div class="text-xs text-muted-foreground">
              {Array.isArray(summary.value?.next_actions) && summary.value.next_actions.length > 0 ? (
                summary.value.next_actions.map((item: string) => (
                  <div key={item}>- {item}</div>
                ))
              ) : (
                <div>- Awaiting PBDEEP report</div>
              )}
            </div>
          </div>

          <div class="glass-surface glass-surface-1 p-4 space-y-3">
            <div class="flex items-center justify-between">
              <div class="text-xs uppercase tracking-[0.2em] text-muted-foreground">Timeline</div>
              <div class="text-[10px] text-muted-foreground">{timelineEvents.value.length} events</div>
            </div>
            <div class="space-y-2 text-xs font-mono text-muted-foreground">
              {timelineEvents.value.length === 0 && <div>No PBDEEP activity yet.</div>}
              {timelineEvents.value.map((event) => {
                const data = asData(event);
                const base = shortIso(event.iso, event.ts);
                if (event.topic === 'operator.pbdeep.progress') {
                  return (
                    <div key={`${event.id}-${event.ts}`}>
                      {base} progress {String(data.stage || '')} {String(data.status || '')} {String(data.percent || 0)}%
                    </div>
                  );
                }
                if (event.topic === 'operator.pbdeep.report') {
                  return (
                    <div key={`${event.id}-${event.ts}`}>
                      {base} report {String(data.mode || '')} {String(data.req_id || '').slice(0, 8)}
                    </div>
                  );
                }
                return (
                  <div key={`${event.id}-${event.ts}`}>
                    {base} request {String(data.req_id || '').slice(0, 8)}
                  </div>
                );
              })}
            </div>
          </div>

          <div class="glass-surface glass-surface-1 p-4 space-y-3">
            <div class="flex items-center justify-between">
              <div class="text-xs uppercase tracking-[0.2em] text-muted-foreground">Report Details</div>
              <div class="text-[10px] text-muted-foreground">
                {reportPath.value ? (
                  <a class="text-cyan-300 hover:text-cyan-200" href={`/api/fs/${encodeFsPath(reportPath.value)}`}>
                    {reportPath.value.split('/').slice(-1)[0]}
                  </a>
                ) : (
                  'no report'
                )}
              </div>
            </div>
            {reportError.value && <div class="text-xs text-red-400">Report load failed: {reportError.value}</div>}
            {!reportError.value && !reportData.value && <div class="text-xs text-muted-foreground">Awaiting report snapshot.</div>}
            {reportData.value && (
              <div class="space-y-3 text-xs text-muted-foreground">
                <div>
                  <div class="text-[10px] uppercase">Lost/Found</div>
                  {lostPaths.value.length === 0 ? (
                    <div class="text-muted-foreground">None detected.</div>
                  ) : (
                    <div class="space-y-1">
                      {lostPaths.value.map((path) => (
                        <div key={path}>{path}</div>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <div class="text-[10px] uppercase">Untracked</div>
                  {untrackedPaths.value.length === 0 ? (
                    <div class="text-muted-foreground">None detected.</div>
                  ) : (
                    <div class="space-y-1">
                      {untrackedPaths.value.map((path) => (
                        <div key={path}>{path}</div>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <div class="text-[10px] uppercase">Doc Drift</div>
                  {docMissing.value.length === 0 && codeMissing.value.length === 0 && (
                    <div class="text-muted-foreground">No doc/code gaps found.</div>
                  )}
                  {docMissing.value.length > 0 && (
                    <div class="space-y-1">
                      {docMissing.value.map((item: any) => {
                        const label = typeof item === 'string'
                          ? item
                          : `${String(item.doc || 'doc')} -> ${String(item.path || '')}`;
                        return (
                          <div key={`${String(item.doc || '')}-${String(item.path || '')}`}>
                            doc missing: {label}
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {codeMissing.value.length > 0 && (
                    <div class="space-y-1">
                      {codeMissing.value.map((item: any) => {
                        const label = typeof item === 'string'
                          ? item
                          : `${String(item.source || 'code')} -> ${String(item.path || '')}`;
                        return (
                          <div key={`${String(item.source || '')}-${String(item.path || '')}`}>
                            code missing: {label}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
                {reportMdPath.value && (
                  <div class="text-[10px] text-muted-foreground">
                    <a class="text-cyan-300 hover:text-cyan-200" href={`/api/fs/${encodeFsPath(reportMdPath.value)}`}>
                      Open markdown report
                    </a>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});
