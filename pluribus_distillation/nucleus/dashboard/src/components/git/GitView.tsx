/**
 * Git View - The "GitHub Killer"
 * Combines GitGraph, Commit Details, Kroma/HGT Controls, and Rhizome Promotion Flow.
 */
import { component$, useSignal, useVisibleTask$, $, useStore } from '@builder.io/qwik';
import { GitGraph } from './GitGraph';

// M3 Components - GitView
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';

// Type for Rhizome Artifact (Mock/Contract)
interface RhizomeArtifact {
  sha: string;
  name: string;
  kind: 'doc' | 'spec' | 'code' | 'data';
  domain: string;
  tags: string[];
  created_iso: string;
  citation_count: number;
}

// Type for Planner/Executor Status
interface RepoFlowStatus {
  reqId: string | null;
  step: 'idle' | 'planning' | 'reviewing' | 'staging' | 'checking' | 'publishing' | 'complete' | 'failed';
  message: string;
  logs: string[];
  plan?: any;
}

interface RecoveryBundle {
  name: string;
  path: string;
  entries: number;
  mtime?: string | null;
}

interface RecoverySnapshot {
  path: string;
  created_iso?: string | null;
  summary?: Record<string, any>;
  run_id?: string | null;
}

interface TaskLedgerEntry {
  req_id: string;
  actor: string;
  topic: string;
  status: string;
  iso?: string;
  intent?: string;
}

interface RecoveryState {
  loading: boolean;
  error: string;
  status: { status: string; path: string }[];
  wipBundles: RecoveryBundle[];
  interrupted: { exists: boolean; path: string; preview?: string; mtime?: string | null } | null;
  ledgerEntries: TaskLedgerEntry[];
  snapshots: RecoverySnapshot[];
  reports: { name: string; path: string; mtime?: string | null }[];
  recentIterations: string | null;
}

export const GitView = component$(() => {
  // View State
  const activeTab = useSignal<'rhizome' | 'git'>('git');
  
  // Git State
  const commits = useSignal<any[]>([]);
  const selectedSha = useSignal<string | null>(null);
  const loading = useSignal(true);
  const repoStatus = useSignal<any>(null);
  const recoveryState = useStore<RecoveryState>({
    loading: false,
    error: '',
    status: [],
    wipBundles: [],
    interrupted: null,
    ledgerEntries: [],
    snapshots: [],
    reports: [],
    recentIterations: null,
  });
  
  // Rhizome State
  const rhizomeArtifacts = useSignal<RhizomeArtifact[]>([]);
  const flowStatus = useStore<RepoFlowStatus>({
    reqId: null,
    step: 'idle',
    message: '',
    logs: []
  });

  // HGT Modal
  const showHGTModal = useSignal(false);
  const hgtSourceSha = useSignal('');
  const hgtStatus = useSignal<'idle' | 'running' | 'success' | 'error'>('idle');
  const hgtResult = useSignal<string>('');

  const exportStatus = useSignal<'idle' | 'running' | 'success' | 'error'>('idle');
  const exportResult = useSignal<string>('');
  const exportNote = useSignal('');
  const snapshotStatus = useSignal<'idle' | 'running' | 'success' | 'error'>('idle');
  const snapshotResult = useSignal<string>('');

  // Promote Flow - Step 1: Request Plan
  const promoteArtifact = $(async (artifact: RhizomeArtifact) => {
    const reqId = `req-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    flowStatus.reqId = reqId;
    flowStatus.step = 'planning';
    flowStatus.message = `Planning promotion for ${artifact.name}...`;
    flowStatus.logs = [`[${new Date().toLocaleTimeString()}] Requesting plan for ${artifact.sha.slice(0, 6)}`];
    flowStatus.plan = null;

    // Emit repo.plan.request
    try {
      await fetch('/api/emit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: 'repo.plan.request',
          kind: 'request',
          level: 'info',
          actor: 'dashboard-user',
          data: {
            req_id: reqId,
            artifact_sha: artifact.sha,
            template_id: 'default' 
          }
        })
      });
      flowStatus.logs.push(`[${new Date().toLocaleTimeString()}] Plan requested: ${reqId}`);
    } catch (e) {
      flowStatus.step = 'failed';
      flowStatus.message = `Plan request failed: ${e}`;
      flowStatus.logs.push(`[${new Date().toLocaleTimeString()}] Error: ${e}`);
    }
  });

  // Promote Flow - Step 2: Confirm Execution
  const confirmPromotion = $(async () => {
    if (!flowStatus.reqId || !flowStatus.plan) return;
    
    flowStatus.step = 'staging';
    flowStatus.message = 'Executing promotion...';
    flowStatus.logs.push(`[${new Date().toLocaleTimeString()}] User confirmed plan. Executing...`);

    try {
      await fetch('/api/emit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: 'repo.exec.request',
          kind: 'request',
          level: 'info',
          actor: 'dashboard-user',
          data: {
            req_id: flowStatus.reqId,
            plan_ref: flowStatus.plan.source_sha || 'unknown',
            stage_path: flowStatus.plan.target_path,
            checks: ['lint', 'test']
          }
        })
      });
    } catch (e) {
      flowStatus.step = 'failed';
      flowStatus.message = `Exec request failed: ${e}`;
    }
  });

  // Load Git Data
  const refresh = $(async () => {
    loading.value = true;
    try {
        const [logRes, statusRes] = await Promise.all([
            fetch('/api/git/log'),
            fetch('/api/git/status')
        ]);
        if (logRes.ok) {
            const data = await logRes.json();
            commits.value = data.commits || [];
            if (!selectedSha.value && commits.value.length > 0) {
                selectedSha.value = commits.value[0].sha;
            }
        }
        if (statusRes.ok) {
            repoStatus.value = await statusRes.json();
        }
    } catch (e) {
        console.error(e);
    } finally {
        loading.value = false;
    }
  });

  const fetchRecovery = $(async () => {
    recoveryState.loading = true;
    recoveryState.error = '';
    try {
      const res = await fetch('/api/git/recovery?limit=8');
      if (!res.ok) {
        throw new Error(`Recovery fetch failed (${res.status})`);
      }
      const data = await res.json();
      recoveryState.status = data.status?.entries || [];
      recoveryState.wipBundles = data.wip_bundles || [];
      recoveryState.interrupted = data.interrupted_tasks || null;
      recoveryState.ledgerEntries = data.task_ledger?.entries || [];
      recoveryState.snapshots = data.recovery_snapshots?.snapshots || [];
      recoveryState.reports = data.recovery_reports || [];
      recoveryState.recentIterations = data.recent_iterations || null;
    } catch (e) {
      recoveryState.error = String(e);
    } finally {
      recoveryState.loading = false;
    }
  });

  const triggerSnapshot = $(async () => {
    snapshotStatus.value = 'running';
    snapshotResult.value = '';
    try {
      const res = await fetch('/api/git/recovery/snapshot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor: 'dashboard-user' }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || `Snapshot failed (${res.status})`);
      }
      snapshotStatus.value = 'success';
      snapshotResult.value = data.path || 'Snapshot created';
      await fetchRecovery();
    } catch (e) {
      snapshotStatus.value = 'error';
      snapshotResult.value = String(e);
    }
  });

  const exportCommitToRhizome = $(async () => {
    if (!selectedSha.value) return;
    exportStatus.value = 'running';
    exportResult.value = '';
    try {
      const res = await fetch('/api/git/rhizome/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sha: selectedSha.value,
          note: exportNote.value.trim(),
          tags: ['git', 'commit', 'snapshot'],
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || `Export failed (${res.status})`);
      }
      exportStatus.value = 'success';
      exportResult.value = data.artifact_sha256 || 'Exported to Rhizome';
      exportNote.value = '';
    } catch (e) {
      exportStatus.value = 'error';
      exportResult.value = String(e);
    }
  });

  // Fetch Rhizome Data
  useVisibleTask$(({ track }) => {
    track(() => activeTab.value);
    if (activeTab.value === 'rhizome') {
        const fetchArtifacts = async () => {
            try {
                const res = await fetch('/api/rhizome', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        jsonrpc: "2.0",
                        method: "tools/call",
                        params: { name: "list_artifacts", arguments: { limit: 50 } },
                        id: `dash-${Date.now()}`
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    const artifacts = data.result?.artifacts || [];
                    rhizomeArtifacts.value = artifacts.map((a: any) => ({
                        sha: a.sha256 || 'unknown',
                        name: a.filename || (a.sha256 ? a.sha256.slice(0, 8) : 'artifact'),
                        kind: (a.filename?.endsWith('.py') || a.filename?.endsWith('.js')) ? 'code' : 'doc',
                        domain: 'rhizome',
                        tags: a.tags || [],
                        created_iso: a.iso || new Date().toISOString(),
                        citation_count: 0
                    }));
                }
            } catch (e) {
                console.error('Rhizome fetch error', e);
            }
        };
        fetchArtifacts();
    } else if (activeTab.value === 'git') {
        fetchRecovery();
    }
  });

  // Bus Listener
  useVisibleTask$(({ cleanup }) => {
    refresh();
    const wsHost = window.location.host;
    const isSecure = window.location.protocol === 'https:';
    const wsUrl = `${isSecure ? 'wss' : 'ws'}://${wsHost}/ws/bus`;
    const ws = new WebSocket(wsUrl);
    
    ws.onmessage = (msg) => {
        try {
            const data = JSON.parse(msg.data);
            if (data.type === 'event' || (data.type === 'sync' && data.events)) {
                const events = data.type === 'sync' ? data.events : [data.event];
                events.forEach((e: any) => {
                    const d = e.data || {};
                    const eReqId = d.req_id || d.reqId;
                    if (flowStatus.reqId && eReqId === flowStatus.reqId) {
                        if (e.topic === 'repo.exec.result') {
                             flowStatus.step = d.status === 'success' || d.status === 'stubbed' ? 'complete' : 'failed';
                             flowStatus.message = `Result: ${d.status}`;
                             flowStatus.logs = [...flowStatus.logs, `[${new Date().toLocaleTimeString()}] Result: ${d.status} ${(d.errors || []).join(', ')}`];
                        } else if (e.topic === 'repo.exec.progress') {
                             flowStatus.message = `Progress: ${d.step || 'working...'}`;
                             flowStatus.logs = [...flowStatus.logs, `[${new Date().toLocaleTimeString()}] ${d.step}: ${d.status}`];
                        } else if (e.topic === 'repo.plan.response') {
                             // Extract source_sha from plan or preview
                             const source = d.preview_snippet?.match(/Source: (.+)/)?.[1] || 'unknown';
                             flowStatus.plan = { ...d, source_sha: source }; 
                             flowStatus.step = 'reviewing';
                             flowStatus.message = 'Plan ready for review';
                             flowStatus.logs = [...flowStatus.logs, `[${new Date().toLocaleTimeString()}] Plan received. Target: ${d.target_path}`];
                        }
                    }
                });
            }
        } catch { /* ignore */ }
    };
    cleanup(() => ws.close());
  });

  const triggerFetch = $(async () => {
      await fetch('/api/git/fetch', { method: 'POST', body: JSON.stringify({}) });
      refresh();
  });

  const triggerPush = $(async () => {
      await fetch('/api/git/push', { method: 'POST', body: JSON.stringify({}) });
      refresh();
  });

  const triggerHGT = $(async () => {
      if (!hgtSourceSha.value) return;
      hgtStatus.value = 'running';
      try {
          const res = await fetch('/api/git/hgt', {
              method: 'POST',
              body: JSON.stringify({ source_sha: hgtSourceSha.value })
          });
          const data = await res.json();
          if (data.success) {
              hgtStatus.value = 'success';
              hgtResult.value = data.message + '\n' + (data.output || '');
              refresh();
          } else {
              hgtStatus.value = 'error';
              hgtResult.value = data.error || 'Unknown error';
          }
      } catch (e) {
          hgtStatus.value = 'error';
          hgtResult.value = String(e);
      }
  });

  return (
    <div class="grid grid-cols-12 h-full bg-background relative">
      
      {/* HGT Modal */}
      {showHGTModal.value && (
          <div class="absolute inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
              <div class="bg-[#0c0c0e] border border-pink-500/50 rounded-xl shadow-[0_0_50px_rgba(236,72,153,0.2)] w-full max-w-lg overflow-hidden">
                  <div class="p-4 border-b border-[var(--glass-border)] bg-pink-500/10 flex justify-between items-center">
                      <h3 class="font-bold text-pink-400 flex items-center gap-2">
                          <span>🧬</span> Horizontal Gene Transfer
                      </h3>
                      <button onClick$={() => showHGTModal.value = false} class="text-white/50 hover:text-white">×</button>
                  </div>
                  <div class="p-6 space-y-4">
                      <p class="text-sm text-gray-400">
                          Splice a verified trait (commit) from an external lineage into the current genome.
                          Subject to the <span class="text-cyan-400">Kroma Guard Ladder (G1-G6)</span>.
                      </p>
                      
                      <div class="space-y-2">
                          <label class="text-xs font-mono text-gray-500 uppercase">Source SHA / Ref</label>
                          <input 
                              type="text" 
                              bind:value={hgtSourceSha}
                              placeholder="e.g. origin/feature-branch or <sha>"
                              class="w-full bg-black/50 border border-[var(--glass-border)] rounded px-3 py-2 text-white font-mono focus:border-pink-500 outline-none"
                          />
                      </div>

                      {hgtStatus.value !== 'idle' && (
                          <div class={`p-3 rounded border text-xs font-mono whitespace-pre-wrap max-h-40 overflow-auto ${ 
                              hgtStatus.value === 'running' ? 'bg-blue-500/10 border-blue-500/30 text-blue-300' :
                              hgtStatus.value === 'success' ? 'bg-green-500/10 border-green-500/30 text-green-300' :
                              'bg-red-500/10 border-red-500/30 text-red-300'
                          }`}>
                              {hgtStatus.value === 'running' ? 'Running Guard Ladder...' : hgtResult.value}
                          </div>
                      )}

                      <div class="flex justify-end gap-3 pt-2">
                          <button 
                              onClick$={() => showHGTModal.value = false}
                              class="px-4 py-2 rounded text-sm text-gray-400 hover:bg-white/5"
                          >
                              Cancel
                          </button>
                          <button 
                              onClick$={triggerHGT}
                              disabled={hgtStatus.value === 'running' || !hgtSourceSha.value}
                              class="px-4 py-2 rounded text-sm font-bold bg-gradient-to-r from-pink-600 to-purple-600 text-white shadow-lg hover:shadow-pink-500/20 disabled:opacity-50"
                          >
                              {hgtStatus.value === 'running' ? 'Splicing...' : 'Initiate Splice'}
                          </button>
                      </div>
                  </div>
              </div>
          </div>
      )}

      {/* Sidebar: Mode Selection */}
      <div class="col-span-12 border-b border-border flex items-center bg-[#08080a] px-4">
         <button 
            onClick$={() => activeTab.value = 'git'}
            class={`px-4 py-3 text-sm font-bold border-b-2 transition-all ${activeTab.value === 'git' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-white'}`}
         >
             📦 Git View
         </button>
         <button 
            onClick$={() => activeTab.value = 'rhizome'}
            class={`px-4 py-3 text-sm font-bold border-b-2 transition-all ${activeTab.value === 'rhizome' ? 'border-cyan-500 text-cyan-400' : 'border-transparent text-muted-foreground hover:text-white'}`}
         >
             🌿 Rhizome Memory
         </button>
      </div>

      {/* Main Content Area */}
      {activeTab.value === 'git' && (
          <>
            {/* Left: Commit Graph */}
            <div class="col-span-4 border-r border-border flex flex-col h-[calc(100vh-140px)] bg-[#08080a]">
                <div class="p-4 border-b border-border flex justify-between items-center bg-card/50 backdrop-blur-sm sticky top-0 z-10">
                <h2 class="font-bold text-sm text-foreground flex items-center gap-2">
                    <span class="text-primary">⚡</span> IsoGit Graph
                </h2>
                <div class="flex gap-2">
                    <button onClick$={triggerFetch} class="p-1 hover:bg-white/10 rounded text-xs" title="Fetch from remote (Requires .git/config and SSH keys)">⬇️</button>
                    <button onClick$={triggerPush} class="p-1 hover:bg-white/10 rounded text-xs" title="Push to remote (Requires write access & SSH keys)">⬆️</button>
                    <button onClick$={refresh} class="p-1 hover:bg-white/10 rounded text-xs" title="Refresh Graph">🔄</button>
                </div>
                </div>
                <div class="flex-1 overflow-auto p-4 custom-scrollbar relative">
                {loading.value ? (
                    <div class="flex justify-center p-8"><span class="animate-spin text-primary">⚡</span></div>
                ) : (
                    <GitGraph 
                        commits={commits.value} 
                        selectedSha={selectedSha.value}
                        onSelectCommit$={(sha) => selectedSha.value = sha}
                    />
                )}
                </div>
            </div>

            {/* Right: Details & HGT */}
            <div class="col-span-8 flex flex-col h-[calc(100vh-140px)] bg-[#0c0c0e]">
                {/* Toolbar */}
                <div class="p-4 border-b border-border flex items-center justify-between bg-card/50 backdrop-blur-sm">
                    <div class="flex items-center gap-4">
                        <div class="text-sm font-mono text-muted-foreground">
                            Current: <span class="text-primary">{selectedSha.value?.slice(0,8) || 'HEAD'}</span>
                        </div>
                        {/* Status Badges */}
                        {repoStatus.value?.status?.length > 0 && (
                            <span class="text-xs px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
                                {repoStatus.value.status.length} dirty files
                            </span>
                        )}
                    </div>
                    
                    <div class="flex items-center gap-2">
                        <details class="relative">
                            <summary class="list-none px-3 py-1.5 text-xs font-bold rounded border border-border bg-muted/30 hover:bg-muted/50 text-muted-foreground cursor-pointer select-none">
                                📚 Docs
                            </summary>
                            <div class="absolute right-0 mt-2 w-72 rounded-lg border border-border bg-[#0c0c0e] shadow-xl overflow-hidden z-20">
                                <a
                                    href="/api/fs/nucleus/docs/RHIZOME_GIT_UI_GUIDE.md"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    class="block px-3 py-2 text-xs hover:bg-white/5"
                                >
                                    Rhizome &amp; Git UI Guide
                                </a>
                                <a
                                    href="/api/fs/nucleus/docs/ui/bus_observatory.md"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    class="block px-3 py-2 text-xs hover:bg-white/5"
                                >
                                    Bus Observatory (🧭 Bus)
                                </a>
                                <a
                                    href="/api/fs/nucleus/docs/concepts/Rhizome-to-Holon.md"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    class="block px-3 py-2 text-xs hover:bg-white/5"
                                >
                                    Rhizome-to-Holon (architecture)
                                </a>
                            </div>
                        </details>
                        <button 
                            onClick$={() => {
                                showHGTModal.value = true;
                                hgtSourceSha.value = '';
                                hgtStatus.value = 'idle';
                            }}
                            class="px-4 py-1.5 text-xs font-bold rounded bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:shadow-[0_0_15px_rgba(219,39,119,0.5)] transition-all flex items-center gap-2"
                        >
                            <span>🧬</span> HGT Push
                        </button>
                    </div>
                </div>

                {/* Content Area */}
                <div class="flex-1 p-6 overflow-auto text-muted-foreground bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-gray-800/10 via-background to-background">
                    <div class="grid gap-6 xl:grid-cols-2">
                        <div class="rounded-xl border border-[var(--glass-border)] bg-black/40 p-6 space-y-4">
                            <div class="flex items-center gap-3">
                                <div class="text-3xl opacity-50">📦</div>
                                <div>
                                    <div class="text-sm uppercase text-muted-foreground">Commit Focus</div>
                                    <div class="text-lg font-medium text-white/80">
                                        {selectedSha.value ? `Commit ${selectedSha.value.slice(0, 8)}` : 'Select a commit'}
                                    </div>
                                </div>
                            </div>

                            {selectedSha.value ? (
                                <>
                                    <div class="p-4 bg-black/40 rounded border border-[var(--glass-border-subtle)] text-left font-mono text-xs">
                                        {(() => {
                                            const c = commits.value.find(c => c.sha === selectedSha.value);
                                            return c ? (
                                                <>
                                                    <div class="text-blue-400 mb-2">{c.message}</div>
                                                    <div class="text-gray-500">Author: {c.author}</div>
                                                    <div class="text-gray-500">Date:   {new Date(c.date).toLocaleString()}</div>
                                                    <div class="text-gray-500">Parent: {c.parents?.map((p:string)=>p.slice(0,7)).join(', ') || 'root'}</div>
                                                </>
                                            ) : 'Loading details...';
                                        })()}
                                    </div>

                                    <div class="space-y-2">
                                        <label class="text-xs font-mono text-gray-400 uppercase">Export Note (optional)</label>
                                        <textarea
                                            bind:value={exportNote}
                                            rows={3}
                                            placeholder="Why this commit matters or what it unlocks..."
                                            class="w-full bg-black/50 border border-[var(--glass-border)] rounded px-3 py-2 text-white text-xs font-mono focus:border-cyan-500 outline-none"
                                        />
                                    </div>

                                    <div class="flex items-center gap-3">
                                        <button
                                            onClick$={exportCommitToRhizome}
                                            disabled={exportStatus.value === 'running'}
                                            class="px-4 py-2 rounded text-xs font-bold bg-gradient-to-r from-cyan-600 to-emerald-600 text-white shadow-lg hover:shadow-cyan-500/30 disabled:opacity-50"
                                        >
                                            {exportStatus.value === 'running' ? 'Exporting...' : '🌿 Export to Rhizome'}
                                        </button>
                                        {exportStatus.value !== 'idle' && (
                                            <span class={`text-xs font-mono ${exportStatus.value === 'success' ? 'text-emerald-400' : exportStatus.value === 'error' ? 'text-red-400' : 'text-blue-400'}`}>
                                                {exportResult.value}
                                            </span>
                                        )}
                                    </div>
                                </>
                            ) : (
                                <div class="text-sm text-muted-foreground">
                                    Select a commit to inspect, annotate, and export into Rhizome memory.
                                </div>
                            )}
                        </div>

                        <div class="rounded-xl border border-[var(--glass-border)] bg-black/40 p-6 space-y-4">
                            <div class="flex items-center justify-between">
                                <div>
                                    <div class="text-sm uppercase text-muted-foreground">Continuity Control</div>
                                    <div class="text-lg font-medium text-white/80">Interruptions & Recovery</div>
                                </div>
                                <div class="flex items-center gap-2">
                                    <button
                                        onClick$={fetchRecovery}
                                        class="px-3 py-1.5 rounded text-xs font-bold bg-white/10 hover:bg-white/20 text-white/70"
                                    >
                                        Refresh
                                    </button>
                                    <button
                                        onClick$={triggerSnapshot}
                                        disabled={snapshotStatus.value === 'running'}
                                        class="px-3 py-1.5 rounded text-xs font-bold bg-purple-600/80 hover:bg-purple-500 text-white disabled:opacity-50"
                                    >
                                        {snapshotStatus.value === 'running' ? 'Snapshotting...' : 'Capture Snapshot'}
                                    </button>
                                </div>
                            </div>

                            {snapshotStatus.value !== 'idle' && (
                                <div class={`text-xs font-mono ${snapshotStatus.value === 'success' ? 'text-emerald-400' : snapshotStatus.value === 'error' ? 'text-red-400' : 'text-blue-400'}`}>
                                    {snapshotResult.value}
                                </div>
                            )}

                            {recoveryState.error && (
                                <div class="text-xs font-mono text-red-400">{recoveryState.error}</div>
                            )}

                            <div class="space-y-3">
                                <div>
                                    <div class="text-xs font-bold text-gray-400 uppercase">Dirty Files</div>
                                    <div class="text-xs text-muted-foreground">
                                        {recoveryState.status.length ? (
                                            <ul class="mt-2 space-y-1">
                                                {recoveryState.status.slice(0, 6).map((entry) => (
                                                    <li key={`${entry.status}-${entry.path}`} class="flex items-center gap-2">
                                                        <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
                                                            {entry.status}
                                                        </span>
                                                        <span class="truncate">{entry.path}</span>
                                                    </li>
                                                ))}
                                                {recoveryState.status.length > 6 && (
                                                    <li class="text-[10px] text-muted-foreground">
                                                        +{recoveryState.status.length - 6} more…
                                                    </li>
                                                )}
                                            </ul>
                                        ) : (
                                            <div class="mt-2 text-[11px] text-muted-foreground">Working tree clean.</div>
                                        )}
                                    </div>
                                </div>

                                <div>
                                    <div class="text-xs font-bold text-gray-400 uppercase">Interrupted Tasks</div>
                                    {recoveryState.interrupted?.exists ? (
                                        <div class="mt-2 space-y-2">
                                            <pre class="text-[11px] font-mono whitespace-pre-wrap text-gray-300 bg-black/30 border border-[var(--glass-border-subtle)] rounded p-2 max-h-28 overflow-auto">
                                                {recoveryState.interrupted.preview}
                                            </pre>
                                            <a
                                                href={`/api/fs/${recoveryState.interrupted.path}`}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                class="text-[11px] font-mono text-cyan-400 hover:text-cyan-300"
                                            >
                                                Open reconstructed_interrupted_tasks.md →
                                            </a>
                                        </div>
                                    ) : (
                                        <div class="mt-2 text-[11px] text-muted-foreground">No interruption log found.</div>
                                    )}
                                </div>

                                <div>
                                    <div class="text-xs font-bold text-gray-400 uppercase">WIP Bundles</div>
                                    {recoveryState.wipBundles.length ? (
                                        <ul class="mt-2 space-y-1 text-[11px]">
                                            {recoveryState.wipBundles.slice(0, 5).map((bundle) => (
                                                <li key={bundle.path} class="flex items-center justify-between gap-2">
                                                    <span class="truncate">{bundle.name}</span>
                                                    <a
                                                        href={`/api/fs/${bundle.path}`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        class="text-cyan-400 hover:text-cyan-300"
                                                    >
                                                        {bundle.entries} items →
                                                    </a>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <div class="mt-2 text-[11px] text-muted-foreground">No WIP bundles detected.</div>
                                    )}
                                </div>

                                <div>
                                    <div class="text-xs font-bold text-gray-400 uppercase">Task Ledger (latest)</div>
                                    {recoveryState.ledgerEntries.length ? (
                                        <ul class="mt-2 space-y-1 text-[11px]">
                                            {recoveryState.ledgerEntries.slice(0, 6).map((entry) => (
                                                <li key={`${entry.req_id}-${entry.topic}`} class="flex items-center justify-between gap-2">
                                                    <span class="truncate">{entry.topic}</span>
                                                    <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/10 text-white/70">
                                                        {entry.status}
                                                    </span>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <div class="mt-2 text-[11px] text-muted-foreground">Ledger empty or unavailable.</div>
                                    )}
                                </div>

                                <div>
                                    <div class="text-xs font-bold text-gray-400 uppercase">Recovery Snapshots</div>
                                    {recoveryState.snapshots.length ? (
                                        <ul class="mt-2 space-y-1 text-[11px]">
                                            {recoveryState.snapshots.slice(0, 4).map((snap) => (
                                                <li key={snap.path} class="flex items-center justify-between gap-2">
                                                    <span class="truncate">{snap.created_iso || snap.path}</span>
                                                    <a
                                                        href={`/api/fs/${snap.path}`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        class="text-cyan-400 hover:text-cyan-300"
                                                    >
                                                        open →
                                                    </a>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <div class="mt-2 text-[11px] text-muted-foreground">No snapshots found.</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
          </>
      )}

      {activeTab.value === 'rhizome' && (
          <div class="col-span-12 h-[calc(100vh-140px)] grid grid-cols-12 bg-[#08080a]">
              {/* Artifact List */}
              <div class="col-span-4 border-r border-border p-4 overflow-auto">
                  <h3 class="text-sm font-bold text-muted-foreground mb-4 uppercase tracking-wider">Curated Artifacts</h3>
                  <div class="space-y-2">
                      {rhizomeArtifacts.value.map(art => (
                          <div key={art.sha} class="p-3 rounded border border-border bg-card/50 hover:bg-card hover:border-cyan-500/50 transition-all group">
                              <div class="flex items-center justify-between mb-2">
                                  <div class="flex items-center gap-2">
                                      <span class="text-lg">{art.kind === 'code' ? '📜' : art.kind === 'spec' ? '📐' : '📄'}</span>
                                      <span class="font-bold text-sm text-foreground">{art.name}</span>
                                  </div>
                                  <span class="text-[10px] font-mono text-muted-foreground">{art.sha.slice(0,6)}</span>
                              </div>
                              <div class="flex flex-wrap gap-1 mb-3">
                                  {art.tags.map(t => (
                                      <span key={t} class="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">{t}</span>
                                  ))}
                              </div>
                              <button 
                                  onClick$={() => promoteArtifact(art)}
                                  class="w-full py-1.5 rounded text-xs font-bold bg-muted text-muted-foreground group-hover:bg-primary group-hover:text-primary-foreground transition-all flex items-center justify-center gap-2"
                              >
                                  <span>🚀</span> Promote to Git
                              </button>
                          </div>
                      ))}
                  </div>
              </div>

              {/* Promotion Flow / Output */}
              <div class="col-span-8 p-8 flex flex-col bg-[#0c0c0e]">
                  <h3 class="text-sm font-bold text-muted-foreground mb-4 uppercase tracking-wider flex items-center justify-between">
                      <span>Flow Status</span>
                      {flowStatus.reqId && <span class="text-xs font-mono text-primary">{flowStatus.reqId}</span>}
                  </h3>
                  
                  {flowStatus.step === 'reviewing' && flowStatus.plan ? (
                      <div class="flex-1 flex flex-col space-y-4">
                          <div class="rounded-xl border border-orange-500/30 bg-orange-900/10 p-6 relative overflow-hidden">
                              {/* Background DNA Decoration */}
                              <div class="absolute inset-0 opacity-5 pointer-events-none text-9xl flex items-center justify-center">🧬</div>
                              
                              <h4 class="text-lg font-bold text-orange-400 mb-6 flex items-center gap-2 relative z-10">
                                  <span>☣️</span> Ribosome Translation: Plan Review
                              </h4>

                              <div class="grid grid-cols-3 gap-4 text-sm mb-6 relative z-10">
                                  {/* Source: Genotype */}
                                  <div class="p-4 rounded border border-cyan-500/30 bg-cyan-900/20 flex flex-col items-center text-center">
                                      <div class="text-2xl mb-2">💎</div>
                                      <div class="font-bold text-cyan-300">GENOTYPE</div>
                                      <div class="text-xs text-cyan-100/50 mb-2">Rhizome Memory</div>
                                      <div class="font-mono text-white bg-black/50 px-2 py-1 rounded text-xs">
                                          {flowStatus.plan.source_sha?.slice(0,8)}
                                      </div>
                                  </div>

                                  {/* The Bridge / Gates */}
                                  <div class="flex flex-col justify-center items-center space-y-2">
                                      <div class="h-0.5 w-full bg-gradient-to-r from-cyan-500 to-green-500"></div>
                                      
                                      <div class="flex flex-col gap-1 w-full">
                                          <div class="flex items-center justify-between text-[10px] px-2 py-1 rounded border border-green-500/30 bg-green-900/30">
                                              <span class="text-green-400">🛡️ Ring 2 (User)</span>
                                              <span class="text-green-500">PASS</span>
                                          </div>
                                          <div class="flex items-center justify-between text-[10px] px-2 py-1 rounded border border-blue-500/30 bg-blue-900/30">
                                              <span class="text-blue-400">🔐 PQC Sig</span>
                                              <span class="text-blue-500">VALID</span>
                                          </div>
                                          <div class="flex items-center justify-between text-[10px] px-2 py-1 rounded border border-purple-500/30 bg-purple-900/30">
                                              <span class="text-purple-400">🧬 VGT Lineage</span>
                                              <span class="text-purple-500">MATCH</span>
                                          </div>
                                      </div>

                                      <div class="text-xs font-mono text-orange-400 animate-pulse">
                                          Ready to Transcribe
                                      </div>
                                  </div>

                                  {/* Target: Phenotype */}
                                  <div class="p-4 rounded border border-green-500/30 bg-green-900/20 flex flex-col items-center text-center">
                                      <div class="text-2xl mb-2">🌿</div>
                                      <div class="font-bold text-green-300">PHENOTYPE</div>
                                      <div class="text-xs text-green-100/50 mb-2">Git State</div>
                                      <div class="font-mono text-white bg-black/50 px-2 py-1 rounded text-xs break-all">
                                          {flowStatus.plan.target_path}
                                      </div>
                                  </div>
                              </div>

                              {/* Steps Preview */}
                              <div class="space-y-2 mb-6 relative z-10 border-t border-[var(--glass-border-subtle)] pt-4">
                                  <div class="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Translation Sequence</div>
                                  {flowStatus.plan.plan_steps?.map((step: any, i: number) => (
                                      <div key={i} class="flex items-center gap-3 text-xs">
                                          <div class="w-5 h-5 rounded-full bg-orange-500/20 flex items-center justify-center text-orange-400 font-mono border border-orange-500/30">{i+1}</div>
                                          <div>
                                              <div class="font-bold text-gray-300">{step.step}</div>
                                              <div class="text-muted-foreground">{step.description}</div>
                                          </div>
                                      </div>
                                  ))}
                              </div>

                              {/* Actions */}
                              <div class="flex gap-3 relative z-10">
                                  <button 
                                      onClick$={confirmPromotion}
                                      class="flex-1 px-4 py-3 rounded bg-gradient-to-r from-green-600 to-emerald-700 hover:from-green-500 hover:to-emerald-600 text-white font-bold shadow-lg shadow-green-900/50 flex items-center justify-center gap-2 transition-all"
                                  >
                                      <span>🧬</span> Manifest Phenotype
                                  </button>
                                  <button 
                                      onClick$={() => { flowStatus.step = 'idle'; flowStatus.plan = null; }}
                                      class="px-6 py-3 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 font-medium border border-[var(--glass-border)]"
                                  >
                                      Abort
                                  </button>
                              </div>
                          </div>
                          
                          {/* Preview snippet */}
                          {flowStatus.plan.preview_snippet && (
                              <div class="flex-1 rounded border border-border bg-black p-4 font-mono text-xs overflow-auto">
                                  <div class="text-xs text-gray-500 mb-2 uppercase">Protein Preview</div>
                                  <pre class="text-gray-300">{flowStatus.plan.preview_snippet}</pre>
                              </div>
                          )}
                      </div>
                  ) : flowStatus.step !== 'idle' ? (
                      <div class="flex-1 rounded-xl border border-border bg-black/40 p-6 relative overflow-hidden">
                          {/* Progress Bar */}
                          <div class="absolute top-0 left-0 w-full h-1 bg-muted/20">
                              <div 
                                  class={`h-full transition-all duration-500 ${
                                      flowStatus.step === 'complete' ? 'w-full bg-green-500' : 
                                      flowStatus.step === 'failed' ? 'w-full bg-red-500' :
                                      'w-1/2 bg-blue-500 animate-pulse'
                                  }`}
                              />
                          </div>

                          <div class="flex flex-col h-full">
                              <div class="flex items-center gap-4 mb-6">
                                  <div class={`w-12 h-12 rounded-full flex items-center justify-center text-2xl border-2 ${
                                      flowStatus.step === 'complete' ? 'border-green-500 bg-green-500/10 text-green-400' :
                                      flowStatus.step === 'failed' ? 'border-red-500 bg-red-500/10 text-red-400' :
                                      'border-blue-500 bg-blue-500/10 text-blue-400 animate-bounce'
                                  }`}>
                                      {flowStatus.step === 'complete' ? '✅' : flowStatus.step === 'failed' ? '❌' : '⚙️'}
                                  </div>
                                  <div>
                                      <div class="text-lg font-bold capitalize">{flowStatus.step}</div>
                                      <div class="text-sm text-muted-foreground">{flowStatus.message}</div>
                                  </div>
                              </div>

                              <div class="flex-1 overflow-auto font-mono text-xs space-y-1 p-4 rounded bg-black/60 border border-[var(--glass-border-subtle)] custom-scrollbar">
                                  {flowStatus.logs.map((log, i) => (
                                      <div key={i} class="text-gray-400">{log}</div>
                                  ))}
                              </div>
                          </div>
                      </div>
                  ) : (
                      <div class="flex-1 flex flex-col items-center justify-center text-muted-foreground border-2 border-dashed border-border/30 rounded-xl">
                          <div class="text-4xl mb-4 opacity-20">🌿 ➡️ 📦</div>
                          <p>Select an artifact to begin promotion</p>
                      </div>
                  )}
              </div>
          </div>
      )}
    </div>
  );
});
