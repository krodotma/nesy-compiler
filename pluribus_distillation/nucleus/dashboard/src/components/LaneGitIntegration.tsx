/**
 * LaneGitIntegration - Git Integration for Lanes
 *
 * Phase 8, Iteration 69 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Git branch linking
 * - PR status display
 * - Commit attribution
 * - Branch health indicators
 * - Merge conflict detection
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface GitBranch {
  name: string;
  sha: string;
  ahead: number;
  behind: number;
  lastCommit: {
    sha: string;
    message: string;
    author: string;
    date: string;
  };
  isProtected: boolean;
}

export interface PullRequest {
  id: number;
  number: number;
  title: string;
  state: 'open' | 'closed' | 'merged';
  author: string;
  sourceBranch: string;
  targetBranch: string;
  createdAt: string;
  updatedAt: string;
  reviewStatus: 'pending' | 'approved' | 'changes_requested' | 'review_required';
  checksStatus: 'pending' | 'success' | 'failure' | 'neutral';
  mergeable: boolean;
  conflicts: boolean;
  additions: number;
  deletions: number;
  comments: number;
}

export interface GitCommit {
  sha: string;
  shortSha: string;
  message: string;
  author: string;
  authorEmail: string;
  date: string;
  laneId?: string;
}

export interface LaneGitLink {
  laneId: string;
  laneName: string;
  branchName?: string;
  pullRequestId?: number;
  commits: GitCommit[];
}

export interface LaneGitIntegrationProps {
  /** Lane ID to show git info for */
  laneId?: string;
  /** Lane name */
  laneName?: string;
  /** Git links data */
  gitLinks?: LaneGitLink[];
  /** Available branches */
  branches?: GitBranch[];
  /** Pull requests */
  pullRequests?: PullRequest[];
  /** Callback when branch is linked */
  onLinkBranch$?: QRL<(laneId: string, branchName: string) => void>;
  /** Callback when PR is linked */
  onLinkPR$?: QRL<(laneId: string, prId: number) => void>;
  /** Callback when link is removed */
  onUnlink$?: QRL<(laneId: string) => void>;
  /** Refresh callback */
  onRefresh$?: QRL<() => void>;
  /** Compact mode */
  compact?: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffHours < 1) return 'just now';
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function getPRStatusColor(pr: PullRequest): string {
  if (pr.state === 'merged') return 'text-purple-400';
  if (pr.state === 'closed') return 'text-red-400';
  if (pr.conflicts) return 'text-orange-400';
  if (pr.checksStatus === 'failure') return 'text-red-400';
  if (pr.reviewStatus === 'approved' && pr.checksStatus === 'success') return 'text-emerald-400';
  if (pr.reviewStatus === 'changes_requested') return 'text-yellow-400';
  return 'text-blue-400';
}

function getPRStatusIcon(pr: PullRequest): string {
  if (pr.state === 'merged') return '⊕';
  if (pr.state === 'closed') return '⊗';
  if (pr.conflicts) return '⚠';
  if (pr.checksStatus === 'failure') return '✗';
  if (pr.reviewStatus === 'approved' && pr.checksStatus === 'success') return '✓';
  return '○';
}

function getBranchHealthColor(branch: GitBranch): string {
  if (branch.behind > 10) return 'text-red-400';
  if (branch.behind > 5) return 'text-yellow-400';
  if (branch.behind > 0) return 'text-blue-400';
  return 'text-emerald-400';
}

// ============================================================================
// Component
// ============================================================================

export const LaneGitIntegration = component$<LaneGitIntegrationProps>(({
  laneId,
  laneName,
  gitLinks = [],
  branches = [],
  pullRequests = [],
  onLinkBranch$,
  onLinkPR$,
  onUnlink$,
  onRefresh$,
  compact = false,
}) => {
  // State
  const showLinkModal = useSignal(false);
  const linkType = useSignal<'branch' | 'pr'>('branch');
  const searchQuery = useSignal('');
  const selectedBranch = useSignal<string | null>(null);
  const selectedPR = useSignal<number | null>(null);
  const isRefreshing = useSignal(false);

  // Get current lane's git link
  const currentLink = gitLinks.find(l => l.laneId === laneId);
  const linkedBranch = branches.find(b => b.name === currentLink?.branchName);
  const linkedPR = pullRequests.find(p => p.id === currentLink?.pullRequestId);

  // Filter branches/PRs by search
  const filteredBranches = branches.filter(b =>
    b.name.toLowerCase().includes(searchQuery.value.toLowerCase())
  );
  const filteredPRs = pullRequests.filter(p =>
    p.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
    p.sourceBranch.toLowerCase().includes(searchQuery.value.toLowerCase())
  );

  // Auto-refresh
  useVisibleTask$(({ cleanup }) => {
    const interval = setInterval(async () => {
      if (onRefresh$) {
        isRefreshing.value = true;
        await onRefresh$();
        isRefreshing.value = false;
      }
    }, 60000); // Refresh every minute

    cleanup(() => clearInterval(interval));
  });

  // Link handlers
  const handleLinkBranch = $(async () => {
    if (!laneId || !selectedBranch.value || !onLinkBranch$) return;
    await onLinkBranch$(laneId, selectedBranch.value);
    showLinkModal.value = false;
    selectedBranch.value = null;
  });

  const handleLinkPR = $(async () => {
    if (!laneId || !selectedPR.value || !onLinkPR$) return;
    await onLinkPR$(laneId, selectedPR.value);
    showLinkModal.value = false;
    selectedPR.value = null;
  });

  const handleUnlink = $(async () => {
    if (!laneId || !onUnlink$) return;
    await onUnlink$(laneId);
  });

  // Compact view
  if (compact) {
    return (
      <div class="flex items-center gap-2 text-[10px]">
        {linkedBranch && (
          <span class={`flex items-center gap-1 ${getBranchHealthColor(linkedBranch)}`}>
            <span>⎇</span>
            <span>{linkedBranch.name}</span>
            {linkedBranch.behind > 0 && (
              <span class="text-[8px] opacity-70">↓{linkedBranch.behind}</span>
            )}
          </span>
        )}
        {linkedPR && (
          <span class={`flex items-center gap-1 ${getPRStatusColor(linkedPR)}`}>
            <span>{getPRStatusIcon(linkedPR)}</span>
            <span>#{linkedPR.number}</span>
          </span>
        )}
        {!linkedBranch && !linkedPR && (
          <button
            onClick$={() => { showLinkModal.value = true; }}
            class="text-muted-foreground hover:text-foreground"
          >
            + Link Git
          </button>
        )}
      </div>
    );
  }

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">GIT INTEGRATION</span>
          {isRefreshing.value && (
            <span class="text-[8px] text-muted-foreground animate-pulse">Syncing...</span>
          )}
        </div>
        <div class="flex items-center gap-1">
          {onRefresh$ && (
            <button
              onClick$={async () => {
                isRefreshing.value = true;
                await onRefresh$();
                isRefreshing.value = false;
              }}
              class="text-[9px] px-2 py-1 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
            >
              ↻ Refresh
            </button>
          )}
          <button
            onClick$={() => { showLinkModal.value = true; }}
            class="text-[9px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30"
          >
            + Link
          </button>
        </div>
      </div>

      {/* Current Links */}
      <div class="p-3 space-y-3">
        {/* Linked Branch */}
        {linkedBranch && (
          <div class="p-2 rounded bg-muted/10 border border-border/30">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="text-[10px] text-muted-foreground">⎇ Branch</span>
                <span class={`text-xs font-mono ${getBranchHealthColor(linkedBranch)}`}>
                  {linkedBranch.name}
                </span>
                {linkedBranch.isProtected && (
                  <span class="text-[8px] px-1 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                    protected
                  </span>
                )}
              </div>
              <button
                onClick$={handleUnlink}
                class="text-[8px] text-muted-foreground hover:text-red-400"
              >
                Unlink
              </button>
            </div>

            <div class="grid grid-cols-2 gap-2 text-[9px]">
              <div>
                <span class="text-muted-foreground">Ahead/Behind: </span>
                <span class="text-emerald-400">+{linkedBranch.ahead}</span>
                <span class="text-muted-foreground"> / </span>
                <span class="text-red-400">-{linkedBranch.behind}</span>
              </div>
              <div class="text-right text-muted-foreground">
                {formatDate(linkedBranch.lastCommit.date)}
              </div>
            </div>

            <div class="mt-2 text-[9px] text-muted-foreground truncate">
              <span class="font-mono text-foreground/70">{linkedBranch.lastCommit.sha.slice(0, 7)}</span>
              {' '}{linkedBranch.lastCommit.message}
            </div>
          </div>
        )}

        {/* Linked PR */}
        {linkedPR && (
          <div class="p-2 rounded bg-muted/10 border border-border/30">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class={getPRStatusColor(linkedPR)}>{getPRStatusIcon(linkedPR)}</span>
                <span class="text-xs font-medium text-foreground">
                  #{linkedPR.number}
                </span>
                <span class="text-xs text-foreground/80 truncate max-w-[200px]">
                  {linkedPR.title}
                </span>
              </div>
              <button
                onClick$={handleUnlink}
                class="text-[8px] text-muted-foreground hover:text-red-400"
              >
                Unlink
              </button>
            </div>

            <div class="flex items-center gap-4 text-[9px]">
              <div>
                <span class="text-muted-foreground">Review: </span>
                <span class={
                  linkedPR.reviewStatus === 'approved' ? 'text-emerald-400' :
                  linkedPR.reviewStatus === 'changes_requested' ? 'text-yellow-400' :
                  'text-muted-foreground'
                }>
                  {linkedPR.reviewStatus}
                </span>
              </div>
              <div>
                <span class="text-muted-foreground">Checks: </span>
                <span class={
                  linkedPR.checksStatus === 'success' ? 'text-emerald-400' :
                  linkedPR.checksStatus === 'failure' ? 'text-red-400' :
                  'text-muted-foreground'
                }>
                  {linkedPR.checksStatus}
                </span>
              </div>
              <div>
                <span class="text-emerald-400">+{linkedPR.additions}</span>
                <span class="text-muted-foreground"> / </span>
                <span class="text-red-400">-{linkedPR.deletions}</span>
              </div>
            </div>

            {linkedPR.conflicts && (
              <div class="mt-2 text-[9px] text-orange-400">
                ⚠ Merge conflicts detected
              </div>
            )}

            <div class="mt-2 text-[9px] text-muted-foreground">
              {linkedPR.sourceBranch} → {linkedPR.targetBranch}
            </div>
          </div>
        )}

        {/* Recent Commits */}
        {currentLink && currentLink.commits.length > 0 && (
          <div>
            <div class="text-[9px] text-muted-foreground mb-2">Recent Commits</div>
            <div class="space-y-1">
              {currentLink.commits.slice(0, 5).map(commit => (
                <div
                  key={commit.sha}
                  class="flex items-center gap-2 text-[9px] p-1 rounded hover:bg-muted/10"
                >
                  <span class="font-mono text-cyan-400 w-14">{commit.shortSha}</span>
                  <span class="flex-1 text-foreground/80 truncate">{commit.message}</span>
                  <span class="text-muted-foreground">{formatDate(commit.date)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No links */}
        {!linkedBranch && !linkedPR && (
          <div class="text-center py-4 text-[10px] text-muted-foreground">
            No git links configured for {laneName || 'this lane'}
          </div>
        )}
      </div>

      {/* Link Modal */}
      {showLinkModal.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-96 max-h-[80vh] overflow-hidden flex flex-col">
            <div class="text-xs font-semibold text-foreground mb-4">
              Link Git to {laneName || 'Lane'}
            </div>

            {/* Tab selector */}
            <div class="flex gap-1 mb-3">
              <button
                onClick$={() => { linkType.value = 'branch'; }}
                class={`flex-1 px-2 py-1.5 text-[10px] rounded ${
                  linkType.value === 'branch'
                    ? 'bg-primary/20 text-primary'
                    : 'bg-muted/20 text-muted-foreground'
                }`}
              >
                Branch
              </button>
              <button
                onClick$={() => { linkType.value = 'pr'; }}
                class={`flex-1 px-2 py-1.5 text-[10px] rounded ${
                  linkType.value === 'pr'
                    ? 'bg-primary/20 text-primary'
                    : 'bg-muted/20 text-muted-foreground'
                }`}
              >
                Pull Request
              </button>
            </div>

            {/* Search */}
            <input
              type="text"
              value={searchQuery.value}
              onInput$={(e) => { searchQuery.value = (e.target as HTMLInputElement).value; }}
              placeholder={linkType.value === 'branch' ? 'Search branches...' : 'Search PRs...'}
              class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 mb-3"
            />

            {/* List */}
            <div class="flex-1 overflow-y-auto space-y-1 min-h-[200px] max-h-[300px]">
              {linkType.value === 'branch' ? (
                filteredBranches.map(branch => (
                  <div
                    key={branch.name}
                    onClick$={() => { selectedBranch.value = branch.name; }}
                    class={`p-2 rounded cursor-pointer ${
                      selectedBranch.value === branch.name
                        ? 'bg-primary/20 border border-primary/30'
                        : 'bg-muted/10 border border-transparent hover:bg-muted/20'
                    }`}
                  >
                    <div class="flex items-center justify-between">
                      <span class={`text-xs font-mono ${getBranchHealthColor(branch)}`}>
                        {branch.name}
                      </span>
                      {branch.isProtected && (
                        <span class="text-[8px] px-1 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                          protected
                        </span>
                      )}
                    </div>
                    <div class="text-[9px] text-muted-foreground mt-1">
                      {branch.lastCommit.message.slice(0, 50)}...
                    </div>
                  </div>
                ))
              ) : (
                filteredPRs.map(pr => (
                  <div
                    key={pr.id}
                    onClick$={() => { selectedPR.value = pr.id; }}
                    class={`p-2 rounded cursor-pointer ${
                      selectedPR.value === pr.id
                        ? 'bg-primary/20 border border-primary/30'
                        : 'bg-muted/10 border border-transparent hover:bg-muted/20'
                    }`}
                  >
                    <div class="flex items-center gap-2">
                      <span class={getPRStatusColor(pr)}>{getPRStatusIcon(pr)}</span>
                      <span class="text-xs font-medium">#{pr.number}</span>
                      <span class="text-xs text-foreground/80 truncate flex-1">
                        {pr.title}
                      </span>
                    </div>
                    <div class="text-[9px] text-muted-foreground mt-1">
                      {pr.sourceBranch} → {pr.targetBranch}
                    </div>
                  </div>
                ))
              )}

              {linkType.value === 'branch' && filteredBranches.length === 0 && (
                <div class="text-center py-4 text-[10px] text-muted-foreground">
                  No branches found
                </div>
              )}
              {linkType.value === 'pr' && filteredPRs.length === 0 && (
                <div class="text-center py-4 text-[10px] text-muted-foreground">
                  No pull requests found
                </div>
              )}
            </div>

            {/* Actions */}
            <div class="flex items-center gap-2 mt-4 pt-3 border-t border-border/50">
              <button
                onClick$={() => {
                  showLinkModal.value = false;
                  selectedBranch.value = null;
                  selectedPR.value = null;
                }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={linkType.value === 'branch' ? handleLinkBranch : handleLinkPR}
                disabled={linkType.value === 'branch' ? !selectedBranch.value : !selectedPR.value}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground disabled:opacity-50"
              >
                Link
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default LaneGitIntegration;
