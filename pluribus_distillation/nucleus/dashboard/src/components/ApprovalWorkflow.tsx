/**
 * ApprovalWorkflow - Request and manage approvals for lane changes
 *
 * Phase 3, Iteration 24 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Request approval for lane operations
 * - Multi-approver support
 * - Approval status tracking
 * - Comments on approvals
 * - Deadline support
 * - Emit bus events for approval state changes
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export type ApprovalStatus = 'pending' | 'approved' | 'rejected' | 'expired';
export type ApprovalType = 'status_change' | 'reassign' | 'archive' | 'priority' | 'blocker' | 'custom';

export interface Approver {
  id: string;
  name: string;
  decision?: 'approved' | 'rejected';
  decidedAt?: string;
  comment?: string;
}

export interface ApprovalRequest {
  id: string;
  type: ApprovalType;
  laneId: string;
  laneName: string;
  requestor: string;
  approvers: Approver[];
  title: string;
  description?: string;
  deadline?: string;
  status: ApprovalStatus;
  createdAt: string;
  resolvedAt?: string;
  data?: Record<string, unknown>;
}

export interface ApprovalEvent {
  type: 'request' | 'approve' | 'reject' | 'cancel' | 'expire';
  request: ApprovalRequest;
  actor: string;
  comment?: string;
  timestamp: string;
}

export interface ApprovalWorkflowProps {
  /** Current approval requests */
  requests: ApprovalRequest[];
  /** Available approvers */
  availableApprovers?: { id: string; name: string }[];
  /** Current actor */
  actor?: string;
  /** Callback when approval state changes */
  onApprovalChange$?: QRL<(event: ApprovalEvent) => void>;
  /** Show only pending requests */
  pendingOnly?: boolean;
  /** Compact mode */
  compact?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: ApprovalStatus): string {
  switch (status) {
    case 'pending': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'approved': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'rejected': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'expired': return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getStatusIcon(status: ApprovalStatus): string {
  switch (status) {
    case 'pending': return '⏳';
    case 'approved': return '✓';
    case 'rejected': return '✕';
    case 'expired': return '⌛';
    default: return '•';
  }
}

function getTypeLabel(type: ApprovalType): string {
  switch (type) {
    case 'status_change': return 'Status Change';
    case 'reassign': return 'Reassignment';
    case 'archive': return 'Archive';
    case 'priority': return 'Priority Change';
    case 'blocker': return 'Blocker Update';
    case 'custom': return 'Custom';
    default: return type;
  }
}

function formatDeadline(deadline: string): { text: string; urgent: boolean } {
  try {
    const date = new Date(deadline);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffMs < 0) return { text: 'Expired', urgent: true };
    if (diffHours < 24) return { text: `${diffHours}h left`, urgent: true };
    return { text: `${diffDays}d left`, urgent: false };
  } catch {
    return { text: deadline.slice(0, 10), urgent: false };
  }
}

function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return ts.slice(0, 10);
  }
}

function generateId(): string {
  return `apr-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

// ============================================================================
// Component
// ============================================================================

export const ApprovalWorkflow = component$<ApprovalWorkflowProps>(({
  requests: initialRequests,
  availableApprovers = [],
  actor = 'dashboard',
  onApprovalChange$,
  pendingOnly = false,
  compact = false,
}) => {
  // State
  const requests = useSignal<ApprovalRequest[]>(initialRequests);
  const showNewRequest = useSignal(false);
  const expandedId = useSignal<string | null>(null);
  const commentText = useSignal('');

  // New request form state
  const newType = useSignal<ApprovalType>('custom');
  const newTitle = useSignal('');
  const newDescription = useSignal('');
  const newApprovers = useSignal<string[]>([]);
  const newDeadlineDays = useSignal(3);

  // Computed
  const filteredRequests = useComputed$(() => {
    let result = [...requests.value];
    if (pendingOnly) {
      result = result.filter(r => r.status === 'pending');
    }
    // Sort by created date (newest first), pending first
    result.sort((a, b) => {
      if (a.status === 'pending' && b.status !== 'pending') return -1;
      if (a.status !== 'pending' && b.status === 'pending') return 1;
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });
    return result;
  });

  const stats = useComputed$(() => ({
    total: requests.value.length,
    pending: requests.value.filter(r => r.status === 'pending').length,
    approved: requests.value.filter(r => r.status === 'approved').length,
    rejected: requests.value.filter(r => r.status === 'rejected').length,
  }));

  // Emit event helper
  const emitEvent = $(async (type: ApprovalEvent['type'], request: ApprovalRequest, comment?: string) => {
    const event: ApprovalEvent = {
      type,
      request,
      actor,
      comment,
      timestamp: new Date().toISOString(),
    };

    console.log(`[ApprovalWorkflow] Emitting: operator.lanes.approval.${type}`, event);

    if (onApprovalChange$) {
      await onApprovalChange$(event);
    }
  });

  // Create new request
  const createRequest = $(async () => {
    if (!newTitle.value.trim() || newApprovers.value.length === 0) return;

    const deadline = new Date();
    deadline.setDate(deadline.getDate() + newDeadlineDays.value);

    const newRequest: ApprovalRequest = {
      id: generateId(),
      type: newType.value,
      laneId: '', // Would be set by parent
      laneName: '',
      requestor: actor,
      approvers: newApprovers.value.map(id => {
        const approver = availableApprovers.find(a => a.id === id);
        return { id, name: approver?.name || id };
      }),
      title: newTitle.value.trim(),
      description: newDescription.value.trim() || undefined,
      deadline: deadline.toISOString(),
      status: 'pending',
      createdAt: new Date().toISOString(),
    };

    requests.value = [...requests.value, newRequest];
    await emitEvent('request', newRequest);

    // Reset form
    newType.value = 'custom';
    newTitle.value = '';
    newDescription.value = '';
    newApprovers.value = [];
    newDeadlineDays.value = 3;
    showNewRequest.value = false;
  });

  // Handle approve/reject
  const handleDecision = $(async (requestId: string, decision: 'approved' | 'rejected') => {
    const idx = requests.value.findIndex(r => r.id === requestId);
    if (idx === -1) return;

    const request = requests.value[idx];
    const approverIdx = request.approvers.findIndex(a => a.id === actor);

    // Update approver's decision
    const updatedApprovers = [...request.approvers];
    if (approverIdx !== -1) {
      updatedApprovers[approverIdx] = {
        ...updatedApprovers[approverIdx],
        decision,
        decidedAt: new Date().toISOString(),
        comment: commentText.value || undefined,
      };
    }

    // Check if all approvers have decided
    const allDecided = updatedApprovers.every(a => a.decision);
    const allApproved = updatedApprovers.every(a => a.decision === 'approved');

    const updatedRequest: ApprovalRequest = {
      ...request,
      approvers: updatedApprovers,
      status: allDecided ? (allApproved ? 'approved' : 'rejected') : 'pending',
      resolvedAt: allDecided ? new Date().toISOString() : undefined,
    };

    const newRequests = [...requests.value];
    newRequests[idx] = updatedRequest;
    requests.value = newRequests;

    await emitEvent(decision === 'approved' ? 'approve' : 'reject', updatedRequest, commentText.value);

    commentText.value = '';
    expandedId.value = null;
  });

  // Cancel request
  const cancelRequest = $(async (requestId: string) => {
    const request = requests.value.find(r => r.id === requestId);
    if (!request) return;

    requests.value = requests.value.filter(r => r.id !== requestId);
    await emitEvent('cancel', request);
  });

  // Toggle approver selection
  const toggleApprover = $((id: string) => {
    if (newApprovers.value.includes(id)) {
      newApprovers.value = newApprovers.value.filter(a => a !== id);
    } else {
      newApprovers.value = [...newApprovers.value, id];
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">APPROVALS</span>
          {stats.value.pending > 0 && (
            <span class="text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 animate-pulse">
              {stats.value.pending} pending
            </span>
          )}
        </div>
        <button
          onClick$={() => { showNewRequest.value = !showNewRequest.value; }}
          class="text-[10px] px-2 py-1 rounded bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-colors"
        >
          + Request Approval
        </button>
      </div>

      {/* New request form */}
      {showNewRequest.value && (
        <div class="p-3 border-b border-border/30 bg-muted/10 space-y-3">
          <div class="text-[10px] font-medium text-foreground">New Approval Request</div>

          {/* Type */}
          <div>
            <label class="text-[9px] text-muted-foreground block mb-1">Type</label>
            <select
              value={newType.value}
              onChange$={(e) => { newType.value = (e.target as HTMLSelectElement).value as ApprovalType; }}
              class="w-full px-2 py-1.5 text-[10px] rounded bg-card border border-border/50 text-foreground"
            >
              <option value="status_change">Status Change</option>
              <option value="reassign">Reassignment</option>
              <option value="archive">Archive</option>
              <option value="priority">Priority Change</option>
              <option value="blocker">Blocker Update</option>
              <option value="custom">Custom</option>
            </select>
          </div>

          {/* Title */}
          <input
            type="text"
            placeholder="What needs approval?"
            value={newTitle.value}
            onInput$={(e) => { newTitle.value = (e.target as HTMLInputElement).value; }}
            class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50"
          />

          {/* Description */}
          <textarea
            placeholder="Additional details (optional)..."
            value={newDescription.value}
            onInput$={(e) => { newDescription.value = (e.target as HTMLTextAreaElement).value; }}
            class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50 resize-none"
            rows={2}
          />

          {/* Approvers */}
          <div>
            <label class="text-[9px] text-muted-foreground block mb-1">Approvers (select at least one)</label>
            <div class="flex flex-wrap gap-1">
              {availableApprovers.map(approver => (
                <button
                  key={approver.id}
                  onClick$={() => toggleApprover(approver.id)}
                  class={`px-2 py-1 text-[9px] rounded border transition-colors ${
                    newApprovers.value.includes(approver.id)
                      ? 'bg-primary/20 text-primary border-primary/30'
                      : 'bg-muted/10 text-muted-foreground border-border/30 hover:bg-muted/20'
                  }`}
                >
                  {newApprovers.value.includes(approver.id) && '✓ '}
                  @{approver.name}
                </button>
              ))}
            </div>
          </div>

          {/* Deadline */}
          <div>
            <label class="text-[9px] text-muted-foreground block mb-1">Deadline</label>
            <div class="flex gap-2">
              {[1, 3, 7, 14].map(days => (
                <button
                  key={days}
                  onClick$={() => { newDeadlineDays.value = days; }}
                  class={`px-2 py-1 text-[9px] rounded border transition-colors ${
                    newDeadlineDays.value === days
                      ? 'bg-primary/20 text-primary border-primary/30'
                      : 'bg-muted/10 text-muted-foreground border-border/30 hover:bg-muted/20'
                  }`}
                >
                  {days}d
                </button>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div class="flex gap-2 justify-end">
            <button
              onClick$={() => { showNewRequest.value = false; }}
              class="px-3 py-1.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick$={createRequest}
              disabled={!newTitle.value.trim() || newApprovers.value.length === 0}
              class="px-3 py-1.5 text-[10px] rounded bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Request Approval
            </button>
          </div>
        </div>
      )}

      {/* Request list */}
      <div class={`overflow-y-auto ${compact ? 'max-h-[200px]' : 'max-h-[350px]'}`}>
        {filteredRequests.value.length === 0 ? (
          <div class="p-6 text-center">
            <div class="text-2xl mb-2">✓</div>
            <div class="text-xs text-muted-foreground">No approval requests</div>
          </div>
        ) : (
          filteredRequests.value.map(request => {
            const isExpanded = expandedId.value === request.id;
            const canDecide = request.status === 'pending' && request.approvers.some(a => a.id === actor && !a.decision);
            const deadlineInfo = request.deadline ? formatDeadline(request.deadline) : null;

            return (
              <div key={request.id} class="border-b border-border/30">
                <div
                  onClick$={() => { expandedId.value = isExpanded ? null : request.id; }}
                  class={`p-3 cursor-pointer hover:bg-muted/5 transition-colors ${
                    request.status === 'pending' ? 'bg-amber-500/5' : ''
                  }`}
                >
                  <div class="flex items-start justify-between">
                    <div class="flex-grow min-w-0">
                      <div class="flex items-center gap-2">
                        <span class={`text-[9px] px-1.5 py-0.5 rounded border ${getStatusColor(request.status)}`}>
                          {getStatusIcon(request.status)} {request.status}
                        </span>
                        <span class="text-[9px] text-muted-foreground">{getTypeLabel(request.type)}</span>
                      </div>
                      <div class="text-xs font-medium text-foreground mt-1">{request.title}</div>
                      <div class="text-[9px] text-muted-foreground mt-0.5">
                        by @{request.requestor} • {formatTimestamp(request.createdAt)}
                      </div>
                    </div>

                    <div class="flex-shrink-0 flex flex-col items-end gap-1">
                      {deadlineInfo && request.status === 'pending' && (
                        <span class={`text-[9px] ${deadlineInfo.urgent ? 'text-red-400' : 'text-muted-foreground'}`}>
                          {deadlineInfo.text}
                        </span>
                      )}
                      {/* Approver dots */}
                      <div class="flex gap-0.5">
                        {request.approvers.map(approver => (
                          <div
                            key={approver.id}
                            class={`w-2 h-2 rounded-full ${
                              approver.decision === 'approved' ? 'bg-emerald-400' :
                              approver.decision === 'rejected' ? 'bg-red-400' :
                              'bg-amber-400'
                            }`}
                            title={`${approver.name}: ${approver.decision || 'pending'}`}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Expanded details */}
                {isExpanded && (
                  <div class="px-3 pb-3 space-y-3">
                    {/* Description */}
                    {request.description && (
                      <div class="text-[10px] text-muted-foreground bg-muted/10 p-2 rounded">
                        {request.description}
                      </div>
                    )}

                    {/* Approvers list */}
                    <div class="space-y-1">
                      <div class="text-[9px] text-muted-foreground">Approvers:</div>
                      {request.approvers.map(approver => (
                        <div key={approver.id} class="flex items-center gap-2 text-[10px]">
                          <div class={`w-2 h-2 rounded-full ${
                            approver.decision === 'approved' ? 'bg-emerald-400' :
                            approver.decision === 'rejected' ? 'bg-red-400' :
                            'bg-amber-400'
                          }`} />
                          <span class="text-foreground">@{approver.name}</span>
                          {approver.decision && (
                            <>
                              <span class={approver.decision === 'approved' ? 'text-emerald-400' : 'text-red-400'}>
                                {approver.decision}
                              </span>
                              {approver.comment && (
                                <span class="text-muted-foreground">- "{approver.comment}"</span>
                              )}
                            </>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* Decision form */}
                    {canDecide && (
                      <div class="pt-2 border-t border-border/30 space-y-2">
                        <textarea
                          placeholder="Comment (optional)..."
                          value={commentText.value}
                          onInput$={(e) => { commentText.value = (e.target as HTMLTextAreaElement).value; }}
                          class="w-full px-2 py-1.5 text-[10px] rounded bg-card border border-border/50 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50 resize-none"
                          rows={2}
                        />
                        <div class="flex gap-2">
                          <button
                            onClick$={() => handleDecision(request.id, 'approved')}
                            class="flex-1 px-3 py-1.5 text-[10px] rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30 transition-colors"
                          >
                            ✓ Approve
                          </button>
                          <button
                            onClick$={() => handleDecision(request.id, 'rejected')}
                            class="flex-1 px-3 py-1.5 text-[10px] rounded bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors"
                          >
                            ✕ Reject
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Cancel button for requestor */}
                    {request.status === 'pending' && request.requestor === actor && (
                      <button
                        onClick$={() => cancelRequest(request.id)}
                        class="w-full px-3 py-1.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
                      >
                        Cancel Request
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {stats.value.approved} approved • {stats.value.rejected} rejected
      </div>
    </div>
  );
});

export default ApprovalWorkflow;
