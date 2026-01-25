/**
 * LaneAccessControl - Manage permissions and access for lanes
 *
 * Phase 3, Iteration 25 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Role-based access control (owner, editor, viewer)
 * - Grant/revoke permissions
 * - Permission inheritance
 * - Audit log of permission changes
 * - Public/private lane toggle
 * - Emit bus events for access changes
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

export type AccessRole = 'owner' | 'editor' | 'viewer' | 'none';
export type Visibility = 'public' | 'private' | 'restricted';

export interface AccessGrant {
  id: string;
  userId: string;
  userName: string;
  role: AccessRole;
  grantedBy: string;
  grantedAt: string;
  expiresAt?: string;
}

export interface AccessAuditEntry {
  id: string;
  action: 'grant' | 'revoke' | 'change' | 'visibility';
  targetUser?: string;
  oldRole?: AccessRole;
  newRole?: AccessRole;
  oldVisibility?: Visibility;
  newVisibility?: Visibility;
  actor: string;
  timestamp: string;
}

export interface AccessEvent {
  type: 'grant' | 'revoke' | 'change' | 'visibility_change';
  laneId: string;
  grant?: AccessGrant;
  visibility?: Visibility;
  actor: string;
  timestamp: string;
}

export interface LaneAccessControlProps {
  /** Lane ID */
  laneId: string;
  /** Lane name for display */
  laneName: string;
  /** Current visibility */
  visibility: Visibility;
  /** Current access grants */
  grants: AccessGrant[];
  /** Available users to grant access to */
  availableUsers?: { id: string; name: string }[];
  /** Current actor */
  actor?: string;
  /** Actor's role (for permission checking) */
  actorRole?: AccessRole;
  /** Audit log entries */
  auditLog?: AccessAuditEntry[];
  /** Callback when access changes */
  onAccessChange$?: QRL<(event: AccessEvent) => void>;
  /** Show audit log */
  showAuditLog?: boolean;
  /** Compact mode */
  compact?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getRoleColor(role: AccessRole): string {
  switch (role) {
    case 'owner': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
    case 'editor': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'viewer': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'none': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getRoleIcon(role: AccessRole): string {
  switch (role) {
    case 'owner': return '👑';
    case 'editor': return '✏️';
    case 'viewer': return '👁';
    case 'none': return '🚫';
    default: return '•';
  }
}

function getVisibilityColor(visibility: Visibility): string {
  switch (visibility) {
    case 'public': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'private': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'restricted': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getVisibilityIcon(visibility: Visibility): string {
  switch (visibility) {
    case 'public': return '🌐';
    case 'private': return '🔒';
    case 'restricted': return '🔐';
    default: return '•';
  }
}

function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return ts.slice(0, 16);
  }
}

function generateId(): string {
  return `acc-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

function canManageAccess(actorRole: AccessRole): boolean {
  return actorRole === 'owner' || actorRole === 'editor';
}

function canChangeVisibility(actorRole: AccessRole): boolean {
  return actorRole === 'owner';
}

// ============================================================================
// Component
// ============================================================================

export const LaneAccessControl = component$<LaneAccessControlProps>(({
  laneId,
  laneName,
  visibility: initialVisibility,
  grants: initialGrants,
  availableUsers = [],
  actor = 'dashboard',
  actorRole = 'viewer',
  auditLog: initialAuditLog = [],
  onAccessChange$,
  showAuditLog = false,
  compact = false,
}) => {
  // State
  const visibility = useSignal<Visibility>(initialVisibility);
  const grants = useSignal<AccessGrant[]>(initialGrants);
  const auditLog = useSignal<AccessAuditEntry[]>(initialAuditLog);
  const showAddGrant = useSignal(false);
  const showAudit = useSignal(showAuditLog);

  // New grant form state
  const newUserId = useSignal('');
  const newRole = useSignal<AccessRole>('viewer');

  // Computed
  const canManage = useComputed$(() => canManageAccess(actorRole));
  const canChangeVis = useComputed$(() => canChangeVisibility(actorRole));

  const sortedGrants = useComputed$(() =>
    [...grants.value].sort((a, b) => {
      const roleOrder: Record<AccessRole, number> = { owner: 0, editor: 1, viewer: 2, none: 3 };
      return roleOrder[a.role] - roleOrder[b.role];
    })
  );

  const availableToGrant = useComputed$(() =>
    availableUsers.filter(u => !grants.value.some(g => g.userId === u.id))
  );

  const stats = useComputed$(() => ({
    owners: grants.value.filter(g => g.role === 'owner').length,
    editors: grants.value.filter(g => g.role === 'editor').length,
    viewers: grants.value.filter(g => g.role === 'viewer').length,
    total: grants.value.length,
  }));

  // Add audit entry
  const addAuditEntry = $((entry: Omit<AccessAuditEntry, 'id' | 'timestamp'>) => {
    const newEntry: AccessAuditEntry = {
      ...entry,
      id: generateId(),
      timestamp: new Date().toISOString(),
    };
    auditLog.value = [newEntry, ...auditLog.value].slice(0, 50); // Keep last 50
  });

  // Emit event helper
  const emitEvent = $(async (type: AccessEvent['type'], grant?: AccessGrant, newVisibility?: Visibility) => {
    const event: AccessEvent = {
      type,
      laneId,
      grant,
      visibility: newVisibility,
      actor,
      timestamp: new Date().toISOString(),
    };

    console.log(`[LaneAccessControl] Emitting: operator.lanes.access.${type}`, event);

    if (onAccessChange$) {
      await onAccessChange$(event);
    }
  });

  // Change visibility
  const changeVisibility = $(async (newVisibility: Visibility) => {
    if (!canChangeVis.value || newVisibility === visibility.value) return;

    addAuditEntry({
      action: 'visibility',
      oldVisibility: visibility.value,
      newVisibility,
      actor,
    });

    visibility.value = newVisibility;
    await emitEvent('visibility_change', undefined, newVisibility);
  });

  // Grant access
  const grantAccess = $(async () => {
    if (!canManage.value || !newUserId.value) return;

    const user = availableUsers.find(u => u.id === newUserId.value);
    if (!user) return;

    const newGrant: AccessGrant = {
      id: generateId(),
      userId: user.id,
      userName: user.name,
      role: newRole.value,
      grantedBy: actor,
      grantedAt: new Date().toISOString(),
    };

    grants.value = [...grants.value, newGrant];

    addAuditEntry({
      action: 'grant',
      targetUser: user.id,
      newRole: newRole.value,
      actor,
    });

    await emitEvent('grant', newGrant);

    // Reset form
    newUserId.value = '';
    newRole.value = 'viewer';
    showAddGrant.value = false;
  });

  // Change role
  const changeRole = $(async (grantId: string, newRole: AccessRole) => {
    if (!canManage.value) return;

    const idx = grants.value.findIndex(g => g.id === grantId);
    if (idx === -1) return;

    const oldGrant = grants.value[idx];

    if (newRole === 'none') {
      // Revoke access
      grants.value = grants.value.filter(g => g.id !== grantId);

      addAuditEntry({
        action: 'revoke',
        targetUser: oldGrant.userId,
        oldRole: oldGrant.role,
        actor,
      });

      await emitEvent('revoke', oldGrant);
    } else {
      // Change role
      const updatedGrant = { ...oldGrant, role: newRole };
      const newGrants = [...grants.value];
      newGrants[idx] = updatedGrant;
      grants.value = newGrants;

      addAuditEntry({
        action: 'change',
        targetUser: oldGrant.userId,
        oldRole: oldGrant.role,
        newRole,
        actor,
      });

      await emitEvent('change', updatedGrant);
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">ACCESS CONTROL</span>
          <span class={`text-[9px] px-1.5 py-0.5 rounded border ${getVisibilityColor(visibility.value)}`}>
            {getVisibilityIcon(visibility.value)} {visibility.value}
          </span>
        </div>
        <span class="text-[9px] text-muted-foreground">{stats.value.total} users</span>
      </div>

      {/* Lane info */}
      <div class="px-3 py-2 border-b border-border/30 flex items-center justify-between">
        <span class="text-xs text-foreground">{laneName}</span>
        <span class={`text-[9px] px-1.5 py-0.5 rounded border ${getRoleColor(actorRole)}`}>
          {getRoleIcon(actorRole)} Your role: {actorRole}
        </span>
      </div>

      {/* Visibility toggle */}
      {canChangeVis.value && (
        <div class="p-3 border-b border-border/30">
          <div class="text-[9px] text-muted-foreground mb-2">Visibility</div>
          <div class="flex gap-2">
            {(['public', 'restricted', 'private'] as const).map(vis => (
              <button
                key={vis}
                onClick$={() => changeVisibility(vis)}
                class={`flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-[10px] rounded border transition-colors ${
                  visibility.value === vis
                    ? getVisibilityColor(vis)
                    : 'bg-muted/10 text-muted-foreground border-border/30 hover:bg-muted/20'
                }`}
              >
                {getVisibilityIcon(vis)} {vis}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Add grant button */}
      {canManage.value && availableToGrant.value.length > 0 && (
        <div class="p-3 border-b border-border/30">
          {!showAddGrant.value ? (
            <button
              onClick$={() => { showAddGrant.value = true; }}
              class="w-full px-3 py-1.5 text-[10px] rounded bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-colors"
            >
              + Add User
            </button>
          ) : (
            <div class="space-y-2">
              <div class="flex gap-2">
                <select
                  value={newUserId.value}
                  onChange$={(e) => { newUserId.value = (e.target as HTMLSelectElement).value; }}
                  class="flex-1 px-2 py-1.5 text-[10px] rounded bg-card border border-border/50 text-foreground"
                >
                  <option value="">Select user...</option>
                  {availableToGrant.value.map(user => (
                    <option key={user.id} value={user.id}>@{user.name}</option>
                  ))}
                </select>
                <select
                  value={newRole.value}
                  onChange$={(e) => { newRole.value = (e.target as HTMLSelectElement).value as AccessRole; }}
                  class="w-24 px-2 py-1.5 text-[10px] rounded bg-card border border-border/50 text-foreground"
                >
                  <option value="viewer">Viewer</option>
                  <option value="editor">Editor</option>
                  {actorRole === 'owner' && <option value="owner">Owner</option>}
                </select>
              </div>
              <div class="flex gap-2">
                <button
                  onClick$={() => { showAddGrant.value = false; }}
                  class="flex-1 px-2 py-1.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick$={grantAccess}
                  disabled={!newUserId.value}
                  class="flex-1 px-2 py-1.5 text-[10px] rounded bg-primary text-primary-foreground font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Add
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Grants list */}
      <div class={`overflow-y-auto ${compact ? 'max-h-[150px]' : 'max-h-[200px]'}`}>
        {sortedGrants.value.length === 0 ? (
          <div class="p-4 text-center text-[10px] text-muted-foreground">
            No users have access
          </div>
        ) : (
          sortedGrants.value.map(grant => (
            <div
              key={grant.id}
              class="flex items-center gap-2 px-3 py-2 border-b border-border/20 hover:bg-muted/5"
            >
              {/* User info */}
              <div class="flex-grow min-w-0">
                <div class="text-xs text-foreground">@{grant.userName}</div>
                <div class="text-[9px] text-muted-foreground">
                  Granted by @{grant.grantedBy}
                </div>
              </div>

              {/* Role selector or badge */}
              {canManage.value && grant.userId !== actor ? (
                <select
                  value={grant.role}
                  onChange$={(e) => changeRole(grant.id, (e.target as HTMLSelectElement).value as AccessRole)}
                  class={`px-2 py-1 text-[9px] rounded border ${getRoleColor(grant.role)}`}
                >
                  <option value="viewer">Viewer</option>
                  <option value="editor">Editor</option>
                  {actorRole === 'owner' && <option value="owner">Owner</option>}
                  <option value="none">Remove</option>
                </select>
              ) : (
                <span class={`px-2 py-1 text-[9px] rounded border ${getRoleColor(grant.role)}`}>
                  {getRoleIcon(grant.role)} {grant.role}
                </span>
              )}
            </div>
          ))
        )}
      </div>

      {/* Audit log toggle */}
      <div class="border-t border-border/50">
        <button
          onClick$={() => { showAudit.value = !showAudit.value; }}
          class="w-full p-2 text-[9px] text-muted-foreground hover:bg-muted/10 transition-colors flex items-center justify-center gap-1"
        >
          <span>{showAudit.value ? '▼' : '▶'}</span>
          <span>Audit Log ({auditLog.value.length})</span>
        </button>

        {showAudit.value && (
          <div class="max-h-[120px] overflow-y-auto border-t border-border/30 bg-muted/5">
            {auditLog.value.length === 0 ? (
              <div class="p-3 text-center text-[9px] text-muted-foreground">
                No audit entries
              </div>
            ) : (
              auditLog.value.map(entry => (
                <div key={entry.id} class="px-3 py-2 border-b border-border/20 text-[9px]">
                  <div class="flex items-center gap-2">
                    <span class={`px-1 py-0.5 rounded ${
                      entry.action === 'grant' ? 'bg-emerald-500/20 text-emerald-400' :
                      entry.action === 'revoke' ? 'bg-red-500/20 text-red-400' :
                      entry.action === 'change' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-amber-500/20 text-amber-400'
                    }`}>
                      {entry.action}
                    </span>
                    <span class="text-muted-foreground">{formatTimestamp(entry.timestamp)}</span>
                  </div>
                  <div class="text-foreground/80 mt-1">
                    {entry.action === 'visibility' && (
                      <>Changed visibility: {entry.oldVisibility} → {entry.newVisibility}</>
                    )}
                    {entry.action === 'grant' && (
                      <>Granted {entry.newRole} to @{entry.targetUser}</>
                    )}
                    {entry.action === 'revoke' && (
                      <>Revoked {entry.oldRole} from @{entry.targetUser}</>
                    )}
                    {entry.action === 'change' && (
                      <>Changed @{entry.targetUser}: {entry.oldRole} → {entry.newRole}</>
                    )}
                    <span class="text-muted-foreground"> by @{entry.actor}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
});

export default LaneAccessControl;
