import { component$ } from '@builder.io/qwik';
import { useTracking } from '../lib/telemetry/use-tracking';

// M3 Components (Step 55 - Avatar with Progress Ring)
import '@material/web/progress/circular-progress.js';
import '@material/web/ripple/ripple.js';

interface AgentAvatarProps {
  actor: string;
  status: string;
  task?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const AgentAvatar = component$<AgentAvatarProps>(({ actor, status, task, size = 'md' }) => {
  useTracking('comp:agent-avatar');

  const isBusy = status === 'busy' || status === 'thinking';
  const isActive = status === 'active' || status === 'running';
  const isError = status === 'error';

  const statusClass =
    isActive ? 'agent-avatar-status-active' :
    isError ? 'agent-avatar-status-error' :
    isBusy ? 'agent-avatar-status-busy' :
    'agent-avatar-status-idle';

  const sizeClass =
    size === 'sm' ? 'agent-avatar-sm' :
    size === 'lg' ? 'agent-avatar-lg' :
    'agent-avatar-md';

  return (
    <div class={`agent-avatar ${sizeClass} ${statusClass}`}>
      {/* M3 Ripple for interaction */}
      <md-ripple class="agent-avatar-ripple"></md-ripple>

      {/* Avatar circle with initials */}
      <div class="agent-avatar-circle">
        {actor.slice(0, 2).toUpperCase()}
      </div>

      {/* M3 Circular Progress for busy/thinking states */}
      {isBusy && (
        <md-circular-progress
          indeterminate
          class="agent-avatar-progress"
        ></md-circular-progress>
      )}

      {/* Status indicator dot (non-busy states) */}
      {!isBusy && (
        <div class="agent-avatar-status-dot" />
      )}

      {/* Tooltip - enhanced styling */}
      <div class="agent-avatar-tooltip">
        <div class="agent-avatar-tooltip-actor">{actor}</div>
        <div class="agent-avatar-tooltip-task">{task || 'Idle'}</div>
        <div class="agent-avatar-tooltip-status">{status}</div>
      </div>
    </div>
  );
});
