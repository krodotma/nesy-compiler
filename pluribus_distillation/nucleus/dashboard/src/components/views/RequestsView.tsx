import { component$, type Signal } from '@builder.io/qwik';
import type { STRpRequest } from '../../lib/state/types';

interface RequestsViewProps {
  requests: Signal<STRpRequest[]>;
}

export const RequestsView = component$<RequestsViewProps>(({ requests }) => {
  return (
    <div class="rounded-lg border border-border bg-card">
      <div class="p-4 border-b border-border">
        <h2 class="font-semibold">Pending Requests</h2>
        <p class="text-sm text-muted-foreground">STRp pipeline requests</p>
      </div>
      <div class="overflow-auto">
        <table class="w-full text-sm">
          <thead class="bg-muted/30">
            <tr>
              <th class="text-left p-3 font-medium">ID</th>
              <th class="text-left p-3 font-medium">Time</th>
              <th class="text-left p-3 font-medium">Kind</th>
              <th class="text-left p-3 font-medium">Actor</th>
              <th class="text-left p-3 font-medium">Goal</th>
              <th class="text-left p-3 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {requests.value.length === 0 ? (
              <tr>
                <td colSpan={6} class="p-8 text-center text-muted-foreground">
                  No pending requests
                </td>
              </tr>
            ) : (
              requests.value.slice().reverse().map((req) => (
                <tr key={req.id} class="border-b border-border/30 hover:bg-muted/20">
                  <td class="p-3 font-mono text-xs">{req.id?.slice(0, 8)}</td>
                  <td class="p-3 text-xs text-muted-foreground">{req.created_iso?.slice(11, 19)}</td>
                  <td class="p-3">{req.kind}</td>
                  <td class="p-3 font-mono">{req.actor}</td>
                  <td class="p-3 max-w-[300px] truncate">{req.goal}</td>
                  <td class="p-3">
                    <span class={`px-2 py-0.5 rounded text-xs ${
                      req.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                      req.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                      req.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                      'bg-blue-500/20 text-blue-400'
                    }`}>
                      {req.status}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
});
