import { component$ } from '@builder.io/qwik';

export default component$(() => {
  return (
    <div class="space-y-6">
      <div class="p-4 border-b border-border flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
            Decentralized Spine
          </h1>
          <p class="text-sm text-muted-foreground">
            Identity (DID), Domains (.crypto/.eth), and AgentFi
          </p>
        </div>
        <span class="px-2 py-1 rounded bg-purple-500/20 text-purple-400 text-xs font-mono">
          RFC: decentralized_spine.md
        </span>
      </div>

      <div class="grid gap-6 md:grid-cols-2">
        {/* Identity Card */}
        <div class="rounded-lg border border-border bg-card p-6">
          <h2 class="text-xl font-semibold mb-4">Self-Sovereign Identity</h2>
          <div class="space-y-4">
            <div class="p-3 rounded bg-muted/30 border border-border/50">
              <div class="text-xs text-muted-foreground mb-1">Active DID (Session)</div>
              <div class="font-mono text-sm text-green-400">did:key:z6MkhaXgBZDvotDkL5257...</div>
            </div>
            <div class="p-3 rounded bg-muted/30 border border-border/50">
              <div class="text-xs text-muted-foreground mb-1">Host Identity</div>
              <div class="font-mono text-sm text-blue-400">did:web:kroma.live</div>
            </div>
          </div>
        </div>

        {/* Resolution Card */}
        <div class="rounded-lg border border-border bg-card p-6">
          <h2 class="text-xl font-semibold mb-4">Name Resolution</h2>
          <div class="flex gap-2 mb-4">
            <input 
              type="text" 
              placeholder="agent.eth or name.crypto" 
              class="flex-1 bg-black/20 border border-border rounded px-3 py-2 text-sm font-mono"
            />
            <button class="px-4 py-2 bg-primary/20 text-primary rounded hover:bg-primary/30">
              Resolve
            </button>
          </div>
          <div class="text-xs text-muted-foreground text-center italic">
            Resolver daemon not connected.
          </div>
        </div>
      </div>
    </div>
  );
});
