/**
 * CloudStoragePanel.tsx - Unified Cloud Storage + VNC OAuth UI
 *
 * Manages Google Drive (and other cloud storage) connections with an
 * EMBEDDED VNC viewer for inline OAuth authentication.
 *
 * Flow:
 * 1. User clicks "Connect" for an account
 * 2. VNC iframe appears inline showing OAuth page
 * 3. User authenticates directly in the embedded VNC
 * 4. Token captured, VNC auto-hides, drive mounted
 * 5. Files accessible at /mnt/gdrive/<account>
 */

import { component$, useSignal, useStore, useVisibleTask$, $ } from '@builder.io/qwik';

// M3 Components - CloudStoragePanel
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/iconbutton/icon-button.js';
import '@material/web/progress/circular-progress.js';

interface CloudAccount {
  name: string;
  email: string;
  provider: string;
  configured: boolean;
  mounted: boolean;
  mountpoint?: string;
}

interface CloudStorageStatus {
  accounts: CloudAccount[];
  pendingAuth: string[];
}

export const CloudStoragePanel = component$(() => {
  const status = useStore<CloudStorageStatus>({
    accounts: [],
    pendingAuth: [],
  });

  const isLoading = useSignal(true);
  const authInProgress = useSignal<string | null>(null);
  const error = useSignal<string | null>(null);
  const showVNC = useSignal(false);
  const browseAccount = useSignal<string | null>(null);
  const files = useStore<{ name: string; size: number; isDir: boolean }[]>([]);

  // noVNC URL for embedded viewer.
  // Avoid embedding the VNC password in the URL; enter it manually (see /pluribus/.pluribus/vnc_password.txt on the VPS).
  // Do not autoconnect: VNC requires a password and autoconnect can get stuck in a failed loop without showing the prompt.
  const noVNCUrl = '/vnc/vnc.html?resize=scale&path=vnc/websockify';

  // Fetch status from cloud storage daemon
  const fetchStatus = $(async () => {
    try {
      const res = await fetch('/api/cloud/status');
      if (res.ok) {
        const data = await res.json();

        // Transform to account list
        const accounts: CloudAccount[] = [];
        const accountConfig: Record<string, { email: string; provider: string }> = {
          peter_herz: { email: 'peter.herz@gmail.com', provider: 'google_drive' },
          peter_kroma: { email: 'peter@kro.ma', provider: 'google_drive' },
          peter_tachy0n: { email: 'peter@tachy0n.com', provider: 'google_drive' },
        };

        for (const [name, config] of Object.entries(accountConfig)) {
          const remote = data.remotes?.[name] || {};
          accounts.push({
            name,
            email: config.email,
            provider: config.provider,
            configured: remote.configured || false,
            mounted: remote.mounted || false,
            mountpoint: remote.mountpoint,
          });
        }

        status.accounts = accounts;
        status.pendingAuth = data.pending_auth || [];

        // Auto-hide VNC if auth completed for the account we're waiting on
        if (authInProgress.value) {
          const account = status.accounts.find(a => a.name === authInProgress.value);
          if (account?.configured) {
            authInProgress.value = null;
            showVNC.value = false;
          }
        }
      }
    } catch (e) {
      console.error('Failed to fetch cloud storage status:', e);
    }
    isLoading.value = false;
  });

  // Start OAuth flow and show VNC
  const startAuth = $(async (account: string) => {
    authInProgress.value = account;
    error.value = null;
    showVNC.value = true;

    try {
      const res = await fetch('/api/cloud/auth/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ account }),
      });

      const data = await res.json();

      if (data.error) {
        error.value = data.error;
        authInProgress.value = null;
        showVNC.value = false;
        return;
      }

      // The daemon automatically opens the OAuth URL in VNC Firefox
      // We just need to show the VNC viewer and poll for completion

    } catch (e) {
      error.value = `Failed to start auth: ${e}`;
      authInProgress.value = null;
      showVNC.value = false;
    }
  });

  // Mount/unmount
  const toggleMount = $(async (account: string, mounted: boolean) => {
    const endpoint = mounted ? 'unmount' : 'mount';
    try {
      await fetch(`/api/cloud/${endpoint}/${account}`, { method: 'POST' });
      await fetchStatus();
    } catch (e) {
      error.value = `Failed to ${endpoint}: ${e}`;
    }
  });

  // Browse files
  const browseFiles = $(async (account: string) => {
    browseAccount.value = account;
    try {
      const res = await fetch(`/api/cloud/browse/${account}`);
      const data = await res.json();
      if (data.files) {
        files.length = 0;
        data.files.forEach((f: { Name?: string; name?: string; Size?: number; size?: number; IsDir?: boolean; isDir?: boolean }) => {
          files.push({
            name: f.Name || f.name || '',
            size: f.Size || f.size || 0,
            isDir: f.IsDir || f.isDir || false,
          });
        });
      }
    } catch (e) {
      error.value = `Failed to browse: ${e}`;
    }
  });

  // Initial fetch and polling
  useVisibleTask$(({ cleanup }) => {
    fetchStatus();
    // Poll more frequently when auth is in progress
    const interval = setInterval(() => {
      fetchStatus();
    }, authInProgress.value ? 2000 : 15000);
    cleanup(() => clearInterval(interval));
  });

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '-';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div class="h-full flex flex-col">
      {/* Header */}
      <div class="flex items-center justify-between p-4 border-b border-border">
        <div class="flex items-center gap-3">
          <span class="text-2xl">☁️</span>
          <div>
            <h2 class="text-lg font-semibold">Cloud Storage</h2>
            <p class="text-xs text-muted-foreground">
              Google Drive mounts with inline VNC OAuth
            </p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          {showVNC.value && (
            <button
              onClick$={() => {
                showVNC.value = false;
                authInProgress.value = null;
              }}
              class="text-xs px-3 py-1.5 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30"
            >
              ✕ Close VNC
            </button>
          )}
          <button
            onClick$={fetchStatus}
            class="text-xs px-3 py-1.5 rounded bg-muted/30 hover:bg-muted/50"
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error.value && (
        <div class="px-4 py-2 bg-red-500/10 border-b border-red-500/20 text-xs text-red-400 flex items-center justify-between">
          <span>{error.value}</span>
          <button
            onClick$={() => error.value = null}
            class="text-red-300 hover:text-red-100"
          >
            ✕
          </button>
        </div>
      )}

      {/* Main content area - split when VNC is shown */}
      <div class={`flex-1 flex ${showVNC.value ? 'flex-row' : 'flex-col'} min-h-0 overflow-hidden`}>

        {/* Accounts Panel */}
        <div class={`${showVNC.value ? 'w-80 border-r border-border' : 'w-full'} flex flex-col overflow-hidden`}>
          <div class="flex-1 overflow-auto p-4 space-y-3">
            {isLoading.value ? (
              <div class="text-center text-muted-foreground py-8">Loading...</div>
            ) : (
              status.accounts.map((account) => (
                <div
                  key={account.name}
                  class={`rounded-lg border p-4 transition-all ${
                    authInProgress.value === account.name
                      ? 'border-amber-500/50 bg-amber-500/10 ring-2 ring-amber-500/30 shadow-[0_0_20px_rgba(245,158,11,0.2)]'
                      : account.configured
                        ? account.mounted
                          ? 'border-green-500/30 bg-green-500/5'
                          : 'border-blue-500/30 bg-blue-500/5'
                        : 'border-border bg-muted/10'
                  }`}
                >
                  <div class="flex items-center justify-between">
                    <div class="flex items-center gap-3">
                      {/* Provider icon */}
                      <span class="text-2xl">
                        {account.provider === 'google_drive' ? '📁' : '☁️'}
                      </span>

                      <div>
                        <div class="font-medium text-sm">{account.email}</div>
                        <div class="text-[10px] text-muted-foreground">
                          {authInProgress.value === account.name ? (
                            <span class="text-amber-400 animate-pulse">
                              🔐 Complete auth in VNC →
                            </span>
                          ) : account.configured ? (
                            account.mounted ? (
                              <span class="text-green-400">
                                ✓ Mounted at {account.mountpoint}
                              </span>
                            ) : (
                              <span class="text-blue-400">✓ Configured (not mounted)</span>
                            )
                          ) : (
                            <span class="text-muted-foreground">Not connected</span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div class="flex items-center gap-2">
                      {account.configured ? (
                        <>
                          {/* Browse button */}
                          <button
                            onClick$={() => browseFiles(account.name)}
                            class="text-[10px] px-2 py-1 rounded bg-muted/30 hover:bg-muted/50"
                          >
                            Browse
                          </button>

                          {/* Mount/Unmount toggle */}
                          <button
                            onClick$={() => toggleMount(account.name, account.mounted)}
                            class={`text-[10px] px-2 py-1 rounded ${
                              account.mounted
                                ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                                : 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30'
                            }`}
                          >
                            {account.mounted ? 'Unmount' : 'Mount'}
                          </button>
                        </>
                      ) : (
                        <button
                          onClick$={() => startAuth(account.name)}
                          disabled={authInProgress.value !== null}
                          class={`text-xs px-4 py-2 rounded font-medium transition-all ${
                            authInProgress.value === account.name
                              ? 'bg-amber-500/30 text-amber-300 animate-pulse shadow-[0_0_15px_rgba(245,158,11,0.3)]'
                              : authInProgress.value !== null
                                ? 'bg-muted/20 text-muted-foreground cursor-not-allowed'
                                : 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-cyan-300 hover:from-cyan-500/30 hover:to-purple-500/30 border border-cyan-500/30'
                          }`}
                        >
                          {authInProgress.value === account.name ? (
                            '🔐 Authenticating...'
                          ) : (
                            '🔗 Connect'
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* File browser section */}
          {browseAccount.value && (
            <div class="border-t border-border p-4 max-h-64 overflow-auto">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-semibold">
                  📂 {browseAccount.value} - Root
                </h3>
                <button
                  onClick$={() => browseAccount.value = null}
                  class="text-xs text-muted-foreground hover:text-foreground"
                >
                  Close
                </button>
              </div>

              <div class="rounded border border-border bg-black/20 overflow-hidden">
                <table class="w-full text-xs">
                  <thead class="bg-muted/30">
                    <tr>
                      <th class="text-left p-2">Name</th>
                      <th class="text-right p-2 w-20">Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {files.length === 0 ? (
                      <tr>
                        <td colSpan={2} class="p-4 text-center text-muted-foreground">
                          No files or loading...
                        </td>
                      </tr>
                    ) : (
                      files.map((file, i) => (
                        <tr key={i} class="border-t border-border/50 hover:bg-muted/10">
                          <td class="p-2">
                            <span class="mr-2">{file.isDir ? '📁' : '📄'}</span>
                            {file.name}
                          </td>
                          <td class="p-2 text-right text-muted-foreground">
                            {file.isDir ? '-' : formatSize(file.size)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Instructions footer */}
          <div class="p-3 border-t border-border bg-muted/10 text-[10px] text-muted-foreground">
            <strong>How it works:</strong> Click Connect → Auth in VNC panel →
            Sign in → Drive auto-mounts to <code>/mnt/gdrive/</code>
          </div>
        </div>

        {/* Embedded VNC Panel */}
        {showVNC.value && (
          <div class="flex-1 flex flex-col min-h-0 bg-black">
            {/* VNC Header */}
            <div class="flex items-center justify-between px-4 py-2 bg-gradient-to-r from-amber-500/20 to-orange-500/20 border-b border-amber-500/30">
              <div class="flex items-center gap-2">
                <span class="animate-pulse">🔐</span>
                <span class="text-sm font-medium text-amber-300">
                  VNC Browser - Complete OAuth for {status.accounts.find(a => a.name === authInProgress.value)?.email}
                </span>
              </div>
              <div class="flex items-center gap-2 text-xs text-amber-300/70">
                <span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                Live session
              </div>
            </div>

            {/* VNC iFrame */}
            <iframe
              src={noVNCUrl}
              class="flex-1 w-full border-0"
              title="VNC OAuth Browser"
            />

            {/* VNC Footer */}
            <div class="px-4 py-2 bg-muted/20 border-t border-border text-[10px] text-muted-foreground flex items-center justify-between">
              <span>
                Sign in with your Google account. Once complete, this panel will auto-close.
              </span>
              <button
                onClick$={() => {
                  showVNC.value = false;
                  authInProgress.value = null;
                }}
                class="px-2 py-1 rounded bg-muted/30 hover:bg-muted/50 text-foreground"
              >
                Cancel & Close
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export default CloudStoragePanel;
