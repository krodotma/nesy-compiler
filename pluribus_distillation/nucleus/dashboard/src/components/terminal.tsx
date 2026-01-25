import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { startComponentTiming, timeAsync } from '../lib/telemetry/load-timing';

interface TerminalProps {
  /** WebSocket endpoint path (e.g., '/terminal' or '/plurichat') */
  endpoint?: string;
  /** Display title */
  title?: string;
  /** Initial command to send after connection */
  initCommand?: string;
}

export const Terminal = component$<TerminalProps>(({
  endpoint = '/terminal',
  title = 'WebSSH (Tmux)',
  initCommand
}) => {
  const terminalRef = useSignal<Element>();
  const pinInput = useSignal('');
  const isAuthenticated = useSignal(false);
  const status = useSignal('Disconnected');
  const dimensions = useSignal({ cols: 80, rows: 24 });

  // Load xterm.js and FitAddon from CDN, then initialize terminal
  useVisibleTask$(({ track, cleanup }) => {
    track(() => isAuthenticated.value);

    let ws: WebSocket | null = null;
    let term: any = null;
    let fitAddon: any = null;
    let resizeObserver: ResizeObserver | null = null;

    if (isAuthenticated.value && terminalRef.value) {
      const initTerminal = async () => {
        const stopTermTiming = startComponentTiming('Terminal');

        // Prefer local bundled deps (no CDN reliance for cloudless / offline-friendly UX).
        // Load xterm CSS dynamically to avoid polluting initial bundle
        const [{ Terminal }, { FitAddon }] = await timeAsync('xterm-bundle', () => Promise.all([
          import('xterm'),
          import('xterm-addon-fit'),
          import('xterm/css/xterm.css'),
        ]));

        // Phase 3 Step 17: Inherit terminal colors from Chroma palette
        const rootStyle = getComputedStyle(document.documentElement);
        const getVar = (name: string, fallback: string) => rootStyle.getPropertyValue(name).trim() || fallback;
        
        // Create terminal with adaptive sizing
        term = new Terminal({
          fontFamily: '"JetBrains Mono", "Fira Code", "SF Mono", Menlo, Monaco, "Courier New", monospace',
          fontSize: 14,
          lineHeight: 1.2,
          cursorBlink: true,
          cursorStyle: 'block',
          scrollback: 5000,
          theme: {
            background: '#0d1117',
            foreground: getVar('--mat-text', '#c9d1d9'),
            cursor: getVar('--mat-primary', '#58a6ff'),
            cursorAccent: '#0d1117',
            selectionBackground: 'rgba(255, 255, 255, 0.1)',
            black: '#0d1117',
            red: getVar('--mat-primary', '#ff7b72'), // Color code based on chroma
            green: getVar('--mat-secondary', '#3fb950'),
            yellow: '#d29922',
            blue: '#58a6ff',
            magenta: '#bc8cff',
            cyan: '#39c5cf',
            white: '#b1bac4',
            brightBlack: '#6e7681',
            brightRed: '#ffa198',
            brightGreen: '#56d364',
            brightYellow: '#e3b341',
            brightBlue: '#79c0ff',
            brightMagenta: '#d2a8ff',
            brightCyan: '#56d4dd',
            brightWhite: '#f0f6fc',
          }
        });

        // Initialize FitAddon if available
        if (FitAddon) {
          fitAddon = new FitAddon();
          term.loadAddon(fitAddon);
        }

        term.open(terminalRef.value as HTMLElement);

        // Initial fit
        if (fitAddon) {
          setTimeout(() => {
            fitAddon.fit();
            dimensions.value = { cols: term.cols, rows: term.rows };
            // Ensure tmux session is resized even if the container never triggers a ResizeObserver event.
            if (ws?.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
            }
          }, 100);
        }

        // Use ResizeObserver to handle container size changes
        resizeObserver = new ResizeObserver(() => {
          if (fitAddon && term) {
            try {
              fitAddon.fit();
              dimensions.value = { cols: term.cols, rows: term.rows };
              // Send resize signal to PTY if connected
              if (ws?.readyState === WebSocket.OPEN) {
                // Send SIGWINCH-style resize (custom protocol)
                ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
              }
            } catch (e) {
              // Ignore resize errors during cleanup
            }
          }
        });
        resizeObserver.observe(terminalRef.value as Element);

        // Connect to WebSocket
        status.value = 'Connecting...';
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const primaryUrl = `${protocol}//${host}${endpoint}?pin=${pinInput.value}`;
        // Fallback only for localhost dev (port 9200 not exposed externally)
        const fallbackUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
          ? `ws://localhost:9200${endpoint}?pin=${pinInput.value}`
          : primaryUrl; // No fallback for remote hosts
        ws = new WebSocket(primaryUrl);

        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
          status.value = 'Connected';
          stopTermTiming(); // Mark terminal fully initialized
          term.write('\r\n\x1b[32m[Pluribus Terminal] Connected.\x1b[0m\r\n');

          // Send initial resize
          if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
          }

          // Send initial command if provided
          if (initCommand && ws?.readyState === WebSocket.OPEN) {
            setTimeout(() => {
              ws!.send(initCommand + '\n');
            }, 500);
          }
        };

        ws.onmessage = (ev) => {
          if (typeof ev.data === 'string') {
            term.write(ev.data);
          } else {
            term.write(new Uint8Array(ev.data));
          }
        };

        ws.onclose = () => {
          status.value = 'Disconnected';
          term.write('\r\n\x1b[31m[Connection Closed]\x1b[0m\r\n');
        };

        ws.onerror = (err) => {
          // If reverse-proxy websocket isn't wired (or port is blocked), fall back to direct bridge port.
          try {
            if (ws && ws.url === primaryUrl) {
              ws.close();
              ws = new WebSocket(fallbackUrl);
              ws.binaryType = 'arraybuffer';
              ws.onopen = () => {
                status.value = 'Connected';
                term.write('\r\n\x1b[33m[Fallback] Connected via :9200 bridge.\x1b[0m\r\n');
                ws!.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
                if (initCommand) {
                  setTimeout(() => ws!.send(initCommand + '\n'), 500);
                }
              };
              ws.onmessage = (ev) => {
                if (typeof ev.data === 'string') term.write(ev.data);
                else term.write(new Uint8Array(ev.data));
              };
              ws.onclose = () => {
                status.value = 'Disconnected';
                term.write('\r\n\x1b[31m[Connection Closed]\x1b[0m\r\n');
              };
              ws.onerror = () => {
                status.value = 'Error';
                term.write('\r\n\x1b[31m[Connection Error]\x1b[0m\r\n');
              };
              return;
            }
          } catch {
            // fall through to error UI
          }
          status.value = 'Error';
          term.write('\r\n\x1b[31m[Connection Error]\x1b[0m\r\n');
        };

        term.onData((data: string) => {
          if (ws?.readyState === WebSocket.OPEN) {
            ws.send(data);
          }
        });
      };

      initTerminal();
    }

    // Cleanup on unmount
    cleanup(() => {
      if (resizeObserver) resizeObserver.disconnect();
      if (ws) ws.close();
      if (term) term.dispose();
    });
  });

  return (
    <div class="h-full w-full bg-[#0c0c0e] flex flex-col term-crt overflow-hidden rounded-lg border border-[var(--glass-border)] shadow-2xl">
      {/* Header / Tabs */}
      <div class="flex items-center justify-between px-4 py-2 border-b border-[var(--glass-border)] bg-white/5 backdrop-blur">
        <div class="flex items-center gap-3">
          <div class="flex gap-1.5">
            <span class="w-3 h-3 rounded-full bg-red-500"></span>
            <span class="w-3 h-3 rounded-full bg-yellow-500"></span>
            <span class="w-3 h-3 rounded-full bg-green-500"></span>
          </div>
          <span class="text-green-400 font-mono text-sm font-medium">{title}</span>
          <span class={`text-xs px-2 py-0.5 rounded ${
            status.value === 'Connected' ? 'bg-green-900/50 text-green-400' :
            status.value === 'Connecting...' ? 'bg-yellow-900/50 text-yellow-400' :
            'bg-red-900/50 text-red-400'
          }`}>
            {status.value}
          </span>
          <span class="text-xs text-gray-500 font-mono">
            {dimensions.value.cols}×{dimensions.value.rows}
          </span>
        </div>

        {!isAuthenticated.value && (
          <div class="flex gap-2 items-center">
            <input
              type="password"
              placeholder="PIN"
              class="w-20 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs font-mono focus:border-green-500 focus:outline-none"
              bind:value={pinInput}
              onKeyDown$={(e) => {
                if (e.key === 'Enter' && pinInput.value === '0912') isAuthenticated.value = true;
              }}
            />
            <button
              onClick$={() => { if (pinInput.value === '0912') isAuthenticated.value = true; }}
              class="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-xs font-medium transition-colors"
            >
              Connect
            </button>
          </div>
        )}
      </div>

      {/* Terminal area - fills remaining space */}
      {isAuthenticated.value ? (
        <div
          ref={terminalRef}
          class="w-full p-1 overflow-hidden"
          style={{ flex: '1 1 0%', minHeight: '300px', height: '100%' }}
        />
      ) : (
        <div class="flex-1 flex items-center justify-center text-gray-500 bg-[#0d1117]">
          <div class="text-center">
            <div class="text-6xl mb-4">🔐</div>
            <div class="text-lg font-medium mb-2">Secure Terminal</div>
            <div class="text-sm text-gray-600">Enter PIN to connect</div>
          </div>
        </div>
      )}
    </div>
  );
});

/**
 * PluriChat Terminal - Connects to a persistent tmux session running plurichat
 */
export const PluriChatTerminal = component$(() => {
  return (
    <Terminal
      endpoint="/plurichat"
      title="PluriChat (Multi-Model Router)"
      initCommand="/status"
    />
  );
});
