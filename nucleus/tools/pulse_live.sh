#!/usr/bin/env bash
set -euo pipefail

# Live Pulse overlay (non-tmux). Renders a 12-line pane at top using pulse_prompt.py.
# Opt-out: export NO_PULSE_LIVE=1 or run under editors (detected heuristically).

if [[ -n "${NO_PULSE_LIVE:-}" ]]; then
  exit 0
fi

# Avoid disturbing full-screen programs.
if [[ -n "${INSIDE_EMACS:-}" ]] || [[ -n "${VIM:-}" ]] || [[ -n "${NVIM:-}" ]] || [[ -n "${TMUX:-}" ]]; then
  exit 0
fi

ROOT="/pluribus"
PULSE_PY="$ROOT/pluribus_next/tools/pulse_prompt.py"
INTERVAL="${PULSE_LIVE_INTERVAL:-2}"

print_panel() {
  # Save cursor
  printf '\033[s'
  # Go to top-left
  printf '\033[H'
  # Print panel
  python3 "$PULSE_PY" 2>/dev/null || true
  # Restore cursor
  printf '\033[u'
}

cleanup() {
  # Clear top pane on exit (optional)
  printf '\033[s\033[H\033[J\033[u'
}

trap cleanup EXIT INT TERM

while true; do
  print_panel
  sleep "$INTERVAL"
done
