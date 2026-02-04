#!/usr/bin/env bash
set -euo pipefail

# Headless-ish helper for `claude setup-token` using a pseudo-TTY (script).
# Usage:
#   HOME=/pluribus/.pluribus/agent_homes/claude /pluribus/pluribus_next/tools/providers/claude_setup_token.sh
# If a code is required, paste it when prompted (interactive), OR pipe it:
#   printf '%s\n' 'PASTE_CODE_HERE' | HOME=/pluribus/.pluribus/agent_homes/claude /pluribus/pluribus_next/tools/providers/claude_setup_token.sh

if ! command -v claude >/dev/null 2>&1; then
  echo "missing claude CLI (install: npm i -g @anthropic-ai/claude-code)" >&2
  exit 2
fi

if ! command -v script >/dev/null 2>&1; then
  echo "missing 'script' (util-linux) for pseudo-tty" >&2
  exit 2
fi

export HOME="${HOME:-/pluribus/.pluribus/agent_homes/claude}"
mkdir -p "$HOME" "$HOME/.config" "$HOME/.local/state" >/dev/null 2>&1 || true
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
export XDG_STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
export TERM="${TERM:-xterm-256color}"

# script allocates a PTY so Ink raw mode works.
script -q -c "claude setup-token" /dev/null

