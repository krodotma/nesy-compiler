#!/bin/bash
# iso_wrapper.sh
# Shim to redirect git commands to iso_git.mjs in sequestered environments.
# Usage: git <command> [args]

set -e

ISO_GIT="node /pluribus/nucleus/tools/iso_git.mjs"

if [[ "$1" == "status" ]]; then
  $ISO_GIT status .
elif [[ "$1" == "commit" ]]; then
  # Basic shim for 'git commit -m "message"'
  if [[ "$2" == "-m" ]] && [[ -n "$3" ]]; then
    # We use commit-paths . for safety if paths aren't specified, 
    # but standard 'commit' stages everything in iso_git.
    # We'll map to 'commit .'
    $ISO_GIT commit . "$3"
  else
    echo "iso_wrapper: Unsupported commit syntax. Use: git commit -m 'message'"
    exit 1
  fi
elif [[ "$1" == "log" ]]; then
  $ISO_GIT log .
else
  # Pass through to native git for read-only/boundary ops (like diff, show) if needed,
  # or block if strict isolation is required. 
  # For Phase 3, we allow fallback for non-mutating commands.
  /usr/bin/git "$@"
fi
