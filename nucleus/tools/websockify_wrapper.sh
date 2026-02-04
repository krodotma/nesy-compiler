#!/usr/bin/env bash
# Wrapper script for websockify to work with PM2.
# If something else already owns the port (e.g. systemd), exit 0 so PM2 can stop.
set -euo pipefail

port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk 'NR>1 {print $4}' | grep -Eq "(:|\\])${port}$"
    return $?
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -tln 2>/dev/null | awk 'NR>2 {print $4}' | grep -Eq "(:|\\])${port}$"
    return $?
  fi
  return 1
}

if port_listening "${WEBSOCKIFY_PORT:-6080}"; then
  echo "[novnc-websockify] Port ${WEBSOCKIFY_PORT:-6080} already in use; not starting a duplicate." >&2
  exit 0
fi

exec /usr/bin/python3 /usr/bin/websockify --web=/usr/share/novnc --wrap-mode=ignore 6080 localhost:5901
