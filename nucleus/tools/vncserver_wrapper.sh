#!/usr/bin/env bash
set -euo pipefail

display_arg="${1:-:1}"
display_num="${display_arg#:}"
lock_file="/tmp/.X${display_num}-lock"

rfb_port="5901"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "-rfbport" ]] && (( i + 1 < ${#args[@]} )); then
    rfb_port="${args[$((i+1))]}"
  fi
done

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

vnc_process_running() {
  if ! command -v pgrep >/dev/null 2>&1; then
    return 1
  fi
  pgrep -f "Xtigervnc[[:space:]]+${display_arg}" >/dev/null 2>&1 && return 0
  pgrep -f "Xvnc[[:space:]]+${display_arg}" >/dev/null 2>&1 && return 0
  return 1
}

# If VNC is already running (process or port), exit successfully so PM2 can stop.
if vnc_process_running || port_listening "$rfb_port"; then
  echo "[novnc-vnc] VNC already running for ${display_arg} (port ${rfb_port}); not starting a duplicate." >&2
  exit 0
fi

# If a stale lock exists but no process/port is active, remove it so vncserver can start.
if [[ -e "$lock_file" ]]; then
  echo "[novnc-vnc] Removing stale lock ${lock_file} (no active VNC detected)." >&2
  rm -f "$lock_file" || true
fi

exec /usr/bin/vncserver "$@"

