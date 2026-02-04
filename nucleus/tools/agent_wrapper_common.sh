#!/usr/bin/env bash
set -euo pipefail

wrapper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

plu_resolve_bus_dir() {
  local bus_dir="$1"
  local bus_events_path="$bus_dir/events.ndjson"
  if ! (
    mkdir -p "$bus_dir" >/dev/null 2>&1 \
      && PYTHONDONTWRITEBYTECODE=1 python3 - "$bus_events_path" >/dev/null 2>&1 <<'PY'
import os, sys
path = sys.argv[1]
os.makedirs(os.path.dirname(path), exist_ok=True)
fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
os.close(fd)
PY
  ); then
    bus_dir="${PLURIBUS_FALLBACK_BUS_DIR:-/pluribus/.pluribus_local/bus}"
    mkdir -p "$bus_dir" >/dev/null 2>&1 || true
  fi
  printf '%s' "$bus_dir"
}

plu_emit_nexus_ack() {
  local agent="$1"
  local bus_dir="$2"
  PYTHONDONTWRITEBYTECODE=1 python3 "$wrapper_dir/nexus_ack.py" --agent "$agent" --bus-dir "$bus_dir" >/dev/null 2>&1 || true
}

plu_emit_session_bootstrap() {
  local bus_dir="$1"
  PYTHONDONTWRITEBYTECODE=1 python3 "$wrapper_dir/session_bootstrap.py" --root /pluribus --bus-dir "$bus_dir" >/dev/null 2>&1 || true
}

plu_prepare_agent_home() {
  local agent="$1"
  local env_var="$2"
  local agent_home="${!env_var:-/pluribus/.pluribus/agent_homes/$agent}"
  if ! mkdir -p "$agent_home" "$agent_home/.config" "$agent_home/.local/state" >/dev/null 2>&1; then
    agent_home="/pluribus/.pluribus_local/agent_homes/$agent"
    mkdir -p "$agent_home" "$agent_home/.config" "$agent_home/.local/state" >/dev/null 2>&1 || true
  fi
  printf '%s' "$agent_home"
}

plu_set_xdg() {
  local agent_home="$1"
  export HOME="$agent_home"
  export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$agent_home/.config}"
  export XDG_STATE_HOME="${XDG_STATE_HOME:-$agent_home/.local/state}"
}

plu_load_secrets() {
  local orig_home="$1"
  local agent_home="$2"
  local secrets_paths=()
  if [[ -n "$orig_home" ]]; then
    secrets_paths+=("$orig_home/.config/nucleus/secrets.env")
    secrets_paths+=("$orig_home/.config/pluribus_next/secrets.env")
  fi
  secrets_paths+=("$agent_home/.config/nucleus/secrets.env")
  secrets_paths+=("$agent_home/.config/pluribus_next/secrets.env")
  for path in "${secrets_paths[@]}"; do
    if [[ -r "$path" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "$path"
      set +a
    fi
  done
  if [[ -r "$wrapper_dir/../.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$wrapper_dir/../.env"
    set +a
  fi
  if [[ -z "${GOOGLE_API_KEY:-}" ]] && [[ -n "${GEMINI_API_KEY:-}" ]]; then
    export GOOGLE_API_KEY="$GEMINI_API_KEY"
  elif [[ -z "${GEMINI_API_KEY:-}" ]] && [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    export GEMINI_API_KEY="$GOOGLE_API_KEY"
  fi
}

plu_load_assimilation_prompt() {
  local agent="$1"
  local prompt_file=""
  local prompt_text=""
  local nexus_file="/pluribus/nexus_bridge/${agent}.md"
  local fallback_file="/pluribus/nucleus/docs/workflows/AGENT_ASSIMILATION_PROMPT.md"
  local legacy_file="/pluribus/pluribus_next/docs/workflows/AGENT_ASSIMILATION_PROMPT.md"
  if [[ -f "$nexus_file" ]]; then
    prompt_file="$nexus_file"
  elif [[ -f "$fallback_file" ]]; then
    prompt_file="$fallback_file"
  elif [[ -f "$legacy_file" ]]; then
    prompt_file="$legacy_file"
  fi
  if [[ -n "$prompt_file" ]]; then
    prompt_text="$(cat "$prompt_file")"
  fi
  export PLURIBUS_ASSIMILATION_PROMPT_FILE="$prompt_file"
  export PLURIBUS_ASSIMILATION_PROMPT="$prompt_text"
}

plu_detect_prompt_flag() {
  local cli="$1"
  local help_text=""
  help_text="$("$cli" --help 2>/dev/null || true)"
  if [[ "$help_text" == *"--append-system-prompt"* ]]; then
    printf '%s' "--append-system-prompt"
    return 0
  fi
  if [[ "$help_text" == *"--system-prompt"* ]]; then
    printf '%s' "--system-prompt"
    return 0
  fi
  if [[ "$help_text" == *"--system"* ]]; then
    printf '%s' "--system"
    return 0
  fi
  if [[ "$help_text" == *"--instruction"* ]]; then
    printf '%s' "--instruction"
    return 0
  fi
  return 1
}
