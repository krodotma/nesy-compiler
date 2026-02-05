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

plu_load_bus_policy() {
  local agent_home="$1"
  local policy_paths=()
  policy_paths+=("/pluribus/.pluribus/config/bus_policy.env")
  if [[ -n "$agent_home" ]]; then
    policy_paths+=("$agent_home/.config/pluribus/bus_policy.env")
    policy_paths+=("$agent_home/.config/nucleus/bus_policy.env")
  fi
  for path in "${policy_paths[@]}"; do
    if [[ -r "$path" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "$path"
      set +a
    fi
  done
  if [[ -z "${PLURIBUS_BUS_BACKEND:-}" ]]; then
    export PLURIBUS_BUS_BACKEND="falkordb"
  fi
  if [[ -z "${PLURIBUS_NDJSON_MODE:-}" ]]; then
    export PLURIBUS_NDJSON_MODE="dr"
  fi
}

plu_set_xdg() {
  local agent_home="$1"
  export HOME="$agent_home"
  export XDG_CONFIG_HOME="$agent_home/.config"
  export XDG_STATE_HOME="$agent_home/.local/state"
  plu_load_bus_policy "$agent_home"
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

  # Load Agent Wrapper Keys (Seed/Manual)
  local wrapper_keys_file="${XDG_CONFIG_HOME:-$HOME/.config}/agent-wrapper/keys"
  if [[ -r "$wrapper_keys_file" ]]; then
    local first_key=""
    while IFS= read -r line; do
      [[ -n "$line" ]] || continue
      [[ "$line" == \#* ]] && continue
      first_key="$line"
      break
    done < "$wrapper_keys_file"
    if [[ -n "$first_key" ]]; then
      export MINDLIKE_API_KEY="$first_key"
      export GLM_API_KEY="$first_key"
      export CODEX_API_KEY="$first_key"
      export OPENAI_API_KEY="$first_key"
    fi
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

# =============================================================================
# EXPONENTIAL BACKOFF WITH RETRY
# =============================================================================
#
# Usage:
#   plu_retry [--max-attempts N] [--initial-delay S] [--max-delay S] [--jitter] -- command [args...]
#
# Environment variables:
#   PLURIBUS_RETRY_MAX_ATTEMPTS (default: 3)
#   PLURIBUS_RETRY_INITIAL_DELAY (default: 1.0 seconds)
#   PLURIBUS_RETRY_MAX_DELAY (default: 30.0 seconds)
#   PLURIBUS_RETRY_JITTER (default: 1 = enabled)
#
# Exit codes for which retry will NOT be attempted (permanent failures):
#   1   - General error (may retry)
#   2   - Misuse of shell command (won't retry)
#   126 - Permission denied (won't retry)
#   127 - Command not found (won't retry)
#   128+ - Fatal signal (won't retry)
#
plu_retry() {
  local max_attempts="${PLURIBUS_RETRY_MAX_ATTEMPTS:-3}"
  local initial_delay="${PLURIBUS_RETRY_INITIAL_DELAY:-1.0}"
  local max_delay="${PLURIBUS_RETRY_MAX_DELAY:-30.0}"
  local jitter="${PLURIBUS_RETRY_JITTER:-1}"
  local cmd=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --max-attempts) max_attempts="$2"; shift 2 ;;
      --initial-delay) initial_delay="$2"; shift 2 ;;
      --max-delay) max_delay="$2"; shift 2 ;;
      --jitter) jitter="1"; shift ;;
      --no-jitter) jitter="0"; shift ;;
      --) shift; cmd=("$@"); break ;;
      *) cmd=("$@"); break ;;
    esac
  done

  if [[ ${#cmd[@]} -eq 0 ]]; then
    echo "plu_retry: no command specified" >&2
    return 2
  fi

  local attempt=0
  local exit_code=0
  local delay="$initial_delay"

  while [[ $attempt -lt $max_attempts ]]; do
    attempt=$((attempt + 1))

    # Execute command
    set +e
    "${cmd[@]}"
    exit_code=$?
    set -e

    # Success - done
    if [[ $exit_code -eq 0 ]]; then
      return 0
    fi

    # Non-retryable exit codes
    case $exit_code in
      2|126|127)
        return $exit_code
        ;;
      *)
        if [[ $exit_code -ge 128 ]]; then
          return $exit_code
        fi
        ;;
    esac

    # Last attempt failed - return error
    if [[ $attempt -ge $max_attempts ]]; then
      return $exit_code
    fi

    # Calculate delay with optional jitter
    local sleep_time="$delay"
    if [[ "$jitter" == "1" ]]; then
      sleep_time="$(PYTHONDONTWRITEBYTECODE=1 python3 - "$delay" <<'PY'
import random, sys
base = float(sys.argv[1])
# Add ±20% jitter
jittered = base * (0.8 + random.random() * 0.4)
print(f"{jittered:.3f}")
PY
)"
    fi

    echo "plu_retry: attempt $attempt/$max_attempts failed (exit $exit_code), retrying in ${sleep_time}s..." >&2
    sleep "$sleep_time"

    # Exponential backoff: double the delay, capped at max
    delay="$(PYTHONDONTWRITEBYTECODE=1 python3 - "$delay" "$max_delay" <<'PY'
import sys
current = float(sys.argv[1])
max_d = float(sys.argv[2])
print(min(current * 2, max_d))
PY
)"
  done

  return $exit_code
}

# Convenience function for retrying bus writes specifically
plu_retry_bus_write() {
  PLURIBUS_RETRY_MAX_ATTEMPTS="${PLURIBUS_RETRY_MAX_ATTEMPTS:-5}" \
  PLURIBUS_RETRY_INITIAL_DELAY="${PLURIBUS_RETRY_INITIAL_DELAY:-0.5}" \
  PLURIBUS_RETRY_MAX_DELAY="${PLURIBUS_RETRY_MAX_DELAY:-10.0}" \
  plu_retry "$@"
}
