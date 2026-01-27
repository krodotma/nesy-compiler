#!/usr/bin/env bash
set -euo pipefail

AGENT_NAME="${1:-unknown}"

export PLURIBUS_BUS_DIR="${PLURIBUS_BUS_DIR:-/pluribus/.pluribus/bus}"
export PLURIBUS_ACTOR="${PLURIBUS_ACTOR:-$AGENT_NAME}"
export PLURIBUS_SESSION="${PLURIBUS_SESSION:-new}"
export PLURIBUS_CELL="${PLURIBUS_CELL:-dia.1.0}"
export PLURIBUS_LANE="${PLURIBUS_LANE:-dialogos}"
export PLURIBUS_DEPTH="${PLURIBUS_DEPTH:-0}"
export PLURIBUS_PROTOCOL="${PLURIBUS_PROTOCOL:-PLURIBUS v1}"
export PLURIBUS_BOOTSTRAP_FILES="${PLURIBUS_BOOTSTRAP_FILES:-/pluribus/AGENTS.md:/pluribus/nucleus/AGENTS.md:/pluribus/nucleus/specs/pluribus_protocol_v1.md:/pluribus/nucleus/specs/repl_header_contract_v1.md}"
export PYTHONPATH="/pluribus:${PYTHONPATH:-}"
export UNIFORM_PANEL_STYLE="${UNIFORM_PANEL_STYLE:-tablet}"
export PLURIBUS_SKILLS_FALLBACK_CMD="${PLURIBUS_SKILLS_FALLBACK_CMD:-python3 /pluribus/nucleus/tools/skills_scanner.py --invoke}"

LOAD_VERTEX_ENV="${PLURIBUS_LOAD_VERTEX_ENV:-1}"
if [ "$AGENT_NAME" = "gemini" ]; then
  # Keep Gemini CLI OAuth-only by default.
  LOAD_VERTEX_ENV=0
fi

if [ "$LOAD_VERTEX_ENV" = "1" ] && [ -f "/pluribus/.pluribus/vertex_env.conf" ]; then
  set -a
  # shellcheck disable=SC1091
  . "/pluribus/.pluribus/vertex_env.conf"
  set +a
fi

MINDLIKE_ENV_FILE="${PLURIBUS_MINDLIKE_ENV:-/pluribus/.pluribus/mindlike_env.conf}"
if [ -f "$MINDLIKE_ENV_FILE" ]; then
  _prev_mindlike="${MINDLIKE_API_KEY-}"
  _prev_glm="${GLM_API_KEY-}"
  _prev_mindlike_base="${MINDLIKE_BASE_URL-}"
  set -a
  # shellcheck disable=SC1091
  . "$MINDLIKE_ENV_FILE"
  set +a
  if [ -n "$_prev_mindlike" ]; then
    export MINDLIKE_API_KEY="$_prev_mindlike"
  fi
  if [ -n "$_prev_glm" ]; then
    export GLM_API_KEY="$_prev_glm"
  fi
  if [ -n "$_prev_mindlike_base" ]; then
    export MINDLIKE_BASE_URL="$_prev_mindlike_base"
  fi
  unset _prev_mindlike _prev_glm _prev_mindlike_base
fi

if [ "${PLURIBUS_SUPPRESS_HEADER:-0}" != "1" ]; then
  if [ -f "/pluribus/nucleus/tools/agent_header.py" ]; then
    python3 /pluribus/nucleus/tools/agent_header.py "$PLURIBUS_ACTOR"
  fi
fi
