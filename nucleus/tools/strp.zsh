#!/usr/bin/env zsh
# STRp Launcher: The Single Point of Entry for the Pluribus Monitor
# Usage: ./strp.zsh [options]

set -e

# --- Configuration & Defaults ---
SCRIPT_DIR=${0:a:h}
REPO_ROOT=${SCRIPT_DIR:h:h} # ../.. from tools/
DEFAULT_BUS_DIR="${REPO_ROOT}/.pluribus/bus"
PYTHON_CMD="python3"
TOOL_MONITOR="${SCRIPT_DIR}/strp_monitor.py"
TOOL_MESH="${SCRIPT_DIR}/mesh_status.py"
TOOL_WORKER="${SCRIPT_DIR}/strp_worker.py"

# --- Colors ---
autoload -U colors && colors
msg_info() { echo "${fg[green]}➜${reset_color} $1"; }
msg_warn() { echo "${fg[yellow]}⚠${reset_color} $1"; }
msg_err() { echo "${fg[red]}✖${reset_color} $1"; }
msg_dim() { echo "${fg[grey]}$1${reset_color}"; }

# --- Help ---
show_help() {
    cat <<EOF
Usage: strp.zsh [OPTIONS]

The "Stir the Pot" (STRp) Launcher.
Prepares the environment, warms up the bus, and launches the TUI.

Options:
  --clean           Kill any existing strp_worker.py or strp_curation_loop.py processes before starting.
  --spawn-worker    Launch a detached background worker (gemini-cli provider) immediately.
  --bus-dir DIR     Override the bus directory (default: .pluribus/bus).
  --no-pipx         Force using system python3 instead of pipx (requires 'pip install textual' manually).
  --dry-run         Print the launch command but do not execute.
  --help            Show this message.

Default Mode:
  - Detects repo root: ${REPO_ROOT}
  - Sets bus dir: ${DEFAULT_BUS_DIR}
  - Emits fresh mesh status to populate the UI.
  - Launches the Monitor via pipx.
EOF
}

# --- Argument Parsing ---
CLEAN_MODE=0
SPAWN_WORKER=0
USE_PIPX=1
DRY_RUN=0
BUS_DIR=$DEFAULT_BUS_DIR

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --clean) CLEAN_MODE=1 ;;
        --spawn-worker) SPAWN_WORKER=1 ;;
        --bus-dir) BUS_DIR="$2"; shift ;;
        --no-pipx) USE_PIPX=0 ;;
        --dry-run) DRY_RUN=1 ;;
        --help) show_help; exit 0 ;;
        *) msg_err "Unknown option: $1"; show_help; exit 1 ;;
    esac
    shift
done

# --- Prerequisite Checks ---
if [[ $USE_PIPX -eq 1 ]] && ! command -v pipx &> /dev/null; then
    msg_warn "pipx not found. Falling back to python3 (ensure 'textual' is installed)."
    USE_PIPX=0
fi

if [[ ! -f "$TOOL_MONITOR" ]]; then
    msg_err "Monitor tool not found at $TOOL_MONITOR"
    exit 1
fi

# --- Execution Logic ---

# 1. Environment Cleanup / Prep
if [[ $CLEAN_MODE -eq 1 ]]; then
    msg_info "Cleaning up stale STRp processes..."
    pkill -f "strp_worker.py" || true
    pkill -f "strp_curation_loop.py" || true
fi

# 2. Bus Warmup
msg_info "Warming up: Emitting fresh mesh status..."
export PLURIBUS_BUS_DIR="$BUS_DIR"
if [[ $DRY_RUN -eq 0 ]]; then
    # Ensure bus dir exists
    mkdir -p "$BUS_DIR"
    # Run mesh_status to emit the initial state so the TUI isn't empty
    # We ignore the exit code because 2 means "partial success" (missing optional tools), which is fine for warmup.
    $PYTHON_CMD "$TOOL_MESH" --emit-bus --root "$REPO_ROOT" --timeout 5 > /dev/null 2>&1 || true
else
    msg_dim "[Dry Run] Would run: $PYTHON_CMD $TOOL_MESH --emit-bus --root $REPO_ROOT"
fi

# 3. Worker Spawning
if [[ $SPAWN_WORKER -eq 1 ]]; then
    msg_info "Spawning background worker..."
    if [[ $DRY_RUN -eq 0 ]]; then
        nohup $PYTHON_CMD "$TOOL_WORKER" --bus-dir "$BUS_DIR" --provider "gemini-cli" > /dev/null 2>&1 &
        msg_dim "Worker spawned (PID $!)"
    else
        msg_dim "[Dry Run] Would run: $PYTHON_CMD $TOOL_WORKER --bus-dir $BUS_DIR &"
    fi
fi

# 4. Launch TUI
# We invoke python3 explicitly inside the pipx environment to ensure dependencies are loaded
# and to avoid "script is already on path" warnings.
CMD_ARGS=("--spec" "textual" "--" "python3" "$TOOL_MONITOR" "--root" "$REPO_ROOT")
if [[ $USE_PIPX -eq 0 ]]; then
    # Direct python invocation
    FINAL_CMD=("$PYTHON_CMD" "$TOOL_MONITOR" "--root" "$REPO_ROOT")
else
    # Pipx invocation
    FINAL_CMD=("pipx" "run" "${CMD_ARGS[@]}")
fi

msg_info "Launching STRp Monitor..."
if [[ $DRY_RUN -eq 1 ]]; then
    echo "${FINAL_CMD[@]}"
else
    exec "${FINAL_CMD[@]}"
fi
