#!/bin/bash
# ensure_tmux.sh - Create a 3-pane tmux session for Pluribus Dashboard

SESSION="pluribus_dash"
BUS_DIR="${PLURIBUS_BUS_DIR:-.pluribus/bus}"
EVENTS_FILE="${BUS_DIR%/}/events.ndjson"

# Check if session exists
tmux has-session -t "$SESSION" 2>/dev/null

if [ $? != 0 ]; then
  # Create new session
  tmux new-session -d -s "$SESSION"
  
  # Split into 3 panes side-by-side
  tmux split-window -h -t "$SESSION:0"
  tmux split-window -h -t "$SESSION:0"
  
  # Layout: even-horizontal (3 columns)
  tmux select-layout -t "$SESSION:0" even-horizontal
  
  # Optional: Start some default tools in panes
  tmux send-keys -t "$SESSION:0.0" "htop" C-m
  tmux send-keys -t "$SESSION:0.1" "tail -f \"$EVENTS_FILE\"" C-m
  tmux send-keys -t "$SESSION:0.2" "python3 /pluribus/pluribus_next/tools/supermotd.py --follow --limit 60" C-m
fi

echo "Tmux session '$SESSION' ready."
