#!/bin/zsh
# Omega Heart Monitor Launcher

# Ensure canonical environment
export PLURIBUS_BUS_DIR="pluribus/remote_pluribus/.pluribus/bus"

echo "Using Bus: $PLURIBUS_BUS_DIR"

# Launch Python TUI
python3 nucleus/tools/ohm.py
