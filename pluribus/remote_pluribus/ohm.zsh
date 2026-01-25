#!/bin/zsh
# Omega Heart Monitor Launcher (VPS Version)

# Canonical bus path for VPS
export PLURIBUS_BUS_DIR="/pluribus/.pluribus/bus"

echo "OHM - Omega Heart Monitor"
echo "Bus: $PLURIBUS_BUS_DIR"
echo ""

# Launch Python TUI (relative to /pluribus)
cd /pluribus
python3 nucleus/tools/ohm.py "$@"
