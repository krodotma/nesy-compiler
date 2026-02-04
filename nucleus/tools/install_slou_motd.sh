#!/bin/bash
# Install the SLOU MOTD as the user's login greeting
# Adds the python script execution to .zshrc if not present

MOTD_CMD="python3 /pluribus/pluribus_next/tools/slou_motd.py --stream --plain"
ZSHRC="$HOME/.zshrc"

if ! grep -q "slou_motd.py" "$ZSHRC"; then
    echo "" >> "$ZSHRC"
    echo "# Pluribus SuperMOTD (SLOU)" >> "$ZSHRC"
    echo "$MOTD_CMD" >> "$ZSHRC"
    echo "Installed SLOU MOTD to $ZSHRC"
else
    echo "SLOU MOTD already installed in $ZSHRC"
fi
