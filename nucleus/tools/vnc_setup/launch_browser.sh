#!/bin/bash
# Launch browser in VNC session for OAuth authentication
# Usage: bash /pluribus/pluribus_next/tools/vnc_setup/launch_browser.sh [url]

export DISPLAY=:1

URL="${1:-about:blank}"

# Common OAuth URLs for reference:
# ChatGPT: https://chat.openai.com
# Claude: https://claude.ai
# Gemini: https://gemini.google.com

# Try Firefox first, then Chromium
if command -v firefox &> /dev/null; then
    firefox "$URL" &
    echo "Firefox launched"
elif command -v chromium-browser &> /dev/null; then
    chromium-browser --no-sandbox "$URL" &
    echo "Chromium launched"
elif command -v chromium &> /dev/null; then
    chromium --no-sandbox "$URL" &
    echo "Chromium launched"
else
    echo "No browser found. Install firefox or chromium-browser."
    exit 1
fi

echo "Browser launched on DISPLAY=:1"
echo ""
echo "Connect via VNC client to: 69.169.104.17:5901"
PASSFILE="/pluribus/.pluribus/vnc_password.txt"
if [ -f "$PASSFILE" ]; then
    echo "Password (from $PASSFILE): $(cat "$PASSFILE")"
else
    echo "Password file not found: $PASSFILE"
fi
