#!/usr/bin/env bash
set -euo pipefail

# install_local_llm.sh
# ====================
# Installs Ollama (local LLM server) and Crush/Mods (TUI chat tools)
# for SUPERWORKER context-aware local inference.

echo "Pluribus Local LLM Installer"
echo "============================="

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Detected: $OS / $ARCH"

# 1. Install Ollama
echo ""
echo "[1/3] Installing Ollama..."
if command -v ollama &> /dev/null; then
    echo "      Ollama already installed: $(ollama --version)"
else
    if [[ "$OS" == "Darwin" ]]; then
        # macOS - use brew if available, else curl
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    elif [[ "$OS" == "Linux" ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi
fi

# 2. Pull default model (llama3.2)
echo ""
echo "[2/3] Pulling default model (llama3.2)..."
if ollama list 2>/dev/null | grep -q "llama3.2"; then
    echo "      llama3.2 already available"
else
    # Start ollama server in background if not running
    if ! pgrep -x ollama &> /dev/null; then
        echo "      Starting ollama server..."
        nohup ollama serve &> /tmp/ollama.log &
        sleep 3
    fi
    ollama pull llama3.2
fi

# 3. Install Crush/Mods (Charmbracelet TUI chat)
echo ""
echo "[3/3] Installing Crush/Mods..."
if command -v mods &> /dev/null; then
    echo "      Mods already installed: $(mods --version 2>/dev/null || echo 'installed')"
else
    if command -v go &> /dev/null; then
        go install github.com/charmbracelet/mods@latest
    elif command -v brew &> /dev/null; then
        brew install mods
    else
        echo "      WARN: Neither Go nor Brew available. Install mods manually:"
        echo "            brew install mods  OR  go install github.com/charmbracelet/mods@latest"
    fi
fi

# 4. Configure Mods to use Ollama by default
echo ""
echo "[+] Configuring Mods for Ollama..."
MODS_CONFIG="${HOME}/.config/mods/mods.yml"
if [[ ! -f "$MODS_CONFIG" ]]; then
    mkdir -p "$(dirname "$MODS_CONFIG")"
    cat > "$MODS_CONFIG" << 'MODSEOF'
# Mods configuration - Pluribus defaults
default-model: ollama
apis:
  ollama:
    base-url: http://localhost:11434/v1
    models:
      llama3.2:
        max-input-chars: 24000
MODSEOF
    echo "      Created $MODS_CONFIG"
else
    echo "      Config exists: $MODS_CONFIG"
fi

echo ""
echo "[✓] Local LLM installation complete."
echo "    - Ollama: $(ollama --version 2>/dev/null || echo 'installed')"
echo "    - Model: llama3.2"
echo "    - Mods: $(mods --version 2>/dev/null || echo 'check PATH')"
