#!/usr/bin/env bash
set -euo pipefail

# install_sota_deps.sh
# ===================
# Recursively installs dependencies for all SOTA tools integrated into the Pluribus Membrane.
# Handles Python (requirements.txt) and Node.js (package.json).

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MEMBRANE_DIR="$ROOT_DIR/membrane"

echo "Pluribus SOTA Dependency Installer"
echo "=================================="
echo "Root: $ROOT_DIR"
echo ""

# 1. Check for Python venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[!] Warning: No active Python virtual environment detected."
    echo "    It is highly recommended to run this inside a venv."
    read -p "    Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Install Membrane Python Dependencies
echo "[+] Scanning membrane/ for Python dependencies..."
find "$MEMBRANE_DIR" -name "requirements.txt" | while read -r req; do
    rel_path=$(realpath --relative-to="$ROOT_DIR" "$req")
    echo "    -> Installing $rel_path..."
    pip install -q -r "$req"
done

# 3. Install Membrane Node.js Dependencies
echo "[+] Scanning membrane/ for Node.js dependencies..."
find "$MEMBRANE_DIR" -name "package.json" -not -path "*/node_modules/*" | while read -r pkg; do
    pkg_dir=$(dirname "$pkg")
    rel_path=$(realpath --relative-to="$ROOT_DIR" "$pkg")
    echo "    -> Installing $rel_path..."
    (cd "$pkg_dir" && npm install --silent)
done

# 4. Special cases
# Graphiti (Submodule)
if [[ -d "$MEMBRANE_DIR/graphiti" ]]; then
    echo "[+] Special: Graphiti (submodule)"
    if [[ -f "$MEMBRANE_DIR/graphiti/requirements.txt" ]]; then
        pip install -q -r "$MEMBRANE_DIR/graphiti/requirements.txt"
    fi
fi

# SQLite-vec (Native extension)
# Requires separate build/download step usually, but we check if it's already there
echo "[+] Verification: SQLite-vec"
python3 -c "import sqlite3; db=sqlite3.connect(':memory:'); try: db.enable_load_extension(True); print('    OK: Load extension supported'); except: print('    WARN: enable_load_extension NOT supported')"

echo ""
echo "[✓] SOTA Dependencies installed."
