#!/bin/bash
# setup_retrieval_native.sh - Unblocks sqlite-vec and neural substrates (v26)
# Mandate: 200% stack optimization

set -e

REPO_ROOT="/pluribus"
VENV_DIR="$REPO_ROOT/.pluribus/venv"

echo "=== Pluribus Retrieval Setup (Native) ==="

# 1. Create/Verify Venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[+] Creating .pluribus/venv..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Install Core Neural Deps
echo "[+] Installing high-performance retrieval dependencies..."
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q sqlite-vec sentence-transformers numpy

# 3. Verify sqlite-vec binary
echo "[+] Verifying sqlite-vec integration..."
"$VENV_DIR/bin/python3" -c "import sqlite3; import sqlite_vec; conn = sqlite3.connect(':memory:'); conn.enable_load_extension(True); sqlite_vec.load(conn); print('  [SUCCESS] sqlite-vec loaded')"

# 4. Bootstrap rag_vector
echo "[+] Initializing Vector RAG schema..."
"$VENV_DIR/bin/python3" "$REPO_ROOT/nucleus/tools/rag_vector.py" init

echo "=== Setup Complete. Stack optimized for v26 Sovereignty ==="
