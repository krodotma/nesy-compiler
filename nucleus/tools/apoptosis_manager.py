#!/usr/bin/env python3
"""
Apoptosis Manager (PAIP GC v2)
==============================
Scans for stale PAIP clones and reaps their resources.
Synchronizes with Cytoplasm registry to mark cells as 'fossilized'.
Part of the InferCell Renaissance (Phase 1, Step 5).
"""

import argparse
import time
import os
import shutil
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

DEFAULT_TTL_SECONDS = 3600  # 1 hour
CYTOPLASM_PATH = "nucleus/state/cytoplasm.json"

def run_reap(clone_id: str):
    """Invoke paip_isolation.py reap."""
    tool_path = Path(__file__).parent / "paip_isolation.py"
    subprocess.run([sys.executable, str(tool_path), "reap", "--id", clone_id], check=False)

def fossilize_cell(cell_id: str):
    """Mark cell as fossilized in Cytoplasm registry."""
    if not os.path.exists(CYTOPLASM_PATH):
        return
    
    try:
        with open(CYTOPLASM_PATH, 'r') as f:
            registry = json.load(f)
        
        for cell in registry['cells']:
            if cell['id'] == cell_id or cell_id.endswith(cell['id']):
                cell['status'] = 'fossilized'
                cell['death_iso'] = datetime.utcnow().isoformat() + "Z"
                print(f"  Fossilized {cell['id']} in registry.")
        
        with open(CYTOPLASM_PATH, 'w') as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"  Error fossilizing cell: {e}")

def scan_and_clean(dry_run: bool = False, ttl: int = DEFAULT_TTL_SECONDS):
    tmp_root = Path("/tmp")
    now = time.time()
    
    print(f"Scanning {tmp_root} for stale PAIP clones (TTL={ttl}s)...")
    
    for item in tmp_root.iterdir():
        if not item.is_dir():
            continue
        
        if item.name.startswith("pluribus_"):
            try:
                mtime = item.stat().st_mtime
                age = now - mtime
                
                if age > ttl:
                    print(f"Found stale clone: {item.name} (Age: {int(age)}s)")
                    
                    if not dry_run:
                        # 1. Reap resources (Ports/Displays)
                        run_reap(item.name)
                        
                        # 2. Fossilize in registry
                        fossilize_cell(item.name)
                        
                        # 3. Delete filesystem
                        print(f"  Removing directory: {item}")
                        shutil.rmtree(item, ignore_errors=True)
                        print("  Cleaned.")
                    else:
                        print("  [Dry Run] Would reap and delete.")
            except Exception as e:
                print(f"Error checking {item.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Apoptosis Manager")
    parser.add_argument("--dry-run", action="store_true", help="Scan only")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL_SECONDS, help="TTL in seconds")
    args = parser.parse_args()
    
    scan_and_clean(args.dry_run, args.ttl)

if __name__ == "__main__":
    main()