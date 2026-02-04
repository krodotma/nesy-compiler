#!/usr/bin/env python3
"""
verify_inception.py - E2E Verification of Phase 3 & 4

Automates the inception of all 16 domains in the Etymon Manifest.
Verifies:
1. Manifest loading
2. Content creation (dummy seed content)
3. PBPORTAL inception (rhizome + bus)
4. CMP scoring
5. Status retrieval

Usage: python3 verify_inception.py
"""

import json
import os
import sys
import time
from pathlib import Path
from nucleus.tools.pbportal_operator import PortalOperator

MANIFEST_PATH = Path("nucleus/specs/etymon_manifest.json")
TEMP_DIR = Path("/tmp/pluribus_verify")

def main():
    print("🚀 Starting WORKTREEPIVOT Final Verification...")
    
    # 1. Load Manifest
    if not MANIFEST_PATH.exists():
        print(f"❌ Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)
        
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
        domains = manifest.get("etymons", [])
        print(f"✅ Loaded {len(domains)} domains from manifest")
    except Exception as e:
        print(f"❌ Failed to load manifest: {e}")
        sys.exit(1)

    # Setup temp dir
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    operator = PortalOperator()
    success_count = 0
    
    # 2. Incept each domain
    print("\n🔮 Incepting Domains...")
    for entry in domains:
        etymon = entry["etymon_id"]
        priority = entry.get("priority", "P?")
        desc = entry.get("description", "")
        
        # Create dummy content
        content_path = TEMP_DIR / f"{etymon}.txt"
        content_path.write_text(f"SEED CONTENT FOR {etymon}\n{desc}\nTimestamp: {time.time()}")
        
        try:
            result = operator.incept(
                etymon=etymon,
                content_path=content_path,
                tags=["verification", "phase6", priority],
                notify_agents=True  # Test HEXIS too
            )
            
            # Verify result
            if result.portal_id and result.cmp_score is not None:
                print(f"  ✅ {etymon:<18} -> {result.portal_id} (CMP: {result.cmp_score:.2f})")
                success_count += 1
            else:
                print(f"  ⚠️ {etymon:<18} -> Inception returned incomplete result")
                
        except Exception as e:
            print(f"  ❌ {etymon:<18} -> Error: {e}")

    # 3. Final Check
    print("\n🔍 Verifying Ledger...")
    portals = operator.list_portals(limit=100)
    print(f"✅ Ledger contains {len(portals)} total portals")
    
    if success_count == len(domains):
        print(f"\n✨ SUCCESS: All {success_count} domains successfully incepted!")
        sys.exit(0)
    else:
        print(f"\n⚠️ WARNING: Only {success_count}/{len(domains)} incepted.")
        sys.exit(1)

if __name__ == "__main__":
    main()
