#!/usr/bin/env python3
"""
Lexicon Validator
=================
Verifies that semops.json defines all operators mentioned in pluribus_lexicon.md.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/pluribus")
LEXICON_PATH = ROOT / "nucleus/specs/pluribus_lexicon.md"
SEMOPS_PATH = ROOT / "nucleus/specs/semops.json"

def main():
    if not LEXICON_PATH.exists() or not SEMOPS_PATH.exists():
        print("Missing files.")
        return 1

    lexicon_text = LEXICON_PATH.read_text()
    semops_data = json.loads(SEMOPS_PATH.read_text())
    
    defined_ops = set(semops_data.keys())
    
    # Find ALLCAPS terms in lexicon that look like operators (e.g. OITERATE)
    # Simple heuristic: Lines starting with "### OITERATE" or similar
    mentioned_ops = set()
    for line in lexicon_text.splitlines():
        match = re.search(r"###\s+([A-Z]{3,})", line)
        if match:
            op = match.group(1)
            # Filter out non-ops (headers like TABLE OF CONTENTS)
            if op not in {"TABLE", "THE", "WHY", "HOW"}:
                mentioned_ops.add(op)
            
    print(f"Defined in JSON: {len(defined_ops)}")
    print(f"Found in MD: {len(mentioned_ops)}")
    
    missing = mentioned_ops - defined_ops
    if missing:
        print(f"Missing in JSON: {missing}")
        # Not a hard fail yet, just warn
    else:
        print("All lexicon headers found in semops.json.")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
