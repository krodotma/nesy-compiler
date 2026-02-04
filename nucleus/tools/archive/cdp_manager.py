#!/usr/bin/env python3
"""
CDP Manager: Cognitive Design Pattern Registry Tool
===================================================
Manages the lifecycle, validation, and retrieval of CDPs.

Usage:
  python3 cdp_manager.py list
  python3 cdp_manager.py validate
  python3 cdp_manager.py get CDP-001
"""
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.dont_write_bytecode = True

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "specs" / "cdp_registry.json"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "specs" / "cdp_schema.json"

def load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"registry_version": "1.0", "patterns": []}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except Exception as e:
        print(f"Error loading registry: {e}", file=sys.stderr)
        sys.exit(1)

def list_patterns(registry: Dict[str, Any]):
    print(f"CDP Registry v{registry.get('registry_version', '?')}")
    print("---------------------------------------------------")
    for p in registry.get("patterns", []):
        print(f"[{p['id']}] {p['name']}")
        print(f"  {p['description']}")
        print(f"  Metrics: {p.get('metrics', {})}")
        print("")

def get_pattern(registry: Dict[str, Any], cdp_id: str):
    for p in registry.get("patterns", []):
        if p['id'] == cdp_id:
            print(json.dumps(p, indent=2))
            return
    print(f"Pattern {cdp_id} not found.", file=sys.stderr)
    sys.exit(1)

def validate_registry(registry: Dict[str, Any]):
    # Simple schema validation (in absence of full jsonschema lib dependency for now)
    # Checks for required fields defined in cdp_schema.json
    required_fields = ["id", "name", "description", "sextet_mapping", "inputs", "outputs", "constraints"]
    sextet_fields = ["object", "process", "type", "shape", "symbol", "observer"]
    
    errors = []
    for i, p in enumerate(registry.get("patterns", [])):
        pid = p.get('id', f'Index {i}')
        for field in required_fields:
            if field not in p:
                errors.append(f"{pid} missing required field: {field}")
        
        if "sextet_mapping" in p:
            for sf in sextet_fields:
                if sf not in p["sextet_mapping"]:
                    errors.append(f"{pid} sextet_mapping missing: {sf}")

    if errors:
        print("Validation Failed:")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("Registry is valid.")

def main():
    parser = argparse.ArgumentParser(description="CDP Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("list", help="List all CDPs")
    subparsers.add_parser("validate", help="Validate registry against schema")
    
    get_p = subparsers.add_parser("get", help="Get CDP details")
    get_p.add_argument("id", help="CDP ID (e.g. CDP-001)")

    args = parser.parse_args()
    
    registry = load_registry()
    
    if args.command == "list":
        list_patterns(registry)
    elif args.command == "validate":
        validate_registry(registry)
    elif args.command == "get":
        get_pattern(registry, args.id)

if __name__ == "__main__":
    main()
