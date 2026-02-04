#!/usr/bin/env python3
"""
AXIOM Operator - Semantic operator for axiom DSL management.

Parses, registers, and checks formal axioms per DKIN v28.
Integrates with bus for evidence emission and motif tracking.

Usage:
    python3 nucleus/tools/axiom_operator.py --parse <file>
    python3 nucleus/tools/axiom_operator.py --registry
    python3 nucleus/tools/axiom_operator.py --check <axiom_name>
    python3 nucleus/tools/axiom_operator.py --load <file>

Authors: codex, claude
Protocol: DKIN v28
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add nucleus to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nucleus.tools.axiom_parser import parse_program, parse_axiom, serialize_ast, ParseError

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BUS_DIR = os.environ.get('PLURIBUS_BUS_DIR', '/pluribus/.pluribus/bus')
DEFAULT_ACTOR = os.environ.get('PLURIBUS_ACTOR', 'axiom_operator')
DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / 'specs' / 'axioms' / 'core.axiom'

# =============================================================================
# Registry
# =============================================================================

class AxiomRegistry:
    """In-memory registry of loaded axioms."""

    def __init__(self):
        self.axioms: Dict[str, Any] = {}
        self.definitions: Dict[str, Any] = {}
        self.rules: Dict[str, Any] = {}
        self.dits: Dict[str, Any] = {}
        self.sources: Dict[str, str] = {}  # name -> source file

    def load_file(self, path: Path) -> int:
        """Load axioms from a .axiom file. Returns count loaded."""
        with open(path) as f:
            source = f.read()

        try:
            program = parse_program(source)
        except ParseError as e:
            raise ValueError(f"Parse error in {path}: {e}")

        count = 0
        for decl in program.get('declarations', []):
            dtype = decl.get('type')
            name = decl.get('name')

            if dtype == 'Axiom':
                self.axioms[name] = decl
                self.sources[name] = str(path)
                count += 1
            elif dtype == 'Definition':
                self.definitions[name] = decl
                self.sources[name] = str(path)
                count += 1
            elif dtype == 'Rule':
                self.rules[name] = decl
                self.sources[name] = str(path)
                count += 1
            elif dtype == 'DiTS':
                self.dits[name] = decl
                self.sources[name] = str(path)
                count += 1

        return count

    def get(self, name: str) -> Optional[Any]:
        """Get axiom/def/rule/dits by name."""
        return (self.axioms.get(name) or
                self.definitions.get(name) or
                self.rules.get(name) or
                self.dits.get(name))

    def list_all(self) -> Dict[str, List[str]]:
        """List all registered items by type."""
        return {
            'axioms': list(self.axioms.keys()),
            'definitions': list(self.definitions.keys()),
            'rules': list(self.rules.keys()),
            'dits': list(self.dits.keys()),
        }

    def count(self) -> int:
        """Total count of registered items."""
        return len(self.axioms) + len(self.definitions) + len(self.rules) + len(self.dits)

# Global registry
_registry = AxiomRegistry()

# =============================================================================
# Bus Integration
# =============================================================================

def emit_bus_event(topic: str, kind: str, data: Dict, bus_dir: str = DEFAULT_BUS_DIR, actor: str = DEFAULT_ACTOR):
    """Emit event to Pluribus bus."""
    event = {
        "id": uuid.uuid4().hex,
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": "info",
        "actor": actor,
        "data": data,
    }

    bus_path = Path(bus_dir) / 'events.ndjson'
    with open(bus_path, 'a') as f:
        f.write(json.dumps(event, separators=(',', ':')) + '\n')

    return event['id']

# =============================================================================
# Commands
# =============================================================================

def cmd_parse(file_path: str, emit_bus: bool = False) -> Dict:
    """Parse an axiom file and return AST."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path) as f:
        source = f.read()

    ast = parse_program(source)

    if emit_bus:
        emit_bus_event('axiom.ast.created', 'artifact', {
            'path': str(path),
            'declaration_count': len(ast.get('declarations', [])),
        })

    return ast


def cmd_load(file_path: str, emit_bus: bool = False) -> int:
    """Load axioms from file into registry."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    count = _registry.load_file(path)

    if emit_bus:
        emit_bus_event('axiom.registry.loaded', 'artifact', {
            'path': str(path),
            'count': count,
            'total': _registry.count(),
        })

    return count


def cmd_registry(emit_bus: bool = False) -> Dict:
    """Show current registry contents."""
    result = {
        'items': _registry.list_all(),
        'total': _registry.count(),
        'sources': _registry.sources,
    }

    if emit_bus:
        emit_bus_event('axiom.registry.queried', 'metric', {
            'total': result['total'],
            'axioms': len(_registry.axioms),
            'definitions': len(_registry.definitions),
            'rules': len(_registry.rules),
            'dits': len(_registry.dits),
        })

    return result


def cmd_check(axiom_name: str, emit_bus: bool = False) -> Dict:
    """Check an axiom (stub - returns binding info)."""
    item = _registry.get(axiom_name)
    if not item:
        raise ValueError(f"Axiom not found: {axiom_name}")

    # For now, just return the axiom info with binding
    # Real enforcement would invoke the bound tool
    binding = item.get('binding') or {}
    result = {
        'name': axiom_name,
        'type': item.get('type'),
        'binding': binding,
        'status': 'info',  # Would be 'pass' or 'fail' with real enforcement
        'message': f"Axiom '{axiom_name}' registered with binding: {binding}",
    }

    if emit_bus:
        topic = binding.get('topic', 'axiom.check.info')
        emit_bus_event(topic, 'metric', {
            'axiom': axiom_name,
            'status': result['status'],
            'binding': binding,
        })

    return result


def cmd_show(axiom_name: str) -> Dict:
    """Show full AST for an axiom."""
    item = _registry.get(axiom_name)
    if not item:
        raise ValueError(f"Axiom not found: {axiom_name}")
    return item

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AXIOM Operator - Formal axiom management for Pluribus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --parse nucleus/specs/axioms/core.axiom
  %(prog)s --load nucleus/specs/axioms/core.axiom --emit-bus
  %(prog)s --registry
  %(prog)s --check auom_lawfulness
  %(prog)s --show code_is_vapor
        """
    )

    parser.add_argument('--parse', metavar='FILE', help='Parse axiom file and show AST')
    parser.add_argument('--load', metavar='FILE', help='Load axioms into registry')
    parser.add_argument('--registry', action='store_true', help='Show registry contents')
    parser.add_argument('--check', metavar='NAME', help='Check an axiom')
    parser.add_argument('--show', metavar='NAME', help='Show axiom AST')
    parser.add_argument('--emit-bus', action='store_true', help='Emit events to bus')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--load-core', action='store_true', help='Load core axioms on startup')

    args = parser.parse_args()

    # Load core axioms if requested or if checking/showing
    if args.load_core or args.check or args.show:
        if DEFAULT_REGISTRY_PATH.exists():
            try:
                _registry.load_file(DEFAULT_REGISTRY_PATH)
            except Exception as e:
                print(f"Warning: Could not load core axioms: {e}", file=sys.stderr)

    try:
        if args.parse:
            result = cmd_parse(args.parse, args.emit_bus)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(serialize_ast(result))

        elif args.load:
            count = cmd_load(args.load, args.emit_bus)
            result = {'loaded': count, 'total': _registry.count()}
            if args.json:
                print(json.dumps(result))
            else:
                print(f"Loaded {count} declarations. Registry total: {_registry.count()}")

        elif args.registry:
            result = cmd_registry(args.emit_bus)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("AXIOM REGISTRY")
                print("=" * 50)
                for dtype, names in result['items'].items():
                    if names:
                        print(f"\n{dtype.upper()} ({len(names)}):")
                        for name in sorted(names):
                            src = _registry.sources.get(name, '?')
                            print(f"  - {name} [{src}]")
                print(f"\nTotal: {result['total']} declarations")

        elif args.check:
            result = cmd_check(args.check, args.emit_bus)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"[{result['status'].upper()}] {result['name']}: {result['message']}")

        elif args.show:
            result = cmd_show(args.show)
            print(json.dumps(result, indent=2))

        else:
            parser.print_help()

    except Exception as e:
        if args.json:
            print(json.dumps({'error': str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
