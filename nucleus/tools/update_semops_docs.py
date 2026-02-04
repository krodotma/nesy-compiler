#!/usr/bin/env python3
"""
SemOps Documentation Updater
============================

This script automatically updates the SemOps registry documentation
whenever changes are made to the canonical semops.json file.
"""

import json
import sys
from pathlib import Path


def load_semops_registry():
    """Load the canonical semops registry."""
    registry_path = Path(__file__).parent.parent / "specs" / "semops.json"
    if not registry_path.exists():
        print(f"ERROR: SemOps registry not found at {registry_path}")
        sys.exit(1)

    with open(registry_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("operators", {})


def generate_operator_section(op_id, op_data):
    """Generate documentation section for a single operator."""
    tool_path = op_data.get("tool", "unknown")
    bus_topic = op_data.get("bus_topic", "unknown")
    description = op_data.get("description", "No description available")

    section = f"### {op_id}\n"
    section += f"- **File**: `{tool_path}`\n"
    section += f"- **Topic**: `{bus_topic}`\n"
    section += f"- **Purpose**: {description}\n\n"

    return section


def update_semops_documentation():
    """Update the SemOps documentation file."""
    operators = load_semops_registry()

    # Generate the documentation content
    content = """# Pluribus Semantic Operators (SemOps) Registry

This document provides a comprehensive reference for all semantic operators in the Pluribus system. Each operator serves a specific purpose in the multi-agent coordination and orchestration ecosystem.

## Operator Reference

"""

    # Sort operators alphabetically for consistent output
    sorted_operators = sorted(operators.items())

    for op_id, op_data in sorted_operators:
        content += generate_operator_section(op_id, op_data)

    content += """## Maintenance

This document should be updated whenever new operators are added or existing operators are modified. The canonical source of truth remains `nucleus/specs/semops.json`, but this document provides a human-readable reference for understanding the purpose and function of each operator.

## Protocol Evolution

Operators evolve with the DKIN protocol versions, with new features and capabilities added in each version. See individual protocol specification documents for version-specific details.
"""

    # Write the updated documentation
    doc_path = Path(__file__).parent.parent / "docs" / "reference" / "semops_registry.md"
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Successfully updated SemOps documentation at {doc_path}")
    print(f"Total operators documented: {len(operators)}")


if __name__ == "__main__":
    update_semops_documentation()